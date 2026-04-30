"""
Validator for text data.

Runs six quality-dimension checks against the feature dictionary produced by
:class:`~src.modules.text.extractor.TextExtractor`, then aggregates the results
into a :class:`~src.models.QualityResult` using the weights from
``dq_config.yaml``.

Text data introduces quality concerns that differ from tabular or semi-structured
data:

* **Completeness** is measured by word count rather than field presence — a
  near-empty document is functionally missing even if the file exists.
* **Accuracy** flags encoding corruption, runaway capitalisation (OCR artefacts),
  and excessive special characters (garbled content).
* **Consistency** flags documents whose language cannot be detected (garbled or
  mixed-language content) and pathological whitespace distributions.
* **Freshness** uses the file's modification timestamp rather than an embedded
  date field, with a 30-day perfect window and a 90-day stale cutoff.
* **Uniqueness** is approximated by char count + word count matching: an exact
  match across both metrics is treated as a likely duplicate.
* **Validity** enforces that the detected language is in the expected set for
  this pipeline (English, French, and four Nigerian languages) and that the
  document contains at least one sentence.
"""

from __future__ import annotations

import os
import time
from typing import Any

from loguru import logger

from src.models import QualityResult
from src.utils import load_config


class TextValidator:
    """Validate a text document's features across six quality dimensions.

    Each public ``check_*`` method accepts the feature dictionary produced by
    :class:`~src.modules.text.extractor.TextExtractor` (plus any extra inputs
    the check needs), performs its checks, and returns a
    ``(score, failed_checks)`` tuple.

    The :meth:`validate` method orchestrates all six checks, computes the
    weighted overall score, and returns a :class:`~src.models.QualityResult`.

    Scoring convention
    ------------------
    Every check returns a *score* in **[0.0, 100.0]** (100 = perfect) and a
    ``failed_checks`` list.  Each item in that list describes one failing
    assertion::

        {
            "check":    str,   # short identifier, e.g. "encoding_errors"
            "column":   str,   # field name or "document" for document-level
            "expected": str,   # threshold or target description
            "actual":   str,   # observed value
            "severity": str,   # "error" | "warning"
        }

    Configuration
    -------------
    Scoring weights and the pass-score threshold are read from *config*, which
    must be the dict returned by :func:`~src.utils.load_config`.

    Attributes:
        COMPLETENESS_FULL_WORDS:  Word-count at or above which completeness = 100.
        COMPLETENESS_WARN_WORDS:  Word-count below which a warning is emitted.
        FRESHNESS_RECENT_DAYS:    Age (days) at or below which freshness = 100.
        FRESHNESS_STALE_DAYS:     Age (days) at or above which freshness = 0.
        EXPECTED_LANGUAGES:       Set of ISO 639-1 codes accepted by validity check.
    """

    COMPLETENESS_FULL_WORDS: int   = 50
    COMPLETENESS_WARN_WORDS: int   = 20
    FRESHNESS_RECENT_DAYS:   int   = 30
    FRESHNESS_STALE_DAYS:    int   = 90
    EXPECTED_LANGUAGES: frozenset[str] = frozenset(
        {"en", "fr", "ha", "yo", "ig", "pcm", "unknown"}
    )

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise the validator with a pre-loaded configuration dict.

        Args:
            config: The pipeline configuration dictionary as returned by
                :func:`~src.utils.load_config`.  Must contain
                ``scoring_weights`` and ``thresholds`` keys.
        """
        self._config     = config
        self._weights    = config["scoring_weights"]
        self._pass_score = config["thresholds"]["pass_score"]
        logger.debug(
            f"TextValidator initialised | "
            f"pass_score={self._pass_score} weights={self._weights}"
        )

    # ==================================================================
    # Public check methods
    # ==================================================================

    def check_completeness(
        self,
        features: dict[str, Any],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Score how much readable content the document contains.

        A document is *complete* when it has enough words to be meaningful.
        The score is zero for empty documents, 100 for those with at least
        :attr:`COMPLETENESS_FULL_WORDS` (50) words, and scales linearly
        in between.

        Algorithm::

            if is_empty:
                score = 0
            elif word_count >= 50:
                score = 100
            else:
                score = (word_count / 50) * 100

        Failed checks are added when:

        * ``is_empty`` is ``True`` (severity ``"error"``).
        * ``word_count`` is less than :attr:`COMPLETENESS_WARN_WORDS` (20)
          and the document is not already flagged as empty (severity
          ``"warning"``).

        Args:
            features: The feature dict returned by
                :class:`~src.modules.text.extractor.TextExtractor`.

        Returns:
            ``(score, failed_checks)`` where *score* is in [0.0, 100.0].

        Example::

            score, failures = validator.check_completeness(features)
            # features["word_count"] == 10  →  score == 20.0
        """
        logger.debug("check_completeness: starting")
        failed_checks: list[dict[str, Any]] = []
        word_count = int(features.get("word_count", 0))
        is_empty   = bool(features.get("is_empty", True))

        if is_empty:
            failed_checks.append({
                "check":    "is_empty",
                "column":   "document",
                "expected": "word_count >= 5",
                "actual":   f"word_count = {word_count}",
                "severity": "error",
            })
            logger.debug("check_completeness: document is empty — score=0")
            return 0.0, failed_checks

        if word_count >= self.COMPLETENESS_FULL_WORDS:
            score = 100.0
        else:
            score = round(word_count / self.COMPLETENESS_FULL_WORDS * 100, 2)
            if word_count < self.COMPLETENESS_WARN_WORDS:
                failed_checks.append({
                    "check":    "low_word_count",
                    "column":   "document",
                    "expected": f"word_count >= {self.COMPLETENESS_WARN_WORDS}",
                    "actual":   f"word_count = {word_count}",
                    "severity": "warning",
                })
                logger.debug(
                    f"check_completeness: low word count ({word_count} words)"
                )

        logger.info(
            f"check_completeness: score={score} word_count={word_count} "
            f"failures={len(failed_checks)}"
        )
        return score, failed_checks

    def check_accuracy(
        self,
        features: dict[str, Any],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Detect signs of corrupted, garbled, or machine-generated noise.

        Starts from a perfect score of 100 and applies point deductions:

        * **Encoding errors** (``\\ufffd`` present): −30 points, severity
          ``"error"``.
        * **Excessive uppercase** (``uppercase_ratio > 0.5``): −20 points,
          flagged as a likely OCR or formatting issue, severity ``"warning"``.
        * **Excessive special characters** (``special_char_ratio > 0.3``):
          −20 points, severity ``"warning"``.

        The final score is floored at 0.

        Args:
            features: The feature dict returned by
                :class:`~src.modules.text.extractor.TextExtractor`.

        Returns:
            ``(score, failed_checks)`` where *score* is in [0.0, 100.0].

        Example::

            # encoding_errors=True, uppercase_ratio=0.6
            score, _ = validator.check_accuracy(features)
            # score == max(0, 100 - 30 - 20) == 50.0
        """
        logger.debug("check_accuracy: starting")
        failed_checks: list[dict[str, Any]] = []
        score = 100.0

        if features.get("encoding_errors", False):
            score -= 30.0
            failed_checks.append({
                "check":    "encoding_errors",
                "column":   "document",
                "expected": "no Unicode replacement characters (\\ufffd)",
                "actual":   "replacement character \\ufffd detected",
                "severity": "error",
            })
            logger.debug("check_accuracy: encoding errors detected (-30)")

        uppercase_ratio = float(features.get("uppercase_ratio", 0.0))
        if uppercase_ratio > 0.5:
            score -= 20.0
            failed_checks.append({
                "check":    "excessive_uppercase",
                "column":   "document",
                "expected": "uppercase_ratio <= 0.5",
                "actual":   f"uppercase_ratio = {uppercase_ratio:.4f} — possible OCR or formatting issue",
                "severity": "warning",
            })
            logger.debug(
                f"check_accuracy: excessive uppercase ({uppercase_ratio:.2%}) (-20)"
            )

        special_char_ratio = float(features.get("special_char_ratio", 0.0))
        if special_char_ratio > 0.3:
            score -= 20.0
            failed_checks.append({
                "check":    "excessive_special_chars",
                "column":   "document",
                "expected": "special_char_ratio <= 0.3",
                "actual":   f"special_char_ratio = {special_char_ratio:.4f}",
                "severity": "warning",
            })
            logger.debug(
                f"check_accuracy: excessive special chars ({special_char_ratio:.2%}) (-20)"
            )

        score = max(0.0, round(score, 2))
        logger.info(
            f"check_accuracy: score={score} failures={len(failed_checks)}"
        )
        return score, failed_checks

    def check_consistency(
        self,
        features: dict[str, Any],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Assess structural consistency of the document's text.

        Two sub-checks:

        1. **Language detectability** — if ``language`` is ``"unknown"`` the
           score is set to 50 (rather than deducted from 100) because the
           document may be too short or garbled to classify.  Severity
           ``"warning"``.
        2. **Whitespace ratio** — a ``whitespace_ratio`` above 0.4 deducts 20
           points and is flagged as excessive whitespace.  Severity
           ``"warning"``.

        The score starts at 100 (or 50 if language is unknown) and is floored
        at 0 after any whitespace deduction.

        Args:
            features: The feature dict returned by
                :class:`~src.modules.text.extractor.TextExtractor`.

        Returns:
            ``(score, failed_checks)`` where *score* is in [0.0, 100.0].

        Example::

            # language="unknown", whitespace_ratio=0.45
            score, _ = validator.check_consistency(features)
            # score == max(0, 50 - 20) == 30.0
        """
        logger.debug("check_consistency: starting")
        failed_checks: list[dict[str, Any]] = []
        score = 100.0

        language = str(features.get("language", "unknown"))
        if language == "unknown":
            score = 50.0
            failed_checks.append({
                "check":    "language_undetected",
                "column":   "document",
                "expected": "detectable language",
                "actual":   "Language could not be detected",
                "severity": "warning",
            })
            logger.debug("check_consistency: language unknown — score capped at 50")

        whitespace_ratio = float(features.get("whitespace_ratio", 0.0))
        if whitespace_ratio > 0.4:
            score = max(0.0, score - 20.0)
            failed_checks.append({
                "check":    "excessive_whitespace",
                "column":   "document",
                "expected": "whitespace_ratio <= 0.4",
                "actual":   f"whitespace_ratio = {whitespace_ratio:.4f} — Excessive whitespace detected",
                "severity": "warning",
            })
            logger.debug(
                f"check_consistency: excessive whitespace ({whitespace_ratio:.2%}) (-20)"
            )

        score = max(0.0, round(score, 2))
        logger.info(
            f"check_consistency: score={score} failures={len(failed_checks)}"
        )
        return score, failed_checks

    def check_freshness(
        self,
        source_path: str,
    ) -> tuple[float, list[dict[str, Any]]]:
        """Assess data recency using the file's last-modification timestamp.

        Uses :func:`os.path.getmtime` to get the modification time of the
        source file.  The staleness score uses a two-zone model:

        * **≤ FRESHNESS_RECENT_DAYS (30)** — score = 100.
        * **≥ FRESHNESS_STALE_DAYS (90)** — score = 0.
        * **Between 30 and 90 days** — linear decay from 100 to 0.

        Args:
            source_path: Path to the source file that was loaded by
                :meth:`~src.modules.text.extractor.TextExtractor.load_data`.
                Must exist on disk.

        Returns:
            ``(score, failed_checks)`` where *score* is in [0.0, 100.0].

        Example::

            # File last modified 45 days ago:
            #   decay_range = 90 - 30 = 60
            #   elapsed     = 45 - 30 = 15
            #   score       = 100 - (15/60)*100 = 75.0
            score, _ = validator.check_freshness("data/raw/article.txt")
        """
        logger.debug(f"check_freshness: checking '{source_path}'")
        failed_checks: list[dict[str, Any]] = []

        mtime_epoch = os.path.getmtime(source_path)
        age_seconds = time.time() - mtime_epoch
        age_days    = age_seconds / 86_400

        if age_days <= self.FRESHNESS_RECENT_DAYS:
            score = 100.0
        elif age_days >= self.FRESHNESS_STALE_DAYS:
            score = 0.0
            failed_checks.append({
                "check":    "stale_file",
                "column":   "document",
                "expected": f"file modified within {self.FRESHNESS_STALE_DAYS} days",
                "actual":   f"file last modified {age_days:.0f} days ago",
                "severity": "error",
            })
            logger.debug(
                f"check_freshness: file is stale ({age_days:.0f} days old) — score=0"
            )
        else:
            decay_range = self.FRESHNESS_STALE_DAYS - self.FRESHNESS_RECENT_DAYS
            elapsed     = age_days - self.FRESHNESS_RECENT_DAYS
            score       = round(100.0 - (elapsed / decay_range) * 100.0, 2)
            failed_checks.append({
                "check":    "aging_file",
                "column":   "document",
                "expected": f"file modified within {self.FRESHNESS_RECENT_DAYS} days",
                "actual":   f"file last modified {age_days:.0f} days ago",
                "severity": "warning",
            })
            logger.debug(
                f"check_freshness: file aging ({age_days:.0f} days old) — score={score}"
            )

        logger.info(
            f"check_freshness: score={score} age_days={age_days:.1f} "
            f"failures={len(failed_checks)}"
        )
        return score, failed_checks

    def check_uniqueness(
        self,
        features: dict[str, Any],
        all_features: list[dict[str, Any]],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Detect likely duplicate documents by comparing char and word counts.

        A document is considered a likely duplicate of another in *all_features*
        when both its ``char_count`` and ``word_count`` are identical to those of
        another entry.  The current document itself is not compared against itself
        (its entry in *all_features* is expected to be present; only *other*
        entries with the same counts are flagged).

        Score:

        * **0** if at least one likely duplicate is found.
        * **100** if no duplicates are found.

        Args:
            features: The feature dict for the document being validated.
            all_features: List of feature dicts for *all* documents in the
                current pipeline run, including *features* itself.  Pass an
                empty list when running in isolation.

        Returns:
            ``(score, failed_checks)`` where *score* is either 0.0 or 100.0.

        Example::

            # Two documents with char_count=500, word_count=80 → duplicate
            score, _ = validator.check_uniqueness(features, all_features)
            # score == 0.0
        """
        logger.debug("check_uniqueness: starting")
        failed_checks: list[dict[str, Any]] = []

        this_chars  = int(features.get("char_count", 0))
        this_words  = int(features.get("word_count", 0))
        this_name   = str(features.get("dataset_name", ""))

        # Count how many *other* documents share both metrics.
        duplicate_count = sum(
            1
            for f in all_features
            if (
                str(f.get("dataset_name", "")) != this_name
                and int(f.get("char_count", -1)) == this_chars
                and int(f.get("word_count", -1)) == this_words
            )
        )

        if duplicate_count > 0:
            failed_checks.append({
                "check":    "likely_duplicate",
                "column":   "document",
                "expected": "unique char_count and word_count combination",
                "actual": (
                    f"char_count={this_chars} and word_count={this_words} "
                    f"matches {duplicate_count} other document(s)"
                ),
                "severity": "warning",
            })
            logger.debug(
                f"check_uniqueness: likely duplicate detected "
                f"(char_count={this_chars}, word_count={this_words})"
            )
            score = 0.0
        else:
            score = 100.0

        logger.info(
            f"check_uniqueness: score={score} failures={len(failed_checks)}"
        )
        return score, failed_checks

    def check_validity(
        self,
        features: dict[str, Any],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Validate language and structural properties of the document.

        Two sub-checks, each worth 50 points:

        1. **Language** — the detected ``language`` code must appear in
           :attr:`EXPECTED_LANGUAGES` (``en``, ``fr``, ``ha``, ``yo``, ``ig``,
           ``pcm``, ``unknown``).  An unexpected code is flagged with severity
           ``"error"``.
        2. **Sentence count** — ``sentence_count`` must be ≥ 1.  A count of
           zero indicates the document could not be segmented into sentences,
           which is flagged with severity ``"warning"``.

        Score::

            score = 100 - (number_of_failing_checks * 50)

        Args:
            features: The feature dict returned by
                :class:`~src.modules.text.extractor.TextExtractor`.

        Returns:
            ``(score, failed_checks)`` where *score* is 100, 50, or 0.

        Example::

            # language="zh" (not in expected set), sentence_count=3
            score, _ = validator.check_validity(features)
            # score == 50.0  (one check fails)
        """
        logger.debug("check_validity: starting")
        failed_checks: list[dict[str, Any]] = []

        language = str(features.get("language", "unknown"))
        if language not in self.EXPECTED_LANGUAGES:
            failed_checks.append({
                "check":    "unexpected_language",
                "column":   "document",
                "expected": f"language in {sorted(self.EXPECTED_LANGUAGES)}",
                "actual":   f"detected language = '{language}'",
                "severity": "error",
            })
            logger.debug(f"check_validity: unexpected language '{language}'")

        sentence_count = int(features.get("sentence_count", 0))
        if sentence_count < 1:
            failed_checks.append({
                "check":    "no_sentences",
                "column":   "document",
                "expected": "sentence_count >= 1",
                "actual":   f"sentence_count = {sentence_count}",
                "severity": "warning",
            })
            logger.debug("check_validity: no sentences found")

        score = max(0.0, round(100.0 - len(failed_checks) * 50.0, 2))
        logger.info(
            f"check_validity: score={score} failures={len(failed_checks)}"
        )
        return score, failed_checks

    # ==================================================================
    # Orchestrating validate() method
    # ==================================================================

    def validate(
        self,
        features: dict[str, Any],
        dataset_name: str,
        source_path: str,
        all_features: list[dict[str, Any]] | None = None,
    ) -> QualityResult:
        """Run all six quality checks and return a populated QualityResult.

        Executes each ``check_*`` method in turn, collects dimension scores and
        failed checks, computes the weighted overall score from
        ``dq_config.yaml`` weights, and sets ``QualityResult.passed`` when the
        overall score meets the configured pass threshold.

        If any individual check raises an unexpected exception, that dimension
        scores 0.0 and an error entry is added to ``failed_checks`` so the
        rest of the pipeline can continue.

        Args:
            features: The feature dict produced by
                :meth:`~src.modules.text.extractor.TextExtractor.extract` for
                this document.
            dataset_name: Human-readable identifier for the dataset, echoed
                into the returned :class:`~src.models.QualityResult`.
            source_path: Path to the source file; forwarded to
                :meth:`check_freshness` for modification-time inspection and
                set on the returned ``QualityResult.source_path``.
            all_features: List of feature dicts for *all* documents in the
                current pipeline run, used by :meth:`check_uniqueness`.
                Defaults to an empty list when ``None``.

        Returns:
            A fully populated :class:`~src.models.QualityResult` with:

            * ``data_type = "text"``
            * ``dimensions`` — one score per quality dimension in [0, 100]
            * ``overall_score`` — weighted sum of dimension scores
            * ``failed_checks`` — merged list from all six checks
            * ``passed`` — ``True`` iff ``overall_score >= pass_score``
            * ``source_path`` — the value of *source_path*
            * ``metadata`` — char count, word count, language, sentence count

        Example::

            from src.utils import load_config
            from src.modules.text.extractor import TextExtractor
            from src.modules.text.validator import TextValidator

            config    = load_config()
            extractor = TextExtractor(config)
            validator = TextValidator(config)

            raw_text, _ = extractor.load_data("data/raw/article.txt")
            features    = extractor.extract(raw_text, dataset_name="article")
            result      = validator.validate(
                features, dataset_name="article",
                source_path="data/raw/article.txt",
            )
            print(result.overall_score)
            print(result.passed)
        """
        if all_features is None:
            all_features = []

        logger.info(f"validate: starting for dataset='{dataset_name}'")

        checks: dict[str, tuple] = {
            "completeness": (self.check_completeness, (features,)),
            "accuracy":     (self.check_accuracy,     (features,)),
            "consistency":  (self.check_consistency,  (features,)),
            "freshness":    (self.check_freshness,    (source_path,)),
            "uniqueness":   (self.check_uniqueness,   (features, all_features)),
            "validity":     (self.check_validity,     (features,)),
        }

        dimensions:    dict[str, float]     = {}
        failed_checks: list[dict[str, Any]] = []

        for dimension, (check_fn, args) in checks.items():
            try:
                score, failures = check_fn(*args)
            except Exception as exc:
                logger.error(
                    f"validate: check '{dimension}' raised an unexpected error "
                    f"for dataset='{dataset_name}': {exc}"
                )
                score    = 0.0
                failures = [{
                    "check":    dimension,
                    "column":   "document",
                    "expected": "check to complete without error",
                    "actual":   str(exc),
                    "severity": "error",
                }]

            dimensions[dimension] = score
            failed_checks.extend(failures)
            logger.debug(f"  {dimension}: score={score}")

        overall_score = round(
            sum(dimensions[dim] * self._weights.get(dim, 0.0) for dim in dimensions),
            2,
        )
        passed = bool(overall_score >= self._pass_score)

        logger.info(
            f"validate: complete | dataset='{dataset_name}' "
            f"overall_score={overall_score} passed={passed} "
            f"total_failures={len(failed_checks)}"
        )

        return QualityResult(
            dataset_name  = dataset_name,
            data_type     = "text",
            overall_score = overall_score,
            dimensions    = dimensions,
            failed_checks = failed_checks,
            passed        = passed,
            source_path   = source_path,
            metadata      = {
                "char_count":     int(features.get("char_count", 0)),
                "word_count":     int(features.get("word_count", 0)),
                "sentence_count": int(features.get("sentence_count", 0)),
                "language":       str(features.get("language", "unknown")),
            },
        )
