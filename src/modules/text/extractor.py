"""
Extractor for text data.

Handles loading from PDF, plain-text, email (.eml), and CSV file formats, then
computes a feature dictionary covering basic length metrics, language detection,
character-level ratios, spaCy named-entity counts, and data-quality flags.  The
resulting dict is consumed by the text validator and scorer.
"""

from __future__ import annotations

import email
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


class TextExtractor:
    """Load and extract quality-relevant features from text-bearing files.

    Supported source formats
    ------------------------
    * **PDF** (``.pdf``) — opened with PyMuPDF (``fitz``); text is extracted
      page-by-page and concatenated.
    * **Plain text** (``.txt``) — read with UTF-8, falling back to latin-1 if a
      decoding error occurs.
    * **Email** (``.eml``) — subject line and decoded body are joined into one
      text blob using :mod:`email`.
    * **CSV** (``.csv``) — read with :func:`pandas.read_csv`; all object-dtype
      (string) columns are concatenated into a single text blob.

    Typical usage::

        from src.utils import load_config

        config    = load_config()
        extractor = TextExtractor(config)

        raw_text, detected_type = extractor.load_data("data/raw/article.pdf")
        features = extractor.extract(raw_text, dataset_name="article")
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise the extractor, store pipeline config, and load the spaCy model.

        The spaCy ``en_core_web_sm`` model is loaded eagerly so that named-entity
        recognition is available on the first call to :meth:`extract`.  If the
        model is not installed, a warning is logged and ``self.nlp`` is set to
        ``None``; entity extraction will return an empty dict in that case.

        Args:
            config: The pipeline configuration dictionary as returned by
                :func:`~src.utils.load_config`.  Stored on ``self._config`` for
                future use by subclasses or format-specific threshold logic.
        """
        self._config = config

        try:
            import spacy  # noqa: PLC0415 — local import to handle optional dep
            self.nlp = spacy.load("en_core_web_sm")
            logger.debug("TextExtractor: spaCy model 'en_core_web_sm' loaded")
        except (ImportError, OSError) as exc:
            logger.warning(
                f"TextExtractor: spaCy model unavailable — entity extraction "
                f"will be skipped ({exc})"
            )
            self.nlp = None

        logger.debug("TextExtractor initialised")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_data(self, source_path: str) -> tuple[str, str]:
        """Load a text-bearing file and return its content as a single string.

        The file type is determined purely from the file extension; no content
        sniffing is performed.  Each format uses a dedicated private loader.

        Args:
            source_path: Path to the source file.  The extension determines the
                loader:

                * ``.pdf``  — PyMuPDF (``fitz``) page-by-page extraction
                * ``.txt``  — UTF-8 read with latin-1 fallback
                * ``.eml``  — Python :mod:`email` subject + body join
                * ``.csv``  — pandas string-column concatenation

        Returns:
            A tuple ``(raw_text, detected_type)`` where:

            * *raw_text* is the full extracted text as a single ``str``.
            * *detected_type* is one of ``"pdf"``, ``"txt"``, ``"eml"``,
              ``"csv"``.

        Raises:
            FileNotFoundError: If *source_path* does not exist.
            ValueError: If the file extension is not one of the four supported
                types.

        Example::

            extractor = TextExtractor(config)
            text, fmt = extractor.load_data("data/raw/newsletter.eml")
            # fmt == "eml", text contains subject + body
        """
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path.resolve()}")

        suffix = path.suffix.lower()

        loaders = {
            ".pdf": self._load_pdf,
            ".txt": self._load_txt,
            ".eml": self._load_eml,
            ".csv": self._load_csv,
        }

        if suffix not in loaders:
            raise ValueError(
                f"Unsupported file extension '{suffix}' for source '{source_path}'. "
                f"Supported extensions: {sorted(loaders.keys())}."
            )

        raw_text, detected_type = loaders[suffix](path)

        logger.info(
            f"load_data: '{path.name}' | type={detected_type} "
            f"chars={len(raw_text):,} | preview={raw_text[:100]!r}"
        )
        return raw_text, detected_type

    def extract(self, raw_text: str, dataset_name: str) -> dict[str, Any]:
        """Extract quality-relevant text features from a raw text string.

        Combines simple character- and word-level statistics with language
        detection (via ``langdetect``) and spaCy named-entity recognition to
        produce a feature dictionary that the text validator can score against
        configured thresholds.

        Args:
            raw_text: The full text content as a single string, as returned by
                :meth:`load_data`.
            dataset_name: Human-readable identifier for the dataset, echoed into
                the returned dict for traceability.

        Returns:
            A dictionary with the following keys::

                {
                    "dataset_name":        str,
                    "char_count":          int,
                    "word_count":          int,
                    "sentence_count":      int,
                    "avg_word_length":     float,
                    "language":            str,        # e.g. "en", "unknown"
                    "language_confidence": float | None,
                    "special_char_ratio":  float,      # 0.0–1.0
                    "whitespace_ratio":    float,      # 0.0–1.0
                    "uppercase_ratio":     float,      # 0.0–1.0
                    "entities":            dict[str, int],  # label → count
                    "is_empty":            bool,
                    "encoding_errors":     bool,
                    "extracted_at":        str,        # ISO-8601 UTC
                }

            Field definitions:

            * **sentence_count** — number of segments produced by splitting on
              ``.`` or newline characters (empty segments are ignored).
            * **special_char_ratio** — non-alphanumeric, non-whitespace characters
              divided by total character count; ``0.0`` when *char_count* is 0.
            * **whitespace_ratio** — whitespace characters divided by total
              character count; ``0.0`` when *char_count* is 0.
            * **uppercase_ratio** — uppercase letters divided by total letter
              count (``[A-Za-z]``); ``0.0`` when there are no letters.
            * **entities** — dict of spaCy NER label → occurrence count, e.g.
              ``{"PERSON": 3, "ORG": 1}``; empty dict when ``self.nlp`` is
              ``None``.
            * **is_empty** — ``True`` when *word_count* is less than 5.
            * **encoding_errors** — ``True`` when the Unicode replacement
              character ``\\ufffd`` (U+FFFD) is present in *raw_text*.

        Example::

            features = extractor.extract(raw_text, dataset_name="blog_post")
            print(features["language"])           # e.g. "en"
            print(features["entities"])           # e.g. {"PERSON": 2, "GPE": 1}
            print(features["is_empty"])           # False
        """
        logger.info(
            f"extract: starting | dataset='{dataset_name}' chars={len(raw_text):,}"
        )

        char_count    = len(raw_text)
        words         = raw_text.split()
        word_count    = len(words)
        sentence_count = self._count_sentences(raw_text)
        avg_word_length = (
            round(sum(len(w) for w in words) / word_count, 4) if word_count else 0.0
        )

        language, language_confidence = self._detect_language(raw_text)
        special_char_ratio  = self._special_char_ratio(raw_text, char_count)
        whitespace_ratio    = self._whitespace_ratio(raw_text, char_count)
        uppercase_ratio     = self._uppercase_ratio(raw_text)
        entities            = self._extract_entities(raw_text)

        features: dict[str, Any] = {
            "dataset_name":        dataset_name,
            "char_count":          char_count,
            "word_count":          word_count,
            "sentence_count":      sentence_count,
            "avg_word_length":     avg_word_length,
            "language":            language,
            "language_confidence": language_confidence,
            "special_char_ratio":  special_char_ratio,
            "whitespace_ratio":    whitespace_ratio,
            "uppercase_ratio":     uppercase_ratio,
            "entities":            entities,
            "is_empty":            word_count < 5,
            "encoding_errors":     "�" in raw_text,
            "extracted_at":        datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"extract: complete | dataset='{dataset_name}' "
            f"words={word_count} lang={language} entities={sum(entities.values())}"
        )
        return features

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_pdf(self, path: Path) -> tuple[str, str]:
        """Extract text from all pages of a PDF using PyMuPDF.

        Args:
            path: Path to the ``.pdf`` file.

        Returns:
            ``(text, "pdf")`` where *text* is all pages joined with a newline.

        Raises:
            ImportError: If ``fitz`` (PyMuPDF) is not installed.
        """
        import fitz  # noqa: PLC0415

        doc = fitz.open(str(path))
        pages = [page.get_text() for page in doc]
        doc.close()
        text = "\n".join(pages)
        logger.debug(f"_load_pdf: extracted {len(pages)} page(s) from '{path.name}'")
        return text, "pdf"

    def _load_txt(self, path: Path) -> tuple[str, str]:
        """Read a plain-text file with UTF-8, falling back to latin-1.

        Args:
            path: Path to the ``.txt`` file.

        Returns:
            ``(text, "txt")``.
        """
        try:
            text = path.read_text(encoding="utf-8")
            logger.debug(f"_load_txt: read '{path.name}' as UTF-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
            logger.debug(f"_load_txt: read '{path.name}' as latin-1 (UTF-8 failed)")
        return text, "txt"

    def _load_eml(self, path: Path) -> tuple[str, str]:
        """Extract subject and body text from an email file.

        Both the ``Subject`` header and the decoded body payload(s) are
        concatenated.  Only ``text/plain`` parts are extracted; HTML parts are
        ignored to avoid markup noise.

        Args:
            path: Path to the ``.eml`` file.

        Returns:
            ``(text, "eml")`` where *text* is ``"Subject: <value>\\n<body>"``.
        """
        raw_bytes = path.read_bytes()
        msg = email.message_from_bytes(raw_bytes)

        subject = msg.get("Subject", "")
        parts: list[str] = [f"Subject: {subject}"] if subject else []

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        parts.append(payload.decode(charset, errors="replace"))
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                parts.append(payload.decode(charset, errors="replace"))

        text = "\n".join(parts)
        logger.debug(f"_load_eml: extracted subject + {len(parts) - 1} body part(s) from '{path.name}'")
        return text, "eml"

    def _load_csv(self, path: Path) -> tuple[str, str]:
        """Concatenate all string columns of a CSV into a single text blob.

        Reads the CSV with :func:`pandas.read_csv` and selects columns whose
        dtype is ``object`` (i.e. string-like).  All values from those columns
        are joined with a single space.  NaN values are dropped before joining.

        Args:
            path: Path to the ``.csv`` file.

        Returns:
            ``(text, "csv")`` where *text* is the concatenated string content.
        """
        df = pd.read_csv(path)
        string_cols = df.select_dtypes(include="object").columns.tolist()
        tokens: list[str] = []
        for col in string_cols:
            tokens.extend(df[col].dropna().astype(str).tolist())
        text = " ".join(tokens)
        logger.debug(
            f"_load_csv: concatenated {len(string_cols)} string column(s) "
            f"from '{path.name}' ({len(df):,} rows)"
        )
        return text, "csv"

    # ------------------------------------------------------------------
    # Private feature helpers
    # ------------------------------------------------------------------

    def _count_sentences(self, text: str) -> int:
        """Count sentences by splitting on periods and newlines.

        Args:
            text: Raw text string.

        Returns:
            Number of non-empty segments after splitting on ``[.\\n]``.
        """
        segments = re.split(r"[.\n]", text)
        return sum(1 for s in segments if s.strip())

    def _detect_language(self, text: str) -> tuple[str, float | None]:
        """Detect the dominant language of *text* using langdetect.

        Args:
            text: Raw text string.  Should contain at least a few words for a
                reliable result.

        Returns:
            A tuple ``(language_code, confidence)`` where *language_code* is a
            BCP-47-style ISO 639-1 code (e.g. ``"en"``) and *confidence* is a
            float in ``[0.0, 1.0]``.  Returns ``("unknown", None)`` if detection
            fails or ``langdetect`` is not installed.
        """
        try:
            from langdetect import detect_langs  # noqa: PLC0415

            results = detect_langs(text)
            if results:
                top = results[0]
                return top.lang, round(top.prob, 4)
            return "unknown", None
        except Exception as exc:
            logger.debug(f"_detect_language: detection failed — {exc}")
            return "unknown", None

    def _special_char_ratio(self, text: str, char_count: int) -> float:
        """Compute the ratio of non-alphanumeric, non-whitespace characters.

        Args:
            text: Raw text string.
            char_count: Pre-computed length of *text* (avoids re-computing).

        Returns:
            Float in ``[0.0, 1.0]``.  ``0.0`` when *char_count* is 0.
        """
        if char_count == 0:
            return 0.0
        special = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
        return round(special / char_count, 6)

    def _whitespace_ratio(self, text: str, char_count: int) -> float:
        """Compute the ratio of whitespace characters to total characters.

        Args:
            text: Raw text string.
            char_count: Pre-computed length of *text*.

        Returns:
            Float in ``[0.0, 1.0]``.  ``0.0`` when *char_count* is 0.
        """
        if char_count == 0:
            return 0.0
        ws = sum(1 for ch in text if ch.isspace())
        return round(ws / char_count, 6)

    def _uppercase_ratio(self, text: str) -> float:
        """Compute the ratio of uppercase letters to all letters.

        Args:
            text: Raw text string.

        Returns:
            Float in ``[0.0, 1.0]``.  ``0.0`` when the text contains no letters.
        """
        letters = [ch for ch in text if ch.isalpha()]
        if not letters:
            return 0.0
        upper = sum(1 for ch in letters if ch.isupper())
        return round(upper / len(letters), 6)

    def _extract_entities(self, text: str) -> dict[str, int]:
        """Run spaCy NER and count occurrences of each entity label.

        Skips processing and returns an empty dict when ``self.nlp`` is
        ``None`` (i.e. the model was not available at startup) or when *text*
        is empty.

        Args:
            text: Raw text string.

        Returns:
            Dict mapping NER label (e.g. ``"PERSON"``, ``"ORG"``, ``"GPE"``)
            to the number of times that label appears across all recognised
            entities.  Returns ``{}`` when NLP is unavailable or text is empty.
        """
        if self.nlp is None or not text.strip():
            return {}

        doc = self.nlp(text)
        counts: dict[str, int] = {}
        for ent in doc.ents:
            counts[ent.label_] = counts.get(ent.label_, 0) + 1

        logger.debug(f"_extract_entities: found {sum(counts.values())} entities across {len(counts)} label(s)")
        return counts
