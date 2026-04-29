"""
Validator for semi-structured data.

Runs six quality-dimension checks against a pandas DataFrame and the profile
produced by :class:`~src.modules.semi_structured.profiler.SemiStructuredProfiler`,
then aggregates the results into a :class:`~src.models.QualityResult` using the
weights from ``dq_config.yaml``.

Semi-structured data differs from tabular data in several ways that shape the
checks implemented here:

* **Schema is optional** — fields may be absent in some records; null rates
  and unexpected-key counts matter more than dtype mismatches.
* **Values may be heterogeneous** — the same key can hold a string in one
  record and a number in the next; the accuracy check flags these columns.
* **Nesting is structural, not a flaw** — depth is only flagged when it
  exceeds a practical processing threshold (> 5 levels by default).
* **Freshness uses a wider window** — event and log data often lags further
  behind real-time than tabular operational data, so a 90-day outer boundary
  is used instead of the 60-day boundary in the structured validator.
"""

from __future__ import annotations

import datetime
import re
from typing import Any

import pandas as pd
from loguru import logger

from src.models import QualityResult
from src.utils import load_config

# ---------------------------------------------------------------------------
# Regex used by check_validity for email field validation
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Regex for checking field-name cleanliness in check_consistency
_FIELD_NAME_RE = re.compile(r"^[\w.]+$")  # allows letters, digits, _, .


class SemiStructuredValidator:
    """Validate a semi-structured DataFrame across six quality dimensions.

    Each public ``check_*`` method is self-contained: it accepts the flattened
    DataFrame (and, where needed, the profile dict from
    :class:`~src.modules.semi_structured.profiler.SemiStructuredProfiler`),
    performs its checks, and returns a ``(score, failed_checks)`` tuple.

    The :meth:`validate` method orchestrates all six checks, computes the
    weighted overall score, and returns a :class:`~src.models.QualityResult`.

    Scoring convention
    ------------------
    Every check returns a *score* in **[0.0, 100.0]** (100 = perfect) and a
    ``failed_checks`` list.  Each item in that list describes one failing
    assertion::

        {
            "check":    str,   # short identifier, e.g. "null_rate"
            "column":   str,   # field name, or "dataset" for dataset-level
            "expected": str,   # threshold or target description
            "actual":   str,   # observed value
            "severity": str,   # "error" | "warning"
        }

    Configuration
    -------------
    The validator reads ``dq_config.yaml`` on construction via the *config*
    dict argument (pre-loaded by the caller).  Key settings used:

    * ``scoring_weights`` — the six dimension weights (must sum to 1.0).
    * ``thresholds.pass_score`` — minimum overall score for ``passed = True``.

    Configurable thresholds are exposed as class attributes so they can be
    overridden per-instance or in tests without changing the config file.

    Attributes:
        NULL_RATE_WARN_THRESHOLD:     Null-rate above which a field is flagged.
        TYPE_INCONSISTENCY_THRESHOLD: Fraction of minority-type values above
                                      which a field is flagged.
        UNEXPECTED_KEY_SCORE_PENALTY: Points deducted per unexpected key.
        FRESHNESS_RECENT_DAYS:        Upper bound for a perfect freshness score.
        FRESHNESS_STALE_DAYS:         Day count at which freshness hits zero.
        FRESHNESS_NO_DATE_SCORE:      Score returned when no date field exists.
        DUPLICATE_RATE_WARN_THRESHOLD: Duplicate rate above which records are
                                       flagged.
        MAX_NESTING_DEPTH:            Nesting levels above which depth is an error.
    """

    NULL_RATE_WARN_THRESHOLD:      float = 0.10
    TYPE_INCONSISTENCY_THRESHOLD:  float = 0.05
    UNEXPECTED_KEY_SCORE_PENALTY:  float = 10.0
    FRESHNESS_RECENT_DAYS:         int   = 30
    FRESHNESS_STALE_DAYS:          int   = 90
    FRESHNESS_NO_DATE_SCORE:       float = 50.0
    DUPLICATE_RATE_WARN_THRESHOLD: float = 0.01
    MAX_NESTING_DEPTH:             int   = 5

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
            f"SemiStructuredValidator initialised | "
            f"pass_score={self._pass_score} weights={self._weights}"
        )

    # ==================================================================
    # Public check methods
    # ==================================================================

    def check_completeness(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Measure the proportion of non-null values across all fields.

        The overall score is the mean non-null rate across all fields scaled to
        [0, 100].  Any field whose null rate exceeds
        :attr:`NULL_RATE_WARN_THRESHOLD` (default 10 %) is added to
        ``failed_checks``.

        Algorithm::

            per_field_completeness = 1 - null_rate   (from profile["null_rates"])
            score = mean(per_field_completeness) * 100

        Null rates are read from the pre-computed profile rather than
        recomputed from the DataFrame so the result is consistent with the
        profiler's measurement.

        Args:
            df: The flattened DataFrame produced by the profiler.
            profile: The profile dict returned by
                :class:`~src.modules.semi_structured.profiler.SemiStructuredProfiler`
                for the same data. Must contain a ``"null_rates"`` key.

        Returns:
            ``(score, failed_checks)`` where *score* is in [0, 100].

        Example::

            score, failures = validator.check_completeness(df, profile)
            # score == 92.5 → average 7.5 % of values are null across fields
        """
        logger.debug("check_completeness: starting")
        null_rates = profile.get("null_rates", {})

        if not null_rates:
            logger.warning("check_completeness: no null_rates in profile — scoring 0")
            return 0.0, []

        failed_checks: list[dict[str, Any]] = []
        completeness_values: list[float] = []

        for field, null_pct in null_rates.items():
            null_rate = null_pct / 100.0
            completeness_values.append(1.0 - null_rate)

            if null_rate > self.NULL_RATE_WARN_THRESHOLD:
                severity = "error" if null_rate > 0.50 else "warning"
                failed_checks.append({
                    "check":    "null_rate",
                    "column":   field,
                    "expected": f"null_rate <= {self.NULL_RATE_WARN_THRESHOLD:.0%}",
                    "actual":   f"{null_rate:.2%}",
                    "severity": severity,
                })
                logger.debug(
                    f"  null_rate [{severity}]: field='{field}' rate={null_rate:.2%}"
                )

        score = round(
            sum(completeness_values) / len(completeness_values) * 100, 2
        )
        logger.info(
            f"check_completeness: score={score} failures={len(failed_checks)}"
        )
        return score, failed_checks

    def check_accuracy(
        self,
        df: pd.DataFrame,
    ) -> tuple[float, list[dict[str, Any]]]:
        """Detect fields whose values contain more than one Python type.

        For each field the *majority type* is the ``type()`` that appears most
        frequently among non-null values.  If the fraction of values whose type
        differs from the majority exceeds :attr:`TYPE_INCONSISTENCY_THRESHOLD`
        (default 5 %), the field is flagged.

        Score::

            score = (passing_fields / total_fields) * 100

        Fields with zero non-null values are treated as passing (type cannot
        be assessed).  An empty DataFrame scores 100.

        Args:
            df: The flattened DataFrame to inspect.

        Returns:
            ``(score, failed_checks)`` where *score* is in [0, 100].

        Example::

            # A column containing [1, "two", 3, 4] has 25 % minority-type values.
            # With threshold 5 %, it is flagged as a warning.
        """
        logger.debug("check_accuracy: starting")
        failed_checks: list[dict[str, Any]] = []

        if df.empty or len(df.columns) == 0:
            logger.warning("check_accuracy: empty DataFrame — scoring 100")
            return 100.0, []

        passing = 0
        total   = len(df.columns)

        for col in df.columns:
            non_null = df[col].dropna()
            if non_null.empty:
                passing += 1
                continue

            type_counts: dict[type, int] = {}
            for val in non_null:
                t = type(val)
                type_counts[t] = type_counts.get(t, 0) + 1

            majority_count    = max(type_counts.values())
            minority_count    = len(non_null) - majority_count
            minority_fraction = minority_count / len(non_null)

            if minority_fraction > self.TYPE_INCONSISTENCY_THRESHOLD:
                majority_type = max(type_counts, key=type_counts.__getitem__)
                minority_types = {
                    t.__name__ for t, c in type_counts.items() if t != majority_type
                }
                severity = (
                    "error" if minority_fraction > 0.25 else "warning"
                )
                failed_checks.append({
                    "check":    "type_inconsistency",
                    "column":   col,
                    "expected": (
                        f"<= {self.TYPE_INCONSISTENCY_THRESHOLD:.0%} minority-type values"
                    ),
                    "actual": (
                        f"{minority_fraction:.1%} of values are "
                        f"{', '.join(sorted(minority_types))} "
                        f"(majority: {majority_type.__name__})"
                    ),
                    "severity": severity,
                })
                logger.debug(
                    f"  type_inconsistency [{severity}]: field='{col}' "
                    f"minority={minority_fraction:.1%}"
                )
            else:
                passing += 1

        score = round(passing / total * 100, 2)
        logger.info(f"check_accuracy: score={score} failures={len(failed_checks)}")
        return score, failed_checks

    def check_consistency(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Check for unexpected keys and invalid field names.

        Two sub-checks:

        1. **Unexpected keys** — field names listed in ``profile["unexpected_keys"]``
           (present in < 10 % of records) are flagged as warnings.  Each one
           deducts :attr:`UNEXPECTED_KEY_SCORE_PENALTY` points from 100.
        2. **Invalid field names** — flattened field names that contain spaces
           or characters other than letters, digits, underscores, and dots are
           flagged as errors.

        Score::

            score = max(0, 100 - len(unexpected_keys) * UNEXPECTED_KEY_SCORE_PENALTY)

        Invalid field names incur an additional 10-point penalty each, floored
        at 0.

        Args:
            df: The flattened DataFrame.
            profile: The profile dict. Must contain ``"unexpected_keys"``.

        Returns:
            ``(score, failed_checks)`` where *score* is in [0, 100].
        """
        logger.debug("check_consistency: starting")
        failed_checks: list[dict[str, Any]] = []
        penalty = 0.0

        # --- unexpected keys ---
        unexpected_keys: list[str] = profile.get("unexpected_keys", [])
        for key in unexpected_keys:
            penalty += self.UNEXPECTED_KEY_SCORE_PENALTY
            failed_checks.append({
                "check":    "unexpected_key",
                "column":   key,
                "expected": "field present in >= 10% of records",
                "actual":   "field present in < 10% of records",
                "severity": "warning",
            })
            logger.debug(f"  unexpected_key [warning]: field='{key}'")

        # --- invalid field names ---
        for col in df.columns:
            if not _FIELD_NAME_RE.fullmatch(col):
                penalty += self.UNEXPECTED_KEY_SCORE_PENALTY
                invalid_chars = set(re.findall(r"[^\w.]", col))
                failed_checks.append({
                    "check":    "invalid_field_name",
                    "column":   col,
                    "expected": "field name with only letters, digits, _ or .",
                    "actual": (
                        f"contains invalid character(s): "
                        f"{', '.join(repr(c) for c in sorted(invalid_chars))}"
                    ),
                    "severity": "error",
                })
                logger.debug(
                    f"  invalid_field_name [error]: field='{col}' "
                    f"chars={invalid_chars}"
                )

        score = max(0.0, round(100.0 - penalty, 2))
        logger.info(
            f"check_consistency: score={score} failures={len(failed_checks)}"
        )
        return score, failed_checks

    def check_freshness(
        self,
        df: pd.DataFrame,
    ) -> tuple[float, list[dict[str, Any]]]:
        """Assess data recency by inspecting date and timestamp fields.

        Identifies date-like columns in two ways:

        1. Columns already of ``datetime64`` dtype.
        2. ``object`` columns whose name contains a freshness keyword
           (``timestamp``, ``ts``, ``date``, ``created``, ``updated``,
           ``modified``, ``time``) and whose non-null values parse as
           dates at a rate ≥ 80 %.

        For each date column the most recent (``max``) value is compared to
        today.  The staleness score uses a two-zone model:

        * **≤ FRESHNESS_RECENT_DAYS (30)** — score = 100.
        * **≥ FRESHNESS_STALE_DAYS (90)** — score = 0.
        * **Between 30 and 90 days** — linear decay from 100 to 0.

        If no date fields are found, returns :attr:`FRESHNESS_NO_DATE_SCORE`
        (50) rather than 100, because absent timestamps in semi-structured data
        is itself a quality concern.

        Args:
            df: The flattened DataFrame to inspect.

        Returns:
            ``(score, failed_checks)`` where *score* is in [0, 100].
        """
        logger.debug("check_freshness: starting")
        failed_checks: list[dict[str, Any]] = []
        today         = datetime.date.today()
        date_columns  = self._find_date_columns(df)

        if not date_columns:
            logger.warning(
                "check_freshness: no date/timestamp fields found — "
                f"scoring {self.FRESHNESS_NO_DATE_SCORE}"
            )
            failed_checks.append({
                "check":    "no_date_fields",
                "column":   "dataset",
                "expected": "at least one date or timestamp field",
                "actual":   "none found",
                "severity": "warning",
            })
            return self.FRESHNESS_NO_DATE_SCORE, failed_checks

        max_staleness_days = 0

        for col, parsed_series in date_columns.items():
            latest = parsed_series.max()
            if pd.isna(latest):
                continue
            if isinstance(latest, pd.Timestamp):
                latest = latest.date()

            staleness = (today - latest).days

            if staleness > self.FRESHNESS_RECENT_DAYS:
                max_staleness_days = max(max_staleness_days, staleness)
                severity = (
                    "error" if staleness >= self.FRESHNESS_STALE_DAYS else "warning"
                )
                failed_checks.append({
                    "check":    "stale_data",
                    "column":   col,
                    "expected": (
                        f"most recent value within {self.FRESHNESS_RECENT_DAYS} days"
                    ),
                    "actual": (
                        f"most recent value is {latest} ({staleness} days ago)"
                    ),
                    "severity": severity,
                })
                logger.debug(
                    f"  stale_data [{severity}]: field='{col}' age={staleness}d"
                )

        if max_staleness_days <= self.FRESHNESS_RECENT_DAYS:
            score = 100.0
        elif max_staleness_days >= self.FRESHNESS_STALE_DAYS:
            score = 0.0
        else:
            decay_range = self.FRESHNESS_STALE_DAYS - self.FRESHNESS_RECENT_DAYS
            elapsed     = max_staleness_days - self.FRESHNESS_RECENT_DAYS
            score       = round(100.0 - (elapsed / decay_range) * 100.0, 2)

        logger.info(f"check_freshness: score={score} failures={len(failed_checks)}")
        return score, failed_checks

    def check_uniqueness(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Measure the proportion of duplicate records in the dataset.

        Reads ``duplicate_record_count`` from the pre-computed profile and
        computes the duplicate rate as a fraction of total records.

        Score::

            duplicate_rate = duplicate_record_count / record_count
            score = (1 - duplicate_rate) * 100

        Args:
            df: The flattened DataFrame.
            profile: The profile dict. Must contain ``"duplicate_record_count"``
                and ``"record_count"``.

        Returns:
            ``(score, failed_checks)`` where *score* is in [0, 100].
        """
        logger.debug("check_uniqueness: starting")
        failed_checks: list[dict[str, Any]] = []

        record_count    = int(profile.get("record_count", len(df)))
        duplicate_count = int(profile.get("duplicate_record_count", 0))

        if record_count == 0:
            logger.warning("check_uniqueness: empty dataset — scoring 100")
            return 100.0, []

        duplicate_rate = duplicate_count / record_count

        if duplicate_rate > self.DUPLICATE_RATE_WARN_THRESHOLD:
            severity = "error" if duplicate_rate > 0.10 else "warning"
            failed_checks.append({
                "check":    "duplicate_records",
                "column":   "dataset",
                "expected": (
                    f"duplicate_rate <= {self.DUPLICATE_RATE_WARN_THRESHOLD:.0%}"
                ),
                "actual":   f"{duplicate_rate:.2%} ({duplicate_count} record(s))",
                "severity": severity,
            })
            logger.debug(
                f"  duplicate_records [{severity}]: "
                f"count={duplicate_count} rate={duplicate_rate:.2%}"
            )

        score = round(float((1.0 - duplicate_rate) * 100), 2)
        logger.info(f"check_uniqueness: score={score} failures={len(failed_checks)}")
        return score, failed_checks

    def check_validity(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Check structural and content validity of semi-structured fields.

        Three sub-checks, each evaluated only when applicable fields exist:

        1. **Nesting depth** — flags the dataset as an error if
           ``profile["max_nesting_depth"]`` exceeds :attr:`MAX_NESTING_DEPTH`
           (default 5).  Always applicable.
        2. **URL / link fields** — any flattened column whose name ends with
           ``_url`` or ``_link`` is expected to contain values starting with
           ``http://`` or ``https://``.  Non-conforming non-null values are
           flagged.
        3. **Email fields** — any column whose name ends with ``_email``
           (case-insensitive) is expected to match the pattern
           ``<local>@<domain>.<tld>``.  Non-conforming non-null values are
           flagged.

        Score::

            score = (passing_checks / applicable_checks) * 100

        Returns 100 when no applicable checks exist (no URL/email fields and
        depth ≤ threshold).

        Args:
            df: The flattened DataFrame.
            profile: The profile dict. Must contain ``"max_nesting_depth"``.

        Returns:
            ``(score, failed_checks)`` where *score* is in [0, 100].
        """
        logger.debug("check_validity: starting")
        failed_checks:    list[dict[str, Any]] = []
        applicable_checks = 0
        passing_checks    = 0

        # --- nesting depth ---
        applicable_checks += 1
        max_depth = int(profile.get("max_nesting_depth", 0))
        if max_depth > self.MAX_NESTING_DEPTH:
            failed_checks.append({
                "check":    "excessive_nesting_depth",
                "column":   "dataset",
                "expected": f"max_nesting_depth <= {self.MAX_NESTING_DEPTH}",
                "actual":   f"max_nesting_depth = {max_depth}",
                "severity": "error",
            })
            logger.debug(
                f"  excessive_nesting_depth [error]: depth={max_depth}"
            )
        else:
            passing_checks += 1

        # --- URL / link fields ---
        url_cols = [
            col for col in df.columns
            if col.lower().endswith(("_url", "_link"))
        ]
        for col in url_cols:
            applicable_checks += 1
            non_null   = df[col].dropna().astype(str)
            bad_values = non_null[
                ~non_null.str.startswith(("http://", "https://"))
            ]
            if bad_values.empty:
                passing_checks += 1
            else:
                n_bad = len(bad_values)
                failed_checks.append({
                    "check":    "invalid_url",
                    "column":   col,
                    "expected": "values starting with http:// or https://",
                    "actual":   f"{n_bad} value(s) do not start with http",
                    "severity": "warning" if n_bad / max(len(non_null), 1) < 0.25 else "error",
                })
                logger.debug(
                    f"  invalid_url [warning]: field='{col}' bad_count={n_bad}"
                )

        # --- email fields ---
        email_cols = [
            col for col in df.columns
            if col.lower().endswith("_email")
        ]
        for col in email_cols:
            applicable_checks += 1
            non_null    = df[col].dropna().astype(str)
            bad_values  = non_null[~non_null.apply(lambda v: bool(_EMAIL_RE.match(v)))]
            if bad_values.empty:
                passing_checks += 1
            else:
                n_bad = len(bad_values)
                failed_checks.append({
                    "check":    "invalid_email",
                    "column":   col,
                    "expected": "values matching <local>@<domain>.<tld>",
                    "actual":   f"{n_bad} value(s) do not match email pattern",
                    "severity": "warning" if n_bad / max(len(non_null), 1) < 0.25 else "error",
                })
                logger.debug(
                    f"  invalid_email: field='{col}' bad_count={n_bad}"
                )

        score = (
            round(passing_checks / applicable_checks * 100, 2)
            if applicable_checks > 0
            else 100.0
        )
        logger.info(f"check_validity: score={score} failures={len(failed_checks)}")
        return score, failed_checks

    # ==================================================================
    # Orchestrating validate() method
    # ==================================================================

    def validate(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
        dataset_name: str,
    ) -> QualityResult:
        """Run all six quality checks and return a populated QualityResult.

        Executes each ``check_*`` method, collects dimension scores and failed
        checks, computes the weighted overall score from ``dq_config.yaml``
        weights, and sets ``QualityResult.passed`` when the overall score meets
        ``thresholds.pass_score``.

        If any individual check raises an unexpected exception, that dimension
        scores 0.0 and an error entry is added to ``failed_checks`` so the
        rest of the pipeline can continue.

        Args:
            df: The flattened DataFrame produced by
                :class:`~src.modules.semi_structured.profiler.SemiStructuredProfiler`.
            profile: The profile dict for the same data. Must contain at least
                ``"null_rates"``, ``"unexpected_keys"``,
                ``"duplicate_record_count"``, ``"record_count"``, and
                ``"max_nesting_depth"``.
            dataset_name: Identifier echoed into the returned
                :class:`~src.models.QualityResult`.

        Returns:
            A fully populated :class:`~src.models.QualityResult` with:

            * ``data_type = "semi_structured"``
            * ``dimensions`` — one score per quality dimension in [0, 100]
            * ``overall_score`` — weighted sum of dimension scores
            * ``failed_checks`` — merged list from all six checks
            * ``passed`` — ``True`` iff ``overall_score >= pass_score``
            * ``metadata`` — record count, field count, max nesting depth

        Example::

            from src.utils import load_config
            from src.modules.semi_structured.profiler import SemiStructuredProfiler
            from src.modules.semi_structured.validator import SemiStructuredValidator

            config    = load_config()
            profiler  = SemiStructuredProfiler(config)
            validator = SemiStructuredValidator(config)

            data, fmt = profiler.load_data("data/raw/events.jsonl")
            df        = profiler.flatten_to_dataframe(data)
            profile   = profiler.profile(df, data, dataset_name="events")
            result    = validator.validate(df, profile, dataset_name="events")

            print(result.overall_score)
            print(result.passed)
        """
        logger.info(
            f"validate: starting for dataset='{dataset_name}'"
        )

        # Map each dimension name to its check callable and argument list.
        checks: dict[str, tuple] = {
            "completeness": (self.check_completeness, (df, profile)),
            "accuracy":     (self.check_accuracy,     (df,)),
            "consistency":  (self.check_consistency,  (df, profile)),
            "freshness":    (self.check_freshness,    (df,)),
            "uniqueness":   (self.check_uniqueness,   (df, profile)),
            "validity":     (self.check_validity,     (df, profile)),
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
                    "column":   "dataset",
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
            data_type     = "semi_structured",
            overall_score = overall_score,
            dimensions    = dimensions,
            failed_checks = failed_checks,
            passed        = passed,
            metadata      = {
                "record_count":      int(profile.get("record_count", len(df))),
                "field_count":       int(profile.get("field_count", len(df.columns))),
                "max_nesting_depth": int(profile.get("max_nesting_depth", 0)),
                "format":            str(profile.get("format", "unknown")),
            },
        )

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _find_date_columns(
        self, df: pd.DataFrame
    ) -> dict[str, pd.Series]:
        """Identify date-like columns and return them as parsed Timestamp Series.

        Two detection strategies:

        1. Columns already of ``datetime64`` dtype — included directly.
        2. ``object`` columns whose name contains a freshness keyword
           (``timestamp``, ``ts``, ``date``, ``created``, ``updated``,
           ``modified``, ``time``) and where ≥ 80 % of non-null values parse
           as dates.

        The keyword check is applied before the parse attempt so that generic
        object columns such as ``"description"`` are not incorrectly classified
        as date columns if they happen to contain some parseable strings.

        Args:
            df: The flattened DataFrame to inspect.

        Returns:
            Dict mapping column name → ``pd.Series`` of ``pd.Timestamp``
            values (NaT where parsing failed).
        """
        _DATE_KEYWORDS = (
            "timestamp", "ts", "date", "created", "updated", "modified", "time"
        )
        date_cols: dict[str, pd.Series] = {}

        for col in df.columns:
            series = df[col]

            if pd.api.types.is_datetime64_any_dtype(series):
                date_cols[col] = series
                continue

            if series.dtype != object:
                continue

            col_lower = col.lower()
            if not any(kw in col_lower for kw in _DATE_KEYWORDS):
                continue

            non_null = series.dropna()
            if len(non_null) == 0:
                continue

            parsed      = pd.to_datetime(non_null, errors="coerce")
            parse_ratio = parsed.notna().sum() / len(non_null)
            if parse_ratio >= 0.80:
                date_cols[col] = pd.to_datetime(series, errors="coerce")

        return date_cols
