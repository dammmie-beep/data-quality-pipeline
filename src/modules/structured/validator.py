"""
Validator for structured (tabular) data.

Runs six quality-dimension checks against a pandas DataFrame and the profile
produced by :class:`~src.modules.structured.profiler.StructuredProfiler`,
then aggregates the results into a :class:`~src.models.QualityResult` using
the weights from ``dq_config.yaml``.
"""

from __future__ import annotations

import datetime
from typing import Any

import pandas as pd
from loguru import logger

from src.models import QualityResult
from src.utils import load_config


# ---------------------------------------------------------------------------
# Sentinel for columns without a recognised date-like dtype.
# ---------------------------------------------------------------------------
_NO_DATE = object()


class StructuredValidator:
    """Validate a structured DataFrame across six quality dimensions.

    Each public ``check_*`` method is self-contained: it receives the
    DataFrame and the profile dict from
    :class:`~src.modules.structured.profiler.StructuredProfiler`, performs its
    checks, and returns a ``(score, failed_checks)`` tuple so results can be
    inspected individually or fed straight into :meth:`validate`.

    Scoring convention
    ------------------
    Every check method returns a *score* in **[0.0, 100.0]** where 100 means
    perfect and 0 means the worst possible result.  ``failed_checks`` is a
    list of dicts, each describing one failing assertion::

        {
            "check":    str,   # short identifier, e.g. "null_rate"
            "column":   str,   # column name, or "dataset" for dataset-level checks
            "expected": Any,   # threshold or target value
            "actual":   Any,   # observed value
            "severity": str,   # "error" | "warning"
        }

    Configuration
    -------------
    The validator reads ``dq_config.yaml`` on construction.  Key settings used:

    * ``scoring_weights`` — the six dimension weights (must sum to 1.0).
    * ``thresholds.pass_score`` — minimum overall score for ``passed = True``.

    Configurable thresholds on the checks themselves are defined as class
    attributes so they can be overridden in a subclass or in tests without
    touching the config file.

    Attributes:
        NULL_RATE_ERROR_THRESHOLD:   Null rate above which a column is an error.
        NULL_RATE_WARN_THRESHOLD:    Null rate above which a column is a warning.
        DUPLICATE_RATE_ERROR_THRESHOLD: Duplicate row rate triggering an error.
        DUPLICATE_RATE_WARN_THRESHOLD:  Duplicate row rate triggering a warning.
        NUMERIC_OUTLIER_Z_THRESHOLD: Z-score magnitude considered an outlier.
        FRESHNESS_MAX_AGE_DAYS:      Maximum acceptable age of the newest date value.
        CARDINALITY_RATIO_THRESHOLD: unique_count/row_count below which a column
                                     flagged as low-cardinality for accuracy checks.
    """

    # ------------------------------------------------------------------
    # Configurable thresholds (class-level so they are easy to override)
    # ------------------------------------------------------------------
    NULL_RATE_ERROR_THRESHOLD:          float = 0.20
    NULL_RATE_WARN_THRESHOLD:           float = 0.05
    DUPLICATE_RATE_ERROR_THRESHOLD:     float = 0.10
    DUPLICATE_RATE_WARN_THRESHOLD:      float = 0.01
    NUMERIC_OUTLIER_Z_THRESHOLD:        float = 3.0
    FRESHNESS_MAX_AGE_DAYS:             int   = 30
    CARDINALITY_RATIO_THRESHOLD:        float = 0.01

    def __init__(self, config_path: str = "dq_config.yaml") -> None:
        """Initialise the validator and load pipeline configuration.

        Args:
            config_path: Path to the YAML configuration file.  Defaults to
                ``dq_config.yaml`` in the current working directory.
        """
        self._config      = load_config(config_path)
        self._weights     = self._config["scoring_weights"]
        self._pass_score  = self._config["thresholds"]["pass_score"]
        logger.debug(
            f"StructuredValidator initialised | pass_score={self._pass_score} "
            f"weights={self._weights}"
        )

    # ==================================================================
    # Public check methods
    # ==================================================================

    def check_completeness(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Measure the proportion of non-null values across all columns.

        Penalises each column whose null rate exceeds the warn or error
        threshold.  The overall score is the mean non-null rate across all
        columns scaled to [0, 100].

        Algorithm::

            per_column_completeness = 1 - null_rate
            score = mean(per_column_completeness) * 100

        Args:
            df: The DataFrame under assessment.
            profile: The profile dict returned by
                :class:`~src.modules.structured.profiler.StructuredProfiler`.

        Returns:
            A tuple ``(score, failed_checks)`` where *score* is in [0, 100]
            and *failed_checks* lists every column whose null rate exceeded
            :attr:`NULL_RATE_WARN_THRESHOLD`.

        Example::

            score, failures = validator.check_completeness(df, profile)
            # score == 95.3   →  4.7 % of values are null on average
        """
        logger.debug("check_completeness: starting")
        columns_profile = profile.get("columns", {})
        if not columns_profile:
            logger.warning("check_completeness: no column profiles found — scoring 0")
            return 0.0, []

        failed_checks: list[dict[str, Any]] = []
        completeness_scores: list[float] = []

        for col, col_profile in columns_profile.items():
            null_rate = col_profile.get("null_rate", 0.0)
            completeness_scores.append(1.0 - null_rate)

            if null_rate > self.NULL_RATE_ERROR_THRESHOLD:
                severity = "error"
            elif null_rate > self.NULL_RATE_WARN_THRESHOLD:
                severity = "warning"
            else:
                continue

            failed_checks.append({
                "check":    "null_rate",
                "column":   col,
                "expected": f"<= {self.NULL_RATE_WARN_THRESHOLD:.0%}",
                "actual":   f"{null_rate:.2%}",
                "severity": severity,
            })
            logger.debug(f"  null_rate [{severity}]: col='{col}' rate={null_rate:.2%}")

        score = round(sum(completeness_scores) / len(completeness_scores) * 100, 2)
        logger.info(f"check_completeness: score={score} failures={len(failed_checks)}")
        return score, failed_checks

    def check_accuracy(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Detect columns with suspiciously low cardinality or numeric outliers.

        Two sub-checks contribute to accuracy:

        1. **Low-cardinality** — a non-boolean ``object`` column whose
           ``unique_count / row_count`` ratio falls below
           :attr:`CARDINALITY_RATIO_THRESHOLD` is flagged as potentially
           corrupt or miscoded.
        2. **Numeric outliers** — for each numeric column a Z-score is
           computed; values beyond :attr:`NUMERIC_OUTLIER_Z_THRESHOLD`
           standard deviations from the mean are counted.  If any outliers
           exist the column is flagged with the outlier count and rate.

        Score::

            accuracy = (1 - outlier_fraction) * 100

        where *outlier_fraction* is ``total_outlier_cells / total_cells``.
        Low-cardinality flags reduce the score by a fixed 5 points per
        column, floored at 0.

        Args:
            df: The DataFrame under assessment.
            profile: Profile dict from the profiler.

        Returns:
            ``(score, failed_checks)`` tuple.
        """
        logger.debug("check_accuracy: starting")
        failed_checks: list[dict[str, Any]] = []
        row_count = len(df)

        if row_count == 0:
            logger.warning("check_accuracy: empty DataFrame — scoring 100")
            return 100.0, []

        total_cells   = row_count * len(df.columns)
        outlier_cells = 0
        penalty       = 0.0

        columns_profile = profile.get("columns", {})

        for col in df.columns:
            col_profile  = columns_profile.get(col, {})
            dtype_str    = col_profile.get("dtype", "")
            unique_count = col_profile.get("unique_count", 0)

            # --- low-cardinality check (object columns only) ---
            if (
                "object" in dtype_str
                and unique_count > 2                        # exclude booleans stored as strings
                and row_count > 0
                and (unique_count / row_count) < self.CARDINALITY_RATIO_THRESHOLD
            ):
                penalty += 5.0
                failed_checks.append({
                    "check":    "low_cardinality",
                    "column":   col,
                    "expected": f">= {self.CARDINALITY_RATIO_THRESHOLD:.1%} unique ratio",
                    "actual":   f"{unique_count / row_count:.2%}",
                    "severity": "warning",
                })
                logger.debug(f"  low_cardinality [warning]: col='{col}'")

            # --- numeric outlier check ---
            if pd.api.types.is_numeric_dtype(df[col]):
                series   = df[col].dropna()
                std      = series.std()
                if len(series) < 2 or std == 0:
                    continue
                z_scores = ((series - series.mean()) / std).abs()
                n_out    = int((z_scores > self.NUMERIC_OUTLIER_Z_THRESHOLD).sum())
                if n_out > 0:
                    outlier_rate = n_out / len(series)
                    outlier_cells += n_out
                    failed_checks.append({
                        "check":    "numeric_outliers",
                        "column":   col,
                        "expected": f"Z-score <= {self.NUMERIC_OUTLIER_Z_THRESHOLD}",
                        "actual":   f"{n_out} outlier(s) ({outlier_rate:.2%})",
                        "severity": "warning" if outlier_rate < 0.05 else "error",
                    })
                    logger.debug(
                        f"  numeric_outliers: col='{col}' n={n_out} rate={outlier_rate:.2%}"
                    )

        outlier_fraction = outlier_cells / total_cells if total_cells > 0 else 0.0
        score = max(0.0, round((1.0 - outlier_fraction) * 100 - penalty, 2))
        logger.info(f"check_accuracy: score={score} failures={len(failed_checks)}")
        return score, failed_checks

    def check_consistency(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Detect inconsistent dtypes and columns whose null pattern is erratic.

        Two sub-checks:

        1. **Mixed-type columns** — an ``object`` column is tested by
           attempting to infer a more specific type with
           :func:`pandas.to_numeric` and :func:`pandas.to_datetime`.  If
           more than 10 % of non-null values parse successfully as one type
           but the column is stored as ``object``, the column is flagged as
           inconsistently typed.
        2. **Constant columns** — a column with ``unique_count == 1`` (or 0
           if fully null) carries no information and is flagged as a warning.

        Score::

            score = (1 - flagged_fraction) * 100

        where *flagged_fraction* is the number of flagged columns divided by
        the total column count.

        Args:
            df: The DataFrame under assessment.
            profile: Profile dict from the profiler.

        Returns:
            ``(score, failed_checks)`` tuple.
        """
        logger.debug("check_consistency: starting")
        failed_checks: list[dict[str, Any]] = []
        columns_profile = profile.get("columns", {})
        flagged = 0

        for col in df.columns:
            col_profile  = columns_profile.get(col, {})
            dtype_str    = col_profile.get("dtype", "")
            unique_count = col_profile.get("unique_count", 0)
            null_rate    = col_profile.get("null_rate", 0.0)

            # --- constant column check ---
            if unique_count <= 1 and null_rate < 1.0:
                flagged += 1
                failed_checks.append({
                    "check":    "constant_column",
                    "column":   col,
                    "expected": "unique_count > 1",
                    "actual":   f"unique_count={unique_count}",
                    "severity": "warning",
                })
                logger.debug(f"  constant_column [warning]: col='{col}'")
                continue

            # --- mixed-type check (object columns only) ---
            if "object" not in dtype_str:
                continue

            series   = df[col].dropna()
            if len(series) == 0:
                continue

            numeric_parsed  = pd.to_numeric(series, errors="coerce").notna().sum()
            datetime_parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True).notna().sum()
            parse_ratio     = max(numeric_parsed, datetime_parsed) / len(series)

            if parse_ratio > 0.10:
                flagged += 1
                inferred = "numeric" if numeric_parsed >= datetime_parsed else "datetime"
                failed_checks.append({
                    "check":    "mixed_type",
                    "column":   col,
                    "expected": f"uniform dtype, not object",
                    "actual":   f"{parse_ratio:.0%} of values parseable as {inferred}",
                    "severity": "warning",
                })
                logger.debug(
                    f"  mixed_type [warning]: col='{col}' inferred={inferred} ratio={parse_ratio:.0%}"
                )

        total_cols = len(df.columns) or 1
        score = round((1.0 - flagged / total_cols) * 100, 2)
        logger.info(f"check_consistency: score={score} failures={len(failed_checks)}")
        return score, failed_checks

    def check_freshness(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Assess how recent the data is by inspecting date/datetime columns.

        Finds all columns with a date or datetime dtype (or object columns
        that parse successfully as dates for > 80 % of non-null values).
        For each date column the maximum (most recent) value is compared
        against today's date.  If the newest value is older than
        :attr:`FRESHNESS_MAX_AGE_DAYS` the column is flagged.

        Score::

            score = max(0, 100 - (max_staleness_days / FRESHNESS_MAX_AGE_DAYS) * 100)

        If no date columns are found the check returns 100.0 (cannot assess)
        with a warning in the failed_checks list.

        Args:
            df: The DataFrame under assessment.
            profile: Profile dict from the profiler.

        Returns:
            ``(score, failed_checks)`` tuple.
        """
        logger.debug("check_freshness: starting")
        failed_checks: list[dict[str, Any]] = []
        today          = datetime.date.today()
        date_columns   = self._find_date_columns(df)

        if not date_columns:
            logger.warning("check_freshness: no date columns found — scoring 100")
            failed_checks.append({
                "check":    "no_date_columns",
                "column":   "dataset",
                "expected": "at least one date/datetime column",
                "actual":   "none found",
                "severity": "warning",
            })
            return 100.0, failed_checks

        max_staleness_days = 0

        for col, parsed_series in date_columns.items():
            max_date = parsed_series.max()
            if pd.isna(max_date):
                continue
            if isinstance(max_date, pd.Timestamp):
                max_date = max_date.date()

            staleness_days = (today - max_date).days

            if staleness_days > self.FRESHNESS_MAX_AGE_DAYS:
                max_staleness_days = max(max_staleness_days, staleness_days)
                severity = (
                    "error" if staleness_days > self.FRESHNESS_MAX_AGE_DAYS * 2
                    else "warning"
                )
                failed_checks.append({
                    "check":    "stale_data",
                    "column":   col,
                    "expected": f"most recent date within {self.FRESHNESS_MAX_AGE_DAYS} days",
                    "actual":   f"most recent date is {max_date} ({staleness_days} days ago)",
                    "severity": severity,
                })
                logger.debug(
                    f"  stale_data [{severity}]: col='{col}' "
                    f"age={staleness_days}d max_allowed={self.FRESHNESS_MAX_AGE_DAYS}d"
                )

        if max_staleness_days == 0:
            score = 100.0
        else:
            score = max(
                0.0,
                round(
                    100.0 - (max_staleness_days / self.FRESHNESS_MAX_AGE_DAYS) * 100,
                    2,
                ),
            )

        logger.info(f"check_freshness: score={score} failures={len(failed_checks)}")
        return score, failed_checks

    def check_uniqueness(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Measure the proportion of fully duplicate rows in the dataset.

        A *duplicate row* is one that is identical to at least one other row
        across all columns.  The check also flags individual columns whose
        ``unique_count / row_count`` ratio falls below a warning threshold,
        which can indicate repeated values that should be unique (e.g. IDs).

        Score::

            score = (1 - duplicate_rate) * 100

        where *duplicate_rate* comes from the pre-computed profile.

        Args:
            df: The DataFrame under assessment.
            profile: Profile dict from the profiler.

        Returns:
            ``(score, failed_checks)`` tuple.
        """
        logger.debug("check_uniqueness: starting")
        failed_checks: list[dict[str, Any]] = []
        row_count      = profile.get("row_count", 0)
        duplicate_rate = profile.get("duplicate_rate", 0.0)

        if duplicate_rate > self.DUPLICATE_RATE_ERROR_THRESHOLD:
            severity = "error"
        elif duplicate_rate > self.DUPLICATE_RATE_WARN_THRESHOLD:
            severity = "warning"
        else:
            severity = None

        if severity:
            failed_checks.append({
                "check":    "duplicate_rows",
                "column":   "dataset",
                "expected": f"duplicate_rate <= {self.DUPLICATE_RATE_WARN_THRESHOLD:.0%}",
                "actual":   f"{duplicate_rate:.2%}",
                "severity": severity,
            })
            logger.debug(
                f"  duplicate_rows [{severity}]: rate={duplicate_rate:.2%}"
            )

        # Flag columns that look like they should be unique (name contains
        # "id" or "key") but have repeated values.
        if row_count > 0:
            columns_profile = profile.get("columns", {})
            for col, col_profile in columns_profile.items():
                col_lower    = col.lower()
                unique_count = col_profile.get("unique_count", 0)
                is_id_col    = any(tok in col_lower for tok in ("id", "key", "uuid", "guid"))
                if is_id_col and unique_count < row_count:
                    repeated = row_count - unique_count
                    failed_checks.append({
                        "check":    "non_unique_identifier",
                        "column":   col,
                        "expected": f"all {row_count} values unique",
                        "actual":   f"{repeated} repeated value(s)",
                        "severity": "error",
                    })
                    logger.debug(
                        f"  non_unique_identifier [error]: col='{col}' repeated={repeated}"
                    )

        score = round(float((1.0 - duplicate_rate) * 100), 2)
        logger.info(f"check_uniqueness: score={score} failures={len(failed_checks)}")
        return score, failed_checks

    def check_validity(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Check that values conform to expected formats and value ranges.

        Three sub-checks:

        1. **Negative values in non-negative columns** — numeric columns
           whose name implies a non-negative quantity (e.g. ``price``,
           ``count``, ``amount``, ``age``, ``quantity``, ``duration``) are
           checked for values below zero.
        2. **Future dates** — date columns are checked for values beyond
           today's date, which typically indicates data entry errors.
        3. **Excessive string length** — object columns with a maximum string
           length > 1 000 characters are flagged as potentially corrupted
           or unintentionally large.

        Score::

            score = (1 - invalid_fraction) * 100

        where *invalid_fraction* is ``total_invalid_cells / total_cells``.

        Args:
            df: The DataFrame under assessment.
            profile: Profile dict from the profiler.

        Returns:
            ``(score, failed_checks)`` tuple.
        """
        logger.debug("check_validity: starting")
        failed_checks:  list[dict[str, Any]] = []
        row_count     = len(df)
        total_cells   = row_count * len(df.columns)
        invalid_cells = 0

        if row_count == 0 or total_cells == 0:
            logger.warning("check_validity: empty DataFrame — scoring 100")
            return 100.0, []

        today = pd.Timestamp.today().normalize()

        # Keywords that imply a column should be >= 0.
        _non_negative_keywords = (
            "price", "cost", "amount", "count", "age",
            "quantity", "duration", "rate", "score", "size",
        )

        date_columns = self._find_date_columns(df)

        for col in df.columns:
            series    = df[col].dropna()
            col_lower = col.lower()

            # --- negative values in non-negative columns ---
            if pd.api.types.is_numeric_dtype(df[col]):
                if any(kw in col_lower for kw in _non_negative_keywords):
                    n_neg = int((series < 0).sum())
                    if n_neg > 0:
                        invalid_cells += n_neg
                        failed_checks.append({
                            "check":    "negative_value",
                            "column":   col,
                            "expected": ">= 0",
                            "actual":   f"{n_neg} negative value(s)",
                            "severity": "error",
                        })
                        logger.debug(
                            f"  negative_value [error]: col='{col}' count={n_neg}"
                        )

            # --- future dates ---
            if col in date_columns:
                parsed  = date_columns[col]
                n_fut   = int((parsed > today).sum())
                if n_fut > 0:
                    invalid_cells += n_fut
                    failed_checks.append({
                        "check":    "future_date",
                        "column":   col,
                        "expected": f"<= today ({today.date()})",
                        "actual":   f"{n_fut} future date(s)",
                        "severity": "warning",
                    })
                    logger.debug(
                        f"  future_date [warning]: col='{col}' count={n_fut}"
                    )

            # --- excessive string length ---
            if df[col].dtype == object:
                str_lengths = series.astype(str).str.len()
                n_long      = int((str_lengths > 1_000).sum())
                if n_long > 0:
                    invalid_cells += n_long
                    failed_checks.append({
                        "check":    "excessive_string_length",
                        "column":   col,
                        "expected": "string length <= 1000 characters",
                        "actual":   f"{n_long} value(s) exceed 1000 characters",
                        "severity": "warning",
                    })
                    logger.debug(
                        f"  excessive_string_length [warning]: col='{col}' count={n_long}"
                    )

        invalid_fraction = invalid_cells / total_cells
        score = round((1.0 - invalid_fraction) * 100, 2)
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

        Executes each ``check_*`` method in sequence, collects the dimension
        scores and failed checks, then computes the weighted overall score
        using the ``scoring_weights`` from ``dq_config.yaml``.  Sets
        ``QualityResult.passed`` to ``True`` when ``overall_score`` meets the
        ``thresholds.pass_score`` in config.

        Args:
            df: The DataFrame to validate.
            profile: The profile dict returned by
                :class:`~src.modules.structured.profiler.StructuredProfiler`
                for the same DataFrame.  Must contain the keys ``row_count``,
                ``duplicate_rate``, and ``columns``.
            dataset_name: Identifier for the dataset, used in the returned
                :class:`~src.models.QualityResult` and in log messages.

        Returns:
            A fully populated :class:`~src.models.QualityResult` with:

            * ``data_type = "structured"``
            * ``dimensions`` — one score per quality dimension in [0, 100]
            * ``overall_score`` — weighted sum of dimension scores
            * ``failed_checks`` — merged list from all six check methods
            * ``passed`` — ``True`` iff ``overall_score >= pass_score``
            * ``metadata`` — row count, column count, duplicate rate
            * ``source_path`` — empty string (set by the caller if needed)

        Example::

            profiler  = StructuredProfiler()
            validator = StructuredValidator()

            df      = profiler.load_data("data/raw/orders.csv")
            profile = profiler.profile(df, dataset_name="orders")
            result  = validator.validate(df, profile, dataset_name="orders")

            print(result.overall_score)
            print(result.passed)
            print(result.to_dict())
        """
        logger.info(f"validate: starting full validation for dataset='{dataset_name}'")

        # ------------------------------------------------------------------
        # Run all six checks
        # ------------------------------------------------------------------
        checks = {
            "completeness": self.check_completeness,
            "accuracy":     self.check_accuracy,
            "consistency":  self.check_consistency,
            "freshness":    self.check_freshness,
            "uniqueness":   self.check_uniqueness,
            "validity":     self.check_validity,
        }

        dimensions:    dict[str, float]           = {}
        failed_checks: list[dict[str, Any]]       = []

        for dimension, check_fn in checks.items():
            try:
                score, failures = check_fn(df, profile)
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

        # ------------------------------------------------------------------
        # Compute weighted overall score
        # ------------------------------------------------------------------
        overall_score = round(
            sum(
                dimensions[dim] * self._weights.get(dim, 0.0)
                for dim in dimensions
            ),
            2,
        )
        passed = overall_score >= self._pass_score

        logger.info(
            f"validate: complete | dataset='{dataset_name}' "
            f"overall_score={overall_score} passed={passed} "
            f"total_failures={len(failed_checks)}"
        )

        # ------------------------------------------------------------------
        # Build and return QualityResult
        # ------------------------------------------------------------------
        return QualityResult(
            dataset_name  = dataset_name,
            data_type     = "structured",
            overall_score = overall_score,
            dimensions    = dimensions,
            failed_checks = failed_checks,
            passed        = passed,
            metadata      = {
                "row_count":      int(profile.get("row_count", 0)),
                "column_count":   int(profile.get("column_count", 0)),
                "duplicate_rate": float(profile.get("duplicate_rate", 0.0)),
            },
        )

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _find_date_columns(
        self, df: pd.DataFrame
    ) -> dict[str, pd.Series]:
        """Identify date/datetime columns and return them as parsed Series.

        First includes columns whose dtype is already ``datetime64``.  Then
        attempts to coerce ``object`` columns; a column is included if ≥ 80 %
        of its non-null values parse successfully as dates.

        Args:
            df: The DataFrame to inspect.

        Returns:
            A dict mapping column name → :class:`pandas.Series` of
            :class:`pandas.Timestamp` values (nulls where parsing failed).
            Only columns that qualify as date columns are included.
        """
        date_cols: dict[str, pd.Series] = {}

        for col in df.columns:
            series = df[col]

            if pd.api.types.is_datetime64_any_dtype(series):
                date_cols[col] = series
                continue

            if series.dtype == object:
                non_null = series.dropna()
                if len(non_null) == 0:
                    continue
                parsed      = pd.to_datetime(non_null, errors="coerce", infer_datetime_format=True)
                parse_ratio = parsed.notna().sum() / len(non_null)
                if parse_ratio >= 0.80:
                    date_cols[col] = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

        return date_cols
