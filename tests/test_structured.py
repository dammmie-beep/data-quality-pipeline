"""
Tests for the structured data profiler and validator.

Covers:
  - StructuredProfiler.load_data  (CSV, Parquet, Excel, SQLite, error paths)
  - StructuredProfiler.profile    (shape, null rates, duplicate rate, numeric stats,
                                   sample values, empty DataFrame)
  - StructuredValidator.check_completeness
  - StructuredValidator.check_uniqueness
  - QualityResult.to_dict         (keys, JSON-safety, value fidelity)
  - Full pipeline                 (CSV → profile → validate → QualityResult)

All file-system operations use pytest's ``tmp_path`` fixture so nothing is
written to the project tree.  The validator reads ``dq_config.yaml`` from the
project root; tests are expected to be run from that directory
(``pytest`` or ``python -m pytest`` at the repo root).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src.models import QualityResult
from src.modules.structured.profiler import StructuredProfiler
from src.modules.structured.validator import StructuredValidator

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def profiler() -> StructuredProfiler:
    """Return a fresh StructuredProfiler instance."""
    return StructuredProfiler()


@pytest.fixture
def validator() -> StructuredValidator:
    """Return a StructuredValidator configured from the project dq_config.yaml."""
    return StructuredValidator()


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """Return a tidy DataFrame with no nulls, no duplicates, and varied dtypes.

    Columns:
        order_id  (int)   — unique integer identifier
        product   (str)   — categorical-ish string
        price     (float) — positive float
        quantity  (int)   — positive integer
        created   (datetime) — recent dates
    """
    return pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5],
            "product":  ["alpha", "beta", "gamma", "delta", "epsilon"],
            "price":    [9.99, 14.99, 4.99, 24.99, 19.99],
            "quantity": [1, 3, 2, 1, 5],
            "created":  pd.to_datetime(
                ["2026-04-20", "2026-04-21", "2026-04-22", "2026-04-22", "2026-04-23"]
            ),
        }
    )


@pytest.fixture
def clean_profile(profiler: StructuredProfiler, clean_df: pd.DataFrame) -> dict:
    """Return the profile dict for clean_df."""
    return profiler.profile(clean_df, dataset_name="orders")


@pytest.fixture
def csv_file(tmp_path: Path, clean_df: pd.DataFrame) -> Path:
    """Write clean_df to a temporary CSV file and return its path."""
    path = tmp_path / "orders.csv"
    clean_df.to_csv(path, index=False)
    return path


@pytest.fixture
def parquet_file(tmp_path: Path, clean_df: pd.DataFrame) -> Path:
    """Write clean_df to a temporary Parquet file and return its path."""
    path = tmp_path / "orders.parquet"
    clean_df.to_parquet(path, index=False)
    return path


@pytest.fixture
def excel_file(tmp_path: Path, clean_df: pd.DataFrame) -> Path:
    """Write clean_df to a temporary Excel file and return its path."""
    path = tmp_path / "orders.xlsx"
    clean_df.to_excel(path, index=False)
    return path


@pytest.fixture
def sqlite_uri(tmp_path: Path, clean_df: pd.DataFrame) -> str:
    """Create a SQLite DB with an 'orders' table and return a pipeline URI.

    The URI format expected by StructuredProfiler is::

        sqlite:///<absolute_path>|<table_or_query>
    """
    import sqlalchemy

    db_path = tmp_path / "test.db"
    engine  = sqlalchemy.create_engine(f"sqlite:///{db_path}")
    # Drop datetime col — SQLite stores it as string, which is fine for SQL tests.
    clean_df[["order_id", "product", "price", "quantity"]].to_sql(
        "orders", engine, index=False, if_exists="replace"
    )
    return f"sqlite:///{db_path}"


# ===========================================================================
# StructuredProfiler — load_data
# ===========================================================================


class TestLoadData:
    """Tests for StructuredProfiler.load_data."""

    def test_load_csv_returns_dataframe(
        self, profiler: StructuredProfiler, csv_file: Path
    ) -> None:
        """load_data on a valid CSV returns a non-empty DataFrame."""
        df = profiler.load_data(str(csv_file))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_load_csv_preserves_columns(
        self, profiler: StructuredProfiler, csv_file: Path, clean_df: pd.DataFrame
    ) -> None:
        """Columns loaded from CSV match the original DataFrame."""
        df = profiler.load_data(str(csv_file))
        assert list(df.columns) == list(clean_df.columns)

    def test_load_parquet_returns_dataframe(
        self, profiler: StructuredProfiler, parquet_file: Path
    ) -> None:
        """load_data on a valid Parquet file returns a non-empty DataFrame."""
        df = profiler.load_data(str(parquet_file))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_load_parquet_preserves_dtypes(
        self, profiler: StructuredProfiler, parquet_file: Path, clean_df: pd.DataFrame
    ) -> None:
        """Parquet preserves numeric dtypes exactly."""
        df = profiler.load_data(str(parquet_file))
        assert df["price"].dtype == clean_df["price"].dtype

    def test_load_excel_returns_dataframe(
        self, profiler: StructuredProfiler, excel_file: Path
    ) -> None:
        """load_data on a valid Excel file returns a non-empty DataFrame."""
        df = profiler.load_data(str(excel_file))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_load_sql_table_name(
        self, profiler: StructuredProfiler, sqlite_uri: str
    ) -> None:
        """load_data accepts a bare table name after the pipe separator."""
        df = profiler.load_data(f"{sqlite_uri}|orders")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "order_id" in df.columns

    def test_load_sql_select_query(
        self, profiler: StructuredProfiler, sqlite_uri: str
    ) -> None:
        """load_data accepts a full SELECT statement after the pipe separator."""
        df = profiler.load_data(f"{sqlite_uri}|SELECT * FROM orders WHERE order_id > 2")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_load_missing_file_raises_file_not_found(
        self, profiler: StructuredProfiler, tmp_path: Path
    ) -> None:
        """load_data raises FileNotFoundError for a path that does not exist."""
        with pytest.raises(FileNotFoundError, match="not found"):
            profiler.load_data(str(tmp_path / "ghost.csv"))

    def test_load_unknown_extension_raises_value_error(
        self, profiler: StructuredProfiler, tmp_path: Path
    ) -> None:
        """load_data raises ValueError for an unrecognised file extension."""
        bad = tmp_path / "data.xyz"
        bad.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError, match="Unrecognised file extension"):
            profiler.load_data(str(bad))

    def test_load_sql_missing_query_raises_value_error(
        self, profiler: StructuredProfiler, sqlite_uri: str
    ) -> None:
        """load_data raises ValueError when no table/query follows the pipe."""
        with pytest.raises(ValueError, match="must not be empty"):
            profiler.load_data(f"{sqlite_uri}|")

    def test_load_sql_missing_pipe_raises_value_error(
        self, profiler: StructuredProfiler
    ) -> None:
        """load_data raises ValueError when SQL URI has no pipe separator."""
        with pytest.raises(ValueError, match="must include a query"):
            profiler.load_data("sqlite:///any.db")


# ===========================================================================
# StructuredProfiler — profile
# ===========================================================================


class TestProfile:
    """Tests for StructuredProfiler.profile."""

    def test_profile_top_level_keys(
        self, clean_profile: dict
    ) -> None:
        """profile() returns all expected top-level keys."""
        expected = {"dataset_name", "row_count", "column_count", "duplicate_rate", "columns"}
        assert expected == set(clean_profile.keys())

    def test_profile_row_and_column_counts(
        self, clean_profile: dict, clean_df: pd.DataFrame
    ) -> None:
        """row_count and column_count match the source DataFrame."""
        assert clean_profile["row_count"]    == len(clean_df)
        assert clean_profile["column_count"] == len(clean_df.columns)

    def test_profile_dataset_name(self, clean_profile: dict) -> None:
        """dataset_name is echoed back from the argument."""
        assert clean_profile["dataset_name"] == "orders"

    def test_profile_no_nulls_gives_zero_null_rate(
        self, clean_profile: dict
    ) -> None:
        """Every column in a null-free DataFrame has null_rate == 0.0."""
        for col, col_prof in clean_profile["columns"].items():
            assert col_prof["null_rate"] == 0.0, f"col='{col}' expected null_rate 0.0"

    def test_profile_null_rate_calculation(
        self, profiler: StructuredProfiler
    ) -> None:
        """null_rate is correctly computed as null_count / row_count."""
        df = pd.DataFrame({"a": [1.0, None, 3.0, None, 5.0]})
        profile = profiler.profile(df, "test")
        assert profile["columns"]["a"]["null_rate"] == pytest.approx(0.4, abs=1e-6)
        assert profile["columns"]["a"]["null_count"] == 2

    def test_profile_no_duplicates_gives_zero_duplicate_rate(
        self, clean_profile: dict
    ) -> None:
        """A DataFrame with all unique rows has duplicate_rate == 0.0."""
        assert clean_profile["duplicate_rate"] == 0.0

    def test_profile_duplicate_rate_calculation(
        self, profiler: StructuredProfiler
    ) -> None:
        """duplicate_rate is the fraction of rows that are exact duplicates."""
        df = pd.DataFrame({"x": [1, 1, 2, 3, 3]})
        profile = profiler.profile(df, "test")
        # Rows 1 and 4 are duplicates of rows 0 and 3 → 2/5 = 0.4
        assert profile["duplicate_rate"] == pytest.approx(0.4, abs=1e-4)

    def test_profile_numeric_stats_present(self, clean_profile: dict) -> None:
        """Numeric columns include min, max, mean, median, and std keys."""
        price_profile = clean_profile["columns"]["price"]
        for stat in ("min", "max", "mean", "median", "std"):
            assert stat in price_profile, f"missing stat '{stat}' for numeric column"

    def test_profile_numeric_stats_values(
        self, profiler: StructuredProfiler
    ) -> None:
        """Numeric statistics are computed correctly."""
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
        profile = profiler.profile(df, "test")
        col = profile["columns"]["v"]
        assert col["min"]    == pytest.approx(1.0)
        assert col["max"]    == pytest.approx(5.0)
        assert col["mean"]   == pytest.approx(3.0)
        assert col["median"] == pytest.approx(3.0)
        assert col["std"]    == pytest.approx(math.sqrt(2.5), rel=1e-4)

    def test_profile_sample_values_at_most_five(
        self, profiler: StructuredProfiler
    ) -> None:
        """sample_values contains at most 5 entries regardless of column length."""
        df = pd.DataFrame({"x": range(100)})
        profile = profiler.profile(df, "test")
        assert len(profile["columns"]["x"]["sample_values"]) <= 5

    def test_profile_sample_values_are_python_natives(
        self, clean_profile: dict
    ) -> None:
        """sample_values contain only JSON-serialisable Python native types."""
        for col, col_prof in clean_profile["columns"].items():
            for val in col_prof["sample_values"]:
                # json.dumps raises TypeError for numpy scalars / Timestamps
                json.dumps(val)  # must not raise

    def test_profile_non_numeric_columns_lack_stats(
        self, clean_profile: dict
    ) -> None:
        """Object-dtype columns do not have numeric stat keys."""
        product_profile = clean_profile["columns"]["product"]
        for stat in ("min", "max", "mean", "median", "std"):
            assert stat not in product_profile

    def test_profile_empty_dataframe_does_not_raise(
        self, profiler: StructuredProfiler
    ) -> None:
        """profile() handles an empty DataFrame without raising."""
        df      = pd.DataFrame({"a": pd.Series([], dtype=float)})
        profile = profiler.profile(df, "empty")
        assert profile["row_count"]      == 0
        assert profile["duplicate_rate"] == 0.0

    def test_profile_unique_count(self, profiler: StructuredProfiler) -> None:
        """unique_count reflects the number of distinct non-null values."""
        df = pd.DataFrame({"cat": ["x", "y", "x", "z", None]})
        profile = profiler.profile(df, "test")
        assert profile["columns"]["cat"]["unique_count"] == 3


# ===========================================================================
# StructuredValidator — check_completeness
# ===========================================================================


class TestCheckCompleteness:
    """Tests for StructuredValidator.check_completeness."""

    def test_perfect_data_scores_100(
        self, validator: StructuredValidator, clean_df: pd.DataFrame, clean_profile: dict
    ) -> None:
        """A DataFrame with no nulls scores 100.0 for completeness."""
        score, failures = validator.check_completeness(clean_df, clean_profile)
        assert score == pytest.approx(100.0)
        assert failures == []

    def test_column_above_warn_threshold_flagged(
        self, validator: StructuredValidator, profiler: StructuredProfiler
    ) -> None:
        """A column with null_rate > NULL_RATE_WARN_THRESHOLD produces a failure."""
        # 3 out of 10 nulls → null_rate = 0.3 > warn (0.05) and error (0.20)
        df      = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, None, None, None]})
        profile = profiler.profile(df, "test")
        score, failures = validator.check_completeness(df, profile)
        assert score < 100.0
        assert len(failures) == 1
        assert failures[0]["check"]    == "null_rate"
        assert failures[0]["column"]   == "a"
        assert failures[0]["severity"] == "error"  # 30 % > error threshold (20 %)

    def test_column_between_warn_and_error_threshold(
        self, validator: StructuredValidator, profiler: StructuredProfiler
    ) -> None:
        """A column with warn_threshold < null_rate <= error_threshold is a warning."""
        # 1 out of 10 nulls → 10 % — above warn (5 %) but below error (20 %)
        df      = pd.DataFrame({"a": [1.0] * 9 + [None]})
        profile = profiler.profile(df, "test")
        _, failures = validator.check_completeness(df, profile)
        assert len(failures) == 1
        assert failures[0]["severity"] == "warning"

    def test_column_below_warn_threshold_not_flagged(
        self, validator: StructuredValidator, profiler: StructuredProfiler
    ) -> None:
        """A column with null_rate <= warn threshold produces no failures."""
        # 0 nulls → null_rate = 0 %
        df      = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        profile = profiler.profile(df, "test")
        _, failures = validator.check_completeness(df, profile)
        assert failures == []

    def test_score_formula(
        self, validator: StructuredValidator, profiler: StructuredProfiler
    ) -> None:
        """completeness score = mean(1 - null_rate) * 100 across all columns."""
        # col_a: 0/5 null → completeness 1.0
        # col_b: 2/5 null → completeness 0.6
        # expected mean = 0.8, score = 80.0
        df = pd.DataFrame({
            "col_a": [1, 2, 3, 4, 5],
            "col_b": [1.0, 2.0, None, None, 5.0],
        })
        profile = profiler.profile(df, "test")
        score, _ = validator.check_completeness(df, profile)
        assert score == pytest.approx(80.0, abs=0.01)

    def test_empty_profile_returns_zero(
        self, validator: StructuredValidator, clean_df: pd.DataFrame
    ) -> None:
        """An empty column profile dict causes check_completeness to return 0.0."""
        score, failures = validator.check_completeness(clean_df, {"columns": {}})
        assert score == 0.0
        assert failures == []

    def test_multiple_null_columns_all_flagged(
        self, validator: StructuredValidator, profiler: StructuredProfiler
    ) -> None:
        """Every column exceeding the warn threshold is included in failures."""
        df = pd.DataFrame({
            "x": [None] * 5 + [1] * 5,   # 50 % null
            "y": [None] * 3 + [2] * 7,   # 30 % null
            "z": [3] * 10,               # 0 % null — should not be flagged
        })
        profile = profiler.profile(df, "test")
        _, failures = validator.check_completeness(df, profile)
        flagged_cols = {f["column"] for f in failures}
        assert "x" in flagged_cols
        assert "y" in flagged_cols
        assert "z" not in flagged_cols


# ===========================================================================
# StructuredValidator — check_uniqueness
# ===========================================================================


class TestCheckUniqueness:
    """Tests for StructuredValidator.check_uniqueness."""

    def test_no_duplicates_scores_100(
        self, validator: StructuredValidator, clean_df: pd.DataFrame, clean_profile: dict
    ) -> None:
        """A DataFrame with all unique rows scores 100.0 for uniqueness."""
        score, failures = validator.check_uniqueness(clean_df, clean_profile)
        assert score == pytest.approx(100.0)
        # No duplicate_rows failure (rate = 0 %)
        dup_failures = [f for f in failures if f["check"] == "duplicate_rows"]
        assert dup_failures == []

    def test_duplicate_rate_above_warn_flagged(
        self, validator: StructuredValidator, profiler: StructuredProfiler
    ) -> None:
        """A dataset with duplicate_rate > DUPLICATE_RATE_WARN_THRESHOLD is flagged."""
        # Rows 0 and 1 are identical → duplicate_rate = 1/4 = 0.25
        df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
        profile = profiler.profile(df, "test")
        score, failures = validator.check_uniqueness(df, profile)
        assert score < 100.0
        dup_failures = [f for f in failures if f["check"] == "duplicate_rows"]
        assert len(dup_failures) == 1

    def test_duplicate_rate_above_error_threshold(
        self, validator: StructuredValidator, profiler: StructuredProfiler
    ) -> None:
        """A duplicate_rate > DUPLICATE_RATE_ERROR_THRESHOLD produces an error."""
        # 3 out of 4 rows duplicated → duplicate_rate = 0.75
        df = pd.DataFrame({"a": [1, 1, 1, 2]})
        profile = profiler.profile(df, "test")
        _, failures = validator.check_uniqueness(df, profile)
        dup_failures = [f for f in failures if f["check"] == "duplicate_rows"]
        assert dup_failures[0]["severity"] == "error"

    def test_duplicate_rate_between_warn_and_error(
        self, validator: StructuredValidator, profiler: StructuredProfiler
    ) -> None:
        """A duplicate_rate between warn and error thresholds produces a warning."""
        # 1 duplicate in 20 rows → 5 %, between warn (1 %) and error (10 %)
        rows = list(range(19)) + [0]   # row 19 duplicates row 0
        df   = pd.DataFrame({"a": rows})
        profile = profiler.profile(df, "test")
        _, failures = validator.check_uniqueness(df, profile)
        dup_failures = [f for f in failures if f["check"] == "duplicate_rows"]
        assert dup_failures[0]["severity"] == "warning"

    def test_id_column_with_repeated_values_flagged(
        self, validator: StructuredValidator, profiler: StructuredProfiler
    ) -> None:
        """A column named '*id*' with repeated values produces an error."""
        df = pd.DataFrame({
            "order_id": [1, 1, 2, 3, 4],   # 1 is repeated
            "value":    [10, 20, 30, 40, 50],
        })
        profile = profiler.profile(df, "test")
        _, failures = validator.check_uniqueness(df, profile)
        id_failures = [f for f in failures if f["check"] == "non_unique_identifier"]
        assert len(id_failures) == 1
        assert id_failures[0]["column"]   == "order_id"
        assert id_failures[0]["severity"] == "error"

    def test_non_id_column_duplicates_not_flagged_as_identifier(
        self, validator: StructuredValidator, profiler: StructuredProfiler
    ) -> None:
        """Repeated values in a non-identifier column are not flagged by uniqueness."""
        df = pd.DataFrame({
            "product": ["alpha", "alpha", "beta"],  # intentional repeats
            "price":   [9.99, 9.99, 14.99],
        })
        profile = profiler.profile(df, "test")
        _, failures = validator.check_uniqueness(df, profile)
        id_failures = [f for f in failures if f["check"] == "non_unique_identifier"]
        assert id_failures == []

    def test_score_reflects_duplicate_rate(
        self, validator: StructuredValidator, profiler: StructuredProfiler
    ) -> None:
        """score == (1 - duplicate_rate) * 100."""
        # Build a df where duplicate_rate is exactly 0.25
        df = pd.DataFrame({"a": [1, 1, 2, 3]})
        profile = profiler.profile(df, "test")
        duplicate_rate = profile["duplicate_rate"]
        score, _ = validator.check_uniqueness(df, profile)
        assert score == pytest.approx((1 - duplicate_rate) * 100, abs=0.01)


# ===========================================================================
# QualityResult — to_dict
# ===========================================================================


class TestQualityResultToDict:
    """Tests for QualityResult.to_dict."""

    @pytest.fixture
    def sample_result(self) -> QualityResult:
        """Return a fully populated QualityResult for serialisation tests."""
        return QualityResult(
            dataset_name  = "orders",
            data_type     = "structured",
            overall_score = 87.5,
            dimensions    = {
                "completeness": 95.0,
                "accuracy":     90.0,
                "consistency":  85.0,
                "freshness":    80.0,
                "uniqueness":   100.0,
                "validity":     75.0,
            },
            failed_checks = [
                {
                    "check":    "null_rate",
                    "column":   "email",
                    "expected": "<= 5%",
                    "actual":   "12.00%",
                    "severity": "warning",
                }
            ],
            metadata    = {"row_count": 500, "column_count": 8},
            passed      = True,
            source_path = "data/raw/orders.csv",
        )

    def test_to_dict_contains_all_expected_keys(
        self, sample_result: QualityResult
    ) -> None:
        """to_dict() returns exactly the nine expected keys."""
        d = sample_result.to_dict()
        expected_keys = {
            "dataset_name", "data_type", "overall_score", "dimensions",
            "failed_checks", "metadata", "passed", "run_timestamp", "source_path",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match_fields(
        self, sample_result: QualityResult
    ) -> None:
        """Scalar values in to_dict() match the corresponding dataclass fields."""
        d = sample_result.to_dict()
        assert d["dataset_name"]   == sample_result.dataset_name
        assert d["data_type"]      == sample_result.data_type
        assert d["overall_score"]  == sample_result.overall_score
        assert d["passed"]         == sample_result.passed
        assert d["source_path"]    == sample_result.source_path

    def test_to_dict_dimensions_match(self, sample_result: QualityResult) -> None:
        """dimensions dict in to_dict() is identical to the field value."""
        d = sample_result.to_dict()
        assert d["dimensions"] == sample_result.dimensions

    def test_to_dict_failed_checks_match(self, sample_result: QualityResult) -> None:
        """failed_checks list in to_dict() is identical to the field value."""
        d = sample_result.to_dict()
        assert d["failed_checks"] == sample_result.failed_checks

    def test_to_dict_is_json_serialisable(
        self, sample_result: QualityResult
    ) -> None:
        """json.dumps() succeeds on the output of to_dict() without a custom encoder."""
        d    = sample_result.to_dict()
        text = json.dumps(d)          # raises TypeError if not JSON-safe
        assert isinstance(text, str)
        assert '"orders"' in text

    def test_to_dict_run_timestamp_is_iso_string(
        self, sample_result: QualityResult
    ) -> None:
        """run_timestamp is an ISO-8601 string that can be parsed back to datetime."""
        from datetime import datetime
        d         = sample_result.to_dict()
        timestamp = d["run_timestamp"]
        assert isinstance(timestamp, str)
        # fromisoformat raises ValueError if the string is malformed
        parsed = datetime.fromisoformat(timestamp)
        assert parsed.year >= 2024

    def test_to_dict_does_not_mutate_original(
        self, sample_result: QualityResult
    ) -> None:
        """Mutating the to_dict() output does not affect the original dataclass."""
        d = sample_result.to_dict()
        d["dataset_name"] = "mutated"
        d["dimensions"]["completeness"] = 0.0
        assert sample_result.dataset_name                    == "orders"
        assert sample_result.dimensions["completeness"]      == 95.0

    def test_to_dict_default_dimensions_are_all_zero(self) -> None:
        """A QualityResult with default dimensions has all six scores at 0.0."""
        result = QualityResult(dataset_name="x", data_type="structured")
        d      = result.to_dict()
        for dim in ("completeness", "accuracy", "consistency",
                    "freshness", "uniqueness", "validity"):
            assert d["dimensions"][dim] == 0.0

    def test_quality_result_rejects_invalid_data_type(self) -> None:
        """QualityResult raises ValueError for an unrecognised data_type."""
        with pytest.raises(ValueError, match="Invalid data_type"):
            QualityResult(dataset_name="x", data_type="spreadsheet")

    def test_quality_result_rejects_out_of_range_score(self) -> None:
        """QualityResult raises ValueError when overall_score is outside [0, 100]."""
        with pytest.raises(ValueError, match="overall_score must be in"):
            QualityResult(dataset_name="x", data_type="structured", overall_score=101.0)


# ===========================================================================
# Full pipeline integration test
# ===========================================================================


class TestFullPipeline:
    """Integration test: CSV file → profiler → validator → QualityResult."""

    def test_full_pipeline_produces_quality_result(
        self,
        profiler:   StructuredProfiler,
        validator:  StructuredValidator,
        csv_file:   Path,
    ) -> None:
        """The full profiler → validator chain produces a valid QualityResult."""
        df      = profiler.load_data(str(csv_file))
        profile = profiler.profile(df, dataset_name="orders")
        result  = validator.validate(df, profile, dataset_name="orders")

        assert isinstance(result, QualityResult)
        assert result.dataset_name == "orders"
        assert result.data_type    == "structured"

    def test_full_pipeline_overall_score_in_range(
        self,
        profiler:  StructuredProfiler,
        validator: StructuredValidator,
        csv_file:  Path,
    ) -> None:
        """overall_score is in [0, 100] after a full run."""
        df      = profiler.load_data(str(csv_file))
        profile = profiler.profile(df, dataset_name="orders")
        result  = validator.validate(df, profile, dataset_name="orders")

        assert 0.0 <= result.overall_score <= 100.0

    def test_full_pipeline_six_dimensions_populated(
        self,
        profiler:  StructuredProfiler,
        validator: StructuredValidator,
        csv_file:  Path,
    ) -> None:
        """All six quality dimensions are present in the result."""
        df      = profiler.load_data(str(csv_file))
        profile = profiler.profile(df, dataset_name="orders")
        result  = validator.validate(df, profile, dataset_name="orders")

        expected_dims = {
            "completeness", "accuracy", "consistency",
            "freshness", "uniqueness", "validity",
        }
        assert set(result.dimensions.keys()) == expected_dims

    def test_full_pipeline_clean_data_passes(
        self,
        profiler:  StructuredProfiler,
        validator: StructuredValidator,
        csv_file:  Path,
    ) -> None:
        """A clean dataset scores above the pass threshold and is marked passed."""
        df      = profiler.load_data(str(csv_file))
        profile = profiler.profile(df, dataset_name="orders")
        result  = validator.validate(df, profile, dataset_name="orders")

        # clean_df is designed to be high quality; it should pass (≥ 80)
        assert result.passed is True
        assert result.overall_score >= 80.0

    def test_full_pipeline_metadata_populated(
        self,
        profiler:  StructuredProfiler,
        validator: StructuredValidator,
        csv_file:  Path,
    ) -> None:
        """metadata contains row_count, column_count, and duplicate_rate."""
        df      = profiler.load_data(str(csv_file))
        profile = profiler.profile(df, dataset_name="orders")
        result  = validator.validate(df, profile, dataset_name="orders")

        assert "row_count"      in result.metadata
        assert "column_count"   in result.metadata
        assert "duplicate_rate" in result.metadata
        assert result.metadata["row_count"] == 5

    def test_full_pipeline_to_dict_round_trip(
        self,
        profiler:  StructuredProfiler,
        validator: StructuredValidator,
        csv_file:  Path,
    ) -> None:
        """to_dict() on a pipeline result is JSON-serialisable."""
        df      = profiler.load_data(str(csv_file))
        profile = profiler.profile(df, dataset_name="orders")
        result  = validator.validate(df, profile, dataset_name="orders")

        payload = result.to_dict()
        text    = json.dumps(payload)   # must not raise
        decoded = json.loads(text)

        assert decoded["dataset_name"]  == "orders"
        assert decoded["data_type"]     == "structured"
        assert 0.0 <= decoded["overall_score"] <= 100.0

    def test_full_pipeline_dirty_data_scores_lower(
        self,
        profiler:  StructuredProfiler,
        validator: StructuredValidator,
        tmp_path:  Path,
    ) -> None:
        """A dataset with nulls and duplicate rows scores lower than clean data."""
        # Half the price values are null; two rows are identical duplicates.
        dirty_df = pd.DataFrame({
            "order_id": [1, 2, 3, 4, 5],
            "product":  ["a", "b", "a", "b", "a"],
            "price":    [9.99, None, 4.99, None, 9.99],
            "quantity": [1, 1, 2, 2, 1],               # rows 0 and 4 are duplicates
        })
        path = tmp_path / "dirty.csv"
        dirty_df.to_csv(path, index=False)

        df_d      = profiler.load_data(str(path))
        profile_d = profiler.profile(df_d, dataset_name="dirty")
        result_d  = validator.validate(df_d, profile_d, dataset_name="dirty")

        # Run the clean version for comparison
        clean_df = pd.DataFrame({
            "order_id": [1, 2, 3, 4, 5],
            "product":  ["alpha", "beta", "gamma", "delta", "epsilon"],
            "price":    [9.99, 14.99, 4.99, 24.99, 19.99],
            "quantity": [1, 3, 2, 1, 5],
        })
        path_c    = tmp_path / "clean.csv"
        clean_df.to_csv(path_c, index=False)
        df_c      = profiler.load_data(str(path_c))
        profile_c = profiler.profile(df_c, dataset_name="clean")
        result_c  = validator.validate(df_c, profile_c, dataset_name="clean")

        assert result_d.overall_score < result_c.overall_score
