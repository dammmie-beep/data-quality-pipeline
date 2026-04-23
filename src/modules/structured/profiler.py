"""
Profiler for structured (tabular) data.

Handles loading from CSV, Parquet, Excel, and SQL sources, then computes a
comprehensive column-level profile — null rates, duplicate rate, numeric
statistics, and sample values — returning everything as a plain dictionary
that the validator and scorer can consume directly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text


class StructuredProfiler:
    """Load and profile tabular data from a variety of source formats.

    Supports four source types, detected automatically from the source path:

    * **CSV** — any path ending in ``.csv``
    * **Parquet** — any path ending in ``.parquet`` or ``.pq``
    * **Excel** — any path ending in ``.xlsx`` or ``.xls``
    * **SQL** — a SQLAlchemy connection string starting with a recognised
      dialect prefix (``postgresql://``, ``mysql://``, ``sqlite://``,
      ``mssql://``, ``oracle://``) followed by an optional query or table
      name appended after a ``|`` separator.

    Example connection string formats::

        "sqlite:///data/raw/orders.db|SELECT * FROM orders"
        "postgresql://user:pass@host/db|orders"   # bare table name also works
        "data/raw/sales.csv"
        "data/raw/snapshot.parquet"
        "data/raw/report.xlsx"

    Typical usage::

        profiler = StructuredProfiler()
        df      = profiler.load_data("data/raw/orders.csv")
        profile = profiler.profile(df, dataset_name="orders")
    """

    # File extensions recognised as each format.
    _CSV_EXTENSIONS     = {".csv"}
    _PARQUET_EXTENSIONS = {".parquet", ".pq"}
    _EXCEL_EXTENSIONS   = {".xlsx", ".xls"}
    _SQL_PREFIXES       = (
        "postgresql://", "mysql://", "sqlite://",
        "mssql://", "oracle://",
    )

    # Maximum number of sample values stored per column in the profile.
    _SAMPLE_SIZE = 5

    def load_data(self, source_path: str) -> pd.DataFrame:
        """Load tabular data from a file or database into a DataFrame.

        Detects the source type from *source_path* and dispatches to the
        appropriate reader.  All readers return a :class:`pandas.DataFrame`
        with its original column names and dtypes inferred by pandas.

        Args:
            source_path: One of:

                * A relative or absolute file path to a ``.csv``,
                  ``.parquet`` / ``.pq``, or ``.xlsx`` / ``.xls`` file.
                * A SQLAlchemy connection string optionally followed by a
                  ``|`` separator and either a bare table name or a full
                  ``SELECT`` statement.  If no query or table is given after
                  ``|``, the loader raises ``ValueError``.

        Returns:
            A :class:`pandas.DataFrame` containing the loaded data, with
            columns and index exactly as returned by the underlying pandas
            reader.

        Raises:
            FileNotFoundError: If a file path is given but the file does not
                exist on disk.
            ValueError: If the source type cannot be determined from
                *source_path*, or if a SQL source is given without a query or
                table name.
            Exception: Propagates any error raised by the underlying pandas
                or SQLAlchemy reader (e.g. a parse error in a malformed CSV).

        Examples::

            profiler = StructuredProfiler()

            # CSV
            df = profiler.load_data("data/raw/orders.csv")

            # Parquet
            df = profiler.load_data("data/raw/snapshot.parquet")

            # Excel
            df = profiler.load_data("data/raw/report.xlsx")

            # SQLite with a full query
            df = profiler.load_data(
                "sqlite:///data/raw/orders.db|SELECT * FROM orders LIMIT 1000"
            )

            # PostgreSQL with a bare table name
            df = profiler.load_data(
                "postgresql://user:pass@localhost/mydb|customers"
            )
        """
        logger.info(f"Loading data from source: '{source_path}'")

        if self._is_sql_source(source_path):
            return self._load_sql(source_path)

        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found: {path.resolve()}"
            )

        suffix = path.suffix.lower()

        if suffix in self._CSV_EXTENSIONS:
            return self._load_csv(path)
        if suffix in self._PARQUET_EXTENSIONS:
            return self._load_parquet(path)
        if suffix in self._EXCEL_EXTENSIONS:
            return self._load_excel(path)

        raise ValueError(
            f"Unrecognised file extension '{suffix}' for source '{source_path}'. "
            f"Supported extensions: "
            f"{self._CSV_EXTENSIONS | self._PARQUET_EXTENSIONS | self._EXCEL_EXTENSIONS}. "
            f"For SQL sources, prefix the connection string with a recognised "
            f"dialect (e.g. 'postgresql://', 'sqlite://')."
        )

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _is_sql_source(self, source_path: str) -> bool:
        """Return True if *source_path* looks like a SQLAlchemy URL."""
        return source_path.startswith(self._SQL_PREFIXES)

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """Read a CSV file into a DataFrame.

        Args:
            path: Validated :class:`pathlib.Path` pointing to the CSV file.

        Returns:
            DataFrame with dtypes inferred by :func:`pandas.read_csv`.
        """
        logger.debug(f"Reading CSV: {path}")
        df = pd.read_csv(path)
        logger.info(f"CSV loaded | rows={len(df):,} cols={len(df.columns)}")
        return df

    def _load_parquet(self, path: Path) -> pd.DataFrame:
        """Read a Parquet file into a DataFrame.

        Args:
            path: Validated :class:`pathlib.Path` pointing to the Parquet file.

        Returns:
            DataFrame with dtypes preserved from the Parquet schema.
        """
        logger.debug(f"Reading Parquet: {path}")
        df = pd.read_parquet(path)
        logger.info(f"Parquet loaded | rows={len(df):,} cols={len(df.columns)}")
        return df

    def _load_excel(self, path: Path) -> pd.DataFrame:
        """Read the first sheet of an Excel workbook into a DataFrame.

        Args:
            path: Validated :class:`pathlib.Path` pointing to the Excel file.

        Returns:
            DataFrame representing the first sheet, with dtypes inferred by
            :func:`pandas.read_excel`.
        """
        logger.debug(f"Reading Excel: {path}")
        df = pd.read_excel(path, sheet_name=0)
        logger.info(f"Excel loaded | rows={len(df):,} cols={len(df.columns)}")
        return df

    def _load_sql(self, source_path: str) -> pd.DataFrame:
        """Execute a SQL query or read a full table into a DataFrame.

        Expects *source_path* in the format::

            "<connection_string>|<query_or_table>"

        If the part after ``|`` starts with ``SELECT`` (case-insensitive) it
        is used verbatim as a SQL query; otherwise it is treated as a table
        name and wrapped in ``SELECT * FROM <table>``.

        Args:
            source_path: A ``|``-separated string of SQLAlchemy connection URL
                and SQL query or table name.

        Returns:
            DataFrame containing the query result.

        Raises:
            ValueError: If ``|`` is absent or the query/table part is empty.
        """
        if "|" not in source_path:
            raise ValueError(
                f"SQL source must include a query or table name after '|', "
                f"e.g. 'sqlite:///db.sqlite|my_table'. Got: '{source_path}'"
            )

        connection_string, query_or_table = source_path.split("|", maxsplit=1)
        query_or_table = query_or_table.strip()

        if not query_or_table:
            raise ValueError(
                "The SQL query or table name after '|' must not be empty."
            )

        if re.match(r"(?i)^select\b", query_or_table):
            sql = query_or_table
        else:
            sql = f"SELECT * FROM {query_or_table}"

        logger.debug(f"SQL query: {sql!r} | connection: {connection_string!r}")
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)

        logger.info(f"SQL loaded | rows={len(df):,} cols={len(df.columns)}")
        return df

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def profile(self, df: pd.DataFrame, dataset_name: str) -> dict[str, Any]:
        """Compute a comprehensive column-level profile of a DataFrame.

        Calculates dataset-level metrics (row count, duplicate rate) and
        per-column metrics (dtype, null rate, numeric statistics, sample
        values).  All values are JSON-safe primitives — no numpy scalars or
        pandas Index objects are returned.

        Args:
            df: The DataFrame to profile.  Must have at least one row and one
                column; an empty DataFrame is accepted but will produce zero
                or ``null`` for most metrics.
            dataset_name: Identifier used in the ``dataset_name`` key of the
                returned dict.  Typically the filename stem or table name.

        Returns:
            A dictionary with the following top-level structure::

                {
                    "dataset_name": str,
                    "row_count":    int,
                    "column_count": int,
                    "duplicate_rate": float,   # proportion of fully duplicate rows [0, 1]
                    "columns": {
                        "<col_name>": {
                            "dtype":       str,    # e.g. "int64", "object", "float64"
                            "null_count":  int,
                            "null_rate":   float,  # proportion of nulls [0, 1]
                            "unique_count": int,
                            "sample_values": list, # up to 5 non-null values
                            # numeric columns only:
                            "min":    float | None,
                            "max":    float | None,
                            "mean":   float | None,
                            "median": float | None,
                            "std":    float | None,
                        },
                        ...
                    }
                }

            ``duplicate_rate`` is the fraction of rows that are exact
            duplicates of at least one other row (computed as
            ``df.duplicated().sum() / len(df)``; returns ``0.0`` for an
            empty DataFrame).

            ``null_rate`` per column is ``null_count / row_count``; returns
            ``0.0`` for an empty DataFrame.

            ``sample_values`` contains up to :attr:`_SAMPLE_SIZE` non-null
            values cast to Python native types so they serialise cleanly to
            JSON.

        Example::

            profiler = StructuredProfiler()
            df       = profiler.load_data("data/raw/orders.csv")
            profile  = profiler.profile(df, dataset_name="orders")

            print(profile["row_count"])
            print(profile["columns"]["price"]["mean"])
        """
        logger.info(
            f"Profiling dataset='{dataset_name}' | "
            f"rows={len(df):,} cols={len(df.columns)}"
        )

        row_count    = len(df)
        column_count = len(df.columns)

        duplicate_rate = (
            round(df.duplicated().sum() / row_count, 6)
            if row_count > 0 else 0.0
        )

        columns_profile: dict[str, dict[str, Any]] = {}
        for col in df.columns:
            series     = df[col]
            null_count = int(series.isna().sum())
            null_rate  = round(null_count / row_count, 6) if row_count > 0 else 0.0

            non_null      = series.dropna()
            unique_count  = int(series.nunique(dropna=True))
            sample_values = [
                self._to_python(v)
                for v in non_null.head(self._SAMPLE_SIZE).tolist()
            ]

            col_profile: dict[str, Any] = {
                "dtype":         str(series.dtype),
                "null_count":    null_count,
                "null_rate":     null_rate,
                "unique_count":  unique_count,
                "sample_values": sample_values,
            }

            if pd.api.types.is_numeric_dtype(series):
                col_profile.update(self._numeric_stats(non_null))

            columns_profile[col] = col_profile
            logger.debug(
                f"  col='{col}' dtype={series.dtype} "
                f"null_rate={null_rate:.2%} unique={unique_count}"
            )

        result = {
            "dataset_name":   dataset_name,
            "row_count":      row_count,
            "column_count":   column_count,
            "duplicate_rate": duplicate_rate,
            "columns":        columns_profile,
        }

        logger.info(
            f"Profile complete | dataset='{dataset_name}' "
            f"duplicate_rate={duplicate_rate:.2%}"
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _numeric_stats(self, series: pd.Series) -> dict[str, float | None]:
        """Compute descriptive statistics for a numeric series.

        Args:
            series: A non-null-filtered numeric :class:`pandas.Series`.

        Returns:
            Dictionary with keys ``min``, ``max``, ``mean``, ``median``,
            and ``std``, each as a Python :class:`float` rounded to 6
            decimal places, or ``None`` if the series is empty.
        """
        if series.empty:
            return {"min": None, "max": None, "mean": None, "median": None, "std": None}

        return {
            "min":    round(float(series.min()),    6),
            "max":    round(float(series.max()),    6),
            "mean":   round(float(series.mean()),   6),
            "median": round(float(series.median()), 6),
            "std":    round(float(series.std()),    6),
        }

    @staticmethod
    def _to_python(value: Any) -> Any:
        """Cast a value to a JSON-safe Python native type.

        Converts numpy scalars and pandas Timestamps to their Python
        equivalents.  All other types are returned unchanged.

        Args:
            value: The value to convert.

        Returns:
            A JSON-serialisable Python object.
        """
        if hasattr(value, "item"):          # numpy scalar → Python scalar
            return value.item()
        if isinstance(value, pd.Timestamp): # Timestamp → ISO string
            return value.isoformat()
        return value
