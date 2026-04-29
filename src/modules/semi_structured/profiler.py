"""
Profiler for semi-structured data.

Handles loading from JSON, JSONL, XML, and log file formats, flattening nested
structures into a pandas DataFrame for analysis, then computing a comprehensive
profile — null rates, nesting depth, type consistency, key presence, and
duplicate detection — returning everything as a plain dictionary that the
validator and scorer can consume.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def get_nesting_depth(obj: Any, depth: int = 0) -> int:
    """Recursively compute the maximum nesting depth of a dict or list.

    The depth of a scalar (string, int, float, bool, None) is 0.  Each layer
    of dict or list wrapping adds 1.

    Args:
        obj: The object to measure.  May be a dict, list, or any scalar.
        depth: Current recursion depth.  Should always be called with the
            default value of 0; the argument exists to carry accumulated depth
            through recursive calls.

    Returns:
        The maximum nesting depth as a non-negative integer.

    Examples::

        get_nesting_depth("hello")                    # 0
        get_nesting_depth({"a": 1})                   # 1
        get_nesting_depth({"a": {"b": 2}})            # 2
        get_nesting_depth({"a": [{"b": {"c": 3}}]})   # 4
        get_nesting_depth([1, 2, 3])                  # 1
        get_nesting_depth([])                         # 1
    """
    if isinstance(obj, dict):
        if not obj:
            return depth + 1
        return max(get_nesting_depth(v, depth + 1) for v in obj.values())
    if isinstance(obj, list):
        if not obj:
            return depth + 1
        return max(get_nesting_depth(item, depth + 1) for item in obj)
    return depth


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SemiStructuredProfiler:
    """Load and profile semi-structured data from JSON, JSONL, XML, and log files.

    Semi-structured data does not conform to a rigid schema: records may have
    different keys, values may be nested to arbitrary depth, and fields may be
    absent or null inconsistently.  This profiler surfaces those irregularities
    so the validator can score them against quality thresholds.

    Supported source formats
    ------------------------
    * **JSON** (``.json``) — a single JSON object or a JSON array of objects.
    * **JSONL** (``.jsonl``) — one JSON object per line (newline-delimited JSON).
    * **XML** (``.xml``) — an XML document whose direct children of the root
      element each represent one record.
    * **Log** (``.log``) — a text file whose lines are either JSON objects or
      raw strings.  Both are loaded; non-JSON lines are stored under the key
      ``"raw_line"``.

    Typical usage::

        from src.utils import load_config

        config   = load_config()
        profiler = SemiStructuredProfiler(config)

        data, fmt = profiler.load_data("data/raw/events.jsonl")
        df        = profiler.flatten_to_dataframe(data)
        profile   = profiler.profile(df, data, dataset_name="events")
    """

    # Fraction of records below which a key is considered "unexpected"
    # (i.e. sparsely present).
    _UNEXPECTED_KEY_THRESHOLD: float = 0.10

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise the profiler and store pipeline configuration.

        Args:
            config: The pipeline configuration dictionary as returned by
                :func:`~src.utils.load_config`.  Currently used to make the
                profiler consistent with the structured module interface;
                future versions will read format-specific thresholds from it.
        """
        self._config = config
        logger.debug("SemiStructuredProfiler initialised")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_data(self, source_path: str) -> tuple[list | dict, str]:
        """Load semi-structured data from a file and detect its format.

        Dispatches to a format-specific private loader based on the file
        extension.  The returned format string is passed back through the
        pipeline so downstream consumers can adapt their behaviour (e.g. the
        validator may apply XML-specific checks).

        Args:
            source_path: Path to the source file.  The extension determines
                the loader:

                * ``.json``  — single JSON document (object or array)
                * ``.jsonl`` — newline-delimited JSON (one object per line)
                * ``.xml``   — XML document (children of root = records)
                * ``.log``   — text log (JSON lines parsed; others kept raw)

        Returns:
            A tuple ``(data, format_str)`` where:

            * *data* is a ``list`` of dicts (one per record) for JSONL, XML,
              and log files; or the raw parsed structure (``list`` or ``dict``)
              for plain JSON files.
            * *format_str* is one of ``"json"``, ``"jsonl"``, ``"xml"``,
              ``"log"``.

        Raises:
            FileNotFoundError: If *source_path* does not exist on disk.
            ValueError: If the file extension is not one of the four supported
                types.
            json.JSONDecodeError: If a ``.json`` file is not valid JSON.
            ET.ParseError: If a ``.xml`` file is not well-formed XML.

        Example::

            profiler = SemiStructuredProfiler(config)
            data, fmt = profiler.load_data("data/raw/events.jsonl")
            # fmt == "jsonl", data is a list of dicts
        """
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path.resolve()}")

        suffix = path.suffix.lower()

        loaders = {
            ".json":  self._load_json,
            ".jsonl": self._load_jsonl,
            ".xml":   self._load_xml,
            ".log":   self._load_log,
        }

        if suffix not in loaders:
            raise ValueError(
                f"Unsupported file extension '{suffix}' for source '{source_path}'. "
                f"Supported extensions: {sorted(loaders.keys())}."
            )

        data, fmt = loaders[suffix](path)

        record_count = len(data) if isinstance(data, list) else 1
        logger.info(
            f"load_data: loaded '{path.name}' | format={fmt} records={record_count:,}"
        )
        return data, fmt

    def flatten_to_dataframe(self, data: list | dict) -> pd.DataFrame:
        """Flatten a nested JSON-like structure into a tidy pandas DataFrame.

        Uses :func:`pandas.json_normalize` to expand nested dicts into
        dot-separated column names (e.g. ``{"user": {"id": 1}}`` becomes
        column ``user.id``).  Lists of scalars within a record are coerced to
        their string representation by ``json_normalize``; lists of objects
        would require explicit explosion and are left as-is.

        Args:
            data: A list of dicts (one per record) or a single dict.  A bare
                dict is wrapped in a one-element list before normalisation so
                the result is always a single-row DataFrame.

        Returns:
            A :class:`pandas.DataFrame` with one row per input record and one
            column per unique flattened field path.  Missing fields appear as
            ``NaN``.

        Example::

            data = [
                {"id": 1, "user": {"name": "Alice", "age": 30}},
                {"id": 2, "user": {"name": "Bob"}},
            ]
            df = profiler.flatten_to_dataframe(data)
            # columns: ["id", "user.name", "user.age"]
            # df["user.age"][1] == NaN  (missing in second record)
        """
        if isinstance(data, dict):
            data = [data]

        df = pd.json_normalize(data)  # type: ignore[arg-type]

        logger.info(
            f"flatten_to_dataframe: shape={df.shape} "
            f"columns={list(df.columns)[:10]}"
            f"{'...' if len(df.columns) > 10 else ''}"
        )
        return df

    def profile(
        self,
        df: pd.DataFrame,
        raw_data: list | dict,
        dataset_name: str,
    ) -> dict[str, Any]:
        """Compute a comprehensive profile of a semi-structured dataset.

        Combines metrics derived from the flattened DataFrame (null rates,
        type consistency, duplicates) with metrics that require the original
        nested structure (nesting depth, unexpected keys).

        Args:
            df: The flattened DataFrame produced by
                :meth:`flatten_to_dataframe` for the same *raw_data*.
            raw_data: The original parsed data as returned by
                :meth:`load_data`.  Used to compute ``max_nesting_depth`` and
                ``unexpected_keys`` without losing structural information that
                flattening discards.
            dataset_name: Human-readable identifier for the dataset, echoed
                back into the profile dict.

        Returns:
            A dictionary with the following keys::

                {
                    "dataset_name":          str,
                    "record_count":          int,
                    "field_count":           int,
                    "fields":                list[str],
                    "null_rates":            dict[str, float],  # 0.0–100.0
                    "max_nesting_depth":     int,
                    "type_consistency":      dict[str, bool],
                    "duplicate_record_count": int,
                    "unexpected_keys":       list[str],
                    "profiled_at":           str,   # ISO-8601 UTC
                }

            Field definitions:

            * **null_rates** — for each flattened column, the percentage of
              records where the value is null or missing, rounded to 2 d.p.
            * **max_nesting_depth** — the deepest level of dict/list nesting
              found anywhere in *raw_data*, computed by
              :func:`get_nesting_depth`.
            * **type_consistency** — ``True`` for a field when every non-null
              value shares the same Python type; ``False`` when multiple types
              are present (e.g. a mix of ``str`` and ``int``).
            * **duplicate_record_count** — the number of rows in *df* that are
              exact duplicates of at least one other row across all fields.
            * **unexpected_keys** — flattened field names present in fewer than
              10 % of records (controlled by ``_UNEXPECTED_KEY_THRESHOLD``).
              A high number here indicates a loosely-schemaed dataset.

        Example::

            data, fmt = profiler.load_data("data/raw/events.jsonl")
            df        = profiler.flatten_to_dataframe(data)
            p         = profiler.profile(df, data, dataset_name="events")

            print(p["max_nesting_depth"])          # e.g. 3
            print(p["unexpected_keys"])            # fields in < 10 % of records
            print(p["type_consistency"]["user.id"]) # True / False
        """
        logger.info(
            f"profile: starting | dataset='{dataset_name}' "
            f"rows={len(df):,} cols={len(df.columns)}"
        )

        record_count = len(df)
        fields       = list(df.columns)
        field_count  = len(fields)

        null_rates        = self._compute_null_rates(df, record_count)
        type_consistency  = self._compute_type_consistency(df)
        duplicate_count   = self._compute_duplicate_count(df)
        unexpected_keys   = self._compute_unexpected_keys(df, record_count)
        max_nesting_depth = self._compute_max_nesting_depth(raw_data)

        profile = {
            "dataset_name":           dataset_name,
            "record_count":           record_count,
            "field_count":            field_count,
            "fields":                 fields,
            "null_rates":             null_rates,
            "max_nesting_depth":      max_nesting_depth,
            "type_consistency":       type_consistency,
            "duplicate_record_count": duplicate_count,
            "unexpected_keys":        unexpected_keys,
            "profiled_at":            datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"profile: complete | dataset='{dataset_name}' "
            f"max_nesting_depth={max_nesting_depth} "
            f"unexpected_keys={len(unexpected_keys)} "
            f"duplicate_records={duplicate_count}"
        )
        return profile

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_json(self, path: Path) -> tuple[list | dict, str]:
        """Load a single JSON document.

        Args:
            path: Path to the ``.json`` file.

        Returns:
            ``(data, "json")`` where *data* is the raw parsed Python object
            (a ``dict`` or ``list``).
        """
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.debug(f"_load_json: parsed '{path.name}'")
        return data, "json"

    def _load_jsonl(self, path: Path) -> tuple[list[dict], str]:
        """Load a newline-delimited JSON file (one JSON object per line).

        Blank lines are silently skipped.

        Args:
            path: Path to the ``.jsonl`` file.

        Returns:
            ``(records, "jsonl")`` where *records* is a list of dicts, one
            per non-blank line.

        Raises:
            json.JSONDecodeError: If any non-blank line is not valid JSON.
        """
        records: list[dict] = []
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(
                        f"_load_jsonl: skipping line {lineno} in '{path.name}' "
                        f"— {exc}"
                    )
        logger.debug(f"_load_jsonl: parsed {len(records):,} records from '{path.name}'")
        return records, "jsonl"

    def _load_xml(self, path: Path) -> tuple[list[dict], str]:
        """Load an XML file and convert each child element of the root to a dict.

        Attributes of each child element are included in its dict.  Text
        content is stored under the key ``"_text"`` when non-empty.  Child
        sub-elements are recursively converted using
        :meth:`_xml_element_to_dict`.

        Args:
            path: Path to the ``.xml`` file.

        Returns:
            ``(records, "xml")`` where *records* is a list of dicts, one per
            direct child of the XML root element.

        Raises:
            ET.ParseError: If the file is not well-formed XML.
        """
        tree = ET.parse(path)
        root = tree.getroot()
        records = [self._xml_element_to_dict(child) for child in root]
        logger.debug(
            f"_load_xml: parsed {len(records):,} records from '{path.name}' "
            f"(root tag: <{root.tag}>)"
        )
        return records, "xml"

    def _load_log(self, path: Path) -> tuple[list[dict], str]:
        """Load a log file, parsing each line as JSON where possible.

        Lines that parse successfully as JSON are included as dicts.  Lines
        that fail JSON parsing are included as ``{"raw_line": "<text>"}`` so
        they appear in the flattened DataFrame and can be counted by the
        profiler.  Blank lines are skipped.

        Args:
            path: Path to the ``.log`` file.

        Returns:
            ``(records, "log")`` where *records* is a list of dicts.
        """
        records: list[dict] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                    records.append(parsed if isinstance(parsed, dict) else {"value": parsed})
                except json.JSONDecodeError:
                    records.append({"raw_line": line})
        logger.debug(
            f"_load_log: parsed {len(records):,} entries from '{path.name}'"
        )
        return records, "log"

    # ------------------------------------------------------------------
    # Private XML helper
    # ------------------------------------------------------------------

    def _xml_element_to_dict(self, element: ET.Element) -> dict[str, Any]:
        """Recursively convert an XML element to a nested Python dictionary.

        Conversion rules:

        * **Attributes** — each attribute becomes a top-level key in the dict.
        * **Text content** — if the element has non-whitespace text and no
          children, the text is stored under ``"_text"``.
        * **Child elements** — recursively converted.  Multiple children with
          the same tag are collected into a list under that tag name.

        Args:
            element: The :class:`xml.etree.ElementTree.Element` to convert.

        Returns:
            A nested dictionary representing the element's attributes,
            text, and children.
        """
        result: dict[str, Any] = dict(element.attrib)

        text = (element.text or "").strip()
        if text and len(element) == 0:
            result["_text"] = text

        for child in element:
            child_dict = self._xml_element_to_dict(child)
            if child.tag in result:
                existing = result[child.tag]
                if isinstance(existing, list):
                    existing.append(child_dict)
                else:
                    result[child.tag] = [existing, child_dict]
            else:
                result[child.tag] = child_dict

        return result

    # ------------------------------------------------------------------
    # Private profiling helpers
    # ------------------------------------------------------------------

    def _compute_null_rates(
        self, df: pd.DataFrame, record_count: int
    ) -> dict[str, float]:
        """Compute the percentage of null values per column.

        Args:
            df: The flattened DataFrame.
            record_count: Total number of rows (passed in to avoid recomputing).

        Returns:
            Dict mapping column name → null percentage in [0.0, 100.0],
            rounded to 2 decimal places.  Returns 0.0 for all columns when
            *record_count* is zero.
        """
        if record_count == 0:
            return {col: 0.0 for col in df.columns}
        return {
            col: round(float(df[col].isna().sum()) / record_count * 100, 2)
            for col in df.columns
        }

    def _compute_type_consistency(
        self, df: pd.DataFrame
    ) -> dict[str, bool]:
        """Check whether all non-null values in each column share one Python type.

        A column is *consistent* (``True``) when every non-null value has
        exactly the same ``type()``.  A column with zero non-null values is
        also treated as consistent.

        Args:
            df: The flattened DataFrame.

        Returns:
            Dict mapping column name → ``True`` (consistent) or ``False``
            (mixed types).
        """
        consistency: dict[str, bool] = {}
        for col in df.columns:
            non_null = df[col].dropna()
            if non_null.empty:
                consistency[col] = True
                continue
            types = set(type(v) for v in non_null)
            consistency[col] = len(types) == 1
            if len(types) > 1:
                logger.debug(
                    f"  type_consistency: col='{col}' mixed types={types}"
                )
        return consistency

    def _compute_duplicate_count(self, df: pd.DataFrame) -> int:
        """Count the number of rows that are exact duplicates of another row.

        Args:
            df: The flattened DataFrame.

        Returns:
            The number of duplicate rows (rows where ``df.duplicated()``
            returns ``True``).  Returns 0 for an empty DataFrame.
        """
        if df.empty:
            return 0
        return int(df.duplicated().sum())

    def _compute_unexpected_keys(
        self, df: pd.DataFrame, record_count: int
    ) -> list[str]:
        """Identify fields present in fewer than the unexpected-key threshold of records.

        A field is "unexpected" if it is non-null in fewer than
        ``_UNEXPECTED_KEY_THRESHOLD`` (default 10 %) of records, suggesting
        it is sparse or schema-inconsistent.

        Args:
            df: The flattened DataFrame.
            record_count: Total number of rows.

        Returns:
            List of column names whose non-null rate falls below the
            threshold.  Returns an empty list when *record_count* is zero.
        """
        if record_count == 0:
            return []

        threshold   = self._UNEXPECTED_KEY_THRESHOLD * record_count
        unexpected  = [
            col
            for col in df.columns
            if df[col].notna().sum() < threshold
        ]
        if unexpected:
            logger.debug(
                f"  unexpected_keys: {len(unexpected)} field(s) present in "
                f"< {self._UNEXPECTED_KEY_THRESHOLD:.0%} of records"
            )
        return unexpected

    def _compute_max_nesting_depth(self, raw_data: list | dict) -> int:
        """Compute the maximum nesting depth across all records in *raw_data*.

        Delegates to the module-level :func:`get_nesting_depth` helper.  For
        a list of records, each record is measured individually and the maximum
        is returned.

        Args:
            raw_data: The original parsed data — a list of dicts or a single
                dict.

        Returns:
            The maximum nesting depth found across all records.
        """
        if isinstance(raw_data, list):
            if not raw_data:
                return 0
            depth = max(get_nesting_depth(record) for record in raw_data)
        else:
            depth = get_nesting_depth(raw_data)
        logger.debug(f"  max_nesting_depth={depth}")
        return depth
