"""
Quality-check stage entry-point.

Runs quality checks across all datasets and appends the results to
``reports/quality_results.json``.

Structured checks
-----------------
For each ``.csv`` file in ``data/raw/`` (files copied there by the ingestion
stage):

1. Load the CSV with :class:`~src.modules.structured.profiler.StructuredProfiler`.
2. Profile it.
3. Validate it with :class:`~src.modules.structured.validator.StructuredValidator`.

Semi-structured checks
----------------------
For each ``_extracted.csv`` file written to ``data/extracted/`` by
:mod:`src.modules.run_extraction`, where the matching raw file still exists in
``data/raw/``:

1. Load the *original* raw file (JSON / JSONL / log) with
   :class:`~src.modules.semi_structured.profiler.SemiStructuredProfiler`.
2. Profile it (producing the dict the validator needs).
3. Load the ``_extracted.csv`` as a flattened DataFrame.
4. Validate with :class:`~src.modules.semi_structured.validator.SemiStructuredValidator`.

Text checks
-----------
For each ``_text_features.json`` file written to ``data/extracted/`` by
:mod:`src.modules.run_extraction`:

1. Load the feature dict from JSON.
2. Collect all feature dicts into a single list (used for the uniqueness check).
3. Validate each document with :class:`~src.modules.text.validator.TextValidator`,
   passing the full list as ``all_features`` so duplicate detection can compare
   across all documents in the run.

All :class:`~src.models.QualityResult` objects are serialised via
:meth:`~src.models.QualityResult.to_dict` and appended to
``reports/quality_results.json``.  Existing results in that file are preserved.

Run directly::

    PYTHONPATH=. python src/modules/run_quality_checks.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.models import QualityResult
from src.modules.semi_structured.profiler import SemiStructuredProfiler
from src.modules.semi_structured.validator import SemiStructuredValidator
from src.modules.structured.profiler import StructuredProfiler
from src.modules.structured.validator import StructuredValidator
from src.modules.text.validator import TextValidator
from src.utils import load_config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR          = Path("data/raw")
EXTRACTED_DIR    = Path("data/extracted")
REPORTS_DIR      = Path("reports")
RESULTS_PATH     = REPORTS_DIR / "quality_results.json"

# Extensions the semi-structured profiler can handle
_SEMI_STRUCTURED_EXTENSIONS = {".json", ".jsonl", ".log"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_existing_results() -> list[dict]:
    """Load any previously written results from ``quality_results.json``.

    Returns an empty list when the file does not exist or cannot be parsed.
    """
    if not RESULTS_PATH.exists():
        return []
    try:
        with RESULTS_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        logger.warning(
            "run_quality_checks: could not read existing results — "
            "starting with an empty list"
        )
        return []


def _save_results(results: list[dict]) -> None:
    """Persist *results* to ``reports/quality_results.json``.

    Args:
        results: List of :meth:`~src.models.QualityResult.to_dict` dicts.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info(f"_save_results: wrote {len(results)} result(s) → {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Structured quality checks
# ---------------------------------------------------------------------------


def run_structured_checks(
    config: dict[str, Any],
) -> list[QualityResult]:
    """Run structured quality checks on every CSV file in ``data/raw/``.

    Args:
        config: Pipeline configuration dict from :func:`~src.utils.load_config`.
            Passed for consistency with the semi-structured runner; the
            :class:`~StructuredValidator` loads its own config from the YAML
            file on disk.

    Returns:
        List of :class:`~src.models.QualityResult` objects, one per CSV file.
    """
    profiler  = StructuredProfiler()
    validator = StructuredValidator()  # loads dq_config.yaml itself
    results: list[QualityResult] = []

    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        logger.warning("run_structured_checks: no CSV files found in data/raw/")
        return results

    for csv_path in csv_files:
        dataset_name = csv_path.stem
        try:
            df      = profiler.load_data(str(csv_path))
            profile = profiler.profile(df, dataset_name=dataset_name)
            result  = validator.validate(df, profile, dataset_name=dataset_name)
            result.source_path = str(csv_path)
            results.append(result)
            logger.info(
                f"run_structured_checks: {dataset_name} | "
                f"score={result.overall_score:.1f} passed={result.passed}"
            )
        except Exception:
            logger.exception(
                f"run_structured_checks: failed for '{csv_path.name}'"
            )

    return results


# ---------------------------------------------------------------------------
# Semi-structured quality checks
# ---------------------------------------------------------------------------


def _find_raw_source(extracted_csv: Path) -> Path | None:
    """Resolve the original raw file for a given ``_extracted.csv``.

    The extraction stage names output files as ``<stem>_extracted.csv``, where
    ``<stem>`` is the raw file's stem.  This function strips the ``_extracted``
    suffix and searches ``data/raw/`` for a matching semi-structured file.

    Args:
        extracted_csv: Path to a ``_extracted.csv`` file.

    Returns:
        The matching raw file path, or ``None`` if no match is found.
    """
    # Strip the "_extracted" suffix that run_extraction.py appends
    if not extracted_csv.stem.endswith("_extracted"):
        return None

    raw_stem = extracted_csv.stem[: -len("_extracted")]

    for ext in _SEMI_STRUCTURED_EXTENSIONS:
        candidate = RAW_DIR / f"{raw_stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def run_semi_structured_checks(
    config: dict[str, Any],
) -> list[QualityResult]:
    """Run semi-structured quality checks on every ``_extracted.csv`` file.

    For each extracted CSV whose matching raw source exists in ``data/raw/``:

    1. Load and profile the *raw* source file.
    2. Load the flattened DataFrame from the extracted CSV.
    3. Validate with :class:`~SemiStructuredValidator`.

    Args:
        config: Pipeline configuration dict from :func:`~src.utils.load_config`.

    Returns:
        List of :class:`~src.models.QualityResult` objects.
    """
    profiler  = SemiStructuredProfiler(config)
    validator = SemiStructuredValidator(config)
    results: list[QualityResult] = []

    extracted_files = sorted(EXTRACTED_DIR.glob("*_extracted.csv"))
    if not extracted_files:
        logger.warning(
            "run_semi_structured_checks: no *_extracted.csv files found "
            "in data/extracted/"
        )
        return results

    for ext_csv in extracted_files:
        raw_path = _find_raw_source(ext_csv)
        if raw_path is None:
            logger.debug(
                f"run_semi_structured_checks: skipping '{ext_csv.name}' "
                "— no matching raw semi-structured source found"
            )
            continue

        dataset_name = raw_path.stem
        try:
            # Profile the original nested structure
            raw_data, fmt = profiler.load_data(str(raw_path))
            df_raw        = profiler.flatten_to_dataframe(raw_data)
            profile       = profiler.profile(df_raw, raw_data, dataset_name=dataset_name)

            # Validate the flattened DataFrame (as written by run_extraction.py)
            df_flat = pd.read_csv(ext_csv)

            result = validator.validate(df_flat, profile, dataset_name=dataset_name)
            result.source_path = str(raw_path)
            result.metadata["extracted_csv"] = str(ext_csv)
            result.metadata["format"]        = fmt
            results.append(result)

            logger.info(
                f"run_semi_structured_checks: {dataset_name} (format={fmt}) | "
                f"score={result.overall_score:.1f} passed={result.passed}"
            )
        except Exception:
            logger.exception(
                f"run_semi_structured_checks: failed for '{raw_path.name}'"
            )

    return results


# ---------------------------------------------------------------------------
# Text quality checks
# ---------------------------------------------------------------------------


def run_text_checks(
    config: dict[str, Any],
) -> list[QualityResult]:
    """Run text quality checks on every ``_text_features.json`` file.

    Loads all feature dicts first so that the full collection can be passed
    to :meth:`~TextValidator.check_uniqueness` as ``all_features``, enabling
    cross-document duplicate detection within a single pipeline run.

    For each feature file:

    1. Load the feature dict from JSON (written by
       :func:`~src.modules.run_extraction.extract_text`).
    2. Validate with :class:`~TextValidator`, passing the full ``all_features``
       list for uniqueness scoring.
    3. Use the ``source_path`` stored in the feature dict for freshness scoring.

    Args:
        config: Pipeline configuration dict from :func:`~src.utils.load_config`.

    Returns:
        List of :class:`~src.models.QualityResult` objects, one per text
        feature file.
    """
    validator = TextValidator(config)
    results: list[QualityResult] = []

    feature_files = sorted(EXTRACTED_DIR.glob("*_text_features.json"))
    if not feature_files:
        logger.warning(
            "run_text_checks: no *_text_features.json files found in "
            "data/extracted/"
        )
        return results

    # Load all feature dicts upfront for cross-document uniqueness comparison
    all_features: list[dict[str, Any]] = []
    for feat_path in feature_files:
        try:
            with feat_path.open("r", encoding="utf-8") as fh:
                all_features.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            logger.warning(
                f"run_text_checks: could not read '{feat_path.name}' — skipping"
            )
            all_features.append({})  # placeholder keeps indices aligned

    for feat_path, features in zip(feature_files, all_features):
        if not features:
            continue  # was unreadable

        dataset_name = str(features.get("dataset_name", feat_path.stem))
        source_path  = str(features.get("source_path", ""))

        try:
            result = validator.validate(
                features,
                dataset_name=dataset_name,
                source_path=source_path,
                all_features=all_features,
            )
            result.metadata["features_file"] = str(feat_path)
            results.append(result)

            logger.info(
                f"run_text_checks: {dataset_name} | "
                f"score={result.overall_score:.1f} passed={result.passed}"
            )
        except Exception:
            logger.exception(
                f"run_text_checks: failed for '{feat_path.name}'"
            )

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all quality checks and persist results to ``reports/quality_results.json``."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()

    logger.info("run_quality_checks: starting structured checks")
    structured_results = run_structured_checks(config)

    logger.info("run_quality_checks: starting semi-structured checks")
    semi_results = run_semi_structured_checks(config)

    logger.info("run_quality_checks: starting text checks")
    text_results = run_text_checks(config)

    all_new = structured_results + semi_results + text_results

    # Preserve any results written by previous pipeline runs
    existing = _load_existing_results()
    combined = existing + [r.to_dict() for r in all_new]
    _save_results(combined)

    pass_count = sum(1 for r in all_new if r.passed)
    fail_count = len(all_new) - pass_count
    logger.info(
        f"run_quality_checks: complete — {len(all_new)} dataset(s) checked | "
        f"passed={pass_count} failed={fail_count} "
        f"({len(structured_results)} structured, {len(semi_results)} semi-structured, "
        f"{len(text_results)} text)"
    )


if __name__ == "__main__":
    main()
