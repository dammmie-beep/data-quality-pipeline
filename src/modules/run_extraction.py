"""
Extraction stage entry-point.

Reads raw data files from ``data/raw/``, applies any necessary
transformations, and writes analysis-ready outputs to ``data/extracted/``.

Structured extraction
---------------------
CSV files in ``data/raw/`` are copied as-is to ``data/extracted/`` (the
structured profiler operates on raw CSV directly, so no flattening is needed).

Semi-structured extraction
--------------------------
For each ``.json``, ``.jsonl``, and ``.log`` file in ``data/raw/``:

1. Load the raw file using :meth:`SemiStructuredProfiler.load_data`.
2. Flatten the nested structure into a tidy DataFrame using
   :meth:`SemiStructuredProfiler.flatten_to_dataframe`.
3. Write the flattened DataFrame as ``<stem>_extracted.csv`` to
   ``data/extracted/``.

Text extraction
---------------
For each ``.txt``, ``.pdf``, and ``.eml`` file in ``data/raw/``, and for any
``.csv`` file whose headers contain a recognisable free-text column (e.g.
``comment_text``, ``body``, ``content``):

1. Load the file using :meth:`TextExtractor.load_data` to obtain ``raw_text``.
2. Extract features using :meth:`TextExtractor.extract`.
3. Write the feature dict as ``<stem>_text_features.json`` to
   ``data/extracted/``.
4. Write the raw extracted text as ``<stem>_raw.txt`` to ``data/extracted/``.

Run directly::

    PYTHONPATH=. python src/modules/run_extraction.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from loguru import logger

from src.modules.semi_structured.profiler import SemiStructuredProfiler
from src.modules.text.extractor import TextExtractor
from src.utils import load_config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR       = Path("data/raw")
EXTRACTED_DIR = Path("data/extracted")

# File extensions handled by each extractor
_SEMI_STRUCTURED_EXTENSIONS = {".json", ".jsonl", ".log"}
_TEXT_DIRECT_EXTENSIONS     = {".txt", ".pdf", ".eml"}

# Column-name substrings that mark a CSV column as free text
_TEXT_COLUMN_INDICATORS: frozenset[str] = frozenset(
    {"text", "body", "content", "comment", "description", "message", "caption", "post", "review"}
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_text_csv(path: Path) -> bool:
    """Return ``True`` if *path* is a CSV that contains a free-text column.

    Reads only the header row to decide, so the cost is negligible even for
    large files.  A column is considered free-text when any of the substrings
    in :data:`_TEXT_COLUMN_INDICATORS` appears in the column name (case-
    insensitive).  This distinguishes comment/social-data CSVs from numeric
    order/transaction CSVs that are handled by the structured module.

    Args:
        path: Path to a ``.csv`` file.

    Returns:
        ``True`` when at least one header matches a text indicator; ``False``
        otherwise or when the file cannot be read.
    """
    import csv  # noqa: PLC0415

    try:
        with path.open(newline="", encoding="utf-8") as fh:
            headers = next(csv.reader(fh), [])
    except (OSError, StopIteration):
        return False

    return any(
        any(indicator in h.lower() for indicator in _TEXT_COLUMN_INDICATORS)
        for h in headers
    )


# ---------------------------------------------------------------------------
# Structured extraction
# ---------------------------------------------------------------------------


def extract_structured() -> list[Path]:
    """Copy CSV files from ``data/raw/`` to ``data/extracted/`` unchanged.

    The structured profiler loads CSV files directly, so no transformation is
    required.  This step ensures that both structured and semi-structured
    artefacts land in the same extraction output directory.

    Returns:
        List of paths written to ``data/extracted/``.
    """
    written: list[Path] = []

    for csv_path in sorted(RAW_DIR.glob("*.csv")):
        dest = EXTRACTED_DIR / csv_path.name
        shutil.copy2(csv_path, dest)
        logger.info(f"extract_structured: copied {csv_path.name} → {dest}")
        written.append(dest)

    if not written:
        logger.warning("extract_structured: no CSV files found in data/raw/")

    return written


# ---------------------------------------------------------------------------
# Semi-structured extraction
# ---------------------------------------------------------------------------


def extract_semi_structured(profiler: SemiStructuredProfiler) -> list[Path]:
    """Flatten semi-structured files and write them as CSV to ``data/extracted/``.

    For each ``.json``, ``.jsonl``, and ``.log`` file found in ``data/raw/``:

    1. Load the file with :meth:`~SemiStructuredProfiler.load_data`.
    2. Flatten it with :meth:`~SemiStructuredProfiler.flatten_to_dataframe`.
    3. Save the resulting DataFrame as ``<stem>_extracted.csv``.

    Args:
        profiler: An initialised :class:`~SemiStructuredProfiler` instance.

    Returns:
        List of paths written to ``data/extracted/``.
    """
    written: list[Path] = []

    candidates = sorted(
        p for p in RAW_DIR.iterdir()
        if p.suffix.lower() in _SEMI_STRUCTURED_EXTENSIONS
    )

    if not candidates:
        logger.warning(
            f"extract_semi_structured: no {_SEMI_STRUCTURED_EXTENSIONS} "
            "files found in data/raw/"
        )
        return written

    for raw_path in candidates:
        try:
            data, fmt = profiler.load_data(str(raw_path))
            df = profiler.flatten_to_dataframe(data)

            dest_name = f"{raw_path.stem}_extracted.csv"
            dest = EXTRACTED_DIR / dest_name
            df.to_csv(dest, index=False)

            logger.info(
                f"extract_semi_structured: {raw_path.name} "
                f"(format={fmt}, rows={len(df)}, cols={len(df.columns)}) "
                f"→ {dest}"
            )
            written.append(dest)

        except Exception:
            logger.exception(
                f"extract_semi_structured: failed to process '{raw_path.name}'"
            )

    return written


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_text(extractor: TextExtractor) -> list[Path]:
    """Extract text features from text-bearing files and write them to ``data/extracted/``.

    Processes three categories of source file found in ``data/raw/``:

    * ``.txt``, ``.pdf``, ``.eml`` — always treated as text sources.
    * ``.csv`` — treated as a text source only when :func:`_is_text_csv`
      returns ``True`` (i.e. the file has at least one free-text column such
      as ``comment_text`` or ``body``).  Plain order/transaction CSVs handled
      by the structured module are excluded.

    For each qualifying file, two artefacts are written to ``data/extracted/``:

    * ``<stem>_text_features.json`` — the feature dict from
      :meth:`~TextExtractor.extract`, augmented with a ``source_path`` key so
      the quality-check stage can locate the original file for freshness
      scoring.
    * ``<stem>_raw.txt`` — the raw extracted text string.

    Args:
        extractor: An initialised :class:`~TextExtractor` instance.

    Returns:
        List of ``.json`` feature-file paths written to ``data/extracted/``.
    """
    written: list[Path] = []

    # Collect candidates: direct text files + text-bearing CSVs
    candidates: list[Path] = []
    for p in sorted(RAW_DIR.iterdir()):
        suffix = p.suffix.lower()
        if suffix in _TEXT_DIRECT_EXTENSIONS:
            candidates.append(p)
        elif suffix == ".csv" and _is_text_csv(p):
            candidates.append(p)

    if not candidates:
        logger.warning(
            "extract_text: no text source files found in data/raw/ "
            f"(looked for {_TEXT_DIRECT_EXTENSIONS | {'.csv'}})"
        )
        return written

    for raw_path in candidates:
        dataset_name = raw_path.stem
        try:
            raw_text, detected_type = extractor.load_data(str(raw_path))
            features = extractor.extract(raw_text, dataset_name=dataset_name)

            # Augment with source_path so run_quality_checks.py can use it
            features["source_path"] = str(raw_path)

            # Write feature dict as JSON
            json_dest = EXTRACTED_DIR / f"{dataset_name}_text_features.json"
            with json_dest.open("w", encoding="utf-8") as fh:
                json.dump(features, fh, indent=2)

            # Write raw extracted text
            txt_dest = EXTRACTED_DIR / f"{dataset_name}_raw.txt"
            txt_dest.write_text(raw_text, encoding="utf-8")

            logger.info(
                f"extract_text: {raw_path.name} (type={detected_type}) | "
                f"words={features['word_count']} lang={features['language']} "
                f"→ {json_dest.name}, {txt_dest.name}"
            )
            written.append(json_dest)

        except Exception:
            logger.exception(
                f"extract_text: failed to process '{raw_path.name}'"
            )

    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all extraction steps and report a summary."""
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    config    = load_config()
    profiler  = SemiStructuredProfiler(config)
    extractor = TextExtractor(config)

    logger.info("run_extraction: starting structured extraction")
    structured_files = extract_structured()

    logger.info("run_extraction: starting semi-structured extraction")
    semi_files = extract_semi_structured(profiler)

    logger.info("run_extraction: starting text extraction")
    text_files = extract_text(extractor)

    total = len(structured_files) + len(semi_files) + len(text_files)
    logger.info(
        f"run_extraction: complete — {total} file(s) written to data/extracted/ "
        f"({len(structured_files)} structured, {len(semi_files)} semi-structured, "
        f"{len(text_files)} text feature files)"
    )


if __name__ == "__main__":
    main()
