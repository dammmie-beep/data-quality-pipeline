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

Run directly::

    PYTHONPATH=. python src/modules/run_extraction.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from loguru import logger

from src.modules.semi_structured.profiler import SemiStructuredProfiler
from src.utils import load_config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR       = Path("data/raw")
EXTRACTED_DIR = Path("data/extracted")

# File extensions handled by the semi-structured profiler
_SEMI_STRUCTURED_EXTENSIONS = {".json", ".jsonl", ".log"}


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
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all extraction steps and report a summary."""
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    config   = load_config()
    profiler = SemiStructuredProfiler(config)

    logger.info("run_extraction: starting structured extraction")
    structured_files = extract_structured()

    logger.info("run_extraction: starting semi-structured extraction")
    semi_files = extract_semi_structured(profiler)

    total = len(structured_files) + len(semi_files)
    logger.info(
        f"run_extraction: complete — {total} file(s) written to data/extracted/ "
        f"({len(structured_files)} structured, {len(semi_files)} semi-structured)"
    )


if __name__ == "__main__":
    main()
