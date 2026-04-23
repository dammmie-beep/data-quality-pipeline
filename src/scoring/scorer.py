"""
Scorer for the Data Quality Pipeline.

Reads the raw per-dataset results written by each module to
``reports/quality_results.json``, applies PASS / WARN / FAIL labels using the
thresholds from ``dq_config.yaml``, computes a pipeline-level summary, and
writes the enriched output to ``reports/quality_scores.json``.

Intended to be run as the fourth DVC stage::

    python src/scoring/scorer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from src.utils import load_config

# ---------------------------------------------------------------------------
# File paths (relative to project root, matching dvc.yaml stage definitions)
# ---------------------------------------------------------------------------
QUALITY_RESULTS_PATH = Path("reports/quality_results.json")
QUALITY_SCORES_PATH  = Path("reports/quality_scores.json")

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------
LABEL_PASS = "PASS"
LABEL_WARN = "WARN"
LABEL_FAIL = "FAIL"


class Scorer:
    """Apply quality labels, compute pipeline summaries, and persist scores.

    The scorer is the bridge between raw module output
    (``reports/quality_results.json``) and everything downstream — the
    reporter, API, and alerting system all read from the file this class
    writes (``reports/quality_scores.json``).

    Labelling rules (read from ``dq_config.yaml``):

    * **PASS** — ``overall_score >= thresholds.pass_score``
    * **WARN** — ``thresholds.warn_score <= overall_score < pass_score``
    * **FAIL** — ``overall_score < thresholds.warn_score``

    The pipeline summary aggregates across all datasets:

    * Mean, min, and max overall score.
    * Count and percentage of PASS / WARN / FAIL datasets.
    * A top-level ``pipeline_passed`` flag that is ``True`` only when every
      dataset is labelled PASS.

    Attributes:
        pass_score: Minimum score for a PASS label.
        warn_score: Minimum score for a WARN label (below → FAIL).
        weights: Dimension weights from config (forwarded into the output for
            reference by downstream consumers).
    """

    def __init__(self, config_path: str = "dq_config.yaml") -> None:
        """Initialise the scorer and load pipeline configuration.

        Args:
            config_path: Path to the YAML configuration file. Defaults to
                ``dq_config.yaml`` in the current working directory.
        """
        self._config    = load_config(config_path)
        thresholds      = self._config["thresholds"]
        self.pass_score = thresholds["pass_score"]
        self.warn_score = thresholds["warn_score"]
        self.weights    = self._config["scoring_weights"]
        logger.debug(
            f"Scorer initialised | pass={self.pass_score} warn={self.warn_score}"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def label(self, overall_score: float) -> str:
        """Return the PASS / WARN / FAIL label for a given overall score.

        Args:
            overall_score: Composite quality score in [0, 100].

        Returns:
            ``"PASS"`` if ``overall_score >= pass_score``,
            ``"WARN"`` if ``warn_score <= overall_score < pass_score``,
            ``"FAIL"`` otherwise.

        Example::

            scorer = Scorer()
            scorer.label(92.0)   # "PASS"
            scorer.label(70.0)   # "WARN"
            scorer.label(50.0)   # "FAIL"
        """
        if overall_score >= self.pass_score:
            return LABEL_PASS
        if overall_score >= self.warn_score:
            return LABEL_WARN
        return LABEL_FAIL

    def score_results(
        self, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Enrich a list of raw QualityResult dicts with labels and a summary.

        Iterates over each result, adds a ``label`` field, recomputes
        ``passed`` to match the label, and builds a ``pipeline_summary``
        block that aggregates statistics across all datasets.

        Args:
            results: List of dicts as produced by
                :meth:`~src.models.QualityResult.to_dict`. Each dict must
                contain at least ``dataset_name``, ``overall_score``, and
                ``dimensions``.

        Returns:
            A dict with the following top-level structure::

                {
                    "pipeline_summary": {
                        "total_datasets":    int,
                        "pass_count":        int,
                        "warn_count":        int,
                        "fail_count":        int,
                        "pass_pct":          float,   # percentage, 0-100
                        "warn_pct":          float,
                        "fail_pct":          float,
                        "mean_score":        float,
                        "min_score":         float,
                        "max_score":         float,
                        "pipeline_passed":   bool,    # True iff all datasets PASS
                        "scoring_weights":   dict,
                        "thresholds":        dict,
                    },
                    "datasets": [
                        {
                            ...all QualityResult fields...,
                            "label": "PASS" | "WARN" | "FAIL",
                        },
                        ...
                    ]
                }

        Raises:
            ValueError: If *results* is empty.
        """
        if not results:
            raise ValueError("Cannot score an empty results list.")

        labelled: list[dict[str, Any]] = []
        for result in results:
            enriched               = dict(result)
            overall_score          = float(enriched.get("overall_score", 0.0))
            enriched["label"]      = self.label(overall_score)
            enriched["passed"]     = enriched["label"] == LABEL_PASS
            labelled.append(enriched)
            logger.debug(
                f"  scored dataset='{enriched.get('dataset_name')}' "
                f"score={overall_score} label={enriched['label']}"
            )

        summary = self._build_summary(labelled)
        logger.info(
            f"score_results: {summary['total_datasets']} dataset(s) | "
            f"PASS={summary['pass_count']} WARN={summary['warn_count']} "
            f"FAIL={summary['fail_count']} pipeline_passed={summary['pipeline_passed']}"
        )

        return {
            "pipeline_summary": summary,
            "datasets":         labelled,
        }

    def run(
        self,
        input_path:  Path = QUALITY_RESULTS_PATH,
        output_path: Path = QUALITY_SCORES_PATH,
    ) -> dict[str, Any]:
        """Read quality_results.json, score it, and write quality_scores.json.

        This is the main entry point called by the DVC ``score`` stage.

        Args:
            input_path: Path to the raw results JSON file written by module
                validators. Defaults to ``reports/quality_results.json``.
            output_path: Path where the scored output will be written.
                Defaults to ``reports/quality_scores.json``. The parent
                directory is created if it does not exist.

        Returns:
            The scored output dict (same structure as :meth:`score_results`
            returns) — useful for testing without touching the filesystem.

        Raises:
            FileNotFoundError: If *input_path* does not exist.
            json.JSONDecodeError: If *input_path* contains invalid JSON.
            ValueError: If the JSON root is not a list or a dict with a
                ``"datasets"`` key.
        """
        logger.info(f"run: reading results from '{input_path}'")

        if not input_path.exists():
            raise FileNotFoundError(
                f"Quality results file not found: {input_path.resolve()}\n"
                "Ensure the quality_check DVC stage has run successfully."
            )

        with input_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        # Accept either a bare list or a dict with a "datasets" key so the
        # scorer is tolerant of both output shapes from module validators.
        if isinstance(raw, list):
            results = raw
        elif isinstance(raw, dict) and "datasets" in raw:
            results = raw["datasets"]
        else:
            raise ValueError(
                f"Expected quality_results.json to contain a JSON list or a dict "
                f"with a 'datasets' key, got {type(raw).__name__}."
            )

        logger.info(f"run: {len(results)} dataset result(s) loaded")
        scored = self.score_results(results)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(scored, fh, indent=2)

        logger.info(f"run: scores written to '{output_path}'")
        return scored

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_summary(
        self, labelled: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compute pipeline-level aggregate statistics from labelled results.

        Args:
            labelled: List of enriched result dicts, each containing a
                ``label`` field (``"PASS"``, ``"WARN"``, or ``"FAIL"``) and
                an ``overall_score`` field.

        Returns:
            A summary dict containing counts, percentages, score statistics,
            and the ``pipeline_passed`` flag.
        """
        total       = len(labelled)
        pass_count  = sum(1 for r in labelled if r["label"] == LABEL_PASS)
        warn_count  = sum(1 for r in labelled if r["label"] == LABEL_WARN)
        fail_count  = sum(1 for r in labelled if r["label"] == LABEL_FAIL)
        scores      = [float(r.get("overall_score", 0.0)) for r in labelled]

        def pct(n: int) -> float:
            return round(n / total * 100, 1) if total else 0.0

        return {
            "total_datasets":  total,
            "pass_count":      pass_count,
            "warn_count":      warn_count,
            "fail_count":      fail_count,
            "pass_pct":        pct(pass_count),
            "warn_pct":        pct(warn_count),
            "fail_pct":        pct(fail_count),
            "mean_score":      round(sum(scores) / total, 2) if total else 0.0,
            "min_score":       round(min(scores), 2)         if scores else 0.0,
            "max_score":       round(max(scores), 2)         if scores else 0.0,
            "pipeline_passed": fail_count == 0 and warn_count == 0,
            "scoring_weights": self.weights,
            "thresholds": {
                "pass_score": self.pass_score,
                "warn_score": self.warn_score,
            },
        }


# ---------------------------------------------------------------------------
# Entry point (invoked by DVC stage: python src/scoring/scorer.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    scorer = Scorer()
    try:
        scored = scorer.run()
        summary = scored["pipeline_summary"]
        passed  = summary["pipeline_passed"]
        logger.info(
            f"Pipeline {'PASSED' if passed else 'FAILED'} | "
            f"mean_score={summary['mean_score']} "
            f"PASS={summary['pass_count']} "
            f"WARN={summary['warn_count']} "
            f"FAIL={summary['fail_count']}"
        )
        sys.exit(0 if passed else 1)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(2)
    except Exception as exc:
        logger.exception(f"Unexpected error in scorer: {exc}")
        sys.exit(3)
