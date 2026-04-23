"""
CML Reporter for the Data Quality Pipeline.

Reads ``reports/quality_scores.json`` (written by the scorer), backs up the
previous run's scores for trend comparison, then generates a rich Markdown
report at ``reports/cml_report.md`` suitable for posting as a CI/CD comment
via the CML CLI.

Report sections
---------------
1. **Pipeline summary** — overall status badge, score statistics, dataset
   counts broken down by PASS / WARN / FAIL.
2. **Dataset summary table** — one row per dataset showing name, type,
   score, label, and total failed checks.
3. **Dimension score table** — heatmap-style table showing all six dimension
   scores for every dataset.
4. **Trend comparison** — score delta vs. the previous run (``+`` / ``-``
   or ``new``) when a backup is available.
5. **Failed checks detail** — collapsible section per dataset listing every
   failed check with its severity, column, expected, and actual value.

Intended to be run as the fifth DVC stage::

    python src/reporting/cml_reporter.py
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.utils import load_config

# ---------------------------------------------------------------------------
# File paths (relative to project root, matching dvc.yaml stage definitions)
# ---------------------------------------------------------------------------
QUALITY_SCORES_PATH  = Path("reports/quality_scores.json")
QUALITY_SCORES_BACKUP = Path("reports/quality_scores.previous.json")
CML_REPORT_PATH      = Path("reports/cml_report.md")

# ---------------------------------------------------------------------------
# Label → emoji badge mapping
# ---------------------------------------------------------------------------
_LABEL_BADGE = {
    "PASS": "✅ PASS",
    "WARN": "⚠️ WARN",
    "FAIL": "❌ FAIL",
}

# ---------------------------------------------------------------------------
# Dimension display order (matches scoring_weights key order)
# ---------------------------------------------------------------------------
_DIMENSIONS = [
    "completeness",
    "accuracy",
    "consistency",
    "freshness",
    "uniqueness",
    "validity",
]


class CMLReporter:
    """Generate a Markdown quality report from ``quality_scores.json``.

    The reporter is deliberately read-only with respect to scoring data — it
    never modifies scores, only reads them and formats them.  The only files
    it writes are ``reports/cml_report.md`` (the report) and
    ``reports/quality_scores.previous.json`` (the backup for trend comparison).

    Attributes:
        pass_score: Minimum score for the PASS label (from config).
        warn_score: Minimum score for the WARN label (from config).
    """

    def __init__(self, config_path: str = "dq_config.yaml") -> None:
        """Initialise the reporter and load pipeline configuration.

        Args:
            config_path: Path to the YAML configuration file. Defaults to
                ``dq_config.yaml`` in the current working directory.
        """
        self._config    = load_config(config_path)
        thresholds      = self._config["thresholds"]
        self.pass_score = thresholds["pass_score"]
        self.warn_score = thresholds["warn_score"]
        logger.debug(
            f"CMLReporter initialised | pass={self.pass_score} warn={self.warn_score}"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        scores_path: Path = QUALITY_SCORES_PATH,
        report_path: Path = CML_REPORT_PATH,
        backup_path: Path = QUALITY_SCORES_BACKUP,
    ) -> str:
        """Read scores, back up previous run, generate and save the report.

        This is the main entry point called by the DVC ``report`` stage.

        Workflow:

        1. Load ``quality_scores.json``.
        2. Load ``quality_scores.previous.json`` if it exists (for trend data).
        3. Back up the current scores file to ``quality_scores.previous.json``
           so the *next* run can diff against it.
        4. Build the Markdown report string.
        5. Write the report to ``reports/cml_report.md``.

        Args:
            scores_path: Path to the scored results JSON. Defaults to
                ``reports/quality_scores.json``.
            report_path: Destination path for the Markdown report. Defaults to
                ``reports/cml_report.md``. The parent directory is created if
                it does not exist.
            backup_path: Path where the current scores file will be copied
                before the next run overwrites it. Defaults to
                ``reports/quality_scores.previous.json``.

        Returns:
            The generated Markdown report as a string.

        Raises:
            FileNotFoundError: If *scores_path* does not exist.
            json.JSONDecodeError: If *scores_path* contains invalid JSON.
        """
        logger.info(f"run: reading scores from '{scores_path}'")

        if not scores_path.exists():
            raise FileNotFoundError(
                f"Quality scores file not found: {scores_path.resolve()}\n"
                "Ensure the score DVC stage has run successfully."
            )

        with scores_path.open("r", encoding="utf-8") as fh:
            scored: dict[str, Any] = json.load(fh)

        # Load previous run for trend comparison (best-effort).
        previous: dict[str, Any] | None = None
        if backup_path.exists():
            try:
                with backup_path.open("r", encoding="utf-8") as fh:
                    previous = json.load(fh)
                logger.info("run: previous scores loaded for trend comparison")
            except Exception as exc:
                logger.warning(f"run: could not load previous scores — {exc}")

        # Back up current scores so the next run can compare against them.
        self._backup_scores(scores_path, backup_path)

        # Build and write the report.
        report_md = self.build_report(scored, previous)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fh:
            fh.write(report_md)

        logger.info(f"run: report written to '{report_path}'")
        return report_md

    def build_report(
        self,
        scored:   dict[str, Any],
        previous: dict[str, Any] | None = None,
    ) -> str:
        """Build the complete Markdown report string.

        Assembles all report sections in order and joins them with blank
        lines.  Each section is generated by a dedicated private method so
        individual sections can be unit-tested in isolation.

        Args:
            scored: The scored output dict produced by
                :class:`~src.scoring.scorer.Scorer` — must contain
                ``pipeline_summary`` and ``datasets`` keys.
            previous: Optional previous-run scored dict used to compute score
                deltas in the trend section.  Pass ``None`` to skip the trend
                section.

        Returns:
            A Markdown string ready to be written to a file or piped to
            ``cml comment create``.
        """
        datasets: list[dict[str, Any]] = scored.get("datasets", [])
        summary:  dict[str, Any]       = scored.get("pipeline_summary", {})

        sections = [
            self._section_header(summary),
            self._section_pipeline_summary(summary),
            self._section_dataset_table(datasets),
            self._section_dimension_table(datasets),
        ]

        if previous is not None:
            sections.append(self._section_trend(datasets, previous))

        sections.append(self._section_failed_checks(datasets))
        sections.append(self._section_footer())

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Report sections
    # ------------------------------------------------------------------

    def _section_header(self, summary: dict[str, Any]) -> str:
        """Generate the top-level report header with status badge.

        Args:
            summary: The ``pipeline_summary`` block from the scored output.

        Returns:
            A Markdown string containing the H1 title and run timestamp.
        """
        pipeline_passed = summary.get("pipeline_passed", False)
        badge           = "✅ PASSED" if pipeline_passed else "❌ FAILED"
        now             = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"# Data Quality Pipeline Report\n\n"
            f"**Status:** {badge}  \n"
            f"**Generated:** {now}"
        )

    def _section_pipeline_summary(self, summary: dict[str, Any]) -> str:
        """Generate the pipeline-level summary block.

        Includes mean / min / max scores, dataset counts, and pass/warn/fail
        percentages.

        Args:
            summary: The ``pipeline_summary`` block from the scored output.

        Returns:
            Markdown string for the summary section.
        """
        total      = summary.get("total_datasets", 0)
        mean_score = summary.get("mean_score",    0.0)
        min_score  = summary.get("min_score",     0.0)
        max_score  = summary.get("max_score",     0.0)
        pass_count = summary.get("pass_count",    0)
        warn_count = summary.get("warn_count",    0)
        fail_count = summary.get("fail_count",    0)
        pass_pct   = summary.get("pass_pct",      0.0)
        warn_pct   = summary.get("warn_pct",      0.0)
        fail_pct   = summary.get("fail_pct",      0.0)
        thresholds = summary.get("thresholds",    {})

        lines = [
            "## Pipeline Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Datasets assessed | {total} |",
            f"| Mean overall score | **{mean_score}** |",
            f"| Min score | {min_score} |",
            f"| Max score | {max_score} |",
            f"| ✅ PASS ({thresholds.get('pass_score', '—')}+) "
            f"| {pass_count} ({pass_pct}%) |",
            f"| ⚠️ WARN ({thresholds.get('warn_score', '—')}–"
            f"{thresholds.get('pass_score', '—')}) "
            f"| {warn_count} ({warn_pct}%) |",
            f"| ❌ FAIL (<{thresholds.get('warn_score', '—')}) "
            f"| {fail_count} ({fail_pct}%) |",
        ]
        return "\n".join(lines)

    def _section_dataset_table(
        self, datasets: list[dict[str, Any]]
    ) -> str:
        """Generate the per-dataset summary table.

        One row per dataset showing name, data type, overall score, label,
        and count of failed checks.

        Args:
            datasets: The ``datasets`` list from the scored output.

        Returns:
            Markdown string for the dataset table section.
        """
        if not datasets:
            return "## Dataset Results\n\n_No datasets found._"

        header = (
            "## Dataset Results\n\n"
            "| Dataset | Type | Score | Label | Failed Checks |\n"
            "|---------|------|------:|-------|:-------------:|"
        )
        rows = []
        for ds in datasets:
            name          = ds.get("dataset_name", "—")
            dtype         = ds.get("data_type", "—")
            score         = ds.get("overall_score", 0.0)
            label         = ds.get("label", "FAIL")
            failed_count  = len(ds.get("failed_checks", []))
            badge         = _LABEL_BADGE.get(label, label)
            rows.append(
                f"| {name} | {dtype} | {score} | {badge} | {failed_count} |"
            )

        return header + "\n" + "\n".join(rows)

    def _section_dimension_table(
        self, datasets: list[dict[str, Any]]
    ) -> str:
        """Generate the dimension score heatmap table.

        Each row is a dataset; columns are the six quality dimensions.
        Scores below the warn threshold are italicised; scores below half the
        warn threshold are bold-italicised to draw the eye.

        Args:
            datasets: The ``datasets`` list from the scored output.

        Returns:
            Markdown string for the dimension table section.
        """
        if not datasets:
            return "## Dimension Scores\n\n_No datasets found._"

        dim_headers = " | ".join(d.capitalize() for d in _DIMENSIONS)
        separator   = " | ".join(["------:"] * len(_DIMENSIONS))
        header = (
            f"## Dimension Scores\n\n"
            f"| Dataset | {dim_headers} |\n"
            f"|---------|{separator}|"
        )

        rows = []
        for ds in datasets:
            name       = ds.get("dataset_name", "—")
            dimensions = ds.get("dimensions", {})
            cells = []
            for dim in _DIMENSIONS:
                score = dimensions.get(dim, 0.0)
                cells.append(self._format_score_cell(score))
            rows.append(f"| {name} | " + " | ".join(cells) + " |")

        return header + "\n" + "\n".join(rows)

    def _section_trend(
        self,
        datasets: list[dict[str, Any]],
        previous: dict[str, Any],
    ) -> str:
        """Generate the score trend table comparing current vs. previous run.

        For each dataset present in the current run, the delta against the
        previous run is computed.  New datasets are shown as ``new``; datasets
        that disappeared are omitted (the diff is one-directional).

        Args:
            datasets: The ``datasets`` list from the current scored output.
            previous: The full scored dict from the previous run.

        Returns:
            Markdown string for the trend section.
        """
        prev_datasets = {
            d["dataset_name"]: d
            for d in previous.get("datasets", [])
        }

        header = (
            "## Score Trend (vs. Previous Run)\n\n"
            "| Dataset | Previous | Current | Delta |\n"
            "|---------|--------:|---------:|------:|"
        )

        rows = []
        for ds in datasets:
            name          = ds.get("dataset_name", "—")
            current_score = float(ds.get("overall_score", 0.0))

            if name in prev_datasets:
                prev_score = float(prev_datasets[name].get("overall_score", 0.0))
                delta      = round(current_score - prev_score, 2)
                delta_str  = f"+{delta}" if delta > 0 else str(delta)
                trend_icon = "📈" if delta > 0 else ("📉" if delta < 0 else "➡️")
                rows.append(
                    f"| {name} | {prev_score} | {current_score} "
                    f"| {trend_icon} {delta_str} |"
                )
            else:
                rows.append(
                    f"| {name} | — | {current_score} | 🆕 new |"
                )

        if not rows:
            return "## Score Trend (vs. Previous Run)\n\n_No comparable datasets found._"

        return header + "\n" + "\n".join(rows)

    def _section_failed_checks(
        self, datasets: list[dict[str, Any]]
    ) -> str:
        """Generate the failed checks detail section.

        Each dataset with at least one failed check gets a collapsible
        ``<details>`` block containing a Markdown table of all failures
        grouped by severity (errors first, then warnings).

        Args:
            datasets: The ``datasets`` list from the scored output.

        Returns:
            Markdown string for the failed checks section.
        """
        lines = ["## Failed Checks Detail"]

        datasets_with_failures = [
            ds for ds in datasets if ds.get("failed_checks")
        ]

        if not datasets_with_failures:
            lines.append("\n_No failed checks — all datasets are clean._ ✅")
            return "\n".join(lines)

        for ds in datasets_with_failures:
            name          = ds.get("dataset_name", "—")
            label         = ds.get("label", "FAIL")
            badge         = _LABEL_BADGE.get(label, label)
            failed_checks = ds.get("failed_checks", [])

            # Sort: errors before warnings.
            sorted_checks = sorted(
                failed_checks,
                key=lambda c: (0 if c.get("severity") == "error" else 1),
            )

            table_rows = []
            for chk in sorted_checks:
                severity = chk.get("severity", "—")
                icon     = "❌" if severity == "error" else "⚠️"
                table_rows.append(
                    f"| {icon} {severity} "
                    f"| {chk.get('check', '—')} "
                    f"| `{chk.get('column', '—')}` "
                    f"| {chk.get('expected', '—')} "
                    f"| {chk.get('actual', '—')} |"
                )

            table = (
                "| Severity | Check | Column | Expected | Actual |\n"
                "|----------|-------|--------|----------|--------|"
            )
            table += "\n" + "\n".join(table_rows)

            block = (
                f"\n<details>\n"
                f"<summary><strong>{name}</strong> — {badge} "
                f"({len(failed_checks)} failure(s))</summary>\n\n"
                f"{table}\n\n"
                f"</details>"
            )
            lines.append(block)

        return "\n".join(lines)

    def _section_footer(self) -> str:
        """Generate the report footer with a link back to the pipeline.

        Returns:
            Markdown string for the footer.
        """
        return (
            "---\n\n"
            "_Report generated by the [Data Quality Pipeline](../README.md). "
            "Scores are computed using weighted dimension checks defined in "
            "`dq_config.yaml`._"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_score_cell(self, score: float) -> str:
        """Format a dimension score with visual emphasis based on severity.

        * Score < half warn threshold → ``***bold italic***`` (critical)
        * Score < warn threshold      → ``*italic*`` (warning)
        * Otherwise                   → plain number

        Args:
            score: Dimension score in [0, 100].

        Returns:
            A Markdown-formatted string representation of the score.
        """
        critical_threshold = self.warn_score / 2
        if score < critical_threshold:
            return f"***{score}***"
        if score < self.warn_score:
            return f"*{score}*"
        return str(score)

    @staticmethod
    def _backup_scores(
        scores_path: Path,
        backup_path: Path,
    ) -> None:
        """Copy the current scores file to the backup path.

        A best-effort operation — logs a warning rather than raising if the
        copy fails (e.g. due to a permissions error).

        Args:
            scores_path: Path to the file to back up.
            backup_path: Destination path for the backup copy.
        """
        try:
            shutil.copy2(scores_path, backup_path)
            logger.debug(f"Backed up scores: '{scores_path}' → '{backup_path}'")
        except Exception as exc:
            logger.warning(f"Could not back up scores file — {exc}")


# ---------------------------------------------------------------------------
# Entry point (invoked by DVC stage: python src/reporting/cml_reporter.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    reporter = CMLReporter()
    try:
        reporter.run()
        sys.exit(0)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(2)
    except Exception as exc:
        logger.exception(f"Unexpected error in reporter: {exc}")
        sys.exit(3)
