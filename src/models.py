"""
Shared data models for the Data Quality Pipeline.

This module defines the canonical result type — QualityResult — that every
module (structured, text, audio, image, video, semi-structured) must produce.
Using a single shared model keeps the scoring, reporting, and API layers
decoupled from any one module's internal representation.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class QualityResult:
    """Represents the complete quality assessment output for one dataset.

    Every pipeline module populates a QualityResult and returns it to the
    scorer. The scorer reads ``overall_score`` and ``dimensions``; the
    reporter reads all fields to build the CML markdown and the JSON artefact
    written to ``reports/quality_scores.json``.

    Attributes:
        dataset_name: Human-readable identifier for the dataset being assessed
            (e.g. ``"customer_orders_2024-06"``).
        data_type: Category of the data. Must be one of ``"structured"``,
            ``"semi_structured"``, ``"text"``, ``"audio"``, ``"image"``,
            or ``"video"``.
        overall_score: Weighted composite quality score in the range [0, 100].
            Computed by the scorer from ``dimensions`` and the weights defined
            in ``dq_config.yaml``. Defaults to ``0.0`` until the scorer sets
            it.
        dimensions: Per-dimension quality scores, each in [0, 100].  The six
            expected keys mirror the ``scoring_weights`` block in
            ``dq_config.yaml``::

                {
                    "completeness": float,
                    "accuracy":     float,
                    "consistency":  float,
                    "freshness":    float,
                    "uniqueness":   float,
                    "validity":     float,
                }

            Individual modules fill whichever dimensions they can evaluate;
            missing keys default to ``0.0`` in the scorer.
        failed_checks: List of dicts describing every check that did not pass.
            Each dict should contain at minimum::

                {
                    "check":    str,   # e.g. "null_rate"
                    "column":   str,   # column / field / segment name
                    "expected": Any,   # threshold or expected value
                    "actual":   Any,   # observed value
                    "severity": str,   # "error" | "warning"
                }

        metadata: Arbitrary key-value pairs set by the module — row count,
            file size, encoding, sample rate, resolution, etc. Not used by
            the scorer but forwarded to the report and the API response.
        passed: ``True`` when ``overall_score`` is at or above the
            ``pass_score`` threshold defined in ``dq_config.yaml``. Set by the
            scorer after computing ``overall_score``. Defaults to ``False``.
        run_timestamp: ISO-8601 UTC timestamp of when the result was created.
            Populated automatically at construction time.
        source_path: Absolute or relative path / URI pointing to the data
            source that was assessed (e.g. ``"data/raw/orders.csv"`` or
            ``"s3://bucket/prefix/"``).
    """

    dataset_name: str
    data_type: str
    overall_score: float = 0.0
    dimensions: dict[str, float] = field(default_factory=lambda: {
        "completeness": 0.0,
        "accuracy":     0.0,
        "consistency":  0.0,
        "freshness":    0.0,
        "uniqueness":   0.0,
        "validity":     0.0,
    })
    failed_checks: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    passed: bool = False
    run_timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    source_path: str = ""

    def __post_init__(self) -> None:
        """Validate field types and log result creation."""
        valid_data_types = {
            "structured", "semi_structured", "text", "audio", "image", "video"
        }
        if self.data_type not in valid_data_types:
            raise ValueError(
                f"Invalid data_type '{self.data_type}'. "
                f"Must be one of: {sorted(valid_data_types)}"
            )
        if not (0.0 <= self.overall_score <= 100.0):
            raise ValueError(
                f"overall_score must be in [0, 100], got {self.overall_score}"
            )
        # Coerce numpy scalars to Python natives so the dataclass is always
        # JSON-serialisable and `is True` / `is False` identity checks work.
        self.overall_score = float(self.overall_score)
        self.passed        = bool(self.passed)
        logger.debug(
            f"QualityResult created | dataset='{self.dataset_name}' "
            f"type='{self.data_type}' score={self.overall_score}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a plain dictionary suitable for JSON output.

        All values are JSON-safe primitives. The returned dict mirrors the
        field layout of this class so it can be round-tripped via
        ``QualityResult(**data)`` if needed.

        Returns:
            A dictionary with the following keys:

            .. code-block:: python

                {
                    "dataset_name":   str,
                    "data_type":      str,
                    "overall_score":  float,
                    "dimensions":     dict[str, float],
                    "failed_checks":  list[dict],
                    "metadata":       dict,
                    "passed":         bool,
                    "run_timestamp":  str,   # ISO-8601 UTC
                    "source_path":    str,
                }

        Example::

            result = QualityResult(
                dataset_name="orders",
                data_type="structured",
                overall_score=87.5,
            )
            payload = result.to_dict()
            json.dumps(payload)   # safe to serialise
        """
        logger.debug(f"Serialising QualityResult for dataset='{self.dataset_name}'")
        # Return shallow copies of mutable fields so callers cannot accidentally
        # mutate the dataclass state through the returned dict.
        return {
            "dataset_name":  self.dataset_name,
            "data_type":     self.data_type,
            "overall_score": self.overall_score,
            "dimensions":    dict(self.dimensions),
            "failed_checks": list(self.failed_checks),
            "metadata":      dict(self.metadata),
            "passed":        self.passed,
            "run_timestamp": self.run_timestamp,
            "source_path":   self.source_path,
        }
