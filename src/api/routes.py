"""
Route definitions for the Data Quality Pipeline API.

All endpoints are registered via register_routes(), which is called by the
application factory in app.py. Routes are read-only stubs for now — write
logic will be wired in once the scoring and DVC stages are implemented.
"""

import json
import os

from flask import Flask, jsonify, request
from loguru import logger

QUALITY_SCORES_PATH = "reports/quality_scores.json"


def _read_quality_scores() -> dict | list | None:
    """Read and parse quality_scores.json from the reports directory.

    Returns:
        Parsed JSON content (dict or list), or None if the file does not exist
        or cannot be parsed.
    """
    if not os.path.exists(QUALITY_SCORES_PATH):
        return None
    with open(QUALITY_SCORES_PATH, "r") as f:
        return json.load(f)


def register_routes(app: Flask) -> None:
    """Register all API routes on the given Flask application.

    Args:
        app: The Flask application instance to attach routes to.
    """

    @app.get("/health")
    def health():
        """Liveness probe — confirms the API process is running.

        Returns:
            200 JSON with status, version, and a human-readable message.
        """
        logger.info("GET /health")
        return jsonify({
            "status": "ok",
            "version": "1.0",
            "message": "Data Quality Pipeline API is running",
        })

    @app.get("/quality/latest")
    def quality_latest():
        """Return the most recent quality scores report.

        Reads reports/quality_scores.json and returns its full contents.
        If the file does not exist the pipeline has not yet produced a report.

        Returns:
            200 JSON with report contents, or 200 JSON with status "pending"
            if no report is available yet.
        """
        logger.info("GET /quality/latest")
        data = _read_quality_scores()
        if data is None:
            logger.warning("Quality scores file not found — returning pending status")
            return jsonify({
                "message": "No quality report available yet",
                "status": "pending",
            })
        return jsonify(data)

    @app.get("/quality/<string:dataset_name>")
    def quality_by_dataset(dataset_name: str):
        """Return the quality report for a specific dataset.

        Filters quality_scores.json by the provided dataset_name field.

        Args:
            dataset_name: Name of the dataset to look up.

        Returns:
            200 JSON with the matching dataset entry, 404 if not found, or
            200 with status "pending" if no report file exists yet.
        """
        logger.info(f"GET /quality/{dataset_name}")
        data = _read_quality_scores()
        if data is None:
            logger.warning("Quality scores file not found — returning pending status")
            return jsonify({
                "message": "No quality report available yet",
                "status": "pending",
            })

        entries = data if isinstance(data, list) else data.get("datasets", [])
        match = next(
            (entry for entry in entries if entry.get("dataset_name") == dataset_name),
            None,
        )
        if match is None:
            logger.warning(f"Dataset '{dataset_name}' not found in quality scores")
            return jsonify({"message": "Dataset not found"}), 404

        return jsonify(match)

    @app.get("/quality/<string:dataset_name>/score")
    def quality_score_by_dataset(dataset_name: str):
        """Return only the overall quality score for a specific dataset.

        Args:
            dataset_name: Name of the dataset to look up.

        Returns:
            200 JSON with dataset name and score, 404 if the dataset is not
            found, or 200 with status "pending" if no report exists yet.
        """
        logger.info(f"GET /quality/{dataset_name}/score")
        data = _read_quality_scores()
        if data is None:
            logger.warning("Quality scores file not found — returning pending status")
            return jsonify({
                "message": "No quality report available yet",
                "status": "pending",
            })

        entries = data if isinstance(data, list) else data.get("datasets", [])
        match = next(
            (entry for entry in entries if entry.get("dataset_name") == dataset_name),
            None,
        )
        if match is None:
            logger.warning(f"Dataset '{dataset_name}' not found when fetching score")
            return jsonify({"message": "Dataset not found"}), 404

        return jsonify({
            "dataset": dataset_name,
            "score": match.get("overall_score"),
        })

    @app.get("/quality/history")
    def quality_history():
        """Return all historical quality score entries.

        Reads quality_scores.json and returns the full history list if
        a "history" key is present, otherwise returns the root-level data.

        Returns:
            200 JSON with a history list, or 200 JSON with an empty history
            and a message if no report file exists yet.
        """
        logger.info("GET /quality/history")
        data = _read_quality_scores()
        if data is None:
            logger.warning("Quality scores file not found — returning empty history")
            return jsonify({
                "history": [],
                "message": "No history available yet",
            })

        if isinstance(data, dict) and "history" in data:
            return jsonify({"history": data["history"]})

        return jsonify({"history": data if isinstance(data, list) else []})

    @app.post("/quality/run")
    def quality_run():
        """Trigger a manual pipeline run (placeholder).

        Will be wired to dvc repro once the execution layer is implemented.
        Accepts an optional JSON body with a "dataset" key to scope the run.

        Returns:
            202 JSON confirming the run has been queued.
        """
        logger.info("POST /quality/run")
        body = request.get_json(silent=True) or {}
        dataset = body.get("dataset", "all")
        logger.info(f"Manual run requested for dataset='{dataset}'")
        return jsonify({
            "message": "Manual run triggered",
            "status": "queued",
            "dataset": dataset,
        }), 202
