"""
Flask application factory for the Data Quality Pipeline API.

Loads configuration from dq_config.yaml, initialises Flask with CORS,
registers all routes, and exposes a run() entry point.
"""

import yaml
from flask import Flask, jsonify
from flask_cors import CORS
from loguru import logger

from src.api.routes import register_routes

CONFIG_PATH = "dq_config.yaml"


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load and return the pipeline configuration from a YAML file.

    Args:
        path: Path to the YAML config file. Defaults to dq_config.yaml.

    Returns:
        Parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file cannot be parsed.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {path}")
    return config


def create_app(config: dict | None = None) -> Flask:
    """Create and configure the Flask application.

    Initialises CORS, registers all API routes, and attaches the pipeline
    config to the application context so routes can access it.

    Args:
        config: Optional pre-loaded config dict. If None, loads from
                dq_config.yaml automatically.

    Returns:
        Configured Flask application instance.
    """
    if config is None:
        config = load_config()

    app = Flask(__name__)
    CORS(app)

    app.config["DQ_CONFIG"] = config

    # Basic health check mounted directly on the app factory so it is always
    # available even if route registration fails.
    @app.get("/health")
    def health():
        """Minimal liveness probe used by load balancers and CI checks."""
        return jsonify({"status": "ok", "version": "1.0"})

    register_routes(app)

    logger.info("Flask application created and routes registered")
    return app


if __name__ == "__main__":
    cfg = load_config()
    flask_cfg = cfg.get("flask", {})

    host = flask_cfg.get("host", "0.0.0.0")
    port = flask_cfg.get("port", 5000)
    debug = flask_cfg.get("debug", False)

    logger.info(f"Starting Data Quality Pipeline API on {host}:{port} (debug={debug})")

    app = create_app(cfg)
    app.run(host=host, port=port, debug=debug)
