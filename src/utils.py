"""
Shared utility helpers for the Data Quality Pipeline.

Keeps cross-cutting concerns — config loading, path resolution — in one place
so every module can import them without duplicating file-I/O logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

# Default config location relative to the project root.
_DEFAULT_CONFIG_PATH = Path("dq_config.yaml")


def load_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and return the pipeline configuration from a YAML file.

    Reads ``dq_config.yaml`` (or the path you supply) and returns its contents
    as a plain Python dictionary. The result is *not* cached — call this once
    at startup and pass the dict around rather than calling it repeatedly.

    Args:
        path: Path to the YAML configuration file. Accepts a :class:`str` or
            :class:`pathlib.Path`. Defaults to ``dq_config.yaml`` in the
            current working directory (i.e. the project root when the pipeline
            is run with ``python src/...`` from that directory).

    Returns:
        Parsed configuration as a nested dictionary. Top-level keys correspond
        to the sections defined in ``dq_config.yaml``::

            {
                "project":         {...},
                "thresholds":      {...},
                "scoring_weights": {...},
                "alerts":          {...},
                "aws":             {...},
                "flask":           {...},
                "dvc":             {...},
                "modules":         {...},
                "logging":         {...},
            }

    Raises:
        FileNotFoundError: If ``path`` does not exist on disk.
        yaml.YAMLError: If the file exists but cannot be parsed as valid YAML.
        TypeError: If the parsed YAML root is not a mapping (dict).

    Example::

        from src.utils import load_config

        cfg = load_config()
        pass_threshold = cfg["thresholds"]["pass_score"]   # 80

        # Or supply an explicit path (useful in tests):
        cfg = load_config("tests/fixtures/test_config.yaml")
    """
    config_path = Path(path)

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path.resolve()}")
        raise FileNotFoundError(
            f"Configuration file not found: {config_path.resolve()}\n"
            "Ensure dq_config.yaml exists in the project root, or pass an "
            "explicit path to load_config()."
        )

    logger.debug(f"Loading configuration from {config_path.resolve()}")

    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if not isinstance(config, dict):
        raise TypeError(
            f"Expected the config file to contain a YAML mapping, "
            f"got {type(config).__name__}."
        )

    logger.info(
        f"Configuration loaded | file='{config_path}' "
        f"project='{config.get('project', {}).get('name', 'unknown')}' "
        f"version='{config.get('project', {}).get('version', 'unknown')}'"
    )
    return config
