"""Project paths and configuration loading."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "survey.yml"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REPORT_DIR = DATA_DIR / "reports"
EXPORT_DIR = DATA_DIR / "exports"
# Committed deliberately: GitHub Pages serves from here.
DOCS_DIR = ROOT / "docs"


@functools.cache
def load_config(path: Path | None = None) -> dict[str, Any]:
    with (path or CONFIG_PATH).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dirs() -> None:
    for directory in (DATA_DIR, RAW_DIR, REPORT_DIR, EXPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
