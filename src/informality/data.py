"""Download the microdata and turn it into a modelling frame.

The EPHC ships as SPSS `.sav` files, which carry their own variable and value
labels. Reading those rather than the parallel CSVs means the codebook travels
with the data instead of living in a separate legacy `.xls` that can drift out
of step with it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyreadstat
import requests

from .config import RAW_DIR


@dataclass
class Provenance:
    source: str
    url: str
    fetched_at: datetime
    http_status: int
    content_length: int
    sha256: str
    from_cache: bool = False


@dataclass
class Dataset:
    frame: pd.DataFrame
    variable_labels: dict[str, str]
    value_labels: dict[str, dict]
    provenance: list[Provenance] = field(default_factory=list)


def download(config: dict, year: int, *, reuse_cache: bool = True) -> tuple[Path, Provenance]:
    """Fetch one year's person-level file, caching it on first use."""
    source = config["source"]
    url = f"{source['base_url']}/{source['files']['person'].format(year=year)}"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / f"REG02_{year}.sav"

    if reuse_cache and target.exists():
        payload = target.read_bytes()
        return target, Provenance(
            source=f"person:{year}", url=url,
            fetched_at=datetime.fromtimestamp(target.stat().st_mtime, tz=timezone.utc),
            http_status=200, content_length=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(), from_cache=True,
        )

    response = requests.get(
        url, headers={"User-Agent": source["user_agent"]}, timeout=600
    )
    if response.status_code != 200:
        raise RuntimeError(f"{url}: HTTP {response.status_code}")
    target.write_bytes(response.content)
    return target, Provenance(
        source=f"person:{year}", url=url,
        fetched_at=datetime.now(timezone.utc),
        http_status=response.status_code,
        content_length=len(response.content),
        sha256=hashlib.sha256(response.content).hexdigest(),
    )


def needed_columns(config: dict) -> list[str]:
    design = config["design"]
    columns = [design["weight"], design["cluster"], *design["strata"]]
    columns.append(config["target"]["variable"])
    columns.append(config["universe"]["employment_category"]["variable"])
    columns += list(config["features"]["categorical"])
    columns += list(config["features"]["numeric"])
    # Derived features need their raw inputs downloaded too.
    for spec in (config["features"].get("derived") or {}).values():
        columns += list(spec.get("from", []))
    # Age is the universe filter as well as a feature; keep one copy.
    return sorted(set(columns))


def load(config: dict, year: int, *, reuse_cache: bool = True) -> Dataset:
    """Read one year into a frame, carrying its labels with it."""
    path, provenance = download(config, year, reuse_cache=reuse_cache)
    frame, meta = pyreadstat.read_sav(path, usecols=needed_columns(config))
    return Dataset(
        frame=frame,
        variable_labels=dict(meta.column_names_to_labels),
        value_labels=dict(meta.variable_value_labels),
        provenance=[provenance],
    )


@dataclass
class UniverseReport:
    """How the sample narrowed, step by step."""

    steps: list[tuple[str, int]] = field(default_factory=list)

    def record(self, label: str, remaining: int) -> None:
        self.steps.append((label, remaining))

    @property
    def final(self) -> int:
        return self.steps[-1][1] if self.steps else 0


def build_universe(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, UniverseReport]:
    """Narrow the person file to the population the model is about.

    Each step is recorded. A sample that shrinks from 57,744 rows to 10,604
    deserves an audit trail: the difference between those numbers is where a
    quiet filtering bug would live.
    """
    report = UniverseReport()
    report.record("person records in the file", len(frame))

    target = config["target"]
    data = frame[frame[target["variable"]].isin(target["valid_values"])]
    report.record(
        f"answered {target['variable']} (asked of employed workers only)", len(data)
    )

    minimum_age = config["universe"]["min_age"]
    data = data[data["P02"] >= minimum_age]
    report.record(f"aged {minimum_age} or over", len(data))

    category = config["universe"]["employment_category"]
    data = data[data[category["variable"]].isin(category["keep"])]
    report.record(category["label"], len(data))

    data = data[data[config["design"]["weight"]] > 0]
    report.record("with a positive expansion factor", len(data))

    return data.copy(), report


def add_target(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Attach the boolean informality flag."""
    target = config["target"]
    out = frame.copy()
    out["informal"] = out[target["variable"]] == target["informal_value"]
    return out


def describe_value(
    labels: dict[str, dict], variable: str, value
) -> str:
    """Render a coded value using the survey's own label."""
    mapping = labels.get(variable) or {}
    try:
        return mapping.get(float(value), str(value))
    except (TypeError, ValueError):
        return str(value)
