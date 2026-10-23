"""Derive the modelling features from the raw survey columns.

Two of the survey's conventions will produce wrong numbers without complaining:

**Binary answers are coded 1/6**, not 1/0 or 1/2. Sex is 1 = male, 6 = female;
area is 1 = urban, 6 = rural. Nothing errors if these are treated as
quantities; the model simply learns from a variable where "female" is six
times "male".

**Daily hours are `H.MM`, not decimal hours.** A respondent who works eight and
a half hours is recorded as `8.30`, and three quarters of an hour is `.45`.
Summing the seven days arithmetically reads that as 8.3 hours and understates
the week. The observed decimal parts in 2024 are .00, .10, .30, .40 and .45 --
minutes, exactly as the variable label ("Horas y Minutos") says.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAILY_HOURS = ["B03LU", "B03MA", "B03MI", "B03JU", "B03VI", "B03SA", "B03DO"]

# Beyond this a weekly total is a reporting artefact rather than a schedule.
# 7 x 18 hours is physically sayable and not credible; the value is kept but
# flagged, never silently clipped.
IMPLAUSIBLE_WEEKLY_HOURS = 112.0


def hours_minutes_to_hours(values: np.ndarray) -> np.ndarray:
    """Convert `H.MM` to decimal hours.

    >>> hours_minutes_to_hours(np.array([8.0, 8.3, 8.45]))
    array([8.  , 8.5 , 8.75])
    """
    whole = np.floor(values)
    minutes = np.round((values - whole) * 100)
    return whole + minutes / 60.0


def weekly_hours(frame: pd.DataFrame) -> pd.Series:
    """Total hours worked across the seven reported days."""
    total = np.zeros(len(frame), dtype=float)
    for column in DAILY_HOURS:
        total += hours_minutes_to_hours(frame[column].to_numpy(dtype=float))
    return pd.Series(total, index=frame.index, name="weekly_hours")


def build(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return the modelling frame plus its categorical and numeric columns.

    `B06` ("hours usually worked") is deliberately not used: it is missing for
    90% of this population because the questionnaire only asks it of
    respondents whose week was atypical. A feature present for one row in ten
    would mostly teach the model the shape of the questionnaire. The seven
    daily-hours columns are complete, and their total says the same thing.
    """
    out = frame.copy()
    out["weekly_hours"] = weekly_hours(out)
    out["hours_implausible"] = out["weekly_hours"] > IMPLAUSIBLE_WEEKLY_HOURS

    categorical = list(config["features"]["categorical"])
    numeric = list(config["features"]["numeric"])
    numeric += [
        name for name, spec in (config["features"].get("derived") or {}).items()
        if spec.get("kind", "numeric") == "numeric"
    ]

    # Categories are modelled as labels, never as quantities, so the 1/6
    # coding cannot leak in as a magnitude.
    for column in categorical:
        out[column] = out[column].astype("object").where(out[column].notna(), "missing")
        out[column] = out[column].astype(str)

    return out, categorical, numeric


def readable_labels(
    config: dict, variable_labels: dict[str, str]
) -> dict[str, str]:
    """Map each feature to a human label, preferring the config's wording."""
    names: dict[str, str] = {}
    for group in ("categorical", "numeric"):
        for column, label in config["features"][group].items():
            names[column] = label
    names["weekly_hours"] = "Hours worked per week (derived from daily entries)"
    for column, label in variable_labels.items():
        names.setdefault(column, label)
    return names
