"""Project constants shared by notebooks, scripts, and tests."""

from __future__ import annotations

import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
MODEL_START = pd.Timestamp("2010-01-01")
ELO_WARMUP_START = pd.Timestamp("2000-01-01")
FORM_WINDOW = 5
RECENCY_HALF_LIFE_YEARS = 4

FORM_FEATURES = [
    "home_points_form",
    "away_points_form",
    "home_gf_form",
    "away_gf_form",
    "home_ga_form",
    "away_ga_form",
]
FINAL_FEATURES = ["elo_diff", "neutral"] + FORM_FEATURES
