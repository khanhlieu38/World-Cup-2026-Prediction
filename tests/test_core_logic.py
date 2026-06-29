from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wc2026_predictor.features import add_elo_features, add_form_features  # noqa: E402
from wc2026_predictor.simulation import (  # noqa: E402
    calculate_table,
    rank_group,
    resolve_round_of_32,
)


def test_elo_feature_records_pre_match_rating() -> None:
    raw = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 0,
                "tournament": "Test",
                "city": "X",
                "country": "X",
                "neutral": True,
                "outcome": "home_win",
            },
            {
                "date": "2020-01-02",
                "home_team": "A",
                "away_team": "B",
                "home_score": 0,
                "away_score": 0,
                "tournament": "Test",
                "city": "X",
                "country": "X",
                "neutral": True,
                "outcome": "draw",
            },
        ]
    )
    raw["date"] = pd.to_datetime(raw["date"])
    matches = raw.copy()
    upcoming = raw.iloc[0:0].copy()

    featured, _, _ = add_elo_features(raw, matches, upcoming)

    assert featured.loc[0, "home_elo_before"] == 1500.0
    assert featured.loc[0, "away_elo_before"] == 1500.0
    assert featured.loc[1, "home_elo_before"] > 1500.0
    assert featured.loc[1, "away_elo_before"] < 1500.0


def test_form_features_do_not_use_same_day_results() -> None:
    matches = pd.DataFrame(
        [
            ("2020-01-01", "A", "B", 1, 0, "home_win"),
            ("2020-01-01", "A", "C", 3, 0, "home_win"),
            ("2020-01-02", "D", "A", 0, 2, "away_win"),
        ],
        columns=["date", "home_team", "away_team", "home_score", "away_score", "outcome"],
    )
    matches["date"] = pd.to_datetime(matches["date"])
    matches["tournament"] = "Test"
    matches["city"] = "X"
    matches["country"] = "X"
    matches["neutral"] = True
    upcoming = matches.iloc[0:0].copy()

    featured, _, _ = add_form_features(matches, upcoming, window=5)

    assert pd.isna(featured.loc[0, "home_points_form"])
    assert pd.isna(featured.loc[1, "home_points_form"])
    assert featured.loc[2, "away_points_form"] == 3.0
    assert featured.loc[2, "away_gf_form"] == 2.0
    assert featured.loc[2, "away_ga_form"] == 0.0


def test_rank_group_uses_head_to_head_before_seeded_lottery() -> None:
    teams = ["A", "B", "C", "D"]
    results = [
        ("A", "B", 1, 0),
        ("A", "C", 2, 0),
        ("A", "D", 0, 1),
        ("B", "C", 1, 0),
        ("B", "D", 2, 0),
        ("C", "D", 0, 0),
    ]

    ranked = rank_group(teams, results, np.random.default_rng(1))

    assert [row["team"] for row in ranked[:2]] == ["A", "B"]


def test_calculate_table_points_and_goal_difference() -> None:
    table = calculate_table(["A", "B"], [("A", "B", 2, 1), ("B", "A", 0, 0)])

    assert table["A"]["points"] == 4
    assert table["A"]["gd"] == 1
    assert table["B"]["points"] == 1
    assert table["B"]["gd"] == -1


def test_round_of_32_resolver_uses_combo_table() -> None:
    r32_skeleton = pd.read_csv(ROOT / "data" / "reference" / "r32_skeleton.csv", keep_default_na=False)
    r32_combos = pd.read_csv(ROOT / "data" / "reference" / "r32_combinations.csv")
    letters = list("ABCDEFGHIJKL")
    winner_by_group = {letter: f"W{letter}" for letter in letters}
    runnerup_by_group = {letter: f"RU{letter}" for letter in letters}
    third_by_group = {letter: f"3{letter}" for letter in letters}
    advancing = set(r32_combos.iloc[0]["advancing_groups"].split(","))

    matches = resolve_round_of_32(
        winner_by_group,
        runnerup_by_group,
        third_by_group,
        advancing,
        r32_skeleton,
        r32_combos,
    )

    assert len(matches) == 16
    assert matches["match_no"].is_unique
    third_place_teams = set(matches["away_team"][matches["away_team"].str.startswith("3")])
    assert third_place_teams <= {f"3{letter}" for letter in advancing}
