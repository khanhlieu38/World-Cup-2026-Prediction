"""Feature engineering helpers for the match-level prediction model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DATA_URL, ELO_WARMUP_START, FORM_FEATURES, FORM_WINDOW, MODEL_START

OUTCOME_SCORE = {"home_win": 1.0, "draw": 0.5, "away_win": 0.0}


@dataclass(frozen=True)
class FeatureBuildResult:
    """Feature tables plus metadata for the processed snapshot."""

    matches_features: pd.DataFrame
    upcoming_matches: pd.DataFrame
    metadata: dict


def get_outcome(row: pd.Series) -> str | pd.NA:
    """Return the result label from home/away goals."""

    if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
        return pd.NA
    if row["home_score"] > row["away_score"]:
        return "home_win"
    if row["home_score"] < row["away_score"]:
        return "away_win"
    return "draw"


def expected_score(elo_a: float, elo_b: float) -> float:
    """Expected Elo score for team A against team B."""

    return 1 / (1 + 10 ** (-(elo_a - elo_b) / 400))


def update_elo(elo: float, expected: float, actual: float, k: float = 20) -> float:
    """Update an Elo rating after one match."""

    return elo + k * (actual - expected)


def prepare_model_scope(
    raw_results: pd.DataFrame,
    model_start: pd.Timestamp = MODEL_START,
    excluded_tournaments: Iterable[str] = ("Friendly",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split raw results into played model rows and future rows."""

    df = raw_results.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["outcome"] = df.apply(get_outcome, axis=1)
    model_scope = df[
        df["date"].ge(model_start)
        & ~df["tournament"].isin(tuple(excluded_tournaments))
    ].copy()
    upcoming_matches = model_scope[
        model_scope[["home_score", "away_score"]].isna().any(axis=1)
    ].copy()
    matches_features = (
        model_scope.dropna(subset=["home_score", "away_score"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    return matches_features, upcoming_matches


def add_elo_features(
    raw_results: pd.DataFrame,
    matches_features: pd.DataFrame,
    upcoming_matches: pd.DataFrame,
    warmup_start: pd.Timestamp = ELO_WARMUP_START,
    model_start: pd.Timestamp = MODEL_START,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Add pre-match Elo features without leaking the current match result."""

    raw = raw_results.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    if "outcome" not in raw:
        raw["outcome"] = raw.apply(get_outcome, axis=1)

    played = matches_features.sort_values("date").reset_index(drop=True).copy()
    future = upcoming_matches.copy()

    warmup_matches = (
        raw[
            raw["date"].between(warmup_start, model_start, inclusive="left")
            & raw["tournament"].ne("Friendly")
        ]
        .dropna(subset=["home_score", "away_score"])
        .sort_values("date")
    )
    all_teams = (
        set(warmup_matches["home_team"])
        | set(warmup_matches["away_team"])
        | set(played["home_team"])
        | set(played["away_team"])
        | set(future["home_team"])
        | set(future["away_team"])
    )
    elo = {team: 1500.0 for team in all_teams}

    def apply_result(row) -> None:
        home_before, away_before = elo[row.home_team], elo[row.away_team]
        expected_home = expected_score(home_before, away_before)
        actual_home = OUTCOME_SCORE[row.outcome]
        elo[row.home_team] = update_elo(home_before, expected_home, actual_home)
        elo[row.away_team] = update_elo(away_before, 1 - expected_home, 1 - actual_home)

    for row in warmup_matches.itertuples(index=False):
        apply_result(row)

    home_elo_before: list[float] = []
    away_elo_before: list[float] = []
    for row in played.itertuples(index=False):
        home_elo_before.append(elo[row.home_team])
        away_elo_before.append(elo[row.away_team])
        apply_result(row)

    played["home_elo_before"] = home_elo_before
    played["away_elo_before"] = away_elo_before
    played["elo_diff"] = played["home_elo_before"] - played["away_elo_before"]

    future["home_elo_now"] = future["home_team"].map(elo)
    future["away_elo_now"] = future["away_team"].map(elo)
    future["elo_diff"] = future["home_elo_now"] - future["away_elo_now"]
    return played, future, elo


def add_form_features(
    matches_features: pd.DataFrame,
    upcoming_matches: pd.DataFrame,
    window: int = FORM_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    """Add rolling form features while keeping same-day matches unseen."""

    played = matches_features.sort_values("date").reset_index(drop=True).copy()
    future = upcoming_matches.copy()
    played["match_id"] = np.arange(len(played))

    home_history = played[
        ["match_id", "date", "home_team", "home_score", "away_score", "outcome"]
    ].rename(columns={"home_team": "team", "home_score": "gf", "away_score": "ga"})
    home_history["points"] = home_history["outcome"].map(
        {"home_win": 3, "draw": 1, "away_win": 0}
    )
    home_history["side"] = "home"

    away_history = played[
        ["match_id", "date", "away_team", "away_score", "home_score", "outcome"]
    ].rename(columns={"away_team": "team", "away_score": "gf", "home_score": "ga"})
    away_history["points"] = away_history["outcome"].map(
        {"home_win": 0, "draw": 1, "away_win": 3}
    )
    away_history["side"] = "away"

    team_history = (
        pd.concat([home_history, away_history], ignore_index=True)
        .sort_values(["team", "date", "match_id"])
        .reset_index(drop=True)
    )
    for column in ["points_form", "gf_form", "ga_form"]:
        team_history[column] = np.nan

    for _, team_matches in team_history.groupby("team", sort=False):
        history: list[tuple[float, float, float]] = []
        for _, same_day in team_matches.groupby("date", sort=True):
            prior = history[-window:]
            if prior:
                team_history.loc[same_day.index, "points_form"] = np.mean(
                    [item[0] for item in prior]
                )
                team_history.loc[same_day.index, "gf_form"] = np.mean(
                    [item[1] for item in prior]
                )
                team_history.loc[same_day.index, "ga_form"] = np.mean(
                    [item[2] for item in prior]
                )
            history.extend(zip(same_day["points"], same_day["gf"], same_day["ga"]))

    for side in ["home", "away"]:
        lookup = team_history[team_history["side"].eq(side)].set_index("match_id")
        for statistic in ["points_form", "gf_form", "ga_form"]:
            played[f"{side}_{statistic}"] = played["match_id"].map(lookup[statistic])

    current_form: dict[str, dict[str, float]] = {}
    for team, team_matches in team_history.groupby("team"):
        recent = team_matches.sort_values(["date", "match_id"]).tail(window)
        current_form[team] = {
            "points_form": float(recent["points"].mean()),
            "gf_form": float(recent["gf"].mean()),
            "ga_form": float(recent["ga"].mean()),
        }

    for side in ["home", "away"]:
        for statistic in ["points_form", "gf_form", "ga_form"]:
            future[f"{side}_{statistic}"] = future[f"{side}_team"].map(
                lambda team: current_form.get(team, {}).get(statistic, np.nan)
            )

    played = played.drop(columns="match_id")
    return played, future, current_form


def build_feature_tables(raw_results: pd.DataFrame, source_url: str = DATA_URL) -> FeatureBuildResult:
    """Build the processed match and upcoming-match tables from raw results."""

    matches_features, upcoming_matches = prepare_model_scope(raw_results)
    matches_features, upcoming_matches, _ = add_elo_features(
        raw_results, matches_features, upcoming_matches
    )
    matches_features, upcoming_matches, _ = add_form_features(matches_features, upcoming_matches)
    metadata = {
        "source_url": source_url,
        "as_of_date": str(matches_features["date"].max().date()),
        "matches_features_rows": int(len(matches_features)),
        "upcoming_matches_rows": int(len(upcoming_matches)),
        "form_features": FORM_FEATURES,
    }
    return FeatureBuildResult(matches_features, upcoming_matches, metadata)
