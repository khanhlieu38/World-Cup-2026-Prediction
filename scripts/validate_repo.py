from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wc2026_predictor.config import FINAL_FEATURES  # noqa: E402
from wc2026_predictor.modeling import fit_goal_models, fit_outcome_model, predict_goal_rates  # noqa: E402
from wc2026_predictor.simulation import (  # noqa: E402
    build_group_data,
    build_knockout_home_probability,
    build_team_state,
    prepare_world_cup_group_matches,
    run_monte_carlo,
)


def ok(message: str) -> None:
    print(f"OK  {message}")


def warn(message: str) -> None:
    print(f"WARN {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    ok(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the World Cup 2026 predictor repo.")
    parser.add_argument(
        "--today",
        default=date.today().isoformat(),
        help="Date used for freshness checks, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--skip-model-smoke",
        action="store_true",
        help="Skip the lightweight Monte Carlo smoke test.",
    )
    parser.add_argument(
        "--smoke-simulations",
        type=int,
        default=250,
        help="Number of simulations for the lightweight model smoke test.",
    )
    return parser.parse_args()


def load_tables() -> dict[str, pd.DataFrame]:
    data_dir = ROOT / "data"
    return {
        "matches": pd.read_csv(data_dir / "processed" / "matches_features.csv", parse_dates=["date"]),
        "upcoming": pd.read_csv(data_dir / "processed" / "upcoming_matches.csv", parse_dates=["date"]),
        "groups": pd.read_csv(data_dir / "reference" / "group_letters.csv"),
        "r32": pd.read_csv(data_dir / "reference" / "r32_skeleton.csv", keep_default_na=False),
        "combos": pd.read_csv(data_dir / "reference" / "r32_combinations.csv"),
        "later": pd.read_csv(data_dir / "reference" / "later_rounds_skeleton.csv", dtype=str),
    }


def validate_static_data(tables: dict[str, pd.DataFrame], today: pd.Timestamp) -> None:
    matches = tables["matches"]
    upcoming = tables["upcoming"]
    groups = tables["groups"]
    r32 = tables["r32"]
    combos = tables["combos"]
    later = tables["later"]

    require(matches["outcome"].notna().all(), "played match outcomes are complete")
    require(
        upcoming[["home_score", "away_score", "outcome"]].isna().all().all(),
        "upcoming snapshot rows have no known results",
    )
    require(groups["group_letter"].nunique() == 12, "reference data has 12 groups")
    require(groups.groupby("group_letter").size().eq(4).all(), "each group has 4 teams")
    require(groups["team"].nunique() == 48, "reference data has 48 unique teams")
    require(len(r32) == 16, "Round-of-32 skeleton has 16 matches")
    require((r32["away_type"] == "3rd").sum() == 8, "Round-of-32 skeleton has 8 third-place slots")
    require(len(combos) == 495 and combos["advancing_groups"].nunique() == 495, "third-place combo table has 495 unique combinations")
    require(
        later["round"].value_counts().to_dict()
        == {"R16": 8, "QF": 4, "SF": 2, "3rd_place": 1, "Final": 1},
        "later-round skeleton has the expected 16 matches",
    )

    group_matches, played, future = prepare_world_cup_group_matches(matches, upcoming, groups)
    require(len(group_matches) == 72, "World Cup group-stage snapshot has 72 matches")
    require(len(played) + len(future) == 72, "played plus future group matches sum to 72")
    require(
        set(group_matches["home_team"]) | set(group_matches["away_team"]) == set(groups["team"]),
        "World Cup group matches cover exactly the 48 reference teams",
    )

    latest_upcoming = upcoming["date"].max()
    if latest_upcoming < today:
        warn(
            f"snapshot is stale for current forecasting: latest upcoming date is "
            f"{latest_upcoming.date()}, today is {today.date()}"
        )
    else:
        ok("snapshot freshness check passed for upcoming match dates")

    metadata_path = ROOT / "data" / "processed" / "metadata.json"
    require(metadata_path.exists(), "processed metadata file exists")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(
        metadata["row_counts"]["matches_features"] == len(matches),
        "metadata matches matches_features row count",
    )
    require(
        metadata["row_counts"]["upcoming_matches"] == len(upcoming),
        "metadata matches upcoming_matches row count",
    )
    require(
        metadata["as_of_date"] == str(matches["date"].max().date()),
        "metadata as_of_date matches latest played match",
    )


def run_model_smoke(tables: dict[str, pd.DataFrame], n_simulations: int) -> None:
    matches = tables["matches"]
    upcoming = tables["upcoming"]
    groups = tables["groups"]
    r32 = tables["r32"]
    combos = tables["combos"]
    later = tables["later"]

    outcome_model = fit_outcome_model(matches, FINAL_FEATURES)
    home_goal_model, away_goal_model = fit_goal_models(matches, FINAL_FEATURES)
    _, played, future = prepare_world_cup_group_matches(matches, upcoming, groups)

    goal_rates = future.apply(
        lambda row: predict_goal_rates(home_goal_model, away_goal_model, row),
        axis=1,
        result_type="expand",
    )
    goal_rates.columns = ["home_goal_rate", "away_goal_rate"]
    future = pd.concat([future.reset_index(drop=True), goal_rates.reset_index(drop=True)], axis=1)

    group_data = build_group_data(groups, played, future)
    elo_now, team_form_now = build_team_state(upcoming)
    teams = sorted(set(groups["team"]))
    knockout_cache = build_knockout_home_probability(
        teams,
        outcome_model,
        elo_now,
        team_form_now,
    )
    probabilities = run_monte_carlo(
        teams,
        group_data,
        r32,
        combos,
        later,
        knockout_cache,
        n_simulations=n_simulations,
        seed=2026,
    )
    totals = probabilities[
        ["semifinal_probability", "final_probability", "champion_probability"]
    ].sum()
    require(np.isclose(totals["semifinal_probability"], 4.0), "semifinal probabilities sum to 4")
    require(np.isclose(totals["final_probability"], 2.0), "final probabilities sum to 2")
    require(np.isclose(totals["champion_probability"], 1.0), "champion probabilities sum to 1")
    ok(
        "model smoke test ran "
        f"{n_simulations} tournament simulations; top champion candidate is "
        f"{probabilities.iloc[0]['team']}"
    )


def main() -> int:
    args = parse_args()
    tables = load_tables()
    validate_static_data(tables, pd.Timestamp(args.today))
    if not args.skip_model_smoke:
        run_model_smoke(tables, args.smoke_simulations)
    print("Validation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
