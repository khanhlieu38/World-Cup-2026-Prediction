"""World Cup group-stage, bracket, and Monte Carlo simulation helpers."""

from __future__ import annotations

from itertools import groupby

import numpy as np
import pandas as pd

from .config import FINAL_FEATURES
from .modeling import predict_outcome_probabilities


def prepare_world_cup_group_matches(
    df_recent: pd.DataFrame,
    upcoming_matches: pd.DataFrame,
    group_letters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return all 72 group-stage matches split into played/future rows."""

    wc_played_all = df_recent[
        df_recent["tournament"].eq("FIFA World Cup")
        & df_recent["date"].ge("2026-01-01")
    ].copy()
    wc_upcoming_all = upcoming_matches[
        upcoming_matches["tournament"].eq("FIFA World Cup")
    ].copy()
    wc_group_matches = (
        pd.concat([wc_played_all, wc_upcoming_all], ignore_index=True, sort=False)
        .sort_values(["date", "home_team", "away_team"])
        .head(72)
        .reset_index(drop=True)
    )
    expected_teams = set(group_letters["team"])
    actual_teams = set(wc_group_matches["home_team"]) | set(wc_group_matches["away_team"])
    if len(wc_group_matches) != 72:
        raise ValueError(f"Expected 72 group matches, found {len(wc_group_matches)}")
    if actual_teams != expected_teams:
        missing = expected_teams - actual_teams
        extra = actual_teams - expected_teams
        raise ValueError(f"World Cup team mismatch. Missing={missing}; extra={extra}")
    played = wc_group_matches.dropna(subset=["home_score", "away_score"]).copy()
    future = wc_group_matches[
        wc_group_matches[["home_score", "away_score"]].isna().any(axis=1)
    ].copy()
    return wc_group_matches, played, future


def calculate_table(teams, results: list[tuple[str, str, int, int]]) -> dict[str, dict]:
    """Calculate points, goals for/against, and goal difference."""

    table = {team: {"team": team, "points": 0, "gf": 0, "ga": 0} for team in teams}
    for home, away, home_goals, away_goals in results:
        if home not in table or away not in table:
            continue
        table[home]["gf"] += home_goals
        table[home]["ga"] += away_goals
        table[away]["gf"] += away_goals
        table[away]["ga"] += home_goals
        if home_goals > away_goals:
            table[home]["points"] += 3
        elif home_goals < away_goals:
            table[away]["points"] += 3
        else:
            table[home]["points"] += 1
            table[away]["points"] += 1

    for row in table.values():
        row["gd"] = row["gf"] - row["ga"]
    return table


def rank_group(teams, results: list[tuple[str, str, int, int]], rng) -> list[dict]:
    """Rank a four-team group with deterministic seeded lottery as last resort."""

    overall = calculate_table(teams, results)
    rows = sorted(
        overall.values(),
        key=lambda row: (row["points"], row["gd"], row["gf"]),
        reverse=True,
    )

    ranked: list[dict] = []
    ranking_key = lambda row: (row["points"], row["gd"], row["gf"])
    for _, tied_rows in groupby(rows, key=ranking_key):
        tied_rows = list(tied_rows)
        if len(tied_rows) == 1:
            ranked.extend(tied_rows)
            continue

        tied_teams = {row["team"] for row in tied_rows}
        head_to_head = calculate_table(tied_teams, results)
        lottery = {team: rng.random() for team in tied_teams}
        tied_rows.sort(
            key=lambda row: (
                head_to_head[row["team"]]["points"],
                head_to_head[row["team"]]["gd"],
                head_to_head[row["team"]]["gf"],
                lottery[row["team"]],
            ),
            reverse=True,
        )
        ranked.extend(tied_rows)
    return ranked


def build_group_data(
    group_letters: pd.DataFrame,
    wc2026_played: pd.DataFrame,
    wc2026_upcoming: pd.DataFrame,
) -> list[dict]:
    """Package played results and future goal rates by group letter."""

    group_data: list[dict] = []
    for letter, mapping in group_letters.groupby("group_letter", sort=True):
        teams = tuple(mapping["team"])
        played = wc2026_played[
            wc2026_played["home_team"].isin(teams)
            & wc2026_played["away_team"].isin(teams)
        ]
        future = wc2026_upcoming[
            wc2026_upcoming["home_team"].isin(teams)
            & wc2026_upcoming["away_team"].isin(teams)
        ]
        played_results = [
            (row.home_team, row.away_team, int(row.home_score), int(row.away_score))
            for row in played.itertuples(index=False)
        ]
        future_matches = [
            (row.home_team, row.away_team, row.home_goal_rate, row.away_goal_rate)
            for row in future.itertuples(index=False)
        ]
        if len(played_results) + len(future_matches) != 6:
            raise ValueError(f"Group {letter} does not have exactly 6 matches")
        group_data.append(
            {
                "letter": letter,
                "teams": teams,
                "played": played_results,
                "future": future_matches,
            }
        )
    return group_data


def simulate_group(group_info: dict, rng) -> list[dict]:
    """Simulate the remaining group games and return ranked standings."""

    results = list(group_info["played"])
    for home, away, home_rate, away_rate in group_info["future"]:
        results.append((home, away, int(rng.poisson(home_rate)), int(rng.poisson(away_rate))))
    return rank_group(group_info["teams"], results, rng)


def best_eight_third_place(third_place_by_group: dict[str, dict], rng) -> set[str]:
    """Return group letters for the eight third-place teams that advance."""

    rows = [{"group": letter, **stats} for letter, stats in third_place_by_group.items()]
    ranking = pd.DataFrame(rows)
    ranking["lottery"] = rng.random(len(ranking))
    ranking = ranking.sort_values(["points", "gd", "gf", "lottery"], ascending=False)
    return set(ranking["group"].iloc[:8])


def resolve_round_of_32(
    winner_by_group: dict[str, str],
    runnerup_by_group: dict[str, str],
    third_by_group: dict[str, str],
    advancing_third_letters: set[str],
    r32_skeleton: pd.DataFrame,
    r32_combos: pd.DataFrame,
) -> pd.DataFrame:
    """Fill the Round-of-32 bracket using the official third-place combo table."""

    advancing_key = ",".join(sorted(advancing_third_letters))
    matching_combo = r32_combos[r32_combos["advancing_groups"].eq(advancing_key)]
    if matching_combo.empty:
        raise ValueError(f"No Round-of-32 combo for advancing groups: {advancing_key}")
    combo = matching_combo.iloc[0]

    matches: list[dict] = []
    for _, slot in r32_skeleton.iterrows():
        home_source = runnerup_by_group if slot["home_type"] == "RU" else winner_by_group
        home = home_source[slot["home_group"]]
        if slot["away_type"] == "3rd":
            third_letter = combo[f"vs_1{slot['home_group']}"]
            away = third_by_group[third_letter]
        else:
            away_source = runnerup_by_group if slot["away_type"] == "RU" else winner_by_group
            away = away_source[slot["away_group"]]
        matches.append(
            {
                "match_no": int(slot["match_no"]),
                "home_team": home,
                "away_team": away,
            }
        )
    return pd.DataFrame(matches)


def run_group_stage_full(
    group_data: list[dict],
    r32_skeleton: pd.DataFrame,
    r32_combos: pd.DataFrame,
    rng,
) -> tuple[set[str], pd.DataFrame]:
    """Simulate all groups and return qualified teams plus Round-of-32 fixtures."""

    standings_by_group = {
        group_info["letter"]: simulate_group(group_info, rng)
        for group_info in group_data
    }
    winner_by_group = {letter: table[0]["team"] for letter, table in standings_by_group.items()}
    runnerup_by_group = {letter: table[1]["team"] for letter, table in standings_by_group.items()}
    third_by_group = {letter: table[2]["team"] for letter, table in standings_by_group.items()}
    third_stats = {
        letter: {"points": table[2]["points"], "gd": table[2]["gd"], "gf": table[2]["gf"]}
        for letter, table in standings_by_group.items()
    }
    advancing_third_letters = best_eight_third_place(third_stats, rng)
    qualified = set(winner_by_group.values()) | set(runnerup_by_group.values()) | {
        third_by_group[letter] for letter in advancing_third_letters
    }
    if len(qualified) != 32:
        raise ValueError(f"Expected 32 qualified teams, found {len(qualified)}")
    r32_matches = resolve_round_of_32(
        winner_by_group,
        runnerup_by_group,
        third_by_group,
        advancing_third_letters,
        r32_skeleton,
        r32_combos,
    )
    if len(r32_matches) != 16:
        raise ValueError(f"Expected 16 Round-of-32 matches, found {len(r32_matches)}")
    return qualified, r32_matches


def build_team_state(upcoming_matches: pd.DataFrame) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Create current Elo and form lookups from upcoming match rows."""

    home_elo = upcoming_matches[["home_team", "home_elo_now"]].rename(
        columns={"home_team": "team", "home_elo_now": "elo"}
    )
    away_elo = upcoming_matches[["away_team", "away_elo_now"]].rename(
        columns={"away_team": "team", "away_elo_now": "elo"}
    )
    elo_now = (
        pd.concat([home_elo, away_elo])
        .drop_duplicates("team")
        .set_index("team")["elo"]
        .to_dict()
    )

    team_form_now: dict[str, dict[str, float]] = {}
    for row in upcoming_matches.itertuples(index=False):
        for side in ["home", "away"]:
            team = getattr(row, f"{side}_team")
            team_form_now[team] = {
                "points_form": getattr(row, f"{side}_points_form"),
                "gf_form": getattr(row, f"{side}_gf_form"),
                "ga_form": getattr(row, f"{side}_ga_form"),
            }
    return elo_now, team_form_now


def knockout_feature_row(
    home_team: str,
    away_team: str,
    elo_now: dict[str, float],
    team_form_now: dict[str, dict[str, float]],
) -> pd.Series:
    """Build model features for a knockout fixture not present in the source data."""

    home_form = team_form_now[home_team]
    away_form = team_form_now[away_team]
    return pd.Series(
        {
            "elo_diff": elo_now[home_team] - elo_now[away_team],
            "neutral": True,
            "home_points_form": home_form["points_form"],
            "away_points_form": away_form["points_form"],
            "home_gf_form": home_form["gf_form"],
            "away_gf_form": away_form["gf_form"],
            "home_ga_form": home_form["ga_form"],
            "away_ga_form": away_form["ga_form"],
        }
    )


def build_knockout_home_probability(
    teams: list[str],
    outcome_model,
    elo_now: dict[str, float],
    team_form_now: dict[str, dict[str, float]],
    features: list[str] = FINAL_FEATURES,
    penalty_home_probability: float = 0.5,
) -> dict[tuple[str, str], float]:
    """Cache ordered knockout advancement probabilities for all team pairs."""

    cache: dict[tuple[str, str], float] = {}
    for home_team in teams:
        for away_team in teams:
            if home_team == away_team:
                continue
            probabilities = predict_outcome_probabilities(
                outcome_model,
                knockout_feature_row(home_team, away_team, elo_now, team_form_now),
                features,
            )
            cache[(home_team, away_team)] = float(
                probabilities["home_win"]
                + penalty_home_probability * probabilities["draw"]
            )
    return cache


def simulate_cached_knockout(
    home_team: str,
    away_team: str,
    knockout_home_probability: dict[tuple[str, str], float],
    rng,
) -> tuple[str, str]:
    """Draw one knockout winner from cached ordered-pair probability."""

    home_wins = rng.random() < knockout_home_probability[(home_team, away_team)]
    return (home_team, away_team) if home_wins else (away_team, home_team)


def simulate_full_tournament(
    group_data: list[dict],
    r32_skeleton: pd.DataFrame,
    r32_combos: pd.DataFrame,
    later_rounds: pd.DataFrame,
    knockout_home_probability: dict[tuple[str, str], float],
    rng,
) -> dict:
    """Simulate from group-stage remainder through the final."""

    _, simulated_r32 = run_group_stage_full(group_data, r32_skeleton, r32_combos, rng)
    results: dict[int, dict[str, str]] = {}

    for row in simulated_r32.itertuples(index=False):
        winner, loser = simulate_cached_knockout(
            row.home_team, row.away_team, knockout_home_probability, rng
        )
        results[int(row.match_no)] = {"winner": winner, "loser": loser}

    semifinalists: set[str] = set()
    finalists: set[str] = set()
    for _, slot in later_rounds.iterrows():
        def source_team(source) -> str:
            source = str(source)
            if source.startswith("loser_"):
                return results[int(source.removeprefix("loser_"))]["loser"]
            return results[int(source)]["winner"]

        home_team = source_team(slot["home_source_match"])
        away_team = source_team(slot["away_source_match"])
        if slot["round"] == "SF":
            semifinalists.update([home_team, away_team])
        elif slot["round"] == "Final":
            finalists.update([home_team, away_team])

        winner, loser = simulate_cached_knockout(
            home_team, away_team, knockout_home_probability, rng
        )
        results[int(slot["match_no"])] = {"winner": winner, "loser": loser}

    return {
        "champion": results[104]["winner"],
        "runner_up": results[104]["loser"],
        "third_place": results[103]["winner"],
        "semifinalists": semifinalists,
        "finalists": finalists,
    }


def run_monte_carlo(
    teams: list[str],
    group_data: list[dict],
    r32_skeleton: pd.DataFrame,
    r32_combos: pd.DataFrame,
    later_rounds: pd.DataFrame,
    knockout_home_probability: dict[tuple[str, str], float],
    n_simulations: int = 10_000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Run full-tournament Monte Carlo and return advancement probabilities."""

    rng = np.random.default_rng(seed)
    champion_count = {team: 0 for team in teams}
    final_count = {team: 0 for team in teams}
    semifinal_count = {team: 0 for team in teams}

    for _ in range(n_simulations):
        simulation = simulate_full_tournament(
            group_data,
            r32_skeleton,
            r32_combos,
            later_rounds,
            knockout_home_probability,
            rng,
        )
        champion_count[simulation["champion"]] += 1
        for team in simulation["finalists"]:
            final_count[team] += 1
        for team in simulation["semifinalists"]:
            semifinal_count[team] += 1

    return (
        pd.DataFrame(
            {
                "team": teams,
                "semifinal_probability": [
                    semifinal_count[team] / n_simulations for team in teams
                ],
                "final_probability": [final_count[team] / n_simulations for team in teams],
                "champion_probability": [
                    champion_count[team] / n_simulations for team in teams
                ],
            }
        )
        .sort_values("champion_probability", ascending=False)
        .reset_index(drop=True)
    )
