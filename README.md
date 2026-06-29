# World Cup 2026 Predictor

A portfolio machine-learning project for forecasting the 2026 FIFA World Cup from an intentionally versioned data snapshot.

The project combines pre-match Elo ratings, recent team form, temporal validation, Poisson goal simulation, and a Monte Carlo tournament engine for the 48-team World Cup format.

## Snapshot Status

This repository is **not a live forecast**.

- Data source: <https://raw.githubusercontent.com/martj42/international_results/master/results.csv>
- Processed snapshot retrieved: `2026-06-23T13:42:00+10:00`
- Latest played match in processed data: `2026-06-21`
- Portfolio audit date: `2026-06-29`
- Snapshot caveat: matches dated `2026-06-22` through `2026-06-27` are still stored as upcoming in `data/processed/upcoming_matches.csv`.

See `data/processed/metadata.json` for row counts and freshness notes.

## What The Model Does

- Builds Elo ratings using only matches before each row being predicted.
- Adds 5-match rolling form features while avoiding same-day result leakage.
- Evaluates candidate classifiers with a temporal split: train before `2024-01-01`, test from `2024-01-01` onward.
- Uses Logistic Regression for win/draw/loss probabilities.
- Uses two Poisson Regression models for home and away goal rates.
- Simulates the 48-team World Cup format: 12 groups, top two plus the eight best third-place teams, Round of 32, and knockout rounds.
- Runs 10,000 Monte Carlo tournament simulations in the notebook output.

Current notebook validation snapshot:

| Model | Test subset | Accuracy | Log loss |
| --- | --- | ---: | ---: |
| Elo + neutral + form | All 2024+ matches | 0.596 | 0.871 |
| Elo + neutral + form | Neutral-site 2024+ matches | 0.553 | 0.944 |
| Elo + neutral + form | World Cup 2024+ matches | 0.500 | 0.977 |
| Majority/prior baseline | All 2024+ matches | 0.468 | 1.057 |

Goal model validation:

| Target | MAE |
| --- | ---: |
| Home goals | 1.059 |
| Away goals | 0.882 |

## Repository Layout

```text
01_eda.ipynb                    # raw data checks, Elo, form features
02_model_training.ipynb         # temporal validation and group-stage simulation
03_knockout_simulation.ipynb    # Round-of-32 and full-tournament Monte Carlo
src/wc2026_predictor/           # reusable feature, model, and simulation helpers
scripts/validate_repo.py        # data/reference/model smoke validation
scripts/smoke_notebooks.py      # in-memory notebook execution smoke test
tests/                          # unit tests for core logic
data/processed/                 # versioned processed snapshot
data/reference/                 # group/bracket reference tables
docs/top_contenders.png         # generated chart from notebook 03
```

## Setup

Python 3.11+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, `pywin32` is included so Jupyter can create secure runtime files. In restricted shells, `scripts/smoke_notebooks.py` sets a temporary Jupyter runtime directory and enables insecure writes only for notebook smoke-test connection files.

## Run The Checks

```powershell
pytest
python scripts\validate_repo.py --today 2026-06-29
python scripts\smoke_notebooks.py
```

`validate_repo.py` checks:

- 48 teams, 12 groups, and 72 group-stage matches.
- 16 Round-of-32 matches and 495 third-place-team combinations.
- Metadata row counts and processed-data freshness.
- A lightweight Monte Carlo smoke test where semifinal probabilities sum to 4, final probabilities sum to 2, and champion probabilities sum to 1.

## Run The Analysis

The Quarto site uses saved notebook outputs:

```powershell
quarto render
```

The project currently has `execute.enabled: false` in `_quarto.yml`, so the website renders the checked-in notebook outputs rather than automatically refreshing data or re-running the model. To make a current forecast, refresh the raw data snapshot first, rerun the notebooks/scripts, and update `data/processed/metadata.json`.

## Limitations

- No betting odds, player-level data, injuries, squads, or market calibration.
- Poisson home/away goals are modeled independently.
- Knockout draws after 90 minutes are advanced with a simple 50/50 extra-time/penalty assumption.
- Fair-play and drawing-of-lots tie-break data is unavailable; unresolved ties use seeded lottery behavior for reproducibility.
- The current processed data is stale relative to `2026-06-29`, so public claims should describe it as a portfolio snapshot.
