"""Model training, validation, and probability helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import FINAL_FEATURES, RECENCY_HALF_LIFE_YEARS


def recency_weights(
    dates: pd.Series,
    half_life_years: float = RECENCY_HALF_LIFE_YEARS,
) -> pd.Series:
    """Exponentially down-weight older matches."""

    age_years = (dates.max() - dates).dt.days / 365.25
    return 0.5 ** (age_years / half_life_years)


def make_classifier():
    """Create the W/D/L classification pipeline."""

    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )


def make_goal_model():
    """Create a Poisson goal-rate regression pipeline."""

    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        PoissonRegressor(alpha=0.1, max_iter=2000),
    )


def classifier_classes(model) -> np.ndarray:
    """Return class labels from the fitted pipeline."""

    return model.named_steps["logisticregression"].classes_


def multiclass_brier_score(
    y_true: pd.Series,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> float:
    """Mean multi-class Brier score across all outcome classes."""

    class_index = {label: index for index, label in enumerate(classes)}
    observed = np.zeros_like(probabilities, dtype=float)
    for row_index, label in enumerate(y_true):
        observed[row_index, class_index[label]] = 1.0
    return float(np.mean(np.sum((probabilities - observed) ** 2, axis=1)))


def evaluate_classifier_candidates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidate_features: dict[str, list[str]],
    subsets: dict[str, pd.Series],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit candidate classifiers and score them on named temporal subsets."""

    train_weights = recency_weights(train["date"])
    rows: list[dict] = []
    validation_models: dict[str, object] = {}
    for model_name, features in candidate_features.items():
        fitted = make_classifier()
        fitted.fit(
            train[features],
            train["outcome"],
            logisticregression__sample_weight=train_weights,
        )
        validation_models[model_name] = fitted
        classes = classifier_classes(fitted)

        for subset_name, mask in subsets.items():
            subset = test.loc[mask]
            probabilities = fitted.predict_proba(subset[features])
            predictions = fitted.predict(subset[features])
            rows.append(
                {
                    "model": model_name,
                    "subset": subset_name,
                    "n_matches": len(subset),
                    "accuracy": accuracy_score(subset["outcome"], predictions),
                    "log_loss": log_loss(subset["outcome"], probabilities, labels=classes),
                    "brier_score": multiclass_brier_score(
                        subset["outcome"], probabilities, classes
                    ),
                }
            )
    return pd.DataFrame(rows), validation_models


def baseline_scores(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float | str]:
    """Score the majority-class/prior-probability baseline."""

    class_order = sorted(train["outcome"].unique())
    train_prior = train["outcome"].value_counts(normalize=True).reindex(class_order)
    probabilities = np.tile(train_prior.to_numpy(), (len(test), 1))
    prediction = train_prior.idxmax()
    return {
        "prediction": prediction,
        "accuracy": float((test["outcome"] == prediction).mean()),
        "log_loss": float(log_loss(test["outcome"], probabilities, labels=class_order)),
        "brier_score": multiclass_brier_score(
            test["outcome"], probabilities, np.array(class_order)
        ),
    }


def fit_outcome_model(
    df_recent: pd.DataFrame,
    features: list[str] = FINAL_FEATURES,
):
    """Fit the final outcome classifier on all known played matches."""

    model = make_classifier()
    model.fit(
        df_recent[features],
        df_recent["outcome"],
        logisticregression__sample_weight=recency_weights(df_recent["date"]),
    )
    return model


def fit_goal_models(
    df_recent: pd.DataFrame,
    features: list[str] = FINAL_FEATURES,
):
    """Fit home and away Poisson goal models on all known played matches."""

    weights = recency_weights(df_recent["date"])
    home_goal_model = make_goal_model()
    away_goal_model = make_goal_model()
    home_goal_model.fit(
        df_recent[features],
        df_recent["home_score"],
        poissonregressor__sample_weight=weights,
    )
    away_goal_model.fit(
        df_recent[features],
        df_recent["away_score"],
        poissonregressor__sample_weight=weights,
    )
    return home_goal_model, away_goal_model


def evaluate_goal_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str] = FINAL_FEATURES,
) -> dict[str, float]:
    """Fit validation goal models and return MAE metrics."""

    weights = recency_weights(train["date"])
    home_goal_model = make_goal_model()
    away_goal_model = make_goal_model()
    home_goal_model.fit(
        train[features],
        train["home_score"],
        poissonregressor__sample_weight=weights,
    )
    away_goal_model.fit(
        train[features],
        train["away_score"],
        poissonregressor__sample_weight=weights,
    )
    return {
        "home_mae": float(mean_absolute_error(test["home_score"], home_goal_model.predict(test[features]))),
        "away_mae": float(mean_absolute_error(test["away_score"], away_goal_model.predict(test[features]))),
    }


def predict_outcome_probabilities(
    model,
    row: pd.Series,
    features: list[str] = FINAL_FEATURES,
) -> dict[str, float]:
    """Convert one feature row into class probability mapping."""

    feature_row = pd.DataFrame([{feature: row[feature] for feature in features}])
    probabilities = model.predict_proba(feature_row)[0]
    return dict(zip(classifier_classes(model), probabilities))


def predict_goal_rates(
    home_goal_model,
    away_goal_model,
    row: pd.Series,
    features: list[str] = FINAL_FEATURES,
    min_rate: float = 0.05,
    max_rate: float = 7.0,
) -> tuple[float, float]:
    """Return clipped Poisson goal rates for one match."""

    feature_row = pd.DataFrame([{feature: row[feature] for feature in features}])
    home_rate = float(home_goal_model.predict(feature_row)[0])
    away_rate = float(away_goal_model.predict(feature_row)[0])
    return (
        float(np.clip(home_rate, min_rate, max_rate)),
        float(np.clip(away_rate, min_rate, max_rate)),
    )


def calibration_table(
    y_true: pd.Series,
    probabilities: np.ndarray,
    classes: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Create one-vs-rest calibration bins for every class."""

    rows: list[dict] = []
    bins = np.linspace(0, 1, n_bins + 1)
    for class_index, class_label in enumerate(classes):
        predicted = probabilities[:, class_index]
        observed = (y_true.to_numpy() == class_label).astype(float)
        bin_ids = np.clip(np.digitize(predicted, bins, right=True) - 1, 0, n_bins - 1)
        for bin_id in range(n_bins):
            mask = bin_ids == bin_id
            if not mask.any():
                continue
            rows.append(
                {
                    "class": class_label,
                    "bin_low": float(bins[bin_id]),
                    "bin_high": float(bins[bin_id + 1]),
                    "n": int(mask.sum()),
                    "mean_predicted": float(predicted[mask].mean()),
                    "observed_rate": float(observed[mask].mean()),
                }
            )
    return pd.DataFrame(rows)
