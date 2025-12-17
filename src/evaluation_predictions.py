import numpy as np
import pandas as pd
from glum import TweedieDistribution
from sklearn.metrics import auc


def evaluate_predictions(
    df,
    outcome_column,
    *,
    preds_column=None,
    model=None,
    tweedie_power=1.5,
    exposure_column=None,
):
    """
    Evaluate model predictions using regression and metrics.
    Computes bias, RMSE, MAE, deviance, and Gini coefficient for model performance.
    """
    metrics = {}

    assert (
        preds_column is not None or model is not None
    ), "Provide either preds_column or model."
    preds = df[preds_column] if preds_column is not None else model.predict(df)
    if exposure_column is not None:
        weights = df[exposure_column].values
    else:
        weights = np.ones(len(df))

    actual = df[outcome_column].values

    metrics["mean_preds"] = np.average(preds, weights=weights)
    metrics["mean_outcome"] = np.average(actual, weights=weights)
    metrics["bias"] = (metrics["mean_preds"] - metrics["mean_outcome"]) / metrics[
        "mean_outcome"
    ]

    metrics["mse"] = np.average((preds - actual) ** 2, weights=weights)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["mae"] = np.average(np.abs(preds - actual), weights=weights)

    metrics["deviance"] = TweedieDistribution(tweedie_power).deviance(
        actual, preds, sample_weight=weights
    ) / np.sum(weights)

    ordered_samples, cum_actuals = lorenz_curve(actual, preds, weights)
    metrics["gini"] = 1 - 2 * auc(ordered_samples, cum_actuals)

    return pd.DataFrame(metrics, index=[0]).T


def lorenz_curve(y_true, y_pred, exposure):
    """
    Compute the Lorenz curve based on predicted risk ranking.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    exposure = np.asarray(exposure)

    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_actuals = y_true[ranking]
    cum_claim_amount = np.cumsum(ranked_actuals * ranked_exposure)
    cum_claim_amount /= cum_claim_amount[-1]  # Normalize to 0–1
    cum_samples = np.linspace(0, 1, len(cum_claim_amount))
    return cum_samples, cum_claim_amount
