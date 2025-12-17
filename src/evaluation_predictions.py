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
    Evaluate predictions against actual outcomes.

    Parameters
    ----------
    df : pd.Dataframe
        Dataframe used for evaluation
    outcome_column : str
        Name of outcome column
    preds_column : str, optional
        Name of predictions column, by default None
    model :
        Fitted model, by default None
    tweedie_power : float, optional
        Power of tweedie distribution for deviance computation, by default 1.5
    exposure_column : str, optional
        Name of exposure column, by default None

    Returns
    -------
    metrics = {}
        DataFrame containing metrics
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
