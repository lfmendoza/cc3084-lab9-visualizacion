"""Error and fit metrics for comparing forecasting models.

Three complementary kinds of metric are reported. The mean absolute error and
the root mean squared error are expressed in visitors, so that management can
size the magnitude of the miss. The scaled error allows series of very
different size to be compared, such as land and sea arrivals. The information
criteria penalise complexity and serve to choose between specifications of the
same family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEASONAL_PERIOD = 12

DEFAULT_RANKING_METRIC = "mean_absolute_scaled_error"


def _align(actual: pd.Series, forecast: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    common = actual.index.intersection(forecast.index)
    if common.empty:
        raise ValueError("The forecast and the observation share no date.")
    return (
        actual.loc[common].to_numpy(dtype="float64"),
        forecast.loc[common].to_numpy(dtype="float64"),
    )


def mean_absolute_error(actual: pd.Series, forecast: pd.Series) -> float:
    """Average absolute error, in visitors."""
    observed, predicted = _align(actual, forecast)
    return float(np.mean(np.abs(observed - predicted)))


def root_mean_squared_error(actual: pd.Series, forecast: pd.Series) -> float:
    """Root mean squared error, which penalises large misses."""
    observed, predicted = _align(actual, forecast)
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def mean_absolute_percentage_error(actual: pd.Series, forecast: pd.Series) -> float:
    """Mean absolute percentage error, undefined against null observations."""
    observed, predicted = _align(actual, forecast)
    valid = observed != 0
    if not valid.any():
        return float("nan")
    return float(
        np.mean(np.abs((observed[valid] - predicted[valid]) / observed[valid])) * 100
    )


def symmetric_percentage_error(actual: pd.Series, forecast: pd.Series) -> float:
    """Symmetric variant, stable when the series takes small values."""
    observed, predicted = _align(actual, forecast)
    denominator = (np.abs(observed) + np.abs(predicted)) / 2
    valid = denominator != 0
    if not valid.any():
        return float("nan")
    return float(
        np.mean(np.abs(observed[valid] - predicted[valid]) / denominator[valid]) * 100
    )


def mean_absolute_scaled_error(
    actual: pd.Series,
    forecast: pd.Series,
    train: pd.Series,
    *,
    period: int = SEASONAL_PERIOD,
) -> float:
    """Mean absolute error scaled by the seasonal naive method.

    A value below one means the model improves on repeating the same month of
    the previous year; a value above one means it adds nothing over that
    trivial benchmark.
    """
    observed, predicted = _align(actual, forecast)
    history = train.to_numpy(dtype="float64")
    if history.size <= period:
        return float("nan")
    scale = np.mean(np.abs(history[period:] - history[:-period]))
    if not scale:
        return float("nan")
    return float(np.mean(np.abs(observed - predicted)) / scale)


def mean_bias(actual: pd.Series, forecast: pd.Series) -> float:
    """Signed average error: positive when the model underestimates."""
    observed, predicted = _align(actual, forecast)
    return float(np.mean(observed - predicted))


def interval_coverage(actual: pd.Series, interval: pd.DataFrame) -> float:
    """Share of observations covered by the prediction interval."""
    common = actual.index.intersection(interval.index)
    if common.empty:
        return float("nan")
    observed = actual.loc[common].to_numpy(dtype="float64")
    lower = interval.loc[common].iloc[:, 0].to_numpy(dtype="float64")
    upper = interval.loc[common].iloc[:, 1].to_numpy(dtype="float64")
    return float(np.mean((observed >= lower) & (observed <= upper)) * 100)


def corrected_akaike(
    log_likelihood: float, parameters: int, observations: int
) -> float:
    """Akaike information criterion corrected for small samples."""
    akaike = 2 * parameters - 2 * log_likelihood
    denominator = observations - parameters - 1
    if denominator <= 0:
        return float("inf")
    return float(akaike + (2 * parameters * (parameters + 1)) / denominator)


def evaluate(
    actual: pd.Series,
    forecast: pd.Series,
    train: pd.Series,
    *,
    model: str = "model",
    interval: pd.DataFrame | None = None,
    akaike: float | None = None,
    bayesian: float | None = None,
) -> dict[str, float | str]:
    """Compute the full set of metrics of one forecast."""
    result: dict[str, float | str] = {
        "model": model,
        "mean_absolute_error": mean_absolute_error(actual, forecast),
        "root_mean_squared_error": root_mean_squared_error(actual, forecast),
        "mean_absolute_percentage_error": mean_absolute_percentage_error(
            actual, forecast
        ),
        "symmetric_percentage_error": symmetric_percentage_error(actual, forecast),
        "mean_absolute_scaled_error": mean_absolute_scaled_error(
            actual, forecast, train
        ),
        "mean_bias": mean_bias(actual, forecast),
    }
    if interval is not None:
        result["interval_coverage"] = interval_coverage(actual, interval)
    result["akaike"] = akaike if akaike is not None else np.nan
    result["bayesian"] = bayesian if bayesian is not None else np.nan
    return result


def comparison_table(
    evaluations: list[dict[str, float | str]],
    *,
    sort_by: str = DEFAULT_RANKING_METRIC,
) -> pd.DataFrame:
    """Sort the evaluations and mark the position of every model."""
    table = pd.DataFrame(evaluations).set_index("model")
    if sort_by in table.columns:
        table = table.sort_values(sort_by)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table


# Number of structural decisions each family imposes on the analyst. It orders
# the models by parsimony when their errors are indistinguishable.
COMPLEXITY: dict[str, int] = {
    "seasonal_naive": 0,
    "holt_winters": 1,
    "state_space_smoothing": 1,
    "sarima": 2,
    "sarima_with_intervention": 3,
    "decomposable_additive": 3,
    "prophet": 4,
}


def select_best(
    table: pd.DataFrame,
    *,
    criterion: str = DEFAULT_RANKING_METRIC,
    tolerance: float = 0.02,
) -> str:
    """Pick the model with the lowest error, breaking ties by parsimony.

    When several specifications fall within a narrow margin of the best error,
    the difference is not distinguishable from sampling noise and the simplest
    one is preferred, being easier to maintain and to explain. This rule
    avoids, for instance, adopting an elaborate model when it reproduces the
    naive forecast exactly.
    """
    if criterion not in table.columns:
        raise KeyError(f"The table has no column '{criterion}'.")

    values = table[criterion].dropna()
    if values.empty:
        raise ValueError(f"No model has a value for '{criterion}'.")

    threshold = float(values.min()) * (1 + tolerance)
    tied = values[values <= threshold].index
    return min(
        tied,
        key=lambda name: (COMPLEXITY.get(str(name), 5), float(values[name])),
    )
