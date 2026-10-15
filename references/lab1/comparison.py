"""Structural comparison between the series of the study.

Answers the questions of the comparative analysis with numeric evidence: which
series shows the strongest seasonality, which grows fastest, which is the most
volatile and which was hit hardest by the pandemic. The seasonal and trend
strength indicators follow the formulation of Hyndman and Athanasopoulos, which
expresses each component as the share of variation it explains within the
series adjusted for the remaining components.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

PERIOD = 12
BASE_YEAR = 2019
SHOCK_START = "2020-03-01"
SHOCK_END = "2021-06-01"

SEASONALITY_QUESTION = "strongest_seasonality"
GROWTH_QUESTION = "strongest_growth"
VOLATILITY_QUESTION = "highest_volatility"
PANDEMIC_QUESTION = "hardest_hit_by_pandemic"


def _decompose(series: pd.Series):
    return STL(series, period=PERIOD, robust=True).fit()


def seasonal_strength(series: pd.Series) -> float:
    """Share of variation attributable to the annual pattern, from zero to one."""
    fit = _decompose(series)
    denominator = float(np.var(fit.seasonal + fit.resid, ddof=1))
    if denominator == 0:
        return 0.0
    return float(max(0.0, 1 - np.var(fit.resid, ddof=1) / denominator))


def trend_strength(series: pd.Series) -> float:
    """Share of variation attributable to the trend, from zero to one."""
    fit = _decompose(series)
    denominator = float(np.var(fit.trend + fit.resid, ddof=1))
    if denominator == 0:
        return 0.0
    return float(max(0.0, 1 - np.var(fit.resid, ddof=1) / denominator))


def volatility(series: pd.Series) -> dict[str, float]:
    """Measure the irregularity left by trend and seasonality."""
    fit = _decompose(series)
    mean = float(series.mean())
    growth = series.pct_change(PERIOD).replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "coefficient_of_variation": float(series.std(ddof=1) / mean * 100)
        if mean
        else np.nan,
        "residual_dispersion": float(fit.resid.std(ddof=1) / mean * 100)
        if mean
        else np.nan,
        "growth_deviation": float(growth.std(ddof=1) * 100),
    }


def trend_growth(series: pd.Series) -> dict[str, float]:
    """Estimate the long-run growth rate of the series.

    A linear regression is fitted on the logarithm of the estimated trend,
    excluding the border closure period so that the shock does not distort the
    structural slope.
    """
    trend = _decompose(series).trend
    valid = trend.loc[
        (trend.index < pd.Timestamp(SHOCK_START))
        | (trend.index > pd.Timestamp(SHOCK_END))
    ]
    valid = valid[valid > 0]
    if valid.size < 24:
        return {
            "trend_annual_growth": np.nan,
            "level_change_end_to_end": np.nan,
        }

    time = np.arange(valid.size, dtype="float64")
    slope = float(np.polyfit(time, np.log(valid.to_numpy()), 1)[0])
    return {
        "trend_annual_growth": (np.exp(slope * 12) - 1) * 100,
        "level_change_end_to_end": float(valid.iloc[-1] / valid.iloc[0] * 100 - 100),
    }


def pandemic_impact(series: pd.Series) -> dict[str, object]:
    """Quantify the depth of the shock and the time to recovery."""
    reference = series.loc[f"{BASE_YEAR}-01-01" : f"{BASE_YEAR}-12-01"]
    base_mean = float(reference.mean())
    base_total = float(reference.sum())

    shock = series.loc[SHOCK_START:SHOCK_END]
    minimum = float(shock.min())
    minimum_date = shock.idxmin()

    annual_cumulative = series.rolling(PERIOD).sum()
    afterwards = annual_cumulative.loc["2021-01-01":]
    recovered = afterwards[afterwards >= base_total]
    recovery_date = recovered.index[0] if len(recovered) else pd.NaT

    months_to_recovery = (
        int((recovery_date.year - 2020) * 12 + recovery_date.month - 3)
        if pd.notna(recovery_date)
        else np.nan
    )

    window = series.loc["2020-03-01":"2021-12-01"]
    expected = _expected_level(series, window.index)
    loss = float(expected.sum() - window.sum())
    relative_loss = float(loss / expected.sum() * 100) if expected.sum() else np.nan

    return {
        "maximum_fall_against_2019": (
            minimum / base_mean * 100 - 100 if base_mean else np.nan
        ),
        "lowest_month": minimum_date,
        "visitors_lost_2020_2021": loss,
        "pandemic_loss": relative_loss,
        "recovery_month": recovery_date,
        "months_to_recovery": months_to_recovery,
    }


def _expected_level(series: pd.Series, window: pd.DatetimeIndex) -> pd.Series:
    """Estimate the flow that would have occurred without the pandemic.

    Takes the average of each calendar month over the three years before the
    shock, so that the benchmark keeps the seasonality of the series instead of
    comparing it against a flat level.
    """
    before = series.loc["2017-01-01":"2020-02-01"]
    profile = before.groupby(before.index.month).mean()
    return pd.Series(
        [float(profile.get(date.month, np.nan)) for date in window],
        index=window,
    )


def compare_series(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Build the full comparative table across the series."""
    from lab_series.series import DEFINITIONS_BY_KEY

    rows = []
    for key, values in series.items():
        row: dict[str, object] = {
            "series": key,
            "category": DEFINITIONS_BY_KEY[key].category,
            "seasonal_strength": seasonal_strength(values),
            "trend_strength": trend_strength(values),
        }
        row.update(volatility(values))
        row.update(trend_growth(values))
        row.update(pandemic_impact(values))
        rows.append(row)

    return pd.DataFrame(rows).set_index("series")


# Each comparative question resolves to the series holding the maximum of one
# indicator; the criterion is stated so that the answer stays auditable.
QUESTIONS: tuple[tuple[str, str], ...] = (
    (SEASONALITY_QUESTION, "seasonal_strength"),
    (GROWTH_QUESTION, "trend_annual_growth"),
    (VOLATILITY_QUESTION, "residual_dispersion"),
    (PANDEMIC_QUESTION, "pandemic_loss"),
)


def answer_questions(comparison: pd.DataFrame) -> pd.DataFrame:
    """Turn the comparative table into direct, verifiable answers."""
    rows = []
    for question, column in QUESTIONS:
        values = comparison[column].dropna()
        if values.empty:
            continue
        winner = values.idxmax()
        rows.append(
            {
                "question": question,
                "series": winner,
                "observed_value": values.loc[winner],
                "indicator_used": column,
            }
        )
    return pd.DataFrame(rows)


def compare_by_category(comparison: pd.DataFrame) -> pd.DataFrame:
    """Answer the same questions within each analysis category."""
    rows = []
    for category, group in comparison.groupby("category"):
        if len(group) < 2:
            continue
        rows.append(
            {
                "category": category,
                SEASONALITY_QUESTION: group["seasonal_strength"].idxmax(),
                GROWTH_QUESTION: group["trend_annual_growth"].idxmax(),
                VOLATILITY_QUESTION: group["residual_dispersion"].idxmax(),
                PANDEMIC_QUESTION: group["pandemic_loss"].idxmax(),
            }
        )
    return pd.DataFrame(rows)
