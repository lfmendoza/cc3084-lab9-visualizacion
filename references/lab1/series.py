"""Construction and description of the monthly series of the study.

These functions are shared by the notebook, the data pipeline and the automated
tests, so that there is a single reference implementation for going from the
individual record to the time series.

The population definition is the most delicate decision of the analysis and is
documented in :mod:`lab_series.loading`: the study works with non-resident
international visitors, that is, tourists and excursionists whose country of
residence is not Guatemala.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from lab_series.loading import (
    REFERENCE_POPULATION,
    TOTAL_REPORTED_POPULATION,
    filter_population,
)

DATE_COLUMN = "date"
VALUE_COLUMN = "visitors"

TOTAL_CATEGORY = "total"
ENTRY_MODE_CATEGORY = "entry_mode"
REGION_CATEGORY = "region"


def build_monthly_series(
    data: pd.DataFrame,
    *,
    filters: dict[str, str] | None = None,
    value_column: str = VALUE_COLUMN,
) -> pd.Series:
    """Build an aggregated monthly series at month-start frequency.

    Applies the requested filters by exact equality, aggregates by ``date`` and
    reindexes to the full calendar, filling months without records with zero,
    so that the resulting series has no temporal gaps.
    """
    if DATE_COLUMN not in data or value_column not in data:
        raise KeyError(f"Columns '{DATE_COLUMN}' and '{value_column}' are required.")
    if not pd.api.types.is_datetime64_any_dtype(data[DATE_COLUMN]):
        raise TypeError("Column 'date' must be of datetime type.")

    unknown = set(filters or {}).difference(data.columns)
    if unknown:
        raise KeyError(f"Unknown filter columns: {sorted(unknown)}")

    calendar = pd.date_range(
        data[DATE_COLUMN].min(), data[DATE_COLUMN].max(), freq="MS"
    )
    filtered = data
    for column, value in (filters or {}).items():
        filtered = filtered.loc[filtered[column].eq(value)]

    series = (
        filtered.groupby(DATE_COLUMN)[value_column]
        .sum()
        .reindex(calendar, fill_value=0.0)
        .astype("float64")
    )
    series.index.name = DATE_COLUMN
    series.index.freq = "MS"
    series.name = value_column
    return series


@dataclass(frozen=True)
class SeriesDefinition:
    """Describe a series of the study and how it is derived from the record.

    The definition holds only internal codes; the name and the description read
    by the audience live in :mod:`lab_series.presentation`.
    """

    key: str
    category: str
    population: str
    filters: dict[str, str] = field(default_factory=dict)

    def build(self, facts: pd.DataFrame) -> pd.Series:
        base = filter_population(facts, self.population)
        series = build_monthly_series(base, filters=self.filters or None)
        series.name = self.key
        return series


DEFINITIONS: tuple[SeriesDefinition, ...] = (
    SeriesDefinition(
        key="total",
        category=TOTAL_CATEGORY,
        population=REFERENCE_POPULATION,
    ),
    SeriesDefinition(
        key="air_entry",
        category=ENTRY_MODE_CATEGORY,
        population=REFERENCE_POPULATION,
        filters={"entry_mode": "Aérea"},
    ),
    SeriesDefinition(
        key="land_entry",
        category=ENTRY_MODE_CATEGORY,
        population=REFERENCE_POPULATION,
        filters={"entry_mode": "Terrestre"},
    ),
    SeriesDefinition(
        # Maritime traffic is recorded as cruise passengers from 2017 onwards,
        # a category outside the reference population, so this series is built
        # on the total reported figure. See docs/adr/0002.
        key="sea_entry",
        category=ENTRY_MODE_CATEGORY,
        population=TOTAL_REPORTED_POPULATION,
        filters={"entry_mode": "Marítima"},
    ),
    SeriesDefinition(
        key="central_america",
        category=REGION_CATEGORY,
        population=REFERENCE_POPULATION,
        filters={"region_group": "América Del Centro"},
    ),
    SeriesDefinition(
        key="north_america",
        category=REGION_CATEGORY,
        population=REFERENCE_POPULATION,
        filters={"region_group": "América Del Norte"},
    ),
    SeriesDefinition(
        key="europe",
        category=REGION_CATEGORY,
        population=REFERENCE_POPULATION,
        filters={"region_group": "Europa"},
    ),
)

DEFINITIONS_BY_KEY: dict[str, SeriesDefinition] = {
    definition.key: definition for definition in DEFINITIONS
}


def build_series(facts: pd.DataFrame) -> dict[str, pd.Series]:
    """Build the seven series of the study from the base record."""
    return {definition.key: definition.build(facts) for definition in DEFINITIONS}


def series_to_long(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Turn the dictionary of series into a persistable long frame."""
    frames = []
    for key, values in series.items():
        definition = DEFINITIONS_BY_KEY[key]
        frames.append(
            pd.DataFrame(
                {
                    "series": key,
                    "category": definition.category,
                    "population": definition.population,
                    "date": values.index,
                    "visitors": values.to_numpy(dtype="float64"),
                }
            )
        )
    long_form = pd.concat(frames, ignore_index=True)
    return long_form.sort_values(["series", "date"]).reset_index(drop=True)


def long_to_series(long_form: pd.DataFrame) -> dict[str, pd.Series]:
    """Rebuild the dictionary of series from the long format."""
    order = [
        definition.key
        for definition in DEFINITIONS
        if definition.key in set(long_form["series"])
    ]
    series: dict[str, pd.Series] = {}
    for key in order:
        group = long_form.loc[long_form["series"] == key].sort_values("date")
        values = pd.Series(
            group["visitors"].to_numpy(dtype="float64"),
            index=pd.DatetimeIndex(group["date"], name="date"),
            name=key,
        )
        values.index.freq = "MS"
        series[key] = values
    return series


def series_profile(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Summarise the span, frequency and shape of every series of the study."""
    rows = []
    for key, values in series.items():
        definition = DEFINITIONS_BY_KEY[key]
        mean = values.mean()
        deviation = values.std(ddof=1)
        rows.append(
            {
                "series": key,
                "category": definition.category,
                "population": definition.population,
                "first_month": values.index.min(),
                "last_month": values.index.max(),
                "observations": int(values.size),
                "cumulative_visitors": values.sum(),
                "monthly_mean": mean,
                "median": values.median(),
                "standard_deviation": deviation,
                "coefficient_of_variation": (
                    deviation / mean * 100 if mean else np.nan
                ),
                "minimum": values.min(),
                "maximum": values.max(),
            }
        )
    return pd.DataFrame(rows).set_index("series")


def stationarity_summary(
    series: pd.Series,
    *,
    kpss_regression: str = "ct",
) -> pd.Series:
    """Test stationarity in mean with Dickey-Fuller and KPSS.

    The two tests state opposite null hypotheses, so they are used together:
    the augmented Dickey-Fuller test assumes a unit root and KPSS assumes
    stationarity. Agreement between them yields a firmer verdict than either
    one on its own.
    """
    clean = series.dropna()
    adf_statistic, adf_pvalue, adf_lags, _, adf_critical, _ = adfuller(
        clean, autolag="AIC"
    )
    kpss_statistic, kpss_pvalue, kpss_lags, kpss_critical = kpss(
        clean, regression=kpss_regression, nlags="auto"
    )

    rejects_adf = adf_pvalue < 0.05
    rejects_kpss = kpss_pvalue < 0.05

    if rejects_adf and not rejects_kpss:
        verdict = "stationary"
    elif not rejects_adf and rejects_kpss:
        verdict = "needs_differencing"
    elif rejects_adf and rejects_kpss:
        verdict = "mixed_evidence"
    else:
        verdict = "insufficient_evidence"

    return pd.Series(
        {
            "adf_statistic": adf_statistic,
            "adf_pvalue": adf_pvalue,
            "adf_lags": int(adf_lags),
            "adf_critical_5pct": adf_critical["5%"],
            "kpss_statistic": kpss_statistic,
            "kpss_pvalue": kpss_pvalue,
            "kpss_lags": int(kpss_lags),
            "kpss_critical_5pct": kpss_critical["5%"],
            "verdict": verdict,
        },
        name=series.name,
    )


def differencing_order(
    series: pd.Series,
    *,
    maximum: int = 2,
    period: int = 12,
) -> pd.Series:
    """Determine how many regular and seasonal differences are needed.

    The seasonal difference is applied first, since it is the one that removes
    the annual pattern, and only then is the remaining series tested for a unit
    root with Dickey-Fuller.
    """
    working = series.dropna()
    seasonal = 0
    if _has_seasonal_root(working, period=period):
        working = working.diff(period).dropna()
        seasonal = 1

    regular = 0
    while regular < maximum:
        pvalue = float(adfuller(working, autolag="AIC")[1])
        if pvalue < 0.05:
            break
        working = working.diff().dropna()
        regular += 1

    return pd.Series(
        {
            "differences": regular,
            "seasonal_differences": seasonal,
            "observations_after_differencing": int(working.size),
            "final_adf_pvalue": float(adfuller(working, autolag="AIC")[1]),
        },
        name=series.name,
    )


def _has_seasonal_root(series: pd.Series, *, period: int = 12) -> bool:
    """Approximate the seasonal unit root test with annual autocorrelation."""
    if series.size <= 2 * period:
        return False
    autocorrelation = series.autocorr(lag=period)
    return bool(autocorrelation is not None and autocorrelation > 0.4)


def recommended_transform(series: pd.Series) -> pd.Series:
    """Assess whether a transformation should stabilise the variance.

    Compares the relation between level and dispersion on the original and on
    the logarithmic scale. When the rolling standard deviation grows together
    with the mean, the variance is not constant and the logarithmic scale is
    the natural correction, besides turning multiplicative growth into additive
    growth.
    """
    window = 12
    before_pandemic = series.loc[:"2019-12-01"]

    def correlation(values: pd.Series) -> float:
        mean = values.rolling(window).mean()
        deviation = values.rolling(window).std()
        return float(mean.corr(deviation))

    original_correlation = correlation(before_pandemic)
    log_correlation = correlation(np.log1p(before_pandemic))
    recommends = original_correlation > 0.5 and log_correlation < original_correlation

    return pd.Series(
        {
            "level_dispersion_correlation": original_correlation,
            "log_level_dispersion_correlation": log_correlation,
            "dispersion_ratio": _dispersion_ratio(series),
            "log_dispersion_ratio": _dispersion_ratio(np.log1p(series)),
            "recommended_transform": "logarithm" if recommends else "none",
        },
        name=series.name,
    )


def _dispersion_ratio(series: pd.Series, *, window: int = 12) -> float:
    """Ratio between the largest and the smallest rolling dispersion."""
    deviation = series.rolling(window).std().dropna()
    minimum = deviation.min()
    if not minimum:
        return float("inf")
    return float(deviation.max() / minimum)
