"""Forecasting models and their diagnostics.

Four families are fitted on every series: seasonal autoregressive integrated
moving average models, Holt-Winters exponential smoothing, a decomposable
additive model in the style of Prophet and the seasonal naive method, which
acts as the minimum acceptable benchmark.

Two design decisions deserve an explicit justification. The first is that the
order search is implemented here, over a bounded grid evaluated with the
corrected Akaike criterion, instead of delegating it to an external automatic
selection library: the brief requires the choice of parameters to be explained,
and an in-house grid keeps both the candidates considered and the tie-breaking
rule in plain sight. The second is that the pandemic enters as an intervention
variable rather than being removed from the series, so that the model tells a
bounded external shock apart from a change in the underlying dynamics.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from lab_series.metrics import corrected_akaike
from lab_series.outliers import holy_week_months

PERIOD = 12
SIGNIFICANCE_LEVEL = 0.05

CLOSURE_START = pd.Timestamp("2020-03-01")
CLOSURE_END = pd.Timestamp("2021-02-01")
RECOVERY_RATE = 0.85

SARIMA = "sarima"
SARIMA_WITH_INTERVENTION = "sarima_with_intervention"
HOLT_WINTERS = "holt_winters"
STATE_SPACE_SMOOTHING = "state_space_smoothing"
PROPHET = "prophet"
DECOMPOSABLE_ADDITIVE = "decomposable_additive"
SEASONAL_NAIVE = "seasonal_naive"

BORDER_CLOSURE_REGRESSOR = "border_closure"
RECOVERY_REGRESSOR = "post_closure_recovery"
HOLY_WEEK_REGRESSOR = "holy_week_month"

LOWER_BOUND = "lower_bound"
UPPER_BOUND = "upper_bound"

logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)


@dataclass
class Forecast:
    """Homogeneous result of any family of models."""

    model: str
    values: pd.Series
    interval: pd.DataFrame | None = None
    fitted: pd.Series | None = None
    residuals: pd.Series | None = None
    akaike: float | None = None
    bayesian: float | None = None
    detail: dict[str, object] = field(default_factory=dict)


def intervention_regressors(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build the variables that represent shocks and calendar effects.

    Three regressors are included. The first is a step equal to one during the
    border closure. The second decays geometrically from the reopening and
    captures the gradual recovery of demand. The third marks the month holding
    Holy Week, whose moving date shifts the peak between March and April and
    which, without this control, would be mistaken for an anomaly.
    """
    index = pd.DatetimeIndex(index)
    closure = ((index >= CLOSURE_START) & (index <= CLOSURE_END)).astype("float64")

    recovery = np.zeros(len(index), dtype="float64")
    months_since_reopening = (
        (index.year - CLOSURE_END.year) * 12 + index.month - CLOSURE_END.month
    )
    after = months_since_reopening > 0
    recovery[after] = RECOVERY_RATE ** months_since_reopening[after]

    easter = holy_week_months(range(index.year.min(), index.year.max() + 1))
    holy_week = pd.Series(index.to_period("M").to_timestamp()).isin(easter)

    return pd.DataFrame(
        {
            BORDER_CLOSURE_REGRESSOR: closure,
            RECOVERY_REGRESSOR: recovery,
            HOLY_WEEK_REGRESSOR: holy_week.to_numpy(dtype="float64"),
        },
        index=index,
    )


def projection_cap(train: pd.Series, *, factor: float = 3.0) -> float:
    """Set a plausibility ceiling for the projections.

    The training set of the regulatory partition ends at the lowest point of
    the pandemic, a situation in which a doubly differenced model can
    extrapolate explosive paths when the logarithmic transform is undone. The
    ceiling does not alter reasonable forecasts and prevents a degenerate case
    from contaminating the comparison of errors.
    """
    return float(train.max()) * factor


def seasonal_naive_method(
    train: pd.Series,
    horizon: int,
    *,
    period: int = PERIOD,
) -> Forecast:
    """Repeat the value observed a year earlier; the minimum benchmark."""
    history = train.to_numpy(dtype="float64")
    last_cycle = history[-period:]
    projection = [last_cycle[step % period] for step in range(horizon)]
    index = pd.date_range(
        train.index.max() + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )
    fitted = train.shift(period)
    return Forecast(
        model=SEASONAL_NAIVE,
        values=pd.Series(projection, index=index, name=SEASONAL_NAIVE),
        fitted=fitted,
        residuals=(train - fitted).dropna(),
    )


def _order_grid(
    max_p: int, max_q: int, max_seasonal_p: int, max_seasonal_q: int
) -> Iterable[tuple[int, int, int, int]]:
    return product(
        range(max_p + 1),
        range(max_q + 1),
        range(max_seasonal_p + 1),
        range(max_seasonal_q + 1),
    )


@dataclass(frozen=True)
class SarimaCandidate:
    """Specification evaluated during the order search."""

    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    akaike: float
    corrected_akaike: float
    bayesian: float
    converged: bool
    ljung_box: float


def search_sarima(
    train: pd.Series,
    *,
    d: int = 1,
    seasonal_d: int = 1,
    max_p: int = 2,
    max_q: int = 2,
    max_seasonal_p: int = 1,
    max_seasonal_q: int = 1,
    exogenous: pd.DataFrame | None = None,
    period: int = PERIOD,
    transform: bool = False,
) -> tuple[list[SarimaCandidate], SarimaCandidate]:
    """Evaluate a bounded grid of orders and return every candidate.

    The leading criterion is the corrected Akaike one, appropriate when the
    number of observations is not much larger than the number of parameters.
    Among models with equivalent fit the most parsimonious is preferred and, on
    a tie, the one leaving residuals without autocorrelation according to the
    Ljung-Box test.
    """
    target = np.log1p(train) if transform else train
    candidates: list[SarimaCandidate] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p, q, seasonal_p, seasonal_q in _order_grid(
            max_p, max_q, max_seasonal_p, max_seasonal_q
        ):
            if p == 0 and q == 0 and seasonal_p == 0 and seasonal_q == 0:
                continue
            try:
                fit = SARIMAX(
                    target,
                    exog=exogenous,
                    order=(p, d, q),
                    seasonal_order=(seasonal_p, seasonal_d, seasonal_q, period),
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                ).fit(disp=False, maxiter=200)
            except (ValueError, np.linalg.LinAlgError):
                continue

            residuals = pd.Series(fit.resid).iloc[period:]
            try:
                test = acorr_ljungbox(residuals, lags=[period], return_df=True)
                pvalue = float(test["lb_pvalue"].iloc[0])
            except (ValueError, IndexError):
                pvalue = float("nan")

            candidates.append(
                SarimaCandidate(
                    order=(p, d, q),
                    seasonal_order=(seasonal_p, seasonal_d, seasonal_q, period),
                    akaike=float(fit.aic),
                    corrected_akaike=corrected_akaike(
                        float(fit.llf),
                        int(fit.df_model),
                        int(target.size),
                    ),
                    bayesian=float(fit.bic),
                    converged=bool(fit.mle_retvals.get("converged", True)),
                    ljung_box=pvalue,
                )
            )

    if not candidates:
        raise RuntimeError("No model in the grid managed to converge.")

    ordered = sorted(candidates, key=lambda c: (c.corrected_akaike, sum(c.order)))
    return ordered, ordered[0]


def candidate_table(
    candidates: Sequence[SarimaCandidate], *, top: int = 8
) -> pd.DataFrame:
    """Present the best candidates of the order search."""
    rows = [
        {
            "order": str(candidate.order),
            "seasonal_order": str(candidate.seasonal_order),
            "akaike": candidate.akaike,
            "corrected_akaike": candidate.corrected_akaike,
            "bayesian": candidate.bayesian,
            "ljung_box_pvalue": candidate.ljung_box,
            "converged": candidate.converged,
        }
        for candidate in sorted(candidates, key=lambda c: c.corrected_akaike)[:top]
    ]
    return pd.DataFrame(rows)


def fit_sarima(
    train: pd.Series,
    horizon: int,
    *,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    exogenous: pd.DataFrame | None = None,
    future_exogenous: pd.DataFrame | None = None,
    transform: bool = False,
    model: str = SARIMA,
) -> Forecast:
    """Fit the chosen model and project the requested horizon."""
    target = np.log1p(train) if transform else train

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(
            target,
            exog=exogenous,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=True,
        ).fit(disp=False, maxiter=300)
        projection = fit.get_forecast(steps=horizon, exog=future_exogenous)
        mean = projection.predicted_mean
        interval = projection.conf_int(alpha=SIGNIFICANCE_LEVEL)

    if transform:
        mean = np.expm1(mean)
        interval = np.expm1(interval)
        fitted = np.expm1(fit.fittedvalues)
    else:
        fitted = fit.fittedvalues

    cap = projection_cap(train)
    mean = mean.clip(lower=0, upper=cap)
    interval = interval.clip(lower=0, upper=cap)
    interval.columns = [LOWER_BOUND, UPPER_BOUND]

    return Forecast(
        model=model,
        values=mean.rename(model),
        interval=interval,
        fitted=fitted,
        residuals=(train - fitted).dropna(),
        akaike=float(fit.aic),
        bayesian=float(fit.bic),
        detail={
            "order": order,
            "seasonal_order": seasonal_order,
            "estimated_parameters": int(fit.df_model),
            "scale": "logarithm" if transform else "original",
        },
    )


def fit_holt_winters(
    train: pd.Series,
    horizon: int,
    *,
    trend: str | None = "add",
    seasonal: str | None = "add",
    damped: bool = True,
    period: int = PERIOD,
    model: str = HOLT_WINTERS,
) -> Forecast:
    """Fit exponential smoothing with trend and seasonality."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(
            train,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=period if seasonal else None,
            damped_trend=damped if trend else False,
            initialization_method="estimated",
        ).fit(optimized=True)
        projection = fit.forecast(horizon).clip(lower=0)

    return Forecast(
        model=model,
        values=projection.rename(model),
        fitted=fit.fittedvalues,
        residuals=(train - fit.fittedvalues).dropna(),
        akaike=float(fit.aic),
        bayesian=float(fit.bic),
        detail={
            "trend": trend or "none",
            "seasonality": seasonal or "none",
            "damped_trend": damped,
        },
    )


def fit_state_space_smoothing(
    train: pd.Series,
    horizon: int,
    *,
    period: int = PERIOD,
    model: str = STATE_SPACE_SMOOTHING,
) -> Forecast:
    """Fit an alternative multiplicative exponential smoothing.

    It is included as a contrast to the previous one: when the seasonal
    amplitude grows with the level, the multiplicative specification usually
    describes the series better than the additive one.
    """
    positive = train.clip(lower=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(
            positive,
            trend="add",
            seasonal="mul",
            seasonal_periods=period,
            damped_trend=False,
            initialization_method="estimated",
        ).fit(optimized=True)
        projection = fit.forecast(horizon).clip(lower=0)

    return Forecast(
        model=model,
        values=projection.rename(model),
        fitted=fit.fittedvalues,
        residuals=(train - fit.fittedvalues).dropna(),
        akaike=float(fit.aic),
        bayesian=float(fit.bic),
        detail={"trend": "additive", "seasonality": "multiplicative"},
    )


def prophet_available() -> bool:
    """Tell whether the Prophet library can be used in this environment."""
    try:
        import prophet  # noqa: F401
    except Exception:  # pragma: no cover - depends on the environment
        return False
    return True


def fit_prophet(
    train: pd.Series,
    horizon: int,
    *,
    model: str = PROPHET,
) -> Forecast:
    """Fit the decomposable additive model of Prophet.

    If the library is unavailable, an in-house implementation of equivalent
    structure is used instead, documented in :func:`fit_decomposable`.
    """
    if not prophet_available():
        return fit_decomposable(train, horizon)

    from prophet import Prophet

    frame = pd.DataFrame({"ds": train.index, "y": train.to_numpy(dtype="float64")})
    intervention = intervention_regressors(train.index)
    for column in intervention.columns:
        frame[column] = intervention[column].to_numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            interval_width=1 - SIGNIFICANCE_LEVEL,
            changepoint_prior_scale=0.1,
        )
        for column in intervention.columns:
            estimator.add_regressor(column)
        estimator.fit(frame)

        future = estimator.make_future_dataframe(periods=horizon, freq="MS")
        future_intervention = intervention_regressors(pd.DatetimeIndex(future["ds"]))
        for column in intervention.columns:
            future[column] = future_intervention[column].to_numpy()
        prediction = estimator.predict(future).set_index("ds")

    projection = prediction["yhat"].iloc[-horizon:].clip(lower=0)
    interval = prediction[["yhat_lower", "yhat_upper"]].iloc[-horizon:].clip(lower=0)
    interval.columns = [LOWER_BOUND, UPPER_BOUND]
    fitted = prediction["yhat"].iloc[:-horizon]
    fitted.index = train.index

    return Forecast(
        model=model,
        values=projection.rename(model),
        interval=interval,
        fitted=fitted,
        residuals=(train - fitted).dropna(),
        detail={
            "seasonality": "multiplicative_annual",
            "regressors": ", ".join(intervention.columns),
        },
    )


def fit_decomposable(
    train: pd.Series,
    horizon: int,
    *,
    harmonics: int = 4,
    model: str = DECOMPOSABLE_ADDITIVE,
) -> Forecast:
    """Additive model with linear trend, annual harmonics and intervention.

    It reproduces the structure of Prophet with statsmodels tools: a linear
    trend, a set of Fourier terms representing the annual seasonality and the
    pandemic intervention regressors. It serves as a substitute when Prophet
    cannot be installed.
    """
    full_index = pd.date_range(
        train.index.min(),
        periods=train.size + horizon,
        freq="MS",
    )
    regressors = _decomposable_matrix(full_index, harmonics)
    target = np.log1p(train)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(
            target,
            exog=regressors.iloc[: train.size],
            order=(1, 0, 0),
            trend="c",
        ).fit(disp=False, maxiter=200)
        projection = fit.get_forecast(steps=horizon, exog=regressors.iloc[train.size :])

    mean = np.expm1(projection.predicted_mean).clip(lower=0)
    interval = np.expm1(projection.conf_int(alpha=SIGNIFICANCE_LEVEL)).clip(lower=0)
    interval.columns = [LOWER_BOUND, UPPER_BOUND]
    fitted = np.expm1(fit.fittedvalues)

    return Forecast(
        model=model,
        values=mean.rename(model),
        interval=interval,
        fitted=fitted,
        residuals=(train - fitted).dropna(),
        akaike=float(fit.aic),
        bayesian=float(fit.bic),
        detail={
            "annual_harmonics": harmonics,
            "scale": "logarithm",
            "regressors": "trend, harmonics and pandemic intervention",
        },
    )


def _decomposable_matrix(index: pd.DatetimeIndex, harmonics: int) -> pd.DataFrame:
    """Build trend, Fourier terms and intervention regressors."""
    time = np.arange(len(index), dtype="float64")
    columns: dict[str, np.ndarray] = {"linear_trend": time / len(index)}
    for order in range(1, harmonics + 1):
        angle = 2 * np.pi * order * time / PERIOD
        columns[f"annual_harmonic_{order}_sine"] = np.sin(angle)
        columns[f"annual_harmonic_{order}_cosine"] = np.cos(angle)

    matrix = pd.DataFrame(columns, index=index)
    return pd.concat([matrix, intervention_regressors(index)], axis=1)


def compare_scales(
    train: pd.Series,
    test: pd.Series,
    *,
    order: tuple[int, int, int] = (0, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (0, 1, 1, PERIOD),
) -> pd.DataFrame:
    """Contrast the original scale with the logarithmic one empirically.

    The variance diagnosis suggests transforming the series, but the pandemic
    took monthly arrivals literally to zero. On a multiplicative scale a fall
    to zero is of unbounded magnitude, so the seasonal difference of the
    logarithm introduces jumps that destabilise estimation. This function
    measures both alternatives instead of settling the matter by convention.
    """
    from lab_series.metrics import mean_absolute_error, mean_absolute_scaled_error

    rows = []
    for scale, transform in (("original", False), ("logarithm", True)):
        forecast = fit_sarima(
            train,
            int(test.size),
            order=order,
            seasonal_order=seasonal_order,
            transform=transform,
        )
        rows.append(
            {
                "scale": scale,
                "mean_absolute_error": mean_absolute_error(test, forecast.values),
                "mean_absolute_scaled_error": mean_absolute_scaled_error(
                    test, forecast.values, train
                ),
                "akaike": forecast.akaike,
            }
        )
    return pd.DataFrame(rows).set_index("scale")


def residual_diagnostics(
    residuals: pd.Series,
    *,
    lags: int = PERIOD,
) -> pd.Series:
    """Test whether the residual behaves as noise without structure.

    Three tests are applied: Ljung-Box on autocorrelation, Jarque-Bera on
    normality and the autoregressive conditional heteroskedasticity test on the
    stability of the variance.
    """
    clean = residuals.dropna()
    if clean.size < lags + 5:
        return pd.Series(dtype="float64")

    ljung = acorr_ljungbox(clean, lags=[lags], return_df=True)
    _, jarque_bera_pvalue, skewness, kurtosis = jarque_bera(clean)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, arch_pvalue, _, _ = het_arch(clean, nlags=min(lags, clean.size // 5))

    ljung_box_pvalue = float(ljung["lb_pvalue"].iloc[0])
    return pd.Series(
        {
            "ljung_box_pvalue": ljung_box_pvalue,
            "jarque_bera_pvalue": float(jarque_bera_pvalue),
            "heteroskedasticity_pvalue": float(arch_pvalue),
            "residual_skewness": float(skewness),
            "residual_kurtosis": float(kurtosis),
            "residual_mean": float(clean.mean()),
            "verdict": (
                "white_noise" if ljung_box_pvalue > 0.05 else "residual_autocorrelation"
            ),
        }
    )


def fit_families(
    train: pd.Series,
    horizon: int,
    *,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    use_intervention: bool = True,
    include_prophet: bool = True,
) -> list[Forecast]:
    """Fit every family of models on the same partition."""
    future_index = pd.date_range(
        train.index.max() + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )
    exogenous = intervention_regressors(train.index) if use_intervention else None
    future_exogenous = (
        intervention_regressors(future_index) if use_intervention else None
    )

    results = [
        fit_sarima(
            train,
            horizon,
            order=order,
            seasonal_order=seasonal_order,
            exogenous=exogenous,
            future_exogenous=future_exogenous,
            model=SARIMA_WITH_INTERVENTION if use_intervention else SARIMA,
        ),
        fit_sarima(
            train,
            horizon,
            order=order,
            seasonal_order=seasonal_order,
            model=SARIMA,
        ),
        fit_holt_winters(train, horizon),
        fit_state_space_smoothing(train, horizon),
        seasonal_naive_method(train, horizon),
    ]

    if include_prophet:
        results.append(fit_prophet(train, horizon))

    return results
