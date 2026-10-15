"""Orchestration of the analysis and persistence of its artefacts.

Gathers in one place the sequence applied identically to the seven series:
stationarity diagnosis, order selection, fitting of the model families,
evaluation on the test set and diagnosis of the residual of the winning model.
The notebook consumes these results to narrate them and the report generator
reads them from disk, so that no figure is transcribed by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from lab_series import comparison as comparison_mod
from lab_series import metrics, models
from lab_series.loading import paths
from lab_series.series import (
    differencing_order,
    recommended_transform,
    stationarity_summary,
)
from lab_series.splitting import (
    CHRONOLOGICAL_SCHEME,
    Partition,
    post_pandemic_holdout,
    split_series,
)

METRICS_FILE = "model_metrics.parquet"
CANDIDATES_FILE = "sarima_candidates.parquet"
FORECASTS_FILE = "forecasts.parquet"
COMPARISON_FILE = "series_comparison.parquet"
DIAGNOSTICS_FILE = "series_diagnostics.parquet"
SUMMARY_FILE = "study_summary.json"


@dataclass
class SeriesResult:
    """Everything the study produces for one series."""

    key: str
    partition: Partition
    diagnostics: pd.Series
    candidates: pd.DataFrame
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    forecasts: list[models.Forecast]
    metrics: pd.DataFrame
    winning_model: str
    winning_residuals: pd.Series
    residual_diagnostics: pd.Series = field(default_factory=pd.Series)

    @property
    def winner(self) -> models.Forecast:
        return next(
            forecast
            for forecast in self.forecasts
            if forecast.model == self.winning_model
        )

    def winning_metrics(self) -> pd.Series:
        return self.metrics.loc[self.winning_model]


def diagnose_series(series: pd.Series) -> pd.Series:
    """Combine stationarity, differencing and variance into a single record."""
    stationarity = stationarity_summary(series)
    differences = differencing_order(series)
    variance = recommended_transform(series)
    record = pd.concat([stationarity, differences, variance])
    record.name = series.name
    return record


def analyse_series(
    key: str,
    series: pd.Series,
    *,
    partition: Partition | None = None,
    max_p: int = 2,
    max_q: int = 2,
    include_prophet: bool = True,
) -> SeriesResult:
    """Run the complete analysis sequence on one series."""
    partition = partition or split_series(series)
    diagnostics = diagnose_series(series)

    d = int(diagnostics["differences"])
    seasonal_d = int(diagnostics["seasonal_differences"])
    candidates, best = models.search_sarima(
        partition.train,
        d=max(d, 1),
        seasonal_d=max(seasonal_d, 1),
        max_p=max_p,
        max_q=max_q,
    )

    forecasts = models.fit_families(
        partition.train,
        partition.horizon,
        order=best.order,
        seasonal_order=best.seasonal_order,
        include_prophet=include_prophet,
    )

    evaluations = [
        metrics.evaluate(
            partition.test,
            forecast.values,
            partition.train,
            model=forecast.model,
            interval=forecast.interval,
            akaike=forecast.akaike,
            bayesian=forecast.bayesian,
        )
        for forecast in forecasts
    ]
    table = metrics.comparison_table(evaluations)
    winner = metrics.select_best(table)
    residuals = next(f for f in forecasts if f.model == winner).residuals

    return SeriesResult(
        key=key,
        partition=partition,
        diagnostics=diagnostics,
        candidates=models.candidate_table(candidates),
        order=best.order,
        seasonal_order=best.seasonal_order,
        forecasts=forecasts,
        metrics=table,
        winning_model=winner,
        winning_residuals=residuals
        if residuals is not None
        else pd.Series(dtype=float),
        residual_diagnostics=(
            models.residual_diagnostics(residuals)
            if residuals is not None
            else pd.Series(dtype=float)
        ),
    )


def run(
    series: dict[str, pd.Series],
    *,
    scheme: str = CHRONOLOGICAL_SCHEME,
    include_prophet: bool = True,
) -> dict[str, SeriesResult]:
    """Analyse every series with the requested partition scheme."""
    results: dict[str, SeriesResult] = {}
    for key, values in series.items():
        partition = (
            split_series(values)
            if scheme == CHRONOLOGICAL_SCHEME
            else post_pandemic_holdout(values)
        )
        results[key] = analyse_series(
            key, values, partition=partition, include_prophet=include_prophet
        )
    return results


def summary_table(results: dict[str, SeriesResult]) -> pd.DataFrame:
    """Consolidate the winning model and its performance for every series."""
    rows = []
    for result in results.values():
        winning = result.winning_metrics()
        rows.append(
            {
                "series": result.key,
                "selected_model": result.winning_model,
                "order": str(result.order),
                "seasonal_order": str(result.seasonal_order),
                "mean_absolute_error": winning["mean_absolute_error"],
                "root_mean_squared_error": winning["root_mean_squared_error"],
                "mean_absolute_percentage_error": winning[
                    "mean_absolute_percentage_error"
                ],
                "mean_absolute_scaled_error": winning["mean_absolute_scaled_error"],
                "beats_naive": bool(
                    winning["mean_absolute_scaled_error"]
                    < result.metrics.loc[
                        models.SEASONAL_NAIVE, "mean_absolute_scaled_error"
                    ]
                ),
                "residuals_without_autocorrelation": bool(
                    result.residual_diagnostics.get("ljung_box_pvalue", 0) > 0.05
                ),
            }
        )
    return pd.DataFrame(rows).set_index("series")


def long_metrics_table(results: dict[str, SeriesResult]) -> pd.DataFrame:
    """Stack the metrics of every model and every series."""
    frames = []
    for result in results.values():
        frame = result.metrics.reset_index()
        frame.insert(0, "series", result.key)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def forecast_table(results: dict[str, SeriesResult]) -> pd.DataFrame:
    """Stack the forecast of the winning model alongside the observed value."""
    frames = []
    for result in results.values():
        winner = result.winner
        frame = pd.DataFrame(
            {
                "series": result.key,
                "model": winner.model,
                "date": winner.values.index,
                "forecast": winner.values.to_numpy(dtype="float64"),
            }
        )
        frame["actual"] = result.partition.test.reindex(winner.values.index).to_numpy(
            dtype="float64"
        )
        if winner.interval is not None:
            frame["lower_bound"] = winner.interval.iloc[:, 0].to_numpy()
            frame["upper_bound"] = winner.interval.iloc[:, 1].to_numpy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def diagnostics_table(results: dict[str, SeriesResult]) -> pd.DataFrame:
    """Consolidate the statistical diagnosis of every series."""
    rows = []
    for result in results.values():
        row = result.diagnostics.to_dict()
        row["series"] = result.key
        rows.append(row)
    frame = pd.DataFrame(rows)
    columns = ["series"] + [name for name in frame.columns if name != "series"]
    return frame[columns]


def long_candidates_table(results: dict[str, SeriesResult]) -> pd.DataFrame:
    """Stack the candidates evaluated during the order search."""
    frames = []
    for result in results.values():
        frame = result.candidates.copy()
        frame.insert(0, "series", result.key)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _portable_paths(figures: dict[str, object], root: Path) -> dict[str, str]:
    """Express figure locations relative to the project root.

    The artefacts are consumed on machines other than the one that produced
    them — continuous integration among them — so an absolute path would render
    them useless outside the originating computer.
    """
    portable: dict[str, str] = {}
    for key, value in figures.items():
        path = Path(str(value))
        if path.is_absolute():
            try:
                path = path.relative_to(root)
            except ValueError:
                pass
        portable[key] = path.as_posix()
    return portable


def save_artifacts(
    results: dict[str, SeriesResult],
    comparison: pd.DataFrame,
    *,
    destination: Path | None = None,
    context: dict[str, object] | None = None,
) -> Path:
    """Persist the results of the study for the report and the tests."""
    destination = destination or paths().artifacts
    destination.mkdir(parents=True, exist_ok=True)

    long_metrics_table(results).to_parquet(destination / METRICS_FILE, index=False)
    long_candidates_table(results).to_parquet(
        destination / CANDIDATES_FILE, index=False
    )
    forecast_table(results).to_parquet(destination / FORECASTS_FILE, index=False)
    diagnostics_table(results).to_parquet(destination / DIAGNOSTICS_FILE, index=False)
    comparison.reset_index().to_parquet(destination / COMPARISON_FILE, index=False)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "series": {
            result.key: {
                "winning_model": result.winning_model,
                "order": list(result.order),
                "seasonal_order": list(result.seasonal_order),
                "test_cutoff": str(result.partition.cutoff.date()),
                "horizon": result.partition.horizon,
                "mean_absolute_scaled_error": float(
                    result.winning_metrics()["mean_absolute_scaled_error"]
                ),
                "mean_absolute_percentage_error": float(
                    result.winning_metrics()["mean_absolute_percentage_error"]
                ),
                "residuals_without_autocorrelation": bool(
                    result.residual_diagnostics.get("ljung_box_pvalue", 0) > 0.05
                ),
            }
            for result in results.values()
        },
        "comparative_answers": comparison_mod.answer_questions(comparison).to_dict(
            orient="records"
        ),
    }
    if context:
        figures = context.get("figures")
        if isinstance(figures, dict):
            context = {
                **context,
                "figures": _portable_paths(figures, paths().root),
            }
        summary["context"] = context

    summary_path = destination / SUMMARY_FILE
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")

    return destination


def load_artifacts(destination: Path | None = None) -> dict[str, object]:
    """Read the artefacts of the study to feed the report."""
    destination = destination or paths().artifacts
    missing = [
        name
        for name in (
            METRICS_FILE,
            FORECASTS_FILE,
            COMPARISON_FILE,
            SUMMARY_FILE,
        )
        if not (destination / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Study artefacts are missing: "
            + ", ".join(missing)
            + ". Run the notebook or 'make analysis' before generating the report."
        )

    with (destination / SUMMARY_FILE).open(encoding="utf-8") as handle:
        summary = json.load(handle)

    return {
        "metrics": pd.read_parquet(destination / METRICS_FILE),
        "candidates": pd.read_parquet(destination / CANDIDATES_FILE),
        "forecasts": pd.read_parquet(destination / FORECASTS_FILE),
        "diagnostics": pd.read_parquet(destination / DIAGNOSTICS_FILE),
        "comparison": pd.read_parquet(destination / COMPARISON_FILE),
        "summary": summary,
    }
