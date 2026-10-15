"""Contract tests for academic evidence and final deliverable formats."""

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pypdf import PdfReader

from build import ACCENT, DATA, GRAY, NAVY, OUT, PALE, ROOT, TEAL, evidence, verify_provenance


def test_snapshot_integrity_and_original_workbook_digest():
    manifest = verify_provenance()
    metadata = json.loads((DATA / "metadata.json").read_text())
    assert manifest["source_xlsx_sha256"] == metadata["source_digest"]
    assert "M reports/artifacts/model_metrics.parquet" in manifest["upstream_worktree_status"]


def test_monthly_calendar_and_populations():
    frame = pd.read_parquet(DATA / "monthly_series.parquet")
    assert frame.series.nunique() == 7
    assert not frame.duplicated(["series", "date"]).any()
    expected = pd.date_range("2009-01-01", "2026-06-01", freq="MS")
    for key, group in frame.groupby("series"):
        assert list(group.date.sort_values()) == list(expected)
        assert np.isfinite(group.visitors).all()
        assert (group.visitors >= 0).all()
        assert group.population.nunique() == 1
        if key != "sea_entry":
            assert group.population.iloc[0] == "reference"


def test_published_kpis_and_recovery():
    *_, total, ranking, forecast, index, stats = evidence()
    assert stats["total_2025"] == pytest.approx(3352288.657508455)
    assert stats["growth_2025_vs_2019_pct"] == pytest.approx(38.021815, rel=1e-5)
    base = total.loc["2019"].sum()
    recovered = total.rolling(12).sum().loc["2021":]
    assert str(recovered[recovered >= base].index[0].date()) == stats["recovery_month"]
    assert index.mean() == pytest.approx(100)
    assert index.idxmax() == 12
    assert ranking.iloc[0].model == "prophet"
    assert len(forecast) == 63
    assert stats["covered_months"] == 29
    assert (forecast.lower_bound <= forecast.upper_bound).all()


@pytest.mark.parametrize(
    ("indicator", "winner"),
    [
        ("seasonal_strength", "sea_entry"),
        ("trend_annual_growth", "central_america"),
        ("residual_dispersion", "sea_entry"),
        ("pandemic_loss", "sea_entry"),
    ],
)
def test_comparative_winners(indicator, winner):
    comparison = pd.read_parquet(DATA / "series_comparison.parquet").set_index("series")
    assert comparison[indicator].idxmax() == winner


def test_png_dimensions():
    with Image.open(OUT / "infografia.png") as img:
        assert img.format == "PNG"
        assert img.width >= 1200 and img.height >= 2400
        assert img.height > img.width
        assert img.info["dpi"][0] == pytest.approx(300, abs=0.1)


@pytest.mark.parametrize(
    ("filename", "width", "height"),
    [
        ("infografia_A3.pdf", 297 / 25.4 * 72, 420 / 25.4 * 72),
        ("infografia_11x17.pdf", 11 * 72, 17 * 72),
    ],
)
def test_print_page_size_and_required_labels(filename, width, height):
    reader = PdfReader(OUT / filename)
    assert len(reader.pages) == 1
    page = reader.pages[0]
    assert float(page.mediabox.width) == pytest.approx(width, abs=0.1)
    assert float(page.mediabox.height) == pytest.approx(height, abs=0.1)
    text = page.extract_text()
    for label in ["Estimación", "95 %", "46,0 %", "INGUAT/IGM", "Fernando Mendoza", "Pablo Escobar"]:
        assert label in text


def test_explanation_is_substantive_and_discloses_limits():
    reader = PdfReader(OUT / "explicacion_diseno.pdf")
    assert len(reader.pages) == 7
    text = "\n".join(page.extract_text() for page in reader.pages)
    for phrase in ["Paleta", "MASE", "arange", "4,5:1", "45 puntos", "55 puntos", "Canvas"]:
        assert phrase.lower() in text.lower()
    assert len(text) > 15000


def luminance(value):
    rgb = [int(value[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
    return sum(a * b for a, b in zip(linear, [0.2126, 0.7152, 0.0722], strict=True))


@pytest.mark.parametrize(
    ("foreground", "background"),
    [
        (NAVY, "#FFFFFF"),
        (TEAL, "#FFFFFF"),
        (ACCENT, "#FFFFFF"),
        (GRAY, "#FFFFFF"),
        (NAVY, PALE),
        (TEAL, PALE),
        (GRAY, PALE),
    ],
)
def test_text_contrast(foreground, background):
    lo, hi = sorted([luminance(foreground), luminance(background)])
    assert (hi + 0.05) / (lo + 0.05) >= 4.5


def test_csv_exports_match_frozen_artifacts():
    for name in ["model_metrics", "forecasts", "series_comparison", "series_diagnostics", "monthly_series"]:
        original = pd.read_parquet(DATA / f"{name}.parquet")
        exported = pd.read_csv(OUT / f"{name}.csv")
        assert original.shape == exported.shape
        assert list(original.columns) == list(exported.columns)
        for column in original.select_dtypes(include="number").columns:
            np.testing.assert_allclose(original[column], exported[column], equal_nan=True)
    stats = json.loads((OUT / "cifras_verificadas.json").read_text())
    assert stats["covered_months"] == 29
    assert (ROOT / "uv.lock").exists()
