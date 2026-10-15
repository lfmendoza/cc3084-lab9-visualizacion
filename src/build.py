"""Build every deliverable from a frozen, hash-verified Lab 1 snapshot."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cc3084-lab9-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "entregables"
DATA = ROOT / "data/lab1"
NAVY = "#17324D"
TEAL = "#006D77"
ACCENT = "#B34422"
GRAY = "#4B5563"
PALE = "#EDF4F4"
MODELS = {
    "prophet": "Prophet",
    "sarima_with_intervention": "SARIMA + intervención",
    "state_space_smoothing": "Suavizamiento (estado)",
    "holt_winters": "Holt–Winters",
    "seasonal_naive": "Ingenuo estacional",
    "sarima": "SARIMA",
}
LABELS = {
    "total": "Total",
    "air_entry": "Aérea",
    "land_entry": "Terrestre",
    "sea_entry": "Marítima*",
    "central_america": "Centroamérica",
    "north_america": "Norteamérica",
    "europe": "Europa",
}
MONTHS = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def number(value, decimals=0):
    return f"{value:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", " ")


def verify_provenance():
    manifest = json.loads((ROOT / "docs/provenance.json").read_text())
    for relative, digest in manifest["files"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Cambió el insumo congelado: {relative}")
    return manifest


def evidence():
    verify_provenance()
    monthly = pd.read_parquet(DATA / "monthly_series.parquet")
    metrics = pd.read_parquet(DATA / "model_metrics.parquet")
    forecasts = pd.read_parquet(DATA / "forecasts.parquet")
    comparison = pd.read_parquet(DATA / "series_comparison.parquet")
    total = monthly[monthly.series == "total"].set_index("date").visitors.sort_index()
    annual = total.groupby(total.index.year).sum()
    recent = total.loc["2023":"2025"]
    profile = recent.groupby(recent.index.month).mean()
    index = profile / profile.mean() * 100
    forecast = forecasts[forecasts.series == "total"].sort_values("date")
    ranking = metrics[metrics.series == "total"].sort_values("rank")
    winner = ranking.iloc[0]
    error = forecast.actual - forecast.forecast
    coverage = (
        (forecast.actual >= forecast.lower_bound) & (forecast.actual <= forecast.upper_bound)
    ).mean() * 100
    assert np.isclose(np.abs(error).mean(), winner.mean_absolute_error)
    assert np.isclose(np.sqrt(np.square(error).mean()), winner.root_mean_squared_error)
    assert np.isclose(coverage, winner.interval_coverage)
    assert np.allclose(total.reindex(forecast.date), forecast.actual)
    stats = {
        "total_2025": float(annual.loc[2025]),
        "growth_2025_vs_2019_pct": float((annual.loc[2025] / annual.loc[2019] - 1) * 100),
        "drop_2020_vs_2019_pct": float((annual.loc[2020] / annual.loc[2019] - 1) * 100),
        "recovery_month": str(comparison.set_index("series").loc["total", "recovery_month"].date()),
        "mae": float(winner.mean_absolute_error),
        "rmse": float(winner.root_mean_squared_error),
        "coverage_pct": float(coverage),
        "covered_months": round(coverage / 100 * len(forecast)),
        "test_months": len(forecast),
        "seasonal_index": {int(k): float(v) for k, v in index.items()},
    }
    return monthly, metrics, comparison, total, ranking, forecast, index, stats


def infographic(bundle):
    _monthly, _metrics, _comparison, total, ranking, forecast, index, s = bundle
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "text.color": NAVY,
            "axes.labelcolor": GRAY,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.edgecolor": "#D8E2E6",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(297 / 25.4, 420 / 25.4), facecolor="white")

    def text(x, y, value, size=10, color=NAVY, weight="normal", **kwargs):
        return fig.text(
            x, y, value, fontsize=size, color=color, weight=weight, va="top", linespacing=1.4, **kwargs
        )

    def section(y, n, title):
        text(0.055, y, n, 11, TEAL, "bold")
        text(0.092, y, title, 14, NAVY, "bold")

    def axes(rect):
        ax = fig.add_axes(rect)
        ax.grid(axis="y", color="#E4E9EC", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=9)
        return ax

    text(0.055, 0.972, "GUATEMALA  /  INTELIGENCIA TURÍSTICA  /  2009–JUN 2026", 10, TEAL, "bold")
    text(0.055, 0.949, "El turismo supera 2019;\nla incertidumbre exige flexibilidad", 25, NAVY, "bold")
    text(
        0.055,
        0.895,
        "17 años y medio de registros: recuperar volumen no elimina la estacionalidad ni el riesgo.",
        11,
    )
    cards = [
        (f"{number(s['total_2025'] / 1e6, 2)} M", "ingresos en 2025", "último año completo"),
        (f"+{number(s['growth_2025_vs_2019_pct'], 1)} %", "frente a 2019", "misma población de referencia"),
        (f"{number(s['drop_2020_vs_2019_pct'], 1)} %", "caída anual en 2020", "respecto a 2019"),
        ("Jun 2023", "recuperación del nivel 2019", "acumulado móvil de 12 meses"),
    ]
    for i, (value, label, detail) in enumerate(cards):
        x = 0.055 + i * 0.227
        fig.add_artist(
            Rectangle((x, 0.815), 0.213, 0.065, transform=fig.transFigure, facecolor=PALE, edgecolor="none")
        )
        text(x + 0.012, 0.873, value, 22, TEAL, "bold")
        text(x + 0.012, 0.844, label, 9.4, NAVY, "bold")
        text(x + 0.012, 0.829, detail, 8.3, GRAY)

    section(0.797, "01", "Una recuperación real, con una definición comparable")
    ax = axes([0.09, 0.666, 0.84, 0.105])
    ax.plot(total.index, total.values / 1000, color=TEAL, lw=1.7)
    ax.set_ylim(0, 510)
    ax.set_ylabel("Miles de ingresos / mes", fontsize=9)
    ax.set_xlim(pd.Timestamp("2009-01-01"), pd.Timestamp("2027-01-01"))
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-09-01"), color=ACCENT, alpha=0.12)
    for date, label, xytext in [
        ("2020-04-01", "2020 · choque\npandémico", ("2016-05-01", 355)),
        ("2023-06-01", "Jun 2023 · recupera\nel acumulado de 2019", ("2021-05-01", 460)),
    ]:
        ax.annotate(
            label,
            xy=(pd.Timestamp(date), total.loc[date] / 1000),
            xytext=(pd.Timestamp(xytext[0]), xytext[1]),
            fontsize=9,
            arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.8},
            color=NAVY,
        )
    text(
        0.09,
        0.640,
        "2023: cambia el registro. Excluimos residentes de Guatemala en todo el período para comparar.\n2026 incluye solo enero–junio; los ingresos no equivalen a personas únicas ni a gasto turístico.",
        9.3,
        GRAY,
    )

    section(0.600, "02", "Planificar por temporada, vía y mercado")
    text(0.065, 0.575, "Diciembre concentra la mayor demanda reciente", 10.5, NAVY, "bold")
    ax = axes([0.085, 0.463, 0.405, 0.092])
    ax.bar(range(1, 13), index.values, color=[ACCENT if x == 12 else TEAL for x in range(1, 13)], width=0.68)
    ax.axhline(100, color=GRAY, lw=0.8, ls="--")
    ax.set_ylim(0, 155)
    ax.set_xticks(range(1, 13), [x[0] for x in MONTHS])
    ax.set_yticks([0, 50, 100, 150])
    ax.set_ylabel("Índice mensual", fontsize=9)
    ax.text(12, index.loc[12] + 4, number(index.loc[12], 0), ha="center", color=ACCENT, weight="bold")
    text(
        0.065,
        0.443,
        "Promedio 2023–2025; media de los 12 meses = 100.\nMar–abr: Semana Santa · jul: vacaciones · dic: fin de año.\nSon coincidencias de calendario, no efectos causales.",
        9,
        GRAY,
    )
    text(0.54, 0.575, "Cuatro respuestas del Laboratorio 1", 10.5, NAVY, "bold")
    answers = [
        ("Mayor estacionalidad", "Marítima* · 0,641", "Fuerza estacional STL (0–1)"),
        ("Mayor crecimiento", "Centroamérica · 7,48 %", "Tendencia anual estimada¹"),
        ("Mayor volatilidad", "Marítima* · 56,30 %", "Desvío residual / media"),
        ("Mayor pérdida pandémica", "Marítima* · 95,55 %", "Mar 2020–dic 2021 vs. patrón previo"),
    ]
    for i, (heading, value, detail) in enumerate(answers):
        x = 0.54 + (i % 2) * 0.225
        y = 0.551 - (i // 2) * 0.064
        text(x, y, heading, 9, GRAY)
        text(x, y - 0.016, value, 10.4, TEAL, "bold")
        text(x, y - 0.033, detail, 8.2, GRAY)
    text(
        0.54,
        0.420,
        "*Marítima incluye todas las categorías de viajeros.\nComparación relativa; no sumar sus niveles al total.\n¹Indicador heredado; no es un pronóstico de crecimiento.",
        8.7,
        GRAY,
    )

    section(0.380, "03", "Prophet gana en el total; su banda subestima el riesgo")
    text(0.065, 0.356, "Error en prueba · menos es mejor", 10.5, NAVY, "bold")
    text(0.065, 0.337, "Modelo", 9, GRAY, "bold")
    text(0.332, 0.337, "MAE", 9, GRAY, "bold")
    text(0.425, 0.337, "RMSE", 9, GRAY, "bold")
    for i, row in enumerate(ranking.itertuples()):
        y = 0.320 - i * 0.017
        color = TEAL if i == 0 else NAVY
        text(0.065, y, MODELS[row.model], 9, color, "bold" if i == 0 else "normal")
        text(0.332, y, number(row.mean_absolute_error), 9, color)
        text(0.425, y, number(row.root_mean_squared_error), 9, color)
    text(
        0.065,
        0.210,
        "Unidades: ingresos/mes. Selección original por MASE\ny parsimonia (2 %). En el total coincide con menor MAE.\nEntrena: ene 2009–mar 2021 · prueba: abr 2021–jun 2026.",
        8.7,
        GRAY,
    )
    ax = axes([0.57, 0.239, 0.36, 0.090])
    ax.fill_between(
        forecast.date,
        forecast.lower_bound / 1000,
        forecast.upper_bound / 1000,
        color=ACCENT,
        alpha=0.20,
        linewidth=0,
    )
    ax.plot(forecast.date, forecast.actual / 1000, color=TEAL, lw=1.3)
    ax.plot(forecast.date, forecast.forecast / 1000, color=ACCENT, lw=1.3, ls="--")
    ax.set_ylim(0, max(forecast.upper_bound.max(), forecast.actual.max()) / 1000 * 1.08)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Miles / mes", fontsize=8.5)
    text(0.57, 0.356, "Estimación · intervalo nominal del 95 %", 10, ACCENT, "bold")
    text(0.57, 0.341, "━ Observado", 8.5, TEAL)
    text(0.74, 0.341, "┄ Estimación Prophet", 8.5, ACCENT)
    text(
        0.55,
        0.218,
        f"Cobertura observada: {s['covered_months']}/{s['test_months']} meses ({number(s['coverage_pct'], 1)} %).\nEvaluación histórica; no es una proyección posterior a junio de 2026.",
        9,
        ACCENT,
    )

    section(0.172, "04", "Decisiones para INGUAT")
    recs = [
        (
            "01  Preparar los picos",
            "Escalonar personal y campañas antes de diciembre;\nrevisar ocupación y tiempos de atención cada mes.",
        ),
        (
            "02  Priorizar Centroamérica",
            "Pilotear campañas regionales por su mayor tendencia;\nmedir conversión antes de ampliar el presupuesto.",
        ),
        (
            "03  Flexibilizar cruceros",
            "Ajustar turnos a arribos confirmados: mayor variabilidad\ny pérdida histórica; evitar capacidad fija excesiva.",
        ),
        (
            "04  Recalibrar antes de comprometer",
            "Actualizar el modelo y validar sus intervalos en ventanas\nmóviles antes de decidir inversiones irreversibles.",
        ),
    ]
    for i, (title, body) in enumerate(recs):
        x, y = 0.065 + (i % 2) * 0.475, 0.146 - (i // 2) * 0.047
        text(x, y, title, 10.3, TEAL, "bold")
        text(x, y - 0.017, body, 9, NAVY)
    text(
        0.055,
        0.039,
        "Fuente: Base de Migración 2009–jun 2026 · INGUAT/IGM, material académico; resultados del Laboratorio 1.\nPoblación: turistas y excursionistas no residentes (excepto marítima). Elaboración: Fernando Mendoza y Pablo Escobar.\nUVG · CC3084 · Laboratorio 9 · Octubre 2026. Cifras analíticas; no constituyen una publicación estadística oficial.",
        8,
        GRAY,
    )
    fig.savefig(OUT / "infografia.png", dpi=300, facecolor="white")
    fig.savefig(
        OUT / "infografia_A3.pdf",
        metadata={
            "Title": "El turismo supera 2019; la incertidumbre exige flexibilidad",
            "Author": "Fernando Mendoza y Pablo Escobar",
        },
    )
    fig.savefig(OUT / "vista_previa.png", dpi=110, facecolor="white")
    plt.close(fig)
    original = PdfReader(OUT / "infografia_A3.pdf").pages[0]
    writer = PdfWriter()
    page = writer.add_blank_page(width=11 * 72, height=17 * 72)
    scale = min(11 * 72 / float(original.mediabox.width), 17 * 72 / float(original.mediabox.height))
    dx = (11 * 72 - float(original.mediabox.width) * scale) / 2
    dy = (17 * 72 - float(original.mediabox.height) * scale) / 2
    page.merge_transformed_page(original, Transformation().scale(scale).translate(dx, dy))
    writer.write(OUT / "infografia_11x17.pdf")


def explanation(bundle):
    _, _metrics, _comparison, _total, ranking, _forecast, index, s = bundle
    font_dir = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, file in [("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir / file)))
    pdfmetrics.registerFontFamily(
        "DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu", boldItalic="DejaVu-Bold"
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "BodyLab",
            fontName="DejaVu",
            fontSize=10,
            leading=15,
            textColor=colors.HexColor(NAVY),
            spaceAfter=9,
        )
    )
    styles.add(
        ParagraphStyle(
            "TitleLab",
            fontName="DejaVu-Bold",
            fontSize=24,
            leading=30,
            textColor=colors.HexColor(NAVY),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            "HeadLab",
            fontName="DejaVu-Bold",
            fontSize=15,
            leading=20,
            textColor=colors.HexColor(TEAL),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            "CellLab",
            fontName="DejaVu",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor(NAVY),
            alignment=TA_LEFT,
        )
    )
    story = []
    source = []

    def p(value):
        story.append(Paragraph(value, styles["BodyLab"]))
        source.append(value)

    def h(value):
        story.append(Paragraph(value, styles["HeadLab"]))
        source.append("\n" + value + "\n")

    def page():
        story.append(PageBreak())

    def table(headers, rows, widths):
        content = [[Paragraph(str(x), styles["CellLab"]) for x in row] for row in [headers] + rows]
        obj = Table(content, colWidths=widths, repeatRows=1, hAlign="LEFT")
        obj.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PALE)),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(TEAL)),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#D8E2E6")),
                ]
            )
        )
        story.append(obj)
        story.append(Spacer(1, 12))
        source.extend([" | ".join(map(str, row)) for row in [headers] + rows])

    story.append(Paragraph("Comunicar la recuperación<br/>sin ocultar la incertidumbre", styles["TitleLab"]))
    p("<b>Memoria de decisiones de diseño · Laboratorio 9 · Visualización estática</b>")
    p(
        "Universidad del Valle de Guatemala · Facultad de Ingeniería<br/>CC3084 – Data Science · Semestre II, 2026<br/><b>Equipo:</b> Fernando Mendoza y Pablo Escobar<br/><b>Avance:</b> 15 de octubre de 2026 · <b>Infografía:</b> 18 de octubre de 2026"
    )
    h("1. Propósito, audiencia y mensaje")
    p(
        "La pieza responde a qué enseñan 17 años de datos migratorios sobre el turismo en Guatemala y cómo anticipar decisiones. La audiencia es la dirección de INGUAT: necesita distinguir capacidad operativa, prioridades comerciales y riesgo sin interpretar un informe estadístico extenso. La cobertura exacta es enero de 2009 a junio de 2026: 210 meses, equivalentes a 17 años y medio."
    )
    p(
        f"El mensaje principal combina recuperación y prudencia: en 2025 hubo {number(s['total_2025'])} ingresos de la población analizada, {number(s['growth_2025_vs_2019_pct'], 1)} % más que en 2019. Sin embargo, el intervalo nominal del 95 % del modelo ganador cubrió solo {s['covered_months']} de {s['test_months']} meses de prueba. Por eso el titular une “El turismo supera 2019” con “la incertidumbre exige flexibilidad”."
    )
    p(
        "La unidad es el ingreso registrado, no el número de personas únicas, divisas ni gasto. No se infieren ingresos económicos a partir del flujo migratorio. Las recomendaciones son propuestas de gestión sustentadas en asociaciones observadas; no son efectos causales probados."
    )
    h("2. Alcance y productos")
    p(
        "Se entrega una infografía estática, una memoria PDF, sus fuentes editables en Python, tablas CSV y evidencia de procedencia. La composición digital mide al menos 1200×2400 píxeles. El enunciado llama “A3” a 11×17 pulgadas: esas medidas no coinciden. Se incluyen un PDF A3 real (297×420 mm) y otro 11×17 pulgadas, ambos verticales y de una sola página, sin deformar el contenido."
    )
    p(
        "El repositorio conserva la historia de edición de código y contenido. La ejecución asistida por IA se declara: el equipo debe revisar y asumir las decisiones académicas. La autoría del equipo no sustituye los aportes individuales verificables exigidos por el curso."
    )

    page()
    h("3. Fidelidad a los datos y al Laboratorio 1")
    p(
        "La fuente es Base_Migracion_2009-2026jun.xlsx, atribuida a INGUAT/IGM en el enunciado. Se verificó su SHA-256 contra los metadatos del Laboratorio 1: cdaba77918ae889c847ba8459503f9cce2441efac41b3d3aa0f05ac9ab283846. Los resultados se leen de un snapshot inmutable; no se reentrenan modelos para escoger cifras más favorables."
    )
    p(
        "El repositorio de origen es github.com/pablouwunya2021/lab1_DS, con HEAD 307da539939bf5b404109db90411c58f234ce842. Había modificaciones locales en métricas y pronósticos. El manifiesto docs/provenance.json registra ese estado y las huellas de los archivos efectivamente copiados; por tanto, no se afirma que todo el snapshot coincida con ese commit."
    )
    p(
        "<b>Comparabilidad histórica.</b> Se conserva la definición del Laboratorio 1: turistas y excursionistas no residentes de Guatemala durante todo el período. El cambio de registro de 2023 se explica junto a la línea temporal. Así se evita presentar una discontinuidad de medición como una caída de demanda. Los resultados ajustados no equivalen a los totales publicados originalmente."
    )
    p(
        "<b>Excepción marítima.</b> La serie incluye todas las categorías de viajeros para conservar cruceristas. Se marca con asterisco en la infografía. Solo se comparan indicadores relativos; sus niveles no se suman al total de población de referencia. El cambio de categorías desde 2017 limita la interpretación de la serie como un fenómeno perfectamente homogéneo."
    )
    table(
        ["Cifra", "Definición reproducible", "Archivo de origen"],
        [
            ["Ingresos 2025", "Suma de los 12 meses de total en 2025.", "monthly_series.parquet"],
            [
                "Crecimiento / caída",
                "100 × (suma anual / suma 2019 − 1). No comparar 2026 parcial con años completos.",
                "monthly_series.parquet",
            ],
            [
                "Recuperación",
                "Primer mes desde 2021 cuyo acumulado móvil de 12 meses alcanza la suma de 2019.",
                "series_comparison.parquet",
            ],
            [
                "Cuadrante",
                "Máximo de cada indicador entre las siete series; valores originales sin recalcular.",
                "series_comparison.parquet",
            ],
            [
                "Modelos e intervalos",
                "Métricas y límites originales de la prueba; MAE, RMSE y cobertura recalculados para verificar.",
                "model_metrics.parquet / forecasts.parquet",
            ],
        ],
        [90, 250, 145],
    )
    p(
        "Los CSV completos se incluyen para revisión. No se transcriben números desde una imagen: el programa calcula los KPIs y lee las comparativas desde los artefactos. El redondeo ocurre únicamente al presentar las cifras."
    )

    page()
    h("4. Orden visual y elección de gráficos")
    table(
        ["Bloque", "Decisión de diseño", "Justificación"],
        [
            [
                "Titular y KPIs",
                "Titular periodístico y cuatro tarjetas: volumen 2025, variación vs. 2019, caída 2020 y mes de recuperación.",
                "Primero establece magnitud, contraste y tiempo. La mayor tipografía se reserva para el mensaje y las cifras.",
            ],
            [
                "Panorama histórico",
                "Línea mensual continua; eje vertical desde cero; anotaciones de pandemia, recuperación y cambio metodológico.",
                "El tiempo es ordenado y continuo: la posición y la pendiente permiten reconocer evolución y rupturas sin una leyenda distante.",
            ],
            [
                "Estacionalidad",
                "Barras por mes del promedio 2023–2025, normalizado a media 100. Diciembre usa el acento.",
                "Compara categorías cíclicas por longitud sobre una base cero. Tres años completos evitan el sesgo de incluir solo medio 2026.",
            ],
            [
                "Vías y mercados",
                "Cuadrante textual con cuatro ganadores y su indicador.",
                "Resume preguntas concretas sin mezclar unidades en un eje ni añadir un radar cuya geometría sugiera equivalencias falsas.",
            ],
            [
                "Modelos y estimación",
                "Tabla de MAE/RMSE para seis modelos del total; línea observada, línea discontinua estimada y banda del 95 %.",
                "La tabla facilita comparación exacta en una unidad común. La banda hace visible incertidumbre y la línea permite contrastar con la prueba.",
            ],
            [
                "Acción y fuentes",
                "Cuatro recomendaciones numeradas; ficha técnica y créditos al final.",
                "Cada propuesta sigue una evidencia ya vista. El pie deja visibles población, período, fuente y autores.",
            ],
        ],
        [85, 205, 195],
    )
    p(
        "La lectura avanza de arriba abajo y de izquierda a derecha dentro de cada franja: qué pasó, cuándo pasó, dónde y cuándo planificar, cuánto confiar y qué hacer. La proximidad agrupa textos con sus gráficos; la alineación y la repetición de encabezados crean continuidad. El espacio blanco separa ideas sin bordes decorativos."
    )
    p(
        "Se evitan gráficos 3D, pictogramas de área, ejes dobles y fondos saturados. No se usa un mapa porque las preguntas comparan series y desempeño, no distribución espacial precisa. Todas las barras y líneas de volumen parten de cero; las bandas no se recortan para hacer parecer más preciso el pronóstico."
    )
    p(
        "La estacionalidad reciente es una síntesis adicional, distinta de la fuerza estacional STL calculada sobre la historia completa. Las referencias a Semana Santa, vacaciones y fin de año son contexto de calendario; no se atribuye a bono 14 o aguinaldo un efecto sobre visitantes internacionales que el análisis no estimó."
    )

    page()
    h("5. Paleta, jerarquía, legibilidad y teoría del color")
    table(
        ["Color", "Función", "Razón"],
        [
            [
                NAVY + " · azul profundo",
                "Titular y texto principal",
                "Ancla sobria y de alta legibilidad; mantiene continuidad entre mensaje y explicación.",
            ],
            [
                TEAL + " · verde azulado",
                "Datos observados y encabezados",
                "Familia fría próxima al azul: armonía por cercanía cromática; agrupa la evidencia histórica.",
            ],
            [
                ACCENT + " · terracota",
                "Estimación, pico y advertencia",
                "Contraste cálido-frío: atrae la atención a la proyección y al riesgo sin usar rojo/verde como único código.",
            ],
            [
                "Blanco y grises neutros",
                "Fondo, notas y guías",
                "Reducen competencia visual y conservan contraste; no agregan categorías cromáticas.",
            ],
        ],
        [145, 140, 200],
    )
    p(
        "Se utilizan dos colores temáticos principales y un acento. La asociación del azul con sobriedad es una decisión editorial, no una propiedad psicológica universal. El acento se aplica con moderación y siempre acompañado por palabras o estilo de línea. El área de incertidumbre usa transparencia del mismo acento, sin una categoría cromática nueva."
    )
    p(
        "La tipografía DejaVu Sans permite incorporar caracteres españoles y se incrusta en los PDF. Los títulos, KPIs, subtítulos y notas tienen una jerarquía consistente. La pieza digital se exporta a 300 ppp, con texto vectorial en PDF; las anotaciones están junto a sus referentes y las series se identifican sobre el gráfico."
    )
    p(
        "Se adopta como referencia de legibilidad el umbral de contraste de 4,5:1 para texto normal de WCAG. El código comprueba los colores de texto contra blanco y las tarjetas claras. Esto verifica contraste, no una certificación completa de accesibilidad de una imagen rasterizada. Los CSV y esta memoria ofrecen un equivalente textual a las cifras principales."
    )
    h("6. Definición de los indicadores comparativos")
    p(
        "<b>Estacionalidad:</b> fuerza STL = máx(0, 1 − Var(residuo) / Var(estacionalidad + residuo)); marítima obtiene 0,641. <b>Volatilidad:</b> desvío estándar residual dividido por la media de la serie, en porcentaje; marítima obtiene 56,30 %. Ambos describen patrones históricos, no certeza futura."
    )
    p(
        "<b>Pérdida pandémica:</b> diferencia relativa entre lo observado en marzo de 2020–diciembre de 2021 y el patrón mensual esperado con datos de enero de 2017–febrero de 2020. Marítima alcanza 95,55 %. Es un contrafactual descriptivo del laboratorio, no una estimación causal de todo el efecto de la pandemia."
    )
    p(
        "<b>Crecimiento:</b> Centroamérica encabeza el indicador heredado con 7,48 % anual. El código base ajusta una recta al logaritmo de la tendencia STL y excluye marzo de 2020–junio de 2021. Después reenumera los meses restantes con arange, comprimiendo el tiempo excluido. Se conserva el valor para fidelidad al laboratorio, pero no se interpreta como CAGR calendario exacto ni como proyección futura."
    )

    page()
    h("7. Modelos, validación e incertidumbre")
    p(
        "El Laboratorio 1 entrenó sobre enero de 2009–marzo de 2021 y evaluó 63 meses, abril de 2021–junio de 2026 (división temporal 70/30). Se conserva esa partición, no el corte alternativo de julio de 2024 que también aparece en metadatos. El pronóstico mostrado corresponde al conjunto de prueba y está rotulado “Estimación”. No se inventa una predicción posterior a la cobertura disponible."
    )
    table(
        ["Modelo · serie total", "MAE", "RMSE", "MASE"],
        [
            [
                MODELS[r.model],
                number(r.mean_absolute_error),
                number(r.root_mean_squared_error),
                number(r.mean_absolute_scaled_error, 3),
            ]
            for r in ranking.itertuples()
        ],
        [225, 90, 90, 80],
    )
    p(
        "MAE es el error absoluto medio; RMSE penaliza más los errores grandes. Ambas métricas están en ingresos por mes. MASE permite comparar error con la escala ingenua estacional calculada en entrenamiento. La regla original elige menor MASE con tolerancia del 2 % y desempate por parsimonia. En una misma serie el denominador de MASE es común: la clasificación coincide con MAE. Los errores en volumen no se comparan directamente entre mercados de distinto tamaño."
    )
    p(
        "AIC/BIC se conservan en el CSV completo, pero no ordenan esta tabla ejecutiva: no están disponibles para todos los modelos ni todas las familias comparten la misma verosimilitud. No se sustituyen los ausentes por ceros. La infografía incluye todas las familias evaluadas para el total, mientras las siete series y 42 evaluaciones permanecen en los anexos CSV."
    )
    p(
        f"<b>La banda del 95 % no está calibrada en esta prueba.</b> Los límites provienen de Prophet con interval_width=0.95 en el código base. La cobertura empírica fue {number(s['coverage_pct'], 2)} % ({s['covered_months']}/{s['test_months']}). Se muestra la banda original, sin ensancharla artificialmente ni ocultar las observaciones exteriores. Es un intervalo de predicción nominal, no un intervalo de confianza del promedio ni una garantía de cobertura."
    )
    p(
        "La selección de familia usa el propio conjunto de prueba; por ello su error no es una evaluación externa independiente del proceso de selección. Además, el diagnóstico de diferenciación del código base consulta la serie completa antes del ajuste. No se corrigen silenciosamente esas decisiones para este laboratorio de comunicación: se declaran como limitaciones y se recomienda validación temporal anidada o ventanas móviles antes de uso operativo."
    )
    p(
        "El MASE de Prophet para el total es 1,243: ser ganador entre estos candidatos no implica superar la escala del ingenuo estacional dentro de entrenamiento. Tampoco equivale a perder frente al ingenuo evaluado sobre esta prueba, que tiene MASE 6,789. La diferencia evita una lectura incorrecta del benchmark."
    )

    page()
    h("8. Recomendaciones y condiciones de aplicación")
    table(
        ["Evidencia", "Acción propuesta / responsable sugerido", "Seguimiento propuesto"],
        [
            [
                f"Diciembre: índice {number(index.loc[12], 1)}; media mensual = 100.",
                "Operaciones: preparar turnos y coordinación con prestadores antes de diciembre; revisar cada mes.",
                "Tiempo de atención y ocupación. Medir línea base antes del piloto.",
            ],
            [
                "Centroamérica: mayor tendencia heredada (7,48 %).",
                "Mercadeo: piloto regional con presupuesto acotado; ampliar solo si hay conversión medida.",
                "Conversiones atribuibles al piloto y costo por conversión; no prometer retorno sin datos de gasto.",
            ],
            [
                "Marítima: volatilidad 56,30 % y pérdida histórica 95,55 %.",
                "Coordinación portuaria: turnos flexibles ligados a arribos confirmados, con revisión de calendario.",
                "Utilización de turnos y cancelaciones. La pérdida pasada no es pronóstico de caída futura.",
            ],
            [
                f"Cobertura nominal 95 %; observada {number(s['coverage_pct'], 1)} %.",
                "Analítica: reentrenar con información reciente, separar selección y evaluación, y recalibrar intervalos.",
                "Cobertura y MAE fuera de muestra por ventana y horizonte; validar antes de comprometer capacidad fija.",
            ],
        ],
        [130, 220, 135],
    )
    h("9. Control de calidad y reproducción")
    p(
        "Flujo en WSL: mise administra Python 3.12.14 y uv 0.12.11; uv.lock fija dependencias. Tras clonar, ejecutar mise trust, mise install, mise run setup, mise run build y mise run check. Los archivos de datos incluidos bastan para reproducir la visualización sin acceso a carpetas personales ni servicios externos."
    )
    p(
        "La verificación comprueba huellas de insumos, continuidad de 210 meses por serie, valores finitos y no negativos, reproducción de MAE/RMSE/cobertura, ganadores del cuadrante, tamaño PNG, páginas y dimensiones PDF, presencia de rótulos críticos y contraste. La revisión visual complementa estas pruebas: no se confunde pasar pruebas con garantizar una calificación."
    )
    h("10. Colaboración y entrega")
    p(
        "Integrantes confirmados: Fernando Mendoza y Pablo Escobar. El historial registra cambios reales. No se crean commits atribuidos a Pablo por acciones que no realizó. docs/colaboracion.txt define una contribución sustantiva pendiente para él: revisar las limitaciones estadísticas, corregir texto y anotaciones, y confirmar esos cambios con su propia identidad. Ambos deben contribuir a la explicación y a la infografía."
    )
    p(
        "El curso exige inscripción en grupo de Canvas, entrega del avance el 15 de octubre y acceso del profesor y auxiliares al historial. El repositorio privado requiere invitaciones: sus usuarios aún no se han proporcionado. Estos pasos administrativos siguen pendientes y no pueden acreditarse mediante generación automática de documentos. La fecha del archivo no demuestra entrega puntual en Canvas."
    )

    page()
    h("11. Matriz de verificación de la rúbrica")
    table(
        ["Criterio", "Evidencia entregada", "Validación / límite"],
        [
            [
                "Documento · 45 puntos",
                "Secciones 1–8: coherencia, orden, paleta, elección de gráficos, alcance e interpretación.",
                "Explicación explícita vinculada a cada componente visible. Sin acceso al material de clase, se aplican principios estándar; contrastar terminología con apuntes.",
            ],
            [
                "Infografía · 55 puntos",
                "Titular y 4 KPIs; historia anotada; estacionalidad; 4 comparativas; 6 modelos; intervalo; 4 recomendaciones; fuente y autores.",
                "Cifras derivadas o conservadas del Laboratorio 1, con límites metodológicos visibles.",
            ],
            [
                "HD y una página",
                "PNG a 300 ppp; PDF A3 y PDF 11×17.",
                "Validación automática de dimensiones y número de páginas; revisión visual.",
            ],
            [
                "Fuentes y código",
                "Snapshot con SHA-256, CSV completos, Python, uv.lock, mise.toml.",
                "Ejecución reproducible y manifiesto con estado local del origen.",
            ],
            [
                "Historia colaborativa",
                "Repositorio con código y texto versionados; tarea real preparada para Pablo.",
                "Pendiente aporte efectivo de Pablo; no se declara completado.",
            ],
            [
                "Acceso y puntualidad",
                "Enlaces y guía de entrega en README.",
                "Pendientes acceso docente, grupo y carga en Canvas por los integrantes.",
            ],
        ],
        [110, 200, 175],
    )
    h("12. Referencias")
    p(
        "Universidad del Valle de Guatemala. (2026). <i>Laboratorio 9. Visualización Estática</i>, CC3084. Enunciado de tres páginas, copia local en references/enunciado.pdf."
    )
    p(
        "INGUAT/IGM. <i>Base de Migración 2009–2026jun</i>. Archivo académico proporcionado para el curso. La atribución proviene del enunciado; no se presenta como una descarga pública verificada."
    )
    p(
        "Mendoza, F., y Escobar, P. (2026). <i>Laboratorio 1 — Series de tiempo</i>. Repositorio y snapshot identificados en la sección 3. Se conservan código metodológico, decisiones de población y resultados originales para auditoría."
    )
    p(
        "Hyndman, R. J., y Athanasopoulos, G. (2021). <i>Forecasting: Principles and Practice</i>, 3.ª ed. OTexts. Evaluación de pronósticos e intervalos: https://otexts.com/fpp3/accuracy.html y https://otexts.com/fpp3/prediction-intervals.html. Consulta: 15 de octubre de 2026."
    )
    p(
        "W3C. <i>Understanding Success Criterion 1.4.3: Contrast (Minimum)</i>. https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html. Referencia para el contraste de texto, no certificación integral de accesibilidad. Consulta: 15 de octubre de 2026."
    )

    def footer(canvas, doc):
        canvas.setFont("DejaVu", 8)
        canvas.setFillColor(colors.HexColor(GRAY))
        canvas.drawString(50, 27, "UVG · CC3084 · Laboratorio 9 · Memoria de diseño · Octubre 2026")
        canvas.drawRightString(A4[0] - 50, 27, str(doc.page))

    doc = SimpleDocTemplate(
        str(OUT / "explicacion_diseno.pdf"),
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=45,
        bottomMargin=50,
        title="Memoria de decisiones de diseño — Laboratorio 9",
        author="Fernando Mendoza y Pablo Escobar",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    (ROOT / "docs/explicacion_diseno.txt").write_text("\n\n".join(source) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    bundle = evidence()
    for filename in [
        "monthly_series",
        "model_metrics",
        "series_comparison",
        "series_diagnostics",
        "forecasts",
    ]:
        pd.read_parquet(DATA / f"{filename}.parquet").to_csv(OUT / f"{filename}.csv", index=False)
    (OUT / "cifras_verificadas.json").write_text(json.dumps(bundle[-1], ensure_ascii=False, indent=2) + "\n")
    infographic(bundle)
    explanation(bundle)
    print("Entregables generados y métricas verificadas:", OUT)


if __name__ == "__main__":
    main()
