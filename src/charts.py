"""
Chart construction (Plotly).

Kept separate from app.py so the Streamlit page code stays focused on
layout, and so chart styling stays consistent across the whole app. Every
function takes plain pandas data (no Streamlit or yfinance objects) and
returns a `plotly.graph_objects.Figure`.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

TEMPLATE = "plotly_white"
COLOR_PRIMARY = "#1f4e79"
COLOR_POSITIVE = "#2e7d32"
COLOR_NEGATIVE = "#c62828"
COLOR_NEUTRAL = "#757575"


def _bar_chart(x, y, title, y_title, color=COLOR_PRIMARY) -> go.Figure:
    fig = go.Figure(go.Bar(x=list(x), y=list(y), marker_color=color))
    fig.update_layout(title=title, yaxis_title=y_title, template=TEMPLATE, height=350,
                       margin=dict(l=40, r=20, t=50, b=40))
    return fig


def _line_chart(x, y, title, y_title, color=COLOR_PRIMARY, as_percent=False) -> go.Figure:
    fig = go.Figure(go.Scatter(x=list(x), y=list(y), mode="lines+markers", line=dict(color=color, width=3)))
    fig.update_layout(title=title, yaxis_title=y_title, template=TEMPLATE, height=350,
                       margin=dict(l=40, r=20, t=50, b=40))
    if as_percent:
        fig.update_yaxes(tickformat=".0%")
    return fig


def revenue_chart(income_df: pd.DataFrame) -> go.Figure:
    years = [str(y.year) if hasattr(y, "year") else str(y) for y in income_df.index]
    return _bar_chart(years, income_df["revenue"], "Revenue", "USD")


def revenue_growth_chart(income_df: pd.DataFrame) -> go.Figure:
    years = [str(y.year) if hasattr(y, "year") else str(y) for y in income_df.index]
    return _line_chart(years, income_df["revenue_growth"], "Revenue Growth (YoY)", "Growth", as_percent=True)


def operating_income_chart(income_df: pd.DataFrame) -> go.Figure:
    years = [str(y.year) if hasattr(y, "year") else str(y) for y in income_df.index]
    return _bar_chart(years, income_df["operating_income"], "Operating Income", "USD")


def net_income_chart(income_df: pd.DataFrame) -> go.Figure:
    years = [str(y.year) if hasattr(y, "year") else str(y) for y in income_df.index]
    return _bar_chart(years, income_df["net_income"], "Net Income", "USD")


def eps_chart(income_df: pd.DataFrame) -> go.Figure:
    years = [str(y.year) if hasattr(y, "year") else str(y) for y in income_df.index]
    return _line_chart(years, income_df["diluted_eps"], "Diluted EPS", "USD per share")


def operating_margin_chart(income_df: pd.DataFrame) -> go.Figure:
    years = [str(y.year) if hasattr(y, "year") else str(y) for y in income_df.index]
    return _line_chart(years, income_df["operating_margin"], "Operating Margin", "Margin", as_percent=True)


def net_margin_chart(income_df: pd.DataFrame) -> go.Figure:
    years = [str(y.year) if hasattr(y, "year") else str(y) for y in income_df.index]
    return _line_chart(years, income_df["net_margin"], "Net Margin", "Margin", as_percent=True)


def fcf_chart(fcf_df: pd.DataFrame) -> go.Figure:
    years = [str(y.year) if hasattr(y, "year") else str(y) for y in fcf_df.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years, y=fcf_df["operating_cash_flow"], name="Operating Cash Flow", marker_color=COLOR_PRIMARY))
    fig.add_trace(go.Bar(x=years, y=-fcf_df["capex"], name="CapEx", marker_color=COLOR_NEGATIVE))
    fig.add_trace(go.Scatter(x=years, y=fcf_df["free_cash_flow"], name="Free Cash Flow", mode="lines+markers",
                              line=dict(color=COLOR_POSITIVE, width=3)))
    fig.update_layout(title="Free Cash Flow", barmode="relative", template=TEMPLATE, height=380,
                       margin=dict(l=40, r=20, t=50, b=40))
    return fig


def roe_chart(roe: pd.Series) -> go.Figure:
    years = [str(y.year) if hasattr(y, "year") else str(y) for y in roe.index]
    return _line_chart(years, roe, "Return on Equity (ROE)", "ROE", as_percent=True)


def roic_chart(roic: pd.Series) -> go.Figure:
    years = [str(y.year) if hasattr(y, "year") else str(y) for y in roic.index]
    return _line_chart(years, roic, "Return on Invested Capital (ROIC)", "ROIC", color=COLOR_POSITIVE, as_percent=True)


def projected_fcf_chart(projection: pd.DataFrame) -> go.Figure:
    years = [f"Year {y}" for y in projection.index]
    return _bar_chart(years, projection["free_cash_flow"], "Projected Free Cash Flow (DCF Forecast)", "USD",
                       color=COLOR_PRIMARY)


def scenario_chart(scenario_results: dict) -> go.Figure:
    names = ["Bear", "Base", "Bull"]
    values = [
        scenario_results["bear"].dcf.intrinsic_value_per_share,
        scenario_results["base"].dcf.intrinsic_value_per_share,
        scenario_results["bull"].dcf.intrinsic_value_per_share,
    ]
    colors = [COLOR_NEGATIVE, COLOR_NEUTRAL, COLOR_POSITIVE]
    fig = go.Figure(go.Bar(x=names, y=values, marker_color=colors, text=[f"${v:,.2f}" for v in values],
                            textposition="outside"))
    current_price = scenario_results["base"].dcf.current_price
    if current_price:
        fig.add_hline(y=current_price, line_dash="dash", line_color="black",
                       annotation_text=f"Current Price: ${current_price:,.2f}")
    fig.update_layout(title="Bull / Base / Bear Intrinsic Value per Share", yaxis_title="USD per share",
                       template=TEMPLATE, height=400, margin=dict(l=40, r=20, t=50, b=40))
    return fig


def sensitivity_heatmap(grid: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=grid.values,
        x=list(grid.columns),
        y=list(grid.index),
        colorscale="RdYlGn",
        text=[[f"${v:,.0f}" if pd.notna(v) else "N/A" for v in row_] for row_ in grid.values],
        texttemplate="%{text}",
        hoverongaps=False,
    ))
    fig.update_layout(title=title, template=TEMPLATE, height=420, margin=dict(l=60, r=20, t=50, b=40))
    return fig


def comparable_multiples_chart(peer_table: pd.DataFrame, target_ticker: str, target_multiples: dict) -> go.Figure:
    if peer_table.empty:
        fig = go.Figure()
        fig.update_layout(title="Comparable Company Multiples (no peer data available)", template=TEMPLATE, height=350)
        return fig

    fig = go.Figure()
    metrics = ["P/E", "EV/EBITDA", "EV/Revenue", "P/S"]
    for metric in metrics:
        if metric in peer_table.columns:
            fig.add_trace(go.Bar(name=metric, x=peer_table["Ticker"], y=peer_table[metric]))
    fig.update_layout(title=f"Peer Valuation Multiples vs. {target_ticker}", barmode="group",
                       template=TEMPLATE, height=400, margin=dict(l=40, r=20, t=50, b=40))
    return fig


def investment_score_gauge(score: float, rating: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": rating},
        gauge={
            "axis": {"range": [0, 1000]},
            "bar": {"color": COLOR_PRIMARY},
            "steps": [
                {"range": [0, 300], "color": "#fddede"},
                {"range": [300, 450], "color": "#fde8d0"},
                {"range": [450, 550], "color": "#fdf6d0"},
                {"range": [550, 650], "color": "#eef7d0"},
                {"range": [650, 750], "color": "#d9f2d0"},
                {"range": [750, 850], "color": "#c1ecc7"},
                {"range": [850, 1000], "color": "#a7e3b4"},
            ],
        },
    ))
    fig.update_layout(template=TEMPLATE, height=320, margin=dict(l=30, r=30, t=50, b=10))
    return fig


def speculative_score_gauge(score: float, rating: str) -> go.Figure:
    """Deliberately styled differently from `investment_score_gauge` (amber/
    purple vs. the Fundamental Score's blue/green) so the two scores are
    visually distinguishable at a glance -- this is a measure of
    speculative OPPORTUNITY, not investment attractiveness, and should
    never be mistaken for the other at a glance.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": rating},
        number={"font": {"color": "#8548c9"}},
        gauge={
            "axis": {"range": [0, 1000]},
            "bar": {"color": "#8548c9"},
            "bordercolor": "#8548c9",
            "steps": [
                {"range": [0, 200], "color": "#f2eefb"},
                {"range": [200, 300], "color": "#e8def8"},
                {"range": [300, 400], "color": "#dccff5"},
                {"range": [400, 500], "color": "#d0bff1"},
                {"range": [500, 600], "color": "#c3aeed"},
                {"range": [600, 700], "color": "#b69ce8"},
                {"range": [700, 800], "color": "#a789e2"},
                {"range": [800, 900], "color": "#9776db"},
                {"range": [900, 1000], "color": "#8548c9"},
            ],
        },
    ))
    fig.update_layout(template=TEMPLATE, height=320, margin=dict(l=30, r=30, t=50, b=10))
    return fig
