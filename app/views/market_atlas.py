"""Interactive city market atlas — a non-geographic view of the investment universe."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core import theme, data

METRICS = [
    "Opportunity index",
    "Avg annual growth (%)",
    "Good-investment rate (%)",
    "Avg price (lakhs)",
    "Median price/sq ft (INR)",
    "Avg 5-yr price (lakhs)",
]


def _score(cs: pd.DataFrame) -> pd.DataFrame:
    out = cs.copy()
    growth = out["Avg annual growth (%)"].rank(pct=True)
    quality = out["Good-investment rate (%)"].rank(pct=True)
    affordability = 1 - out["Median price/sq ft (INR)"].rank(pct=True)
    out["Opportunity index"] = (100 * (.40 * growth + .35 * quality + .25 * affordability)).round(1)
    return out


def _universe(cs: pd.DataFrame, metric: str) -> go.Figure:
    cs = cs.sort_values("Opportunity index", ascending=False).reset_index(drop=True)
    values = cs[metric].astype(float)
    size = 9 + 17 * np.sqrt(cs["Listings"] / cs["Listings"].max())
    custom = np.column_stack([
        cs["State"], cs["Tier"], cs["Listings"], cs["Opportunity index"],
        cs["Avg annual growth (%)"], cs["Good-investment rate (%)"], cs["Median price/sq ft (INR)"],
    ])
    fig = go.Figure(go.Scatterpolar(
        r=cs["Opportunity index"], theta=cs["City"], mode="markers+text",
        text=[city if i < 12 else "" for i, city in enumerate(cs["City"])],
        textposition="top center", textfont=dict(size=10, color=theme.INK),
        marker=dict(
            size=size, sizemode="diameter", color=values, colorscale=theme.MINT_SCALE,
            showscale=True, colorbar=dict(title=metric.replace("Avg ", "").replace(" (%)", ""), thickness=10, x=1.08),
            line=dict(color="rgba(240,255,248,.72)", width=1), opacity=.94,
        ),
        customdata=custom,
        hovertemplate=(
            "<b>%{theta}</b> · %{customdata[0]}<br>%{customdata[1]} · %{customdata[2]:,} listings"
            "<br>Opportunity index: %{customdata[3]:.1f}/100"
            "<br>Growth: %{customdata[4]:.2f}% / yr"
            "<br>Good investments: %{customdata[5]:.1f}%"
            "<br>Median price: ₹%{customdata[6]:,.0f}/sq ft<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="City opportunity universe · angle = rank, radius = opportunity, bubble = listings",
        polar=dict(
            bgcolor="rgba(8,13,11,.5)",
            radialaxis=dict(range=[0, 108], showticklabels=False, gridcolor="rgba(190,215,203,.12)", linecolor="rgba(190,215,203,.12)"),
            angularaxis=dict(showticklabels=False, gridcolor="rgba(190,215,203,.09)", linecolor="rgba(190,215,203,.12)", rotation=90, direction="clockwise"),
        ),
    )
    return theme.style_fig(fig, 650, False)


def _signature_card(row: pd.Series, rank: int):
    st.markdown(f"""
    <div class="atlas-note">
      <div class="lbl">0{rank} · {row['Tier']} · {row['State']}</div>
      <div style="font-family:Talina,Manrope,sans-serif;font-size:2rem;margin:.35rem 0">{row['City']}</div>
      <div style="display:flex;align-items:end;justify-content:space-between;border-bottom:1px solid var(--hair);padding-bottom:.75rem">
        <span style="font-size:2.5rem;font-weight:300;color:var(--mint)">{row['Opportunity index']:.1f}</span>
        <span class="lbl">Opportunity / 100</span>
      </div>
      <div class="spec"><span>Assumed growth</span><b>{row['Avg annual growth (%)']:.2f}% / yr</b></div>
      <div class="spec"><span>Good investments</span><b>{row['Good-investment rate (%)']:.1f}%</b></div>
      <div class="spec"><span>Median entry</span><b>₹{row['Median price/sq ft (INR)']:,.0f}/sq ft</b></div>
    </div>""", unsafe_allow_html=True)


def _comparison(cs: pd.DataFrame, cities: list[str]) -> go.Figure:
    dimensions = [
        ("Growth", "Avg annual growth (%)", True),
        ("Good rate", "Good-investment rate (%)", True),
        ("Affordability", "Median price/sq ft (INR)", False),
        ("5-yr value", "Avg 5-yr price (lakhs)", True),
        ("Market size", "Listings", True),
    ]
    labels = [d[0] for d in dimensions]
    fig = go.Figure()
    colors = [theme.MINT, theme.INK]
    for city, color in zip(cities, colors):
        row = cs[cs["City"] == city].iloc[0]
        vals = []
        for _, col, higher in dimensions:
            ranked = cs[col].rank(pct=True)
            value = float(ranked.loc[row.name] if higher else 1 - ranked.loc[row.name])
            vals.append(round(value * 100, 1))
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=labels + labels[:1], name=city,
            fill="toself", fillcolor="rgba(85,245,184,.12)" if color == theme.MINT else "rgba(240,245,243,.06)",
            line=dict(color=color, width=2), marker=dict(size=6, color=color),
            hovertemplate="%{theta}: %{r:.0f}/100<extra>" + city + "</extra>",
        ))
    fig.update_layout(
        title="City DNA · percentile score against all filtered cities",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75, 100], ticktext=["25", "50", "75", "100"], gridcolor="rgba(190,215,203,.12)", tickfont=dict(size=9, color=theme.GRAPHITE)),
            angularaxis=dict(gridcolor="rgba(190,215,203,.12)", tickfont=dict(color=theme.INK)),
        ),
    )
    return theme.style_fig(fig, 470, True)


def render():
    cs = _score(data.city_stats())
    theme.page_title(
        "Market Atlas", "The city opportunity universe",
        "A non-geographic view of 42 markets. Rank cities by value, quality and assumed growth, then compare their investment signatures.",
    )
    c1, c2, c3 = st.columns([1.25, 1, 1])
    metric = c1.selectbox("Bubble colour shows", METRICS)
    tiers = c2.multiselect("City tier", ["Tier-1", "Tier-2", "Tier-3"], default=["Tier-1", "Tier-2", "Tier-3"])
    states = c3.multiselect("State", sorted(cs["State"].unique()))
    f = cs[cs["Tier"].isin(tiers)]
    if states:
        f = f[f["State"].isin(states)]
    if len(f) < 2:
        st.warning("Select at least two cities by widening the tier or state filters.")
        return

    st.plotly_chart(_universe(f, metric), width="stretch")
    best = f.loc[f["Opportunity index"].idxmax()]
    theme.insight(
        "Opportunity index",
        f"{best['City']} leads this filtered universe at {best['Opportunity index']:.1f}/100. The index blends assumed growth (40%), good-investment rate (35%) and affordability (25%); it is a screening score, not a return forecast.",
    )

    theme.section("02 / City signatures", "The strongest markets in this lens")
    top = f.nlargest(3, "Opportunity index")
    for col, (rank, (_, row)) in zip(st.columns(3, gap="medium"), enumerate(top.iterrows(), 1)):
        with col:
            _signature_card(row, rank)

    theme.section("03 / Compare", "Two cities, five dimensions")
    cities = f["City"].tolist()
    default_second = cities.index("Pune") if "Pune" in cities and cities[0] != "Pune" else min(1, len(cities) - 1)
    a, b = st.columns(2)
    city_a = a.selectbox("First city", cities, index=0)
    city_b = b.selectbox("Second city", cities, index=default_second)
    if city_a == city_b:
        st.info("Choose two different cities to draw the comparison.")
    else:
        st.plotly_chart(_comparison(f, [city_a, city_b]), width="stretch")
        ra = f[f["City"] == city_a].iloc[0]
        rb = f[f["City"] == city_b].iloc[0]
        theme.insight(
            "Comparison",
            f"{city_a} scores {ra['Opportunity index']:.1f} versus {rb['Opportunity index']:.1f} for {city_b}. The radar shows relative position inside the filtered set, so change the filters and the comparison baseline changes too.",
        )
    with st.expander("View the underlying city data"):
        st.dataframe(
            f[["City", "State", "Tier", "Listings", *METRICS]].sort_values("Opportunity index", ascending=False).round(2),
            hide_index=True, width="stretch", height=430,
        )
    theme.note("Opportunity scores are relative to the current filter selection. Growth remains assumption-based; no historical price series is available.")
    theme.footer()


render()
