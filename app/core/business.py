"""Business questions for the project. Each handler answers from the data and returns (markdown, plotly figure)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from . import theme
from .theme import INK, MINT, MINT_DEEP, GRAPHITE, CHROME


def L(x):  # lakhs -> text
    return f"₹{x / 100:,.2f} cr" if x >= 100 else f"₹{x:,.1f} L"


def _heat(z, x, y, title, fmt="{:.0f}%", height=360):
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=theme.MINT_SCALE, text=[[fmt.format(v) for v in r] for r in z], texttemplate="%{text}",
                               showscale=False, hovertemplate="%{y} · %{x}: %{text}<extra></extra>"))
    fig.update_layout(title=title); fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return theme.style_fig(fig, height, False)


def bq_city_value(d, cs, m, lp):
    s = cs.copy()
    s["score"] = s["Avg annual growth (%)"].rank(pct=True) + (1 - s["Median price/sq ft (INR)"].rank(pct=True))
    top = s.nlargest(5, "score"); rest = s[~s["City"].isin(top["City"])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rest["Median price/sq ft (INR)"], y=rest["Avg annual growth (%)"], mode="markers", name="Other cities", text=rest["City"], marker=dict(size=10, color="#9AA3AB", opacity=.75), hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(go.Scatter(x=top["Median price/sq ft (INR)"], y=top["Avg annual growth (%)"], mode="markers+text", name="Best balance", text=top["City"], textposition=["top center", "bottom center", "top right", "bottom left", "middle right"][:len(top)], marker=dict(size=15, color=MINT, line=dict(color=INK, width=1.5))))
    fig.update_layout(title="Entry price vs assumed growth, by city", xaxis_title="Median price per sq ft (INR) — lower is cheaper", yaxis_title="Avg assumed growth (% / yr)")
    spread = (s["Median price/sq ft (INR)"].max() / s["Median price/sq ft (INR)"].min() - 1) * 100
    text = (f"**Best balance of low entry price and high growth:** {', '.join(top['City'])}.\n\nThese cities combine an above-average assumed growth rate (Tier-2 markets get the highest rate in the growth model) with a below-average price per sq ft. "
            f"Caution: median price per sq ft differs by only {spread:.1f}% across all 42 cities, so city choice is a weak lever in this dataset; property-level factors matter more.")
    return text, theme.style_fig(fig, 420)


def bq_ready_premium(d, cs, m, lp):
    g = d.groupby("Availability_Status", observed=True).agg(good=("Good_Investment", "mean"), growth=("Annual_Growth_Rate", "mean"), gain=("Gain_pct", "median"), price=("Price_in_Lakhs", "median"))
    names = {"Ready_to_Move": "Ready to move", "Under_Construction": "Under construction"}
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Good-investment rate (%)", "Assumed growth (% / yr)", "Median 5-yr gain (%)"))
    for i, (col, mult) in enumerate([("good", 100), ("growth", 100), ("gain", 1)], 1):
        for st_, c in zip(["Ready_to_Move", "Under_Construction"], [INK, MINT_DEEP]):
            fig.add_trace(go.Bar(x=[names[st_]], y=[g.loc[st_, col] * mult], marker_color=c, showlegend=False, text=[f"{g.loc[st_, col] * mult:.1f}"], textposition="outside"), row=1, col=i)
    fig.update_yaxes(rangemode="tozero")
    rm, uc = g.loc["Ready_to_Move"], g.loc["Under_Construction"]
    text = (f"**Ready to move** has a {rm['good'] * 100:.1f}% good-investment rate versus {uc['good'] * 100:.1f}% for **under construction**, at almost the same median price ({L(rm['price'])} vs {L(uc['price'])}). "
            f"Under-construction units carry a small assumed growth premium ({uc['growth'] * 100:.2f}% vs {rm['growth'] * 100:.2f}% a year) for the completion uplift.\n\n"
            "Business reading: ready-to-move listings are the safer screen (no delivery risk); under-construction ones only win on assumed upside, which is not verified by sale history.")
    return text, theme.style_fig(fig, 340, False)


def bq_bhk_budget(d, cs, m, lp):
    p = d.pivot_table(index="BHK", columns="Price_band", values="Good_Investment", aggfunc="mean", observed=True) * 100
    fig = _heat(p.values, list(p.columns.astype(str)), [f"{i} BHK" for i in p.index], "Good-investment rate (%) by BHK and budget band (₹ lakhs)")
    best = p.stack().idxmax(); worst = p.stack().idxmin()
    text = (f"**Best segment:** {best[0]} BHK in the {best[1]} lakh band ({p.loc[best]:.0f}% qualify). **Weakest:** {worst[0]} BHK in the {worst[1]} lakh band ({p.loc[worst]:.0f}%).\n\n"
            "Two rules drive this: 3+ BHK homes earn a criterion, and lower prices are more likely to sit below their city median. Buyers with a budget cap should shortlist 3+ BHK units in the cheaper bands.")
    return text, fig


def bq_undervalued(d, cs, m, lp):
    u = d[d["PPSF_ratio"] <= 0.8]
    share = u.groupby("City", observed=True).size() / d.groupby("City", observed=True).size() * 100
    top = share.sort_values(ascending=False).head(10)
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=MINT_DEEP, text=[f"{v:.1f}%" for v in top.values], textposition="outside"))
    fig.update_layout(title="Share of listings priced ≥20% below their city's median ₹/sq ft", xaxis_title="% of city listings"); fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, top.max() * 1.15])
    text = (f"**{len(u):,} listings ({len(u) / len(d) * 100:.1f}%)** are priced at least 20% below their city's median price per sq ft; {u['Good_Investment'].mean() * 100:.0f}% of them qualify as Good Investments.\n\n"
            f"The cities with the most such listings are {', '.join(top.index[:3])}. For a listing platform this is a ready-made 'Value picks' shelf. Their median size is {u['Size_in_SqFt'].median():,.0f} sq ft versus {d['Size_in_SqFt'].median():,.0f} sq ft overall, so a low price per sq ft mostly reflects larger units.")
    return text, theme.style_fig(fig, 400, False)


def bq_tier_type(d, cs, m, lp):
    p = d.pivot_table(index="City_Tier", columns="Property_Type", values="Annual_Growth_Rate", aggfunc="mean", observed=True) * 100
    fig = _heat(p.values, list(p.columns), list(p.index), "Assumed annual growth (%) by city tier and property type", "{:.2f}%", 300)
    best = p.stack().idxmax()
    text = (f"**Highest assumed growth:** {best[1]} in {best[0]} cities ({p.loc[best]:.2f}% a year); lowest is {p.stack().idxmin()[1]} in {p.stack().idxmin()[0]} cities ({p.stack().min():.2f}%).\n\n"
            "For a developer this says where to build: villas and houses in emerging Tier-2 markets. These rates come from the project's growth assumptions (no price history exists), so validate with real transaction data.")
    return text, fig


def bq_infra(d, cs, m, lp):
    d = d.assign(band=pd.qcut(d["Infrastructure_Score"], 4, labels=["Lowest 25%", "Lower-mid", "Upper-mid", "Top 25%"]))
    g = d.groupby("band", observed=True).agg(good=("Good_Investment", "mean"), growth=("Annual_Growth_Rate", "mean")) * 100
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=g.index.astype(str), y=g["good"], name="Good-investment rate (%)", marker_color=CHROME, text=[f"{v:.0f}%" for v in g["good"]], textposition="outside"))
    fig.add_trace(go.Scatter(x=g.index.astype(str), y=g["growth"], name="Assumed growth (%/yr)", mode="lines+markers", line=dict(color=INK, width=3), marker=dict(size=9, color=MINT)), secondary_y=True)
    fig.update_layout(title="Infrastructure score quartile vs investment quality and growth"); fig.update_yaxes(rangemode="tozero", secondary_y=False); fig.update_yaxes(secondary_y=True, showgrid=False)
    text = (f"Moving from the lowest to the top infrastructure quartile lifts the good-investment rate from {g['good'].iloc[0]:.0f}% to {g['good'].iloc[-1]:.0f}% and assumed growth from {g['growth'].iloc[0]:.2f}% to {g['growth'].iloc[-1]:.2f}% a year.\n\n"
            "Business reading: schools, hospitals and transport access are the most 'investable' neighbourhood features here, which supports pitching infrastructure in listings and city-planning conversations. Both effects are built into the labelling rules.")
    return text, theme.style_fig(fig, 380)


def bq_scenario(d, cs, m, lp):
    price = lp["price"] if lp else float(d["Price_in_Lakhs"].median())
    model_r = lp["growth"] if lp else float(d["Annual_Growth_Rate"].mean())
    sc = [("Conservative 5%", .05), (f"Model {model_r * 100:.1f}%", model_r), ("Base 8%", .08), ("Optimistic 10%", .10)]
    vals = [price * (1 + r) ** 5 for _, r in sc]
    fig = go.Figure(go.Bar(x=[n for n, _ in sc], y=vals, marker_color=[CHROME, MINT_DEEP, INK, MINT], text=[L(v) for v in vals], textposition="outside"))
    fig.add_hline(y=price, line_dash="dot", line_color=GRAPHITE, annotation_text=f"Today {L(price)}", annotation_position="bottom right")
    fig.update_layout(title=f"5-year value of a {L(price)} property under different growth assumptions"); fig.update_yaxes(range=[0, max(vals) * 1.18])
    text = (f"For {'your last analysed property' if lp else 'the median listing'} ({L(price)}), the 5-year value ranges from **{L(vals[0])}** (5%) to **{L(vals[3])}** (10%): a spread of {L(vals[3] - vals[0])}.\n\n"
            "Business reading: the growth assumption moves the outcome more than any single property feature, so always present a range rather than one forecast.")
    return text, theme.style_fig(fig, 380, False)


def bq_overpriced(d, cs, m, lp):
    bands = pd.cut(d["PPSF_ratio"], [0, .7, .9, 1.1, 1.3, 20], labels=["<70%", "70-90%", "90-110%", "110-130%", ">130%"])
    g = d.groupby(bands, observed=True)["Good_Investment"].mean() * 100
    share = (d["PPSF_ratio"] > 1).mean() * 100
    fig = go.Figure(go.Bar(x=g.index.astype(str), y=g.values, marker_color=[MINT_DEEP, MINT_DEEP, CHROME, INK, INK], text=[f"{v:.0f}%" for v in g.values], textposition="outside"))
    fig.update_layout(title="Good-investment rate by price vs city median (₹/sq ft)", xaxis_title="Price per sq ft as % of the city median", yaxis_title="% Good Investment"); fig.update_yaxes(range=[0, 110])
    text = (f"**{share:.0f}% of listings** are priced above their city's median per sq ft. Their good-investment rate falls from {g.iloc[0]:.0f}% (priced under 70% of median) to {g.iloc[-1]:.0f}% (over 130%).\n\n"
            "Business reading: sellers priced well above the local median lose the 'value' criteria and qualify far less often. A pricing nudge toward the median is the quickest way to improve a listing's investment score.")
    return text, theme.style_fig(fig, 360, False)


def bq_drivers(d, cs, m, lp):
    fi = pd.Series(m["clf"]["top_features"]).sort_values().tail(8)
    fig = go.Figure(go.Bar(x=fi.values, y=[i.replace("_", " ") for i in fi.index], orientation="h", marker_color=[MINT_DEEP if v > .05 else CHROME for v in fi.values]))
    fig.update_layout(title="What drives the Good-Investment verdict (permutation importance)", xaxis_title="Drop in F1 when shuffled")
    text = ("**Five factors do all the work:** price, price relative to the city median, ready-to-move status, infrastructure score and BHK. Everything else (facing, furnishing, owner type, amenities) has almost no effect.\n\n"
            "These are exactly the inputs of the labelling rule, which confirms the model learned the intended logic. Business reading: to improve a listing's score, change price positioning, readiness, BHK mix or neighbourhood access, not cosmetics.")
    return text, theme.style_fig(fig, 360, False)


def bq_portfolio(d, cs, m, lp):
    g = d.groupby("City_Tier", observed=True)["Annual_Growth_Rate"].mean()
    vals = {t: 100 * (1 + r) ** 5 for t, r in g.items()}
    fig = go.Figure(go.Bar(x=list(vals), y=list(vals.values()), marker_color=[INK, MINT_DEEP, CHROME], text=[f"₹{v:.0f} L" for v in vals.values()], textposition="outside"))
    fig.add_hline(y=100, line_dash="dot", line_color=GRAPHITE, annotation_text="Invested ₹100 L", annotation_position="bottom right")
    fig.update_layout(title="₹1 crore (100 lakh) after 5 years, by city tier"); fig.update_yaxes(range=[0, max(vals.values()) * 1.2])
    best = max(vals, key=vals.get)
    text = (f"A ₹1 crore ticket grows to about **₹{vals[best]:.0f} lakh** in {best} cities, versus ₹{min(vals.values()):.0f} lakh in the weakest tier, at the model's average assumed rates.\n\n"
            "Business reading: tilt a portfolio toward Tier-2 markets for growth and keep some ready-to-move Tier-1 stock for stability. The tier gaps reflect assumptions, so treat this as a framework, not a forecast.")
    return text, theme.style_fig(fig, 360, False)


BQS = [
    dict(id="city_value", persona="Investor", q="Which cities balance low entry price and high growth best?", fn=bq_city_value, kw=["balance", "best value", "entry price", "value for money", "cheap and growing", "best city"]),
    dict(id="ready_premium", persona="Investor", q="Is ready-to-move worth it versus under-construction?", fn=bq_ready_premium, kw=["worth it", "premium", "ready to move vs", "ready vs", "ready-to-move", "under construction vs"]),
    dict(id="portfolio", persona="Investor", q="What could ₹1 crore become in 5 years across city tiers?", fn=bq_portfolio, kw=["1 crore", "portfolio", "ticket", "10 million", "diversif"]),
    dict(id="scenario", persona="Risk", q="How sensitive is the 5-year value to the growth assumption?", fn=bq_scenario, kw=["sensitiv", "scenario", "assumption", "what if growth", "best case", "worst case"]),
    dict(id="bhk_budget", persona="Buyer", q="Which BHK and budget bands have the most good investments?", fn=bq_bhk_budget, kw=["bhk and budget", "which bhk", "budget band", "which budget", "best bhk"]),
    dict(id="overpriced", persona="Seller", q="Does pricing above the city median hurt investment quality?", fn=bq_overpriced, kw=["overpriced", "above median", "pricing", "price too high", "priced above"]),
    dict(id="undervalued", persona="Platform", q="Which listings are priced well below their city median?", fn=bq_undervalued, kw=["undervalued", "underpriced", "bargain", "below median", "value picks", "deal"]),
    dict(id="tier_type", persona="Developer", q="Which property type and city tier appreciates fastest?", fn=bq_tier_type, kw=["appreciates", "what to build", "villa or apartment", "property type and", "fastest growth", "developer"]),
    dict(id="infra", persona="Planner", q="How much does infrastructure change investment quality?", fn=bq_infra, kw=["infrastructure", "schools and hospitals", "transport matter", "connectivity"]),
    dict(id="drivers", persona="Product", q="Which factors drive the model's verdict the most?", fn=bq_drivers, kw=["drive the model", "which factors", "feature importance", "what matters most", "key drivers", "drivers"]),
]
BY_ID = {b["id"]: b for b in BQS}


def run(bq_id, d, cs, m, lp):
    text, fig = BY_ID[bq_id]["fn"](d, cs, m, lp)
    others = [b["id"] for b in BQS if b["id"] != bq_id]
    i = [b["id"] for b in BQS].index(bq_id)
    follow = [BQS[(i + k) % len(BQS)]["id"] for k in (1, 2, 3)]
    return {"text": f"**{BY_ID[bq_id]['q']}**\n\n{text}", "fig": fig, "followups": follow}


def match(ql):
    best, score = None, 0
    for b in BQS:
        s = sum(1 for k in b["kw"] if k in ql)
        if s > score: best, score = b["id"], s
    return best


def listing():
    rows = [f"{i}. **{b['persona']}**: {b['q']}" for i, b in enumerate(BQS, 1)]
    return {"text": "Here are business questions worth asking about this project, each answerable from the data. Tap one below, or ask in your own words.\n\n" + "\n".join(rows),
            "fig": None, "followups": ["city_value", "ready_premium", "overpriced", "scenario"]}
