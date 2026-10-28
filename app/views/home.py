import base64
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
from core import theme, data, business

HERO_IMAGE = Path(__file__).resolve().parents[1] / "assets" / "hero_skyline.webp"


def _ask(bid):
    st.session_state["pending_q"] = bid
    st.switch_page("views/advisor_page.py")


def render():
    df, cs, m = data.listings(), data.city_stats(), data.metrics()
    x, rg = m["clf"]["results"]["XGBoost_Tuned"], m["reg"]["results"]["LinearRegression_Ridge"]

    with st.container(key="hero"):
        l, r = st.columns([1.05, 1], gap="large")
        with l:
            st.markdown("""
            <div class="fade"><div class="lbl">Property intelligence &nbsp;/&nbsp; India &nbsp;/&nbsp; 2026</div>
            <div class="hero-h">Know its <span class="hl">worth</span>,<br>five years out.</div>
            <div class="rule"></div>
            <div class="hero-sub">Every listing tells a story.<br>Every price leaves a pattern.</div>
            <div class="hero-p">An exploration of 250,000 Indian property listings. Turning price, location and neighbourhood data into a clearer investment decision,
            with a verdict, a five-year forecast and the reasoning behind both.</div>
            <div class="idx"><div><b>01</b>/ Price</div><div><b>02</b>/ Location</div><div><b>03</b>/ Infrastructure</div><div><b>04</b>/ Growth</div></div></div>""", unsafe_allow_html=True)
            with st.container(key="cta"):
                st.page_link("views/predict.py", label="Analyse a property  ↗")
            st.markdown('<div class="note" style="margin-top:1.2rem">Real data. Honest assumptions. A clearer view of value.</div>', unsafe_allow_html=True)
        with r:
            image64 = base64.b64encode(HERO_IMAGE.read_bytes()).decode()
            st.markdown(f'''<div class="hero-art fade">
                <img src="data:image/webp;base64,{image64}" alt="Premium contemporary glass skyline">
                <div class="art-caption"><span>Architectural intelligence</span><span>India · 2026</span></div>
            </div>''', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stats">
      <div class="cell"><div class="lbl" style="color:#55F5B8">Behind the patterns</div><div class="big">A large dataset.<br>A closer look.</div></div>
      <div class="cell"><div class="n">{len(df):,}</div><div class="t">Listings analysed</div></div>
      <div class="cell"><div class="n">{df['City'].nunique()}</div><div class="t">Cities · {df['State'].nunique()} states</div></div>
      <div class="cell"><div class="n">2</div><div class="t">Trained models</div></div>
      <div class="cell"><div class="n">{df['Annual_Growth_Rate'].mean() * 100:.1f}%</div><div class="t">Avg assumed growth <em>/ yr</em></div></div>
      <div class="cell"><div class="mut" style="font-size:.78rem">Price is more than a number.</div><div style="font-weight:700;font-size:.88rem">Let's understand the pattern.</div></div>
    </div>""", unsafe_allow_html=True)

    theme.section("02 / The models", "Two questions, answered")
    a, b, c = st.columns(3, gap="medium")
    a.markdown(f"""<div class="card"><div class="lbl">Classification</div><h4>Is it a good investment?</h4>
      <p>Scores five criteria: price and price per sq ft versus the city median, 3+ BHK, ready to move, strong infrastructure. Three of five qualifies.</p>
      <div class="spec"><span>Model</span><span>HGB runtime · XGB research</span></div><div class="spec"><span>Research test accuracy</span><span>{x['test_accuracy'] * 100:.2f}%</span></div>
      <div class="spec"><span>F1 score</span><span>{x['test_f1']:.4f}</span></div></div>""", unsafe_allow_html=True)
    b.markdown(f"""<div class="card"><div class="lbl">Regression</div><h4>What is it worth in five years?</h4>
      <p>Compounds today's price at a growth rate set by city tier, property type, readiness, infrastructure and age.</p>
      <div class="spec"><span>Model</span><span>Ridge, log target</span></div><div class="spec"><span>RMSE</span><span>{rg['test_rmse']:.2f} lakhs</span></div>
      <div class="spec"><span>Growth range</span><span>{df['Annual_Growth_Rate'].min() * 100:.1f}% to {df['Annual_Growth_Rate'].max() * 100:.1f}% / yr</span></div></div>""", unsafe_allow_html=True)
    c.markdown(f"""<div class="card"><div class="lbl">Data and honesty</div><h4>What to keep in mind</h4>
      <p>The dataset is one price snapshot, so both targets come from documented rules. High scores show the rules were learned, not that markets are predicted.</p>
      <div class="spec"><span>Source</span><span>Single snapshot</span></div><div class="spec"><span>Targets</span><span>Rule-based</span></div>
      <div class="spec"><span>Rows fixed</span><span>{m['dq']['floor_gt_total_pct']}% floor errors</span></div></div>""", unsafe_allow_html=True)

    theme.section("03 / Business questions", "Questions worth asking", "Tap one and the advisor answers from the data, with a chart.")
    qs = [business.BY_ID[i] for i in ("city_value", "ready_premium", "overpriced", "scenario")]
    cols = st.columns(4, gap="medium")
    for col, q in zip(cols, qs):
        col.markdown(f'<div class="qcard"><div class="p">{q["persona"]}</div><div class="q">{q["q"]}</div></div>', unsafe_allow_html=True)
        col.button("Ask the advisor  ↗", key=f"hq_{q['id']}", on_click=_ask, args=(q["id"],))

    theme.section("04 / Market snapshot", "What moves the verdict")
    d = data.enriched()
    crit = [("Ready to move", d["Availability_Status"] == "Ready_to_Move"), ("3 BHK or larger", d["BHK"] >= 3), ("Infrastructure score ≥ 0.5", d["Infrastructure_Score"] >= 0.5),
            ("Price/sq ft ≤ city median", d["PPSF_ratio"] <= 1)]
    rows = [(n, d.loc[~mk, "Good_Investment"].mean() * 100, d.loc[mk, "Good_Investment"].mean() * 100) for n, mk in crit]
    fig = go.Figure()
    for n, no, yes in rows:
        fig.add_trace(go.Scatter(x=[no, yes], y=[n, n], mode="lines", line=dict(color="rgba(190,215,203,.22)", width=2), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[r[1] for r in rows], y=[r[0] for r in rows], mode="markers+text", name="Criterion not met", marker=dict(size=15, color=theme.CHROME, line=dict(color=theme.INK, width=1.5)),
                             text=[f"{r[1]:.0f}%" for r in rows], textposition="middle left"))
    fig.add_trace(go.Scatter(x=[r[2] for r in rows], y=[r[0] for r in rows], mode="markers+text", name="Criterion met", marker=dict(size=15, color=theme.MINT, line=dict(color=theme.INK, width=1.5)),
                             text=[f"{r[2]:.0f}%" for r in rows], textposition="middle right"))
    fig.update_layout(title="Share of listings that are Good Investments, when each criterion is met or not (%)", xaxis_title="Good-investment rate (%)")
    fig.update_xaxes(range=[-5, 105]); fig.update_yaxes(autorange="reversed", showgrid=False)
    g1, g2 = st.columns([1.5, 1], gap="large")
    g1.plotly_chart(theme.style_fig(fig, 400), width="stretch")
    with g2:
        st.write(""); st.write("")
        big = max(rows, key=lambda r: r[2] - r[1])
        theme.insight("Drivers", f"{big[0]} moves the verdict the most: {big[2]:.0f}% of listings that meet it qualify, against {big[1]:.0f}% of those that do not. "
                                 "A property needs three of the five criteria, so each one can tip the result.")
        theme.note()
    theme.footer()


render()
