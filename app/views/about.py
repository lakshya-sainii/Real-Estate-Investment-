import streamlit as st
from core import theme, data


def render():
    m = data.metrics(); dq = m["dq"]
    theme.page_title("About", "The project, plainly", "What was built, how, and what to keep in mind before relying on it.")
    a, b, c = st.columns(3, gap="medium")
    a.markdown(f"""<div class="card"><div class="lbl">Problem</div><h4>Is it worth buying?</h4><p>Help investors judge whether a property is a good buy and estimate what it could be worth in five years.</p>
      <div class="spec"><span>Rows</span><span>{dq['raw_rows']:,}</span></div><div class="spec"><span>Cities</span><span>42 in 20 states</span></div><div class="spec"><span>Missing values</span><span>None</span></div></div>""", unsafe_allow_html=True)
    b.markdown("""<div class="card"><div class="lbl">Method</div><h4>From data to verdict</h4><p>Cleaning and feature engineering feed the research models. The app uses a portable HistGradientBoosting classifier distilled from the same five investment criteria, plus the selected Ridge regressor, avoiding XGBoost serialisation issues at inference time.</p>
      <div class="spec"><span>Runtime classifier</span><span>Portable HGB</span></div><div class="spec"><span>Regressor</span><span>Ridge, log target</span></div><div class="spec"><span>Tracking</span><span>MLflow</span></div></div>""", unsafe_allow_html=True)
    c.markdown("""<div class="card"><div class="lbl">Rules</div><h4>How targets are defined</h4><p><b>Good Investment:</b> at least 3 of 5: price/sq ft ≤ city median, price ≤ city median, 3+ BHK, ready to move, infrastructure ≥ 0.5.<br><b>5-year price:</b> price × (1 + r)^5, r = 6% base plus tier, type, readiness, infrastructure and age adjustments.</p></div>""", unsafe_allow_html=True)
    theme.section("Limitations", "Stated plainly")
    st.markdown(f"""
- The dataset is a single snapshot with no price history, so both targets are rule-based. Scores show the rules were learned, not real-market accuracy.
- Growth rates are assumptions; replace them with estimates from historical data before any real use.
- Price is almost independent of location, size and facilities here, and {dq['floor_gt_total_pct']}% of rows had a floor number above the building height (corrected). There is no crime-rate field.
- The architectural hero image is an editorial visual, not a photograph of any analysed listing.
- Market Atlas and Opportunity Lab scores are relative screening aids, not predicted returns.
- Display font: Talina DEMO by Pandeka Foundry, free for personal use only.
""")
    theme.note(); theme.footer()


render()
