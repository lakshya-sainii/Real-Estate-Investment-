"""Estate IQ - Real Estate Investment Advisor (run: streamlit run app/app.py)."""
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
st.set_page_config(
    page_title="Estate IQ · Real Estate Investment Advisor",
    page_icon=str(ROOT / "assets" / "estate_iq_mark.png"),
    layout="wide",
    initial_sidebar_state="collapsed",
)

from core import theme
theme.inject_css()

PAGES = [
    ("Home", st.Page("views/home.py", title="Home", default=True)),
    ("Analyse", st.Page("views/predict.py", title="Analyse Property")),
    ("Market Atlas", st.Page("views/market_atlas.py", title="Market Atlas")),
    ("Insights", st.Page("views/insights.py", title="Insights")),
    ("AI Advisor", st.Page("views/advisor_page.py", title="AI Advisor")),
    ("Performance", st.Page("views/performance.py", title="Performance")),
    ("About", st.Page("views/about.py", title="About")),
]

nav = st.navigation(
    [page for _, page in PAGES],
    position="hidden"
)

with st.container(key="topnav"):
    cols = st.columns(
        [1.7, .8, .8, 1.15, .9, 1.1, 1.1, .8],
        vertical_alignment="center"
    )

    cols[0].markdown(
        '<div class="brand">'
        '<span class="brand-mark"></span>'
        '<span class="brand-text">Estate IQ</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    for c, (label, page) in zip(cols[1:], PAGES):
        with c:
            st.page_link(
                page,
                label=label,
                use_container_width=True
            )

nav.run()

