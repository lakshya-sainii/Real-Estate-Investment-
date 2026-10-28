import streamlit as st
from core import theme, data, advisor, business


def _ask(question: str, bid: str | None = None):
    df, cs, m = data.enriched(), data.city_stats(), data.metrics(); lp = st.session_state.get("last_pred")
    res = business.run(bid, df, cs, m, lp) if bid else advisor.reply(question, df, cs, m, lp)
    st.session_state.chat += [{"role": "user", "text": question}, {"role": "assistant", **res}]


def render():
    theme.page_title("AI Advisor", "Ask a business question", "Rule-based and data-driven: it reads your question and answers from this project's dataset and models, with a chart. "
                                                              "It can also suggest the business questions worth asking.")
    if "chat" not in st.session_state:
        st.session_state.chat = [{"role": "assistant", "text": "Hello! I'm the Estate IQ advisor. Pick a business question on the right, ask in your own words "
                                                               "(*2 BHK under 80 lakh in Pune*, *Compare Pune and Jaipur*), or type **business questions** and I'll suggest more.",
                                  "fig": None, "followups": ["city_value", "ready_premium", "overpriced", "scenario"]}]
    pending = st.session_state.pop("pending_q", None)
    if pending: _ask(business.BY_ID[pending]["q"], pending)
    q = st.chat_input("Ask the advisor, e.g. 'Which cities offer the best value?'")
    if q: _ask(q)

    left, right = st.columns([1.7, 1], gap="large")
    with right:
        st.markdown('<div class="lbl">Business questions</div><div class="mut" style="font-size:.85rem;margin:.3rem 0 .8rem">Each answer comes with a chart and a plain reading.</div>', unsafe_allow_html=True)
        for b in business.BQS:
            st.markdown(f'<div class="lbl" style="color:#55F5B8;margin-top:.6rem">{b["persona"]}</div>', unsafe_allow_html=True)
            st.button(b["q"], key=f"bq_{b['id']}", on_click=_ask, args=(b["q"], b["id"]), width="stretch")
        st.button("Suggest more business questions", key="bq_list", on_click=_ask, args=("business questions",), width="stretch")
        st.button("Explain my last result", key="bq_exp", on_click=_ask, args=("Explain my result",), width="stretch")
    with left:
        for i, msg in enumerate(st.session_state.chat):
            with st.chat_message(msg["role"]):
                st.markdown(msg["text"])
                if msg.get("fig") is not None: st.plotly_chart(msg["fig"], key=f"chat_fig_{i}", width="stretch")
        last = st.session_state.chat[-1]
        if last["role"] == "assistant" and last.get("followups"):
            st.markdown('<div class="lbl" style="margin:.8rem 0 .4rem">Questions worth taking next</div>', unsafe_allow_html=True)
            cols = st.columns(len(last["followups"]))
            for c, bid in zip(cols, last["followups"]):
                c.button(business.BY_ID[bid]["q"], key=f"fu_{len(st.session_state.chat)}_{bid}", on_click=_ask, args=(business.BY_ID[bid]["q"], bid), width="stretch")
    theme.note("This advisor is rule-based, not a generative AI: it answers only from the project's dataset and trained models, and cannot answer unrelated questions.")
    theme.footer()


render()
