import base64
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
from core import theme, data, predictor

PROPERTY_IMAGE = Path(__file__).resolve().parents[1] / "assets" / "hero_skyline.webp"


def _form(cs):
    cities = sorted(cs["City"].tolist())
    with st.form("property_form"):
        st.markdown('<div class="lbl">01 / Property</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        city = a.selectbox("City", cities, index=cities.index("Pune"))
        ptype = b.selectbox("Property type", ["Apartment", "Villa", "Independent House"])
        bhk = c.selectbox("BHK", [1, 2, 3, 4, 5], index=2)
        a, b, c = st.columns(3)
        size = a.slider("Size (sq ft)", 500, 5000, 1800, 50)
        price = b.number_input("Asking price (lakhs)", 10.0, 500.0, 120.0, 5.0)
        year = c.slider("Year built", 1990, 2025, 2018)
        a, b, c = st.columns(3)
        floor = a.number_input("Floor no.", 0, 30, 8); total = b.number_input("Total floors", 1, 30, 15)
        avail = c.selectbox("Availability", ["Ready_to_Move", "Under_Construction"], format_func=lambda x: x.replace("_", " "))
        st.markdown('<div class="lbl" style="margin-top:.6rem">02 / Neighbourhood and features</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        schools = a.slider("Nearby schools", 0, 10, 8); hosp = b.slider("Nearby hospitals", 0, 10, 6)
        transport = c.select_slider("Public transport", ["Low", "Medium", "High"], value="High")
        a, b, c = st.columns(3)
        furn = a.selectbox("Furnishing", ["Unfurnished", "Semi-furnished", "Furnished"], index=1)
        facing = b.selectbox("Facing", ["East", "West", "North", "South"])
        owner = c.selectbox("Owner type", ["Builder", "Owner", "Broker"])
        a, b, c = st.columns([1, 1, 2])
        parking = a.toggle("Parking", True); security = b.toggle("Security", True)
        amen = c.multiselect("Amenities", ["Gym", "Pool", "Clubhouse", "Playground", "Garden"], default=["Gym", "Pool", "Garden"])
        go_ = st.form_submit_button("Analyse property  ↗", width="stretch")
    ref = cs.loc[cs["City"] == city, "Median price/sq ft (INR)"].iloc[0] * size / 1e5
    st.caption(f"Reference: the median in {city} is about ₹{ref * 1e5 / size:,.0f} per sq ft, roughly {ref:,.1f} lakhs for {size:,} sq ft.")
    return go_, {"City": city, "Property_Type": ptype, "BHK": bhk, "Size_in_SqFt": size, "Price_in_Lakhs": float(price), "Year_Built": year, "Furnished_Status": furn,
                 "Floor_No": int(floor), "Total_Floors": int(total), "Nearby_Schools": schools, "Nearby_Hospitals": hosp, "Public_Transport_Accessibility": transport,
                 "Parking_Space": "Yes" if parking else "No", "Security": "Yes" if security else "No", "Amenities": ", ".join(amen), "Facing": facing,
                 "Owner_Type": owner, "Availability_Status": avail}


def _result(r):
    good = r["good"]
    theme.section("03 / Verdict", "Your property, analysed")
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        image64 = base64.b64encode(PROPERTY_IMAGE.read_bytes()).decode()
        badge = "Good investment" if good else "Review carefully"
        st.markdown(f'''<div class="property-visual fade" style="background-image:url(data:image/webp;base64,{image64})">
            <div class="pv-badge">{badge}</div>
            <div class="pv-caption">{r['inputs']['Property_Type']} · {r['inputs']['City']}</div>
        </div>''', unsafe_allow_html=True)
    with right:
        label_color = "#050707" if good else "#FFB0B0"
        st.markdown(f'<div class="verdict {"good" if good else "bad"} fade"><div class="lbl" style="color:{label_color}">Investment verdict</div>'
                    f'<div class="big">{"A good investment" if good else "Not a good investment"}</div>'
                    f'<span class="pill">{r["met"]} of 5 criteria</span><span class="pill">Confidence {r["confidence"] * 100:.1f}%</span><span class="pill">{r["tier"]}</span></div>', unsafe_allow_html=True)
        st.write("")
        rows = "".join(f'<div class="crit"><div><b>{n}</b><br><small>{d}</small></div><span class="{"ok" if ok else "no"}">{"Met" if ok else "Not met"}</span></div>' for n, ok, d in r["criteria"])
        st.markdown(f'<div class="card" style="padding-top:.8rem">{rows}</div>', unsafe_allow_html=True)

    st.write("")
    k = st.columns(4)
    k[0].metric("Asking price", f"{r['price']:,.1f} L"); k[1].metric("Estimated in 5 years", f"{r['future']:,.1f} L")
    k[2].metric("Expected gain", f"{r['gain']:,.1f} L", f"{(r['future'] / r['price'] - 1) * 100:.1f}%"); k[3].metric("Implied growth / year", f"{r['growth'] * 100:.2f}%")

    c1, c2 = st.columns(2, gap="large")
    yrs = list(range(0, 6)); model = [r["price"] * (1 + r["growth"]) ** y for y in yrs]
    lo = [r["price"] * 1.05 ** y for y in yrs]; hi = [r["price"] * 1.10 ** y for y in yrs]
    with c1:
        f = go.Figure()
        f.add_trace(go.Scatter(x=yrs, y=hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        f.add_trace(go.Scatter(x=yrs, y=lo, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(85,245,184,.35)", name="5% to 10% range", hoverinfo="skip"))
        f.add_trace(go.Scatter(x=yrs, y=model, mode="lines+markers", name="Model forecast", line=dict(color=theme.INK, width=3), marker=dict(size=8, color=theme.MINT, line=dict(color=theme.INK, width=1.5))))
        theme.annotate(f, 5, model[-1], f"{model[-1]:,.0f} L", ax=-20, ay=-46)
        f.update_layout(title="Price journey over five years (lakhs)", xaxis_title="Years from now", xaxis=dict(dtick=1))
        st.plotly_chart(theme.style_fig(f, 380), width="stretch")
        theme.insight("Forecast", f"The model implies {r['growth'] * 100:.2f}% a year. Across a conservative 5% to an optimistic 10%, the value in five years ranges from {lo[-1]:,.0f} to {hi[-1]:,.0f} lakhs. Present it as a range.")
    with c2:
        cats = ["Price per sq ft", "Total price"]
        mine = [r["ppsf"], r["price"]]; med = [r["city_ppsf"], r["city_price"]]; pct = [(a / b - 1) * 100 for a, b in zip(mine, med)]
        f2 = go.Figure(go.Bar(x=pct, y=cats, orientation="h", marker_color=[theme.MINT_DEEP if p <= 0 else theme.INK for p in pct], text=[f"{p:+.0f}%" for p in pct], textposition="outside"))
        f2.add_vline(x=0, line_color=theme.INK, line_width=1)
        f2.update_layout(title=f"Your property vs the {r['inputs']['City']} median (%)", xaxis_title="Above (+) or below (−) the city median"); f2.update_yaxes(autorange="reversed")
        f2.update_xaxes(range=[min(pct + [-10]) * 1.4, max(pct + [10]) * 1.4])
        st.plotly_chart(theme.style_fig(f2, 380, False), width="stretch")
        theme.insight("Value", f"Price per sq ft is {abs(pct[0]):.0f}% {'below' if pct[0] <= 0 else 'above'} the city median and total price is {abs(pct[1]):.0f}% {'below' if pct[1] <= 0 else 'above'}. "
                               "Being at or below the median earns two of the five criteria.")
    theme.note("The 5-year estimate compounds an assumed, feature-based growth rate (the dataset has no price history). Indicative only; not financial advice.")


def render():
    cs = data.city_stats()
    theme.page_title("Predict", "Analyse a property", "Fill in the details to get a verdict, a five-year price estimate and the reasoning behind both.")
    submitted, p = _form(cs)
    if submitted:
        try:
            st.session_state["last_pred"] = predictor.predict(p)
        except predictor.ModelLoadError as exc:
            st.error(str(exc))
            st.code("python -m pip install -r app/requirements.txt\nstreamlit run app/app.py", language="bash")
            st.stop()
    if "last_pred" in st.session_state:
        _result(st.session_state["last_pred"])
    theme.footer()


render()
