import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from core import theme, data
from core.theme import INK, MINT, MINT_DEEP, CHROME, GRAPHITE


def _heat(z, x, y, title, fmt="{:.0f}%", h=340):
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y, colorscale=theme.MINT_SCALE,
        text=[[fmt.format(v) for v in r] for r in z], texttemplate="%{text}", showscale=False,
        hovertemplate="%{y} · %{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(title=title)
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return theme.style_fig(fig, h, False)


def _opportunity_mosaic(cs):
    # Explicit state nodes are required: implicit, missing parents can leave a
    # Plotly treemap blank even though the city-level data is valid.
    states = cs.groupby("State", observed=True).agg(
        score=("score", "mean"), cities=("City", "size"),
    ).reset_index()
    city_values = cs["score"].clip(lower=0) + 1
    state_values = cs.assign(_value=city_values).groupby("State", observed=True)["_value"].sum()
    labels = states["State"].tolist() + cs["City"].tolist()
    parents = [""] * len(states) + cs["State"].tolist()
    values = [state_values[s] for s in states["State"]] + city_values.tolist()
    colors = states["score"].tolist() + cs["score"].tolist()
    text = [f"{n} cities" for n in states["cities"]] + [f"{v:.1f}" for v in cs["score"]]
    hover = [f"{n} cities<br>Mean opportunity: {v:.1f}/100" for n, v in zip(states["cities"], states["score"])]
    hover += [
        f"{r.State} · {r.Tier}<br>Opportunity: {r.score:.1f}/100"
        f"<br>Growth: {r.growth:.2f}% / yr<br>Good investments: {r.good:.1f}%"
        f"<br>Median: ₹{r.ppsf:,.0f}/sq ft"
        for r in cs.itertuples()
    ]
    fig = go.Figure(go.Treemap(
        labels=labels, parents=parents, values=values, branchvalues="total",
        text=text, textinfo="label+text", hovertext=hover,
        hovertemplate="<b>%{label}</b><br>%{hovertext}<extra></extra>",
        marker=dict(colors=colors, colorscale=theme.MINT_SCALE, showscale=True,
                    cmin=0, cmax=100, colorbar=dict(title="Score", thickness=10),
                    line=dict(color="#07100D", width=2)),
        textfont=dict(family="Manrope", size=13, color="#F1F5F3"),
        pathbar=dict(visible=False), root=dict(color="#0B0F0E"),
    ))
    fig.update_layout(title="Opportunity mosaic · grouped by state, sized and coloured by your weighted score")
    return theme.style_fig(fig, 530, False)


def _mobility_sankey(f):
    work = f.assign(
        Infra=pd.cut(f["Infrastructure_Score"], [-0.01, .33, .66, 1.01], labels=["Limited", "Connected", "Strong"]),
        Verdict=np.where(f["Good_Investment"] == 1, "Good investment", "Does not qualify"),
    )
    nodes = ["Low transport", "Medium transport", "High transport", "Limited infrastructure", "Connected infrastructure", "Strong infrastructure", "Good investment", "Does not qualify"]
    transport = ["Low", "Medium", "High"]
    infra = ["Limited", "Connected", "Strong"]
    flows = []
    for i, t in enumerate(transport):
        for j, band in enumerate(infra):
            n = len(work[(work["Public_Transport_Accessibility"] == t) & (work["Infra"] == band)])
            if n:
                flows.append((i, 3 + j, n))
    for j, band in enumerate(infra):
        for k, verdict in enumerate(["Good investment", "Does not qualify"]):
            n = len(work[(work["Infra"] == band) & (work["Verdict"] == verdict)])
            if n:
                flows.append((3 + j, 6 + k, n))
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=nodes, pad=18, thickness=18,
                  color=[CHROME, "#91A49B", MINT, "#27362F", "#32644F", MINT_DEEP, MINT, "#5B4142"],
                  line=dict(color="rgba(240,245,243,.22)", width=1),
                  hovertemplate="%{label}<extra></extra>"),
        link=dict(source=[v[0] for v in flows], target=[v[1] for v in flows], value=[v[2] for v in flows],
                  color="rgba(85,245,184,.16)", hovertemplate="%{value:,} listings<extra></extra>"),
    ))
    fig.update_layout(title="Connectivity flow · transport access → infrastructure strength → investment verdict")
    return theme.style_fig(fig, 460, False)


def render():
    df = data.enriched()
    theme.page_title(
        "Insights", "Market intelligence",
        "Filter 250,000 listings, build your own opportunity index, trace neighbourhood signals and export the evidence behind the charts.",
    )
    with st.expander("Filters", expanded=False):
        a, b, c, d = st.columns(4)
        states = a.multiselect("State", sorted(df["State"].unique()))
        types = b.multiselect("Property type", sorted(df["Property_Type"].unique()))
        bhk = c.multiselect("BHK", [1, 2, 3, 4, 5])
        avail = d.multiselect("Availability", sorted(df["Availability_Status"].unique()))
        a, b = st.columns(2)
        pr = a.slider("Price (lakhs)", 10, 500, (10, 500))
        sz = b.slider("Size (sq ft)", 500, 5000, (500, 5000))
    f = df[(df["Price_in_Lakhs"].between(*pr)) & (df["Size_in_SqFt"].between(*sz))]
    if states:
        f = f[f["State"].isin(states)]
    if types:
        f = f[f["Property_Type"].isin(types)]
    if bhk:
        f = f[f["BHK"].isin(bhk)]
    if avail:
        f = f[f["Availability_Status"].isin(avail)]
    if len(f) < 200:
        st.warning("Too few listings match these filters. Widen them to see charts.")
        return

    views = ["Overview", "Opportunity Lab", "Value & Growth", "Cities", "Property Mix", "Livability", "Signals", "Data Explorer"]
    k = st.columns(4)
    k[0].metric("Listings", f"{len(f):,}")
    k[1].metric("Median price", f"{f['Price_in_Lakhs'].median():,.1f} L")
    k[2].metric("Good-investment rate", f"{f['Good_Investment'].mean() * 100:.1f}%")
    k[3].metric("Avg growth / yr", f"{f['Annual_Growth_Rate'].mean() * 100:.2f}%")
    st.write("")
    view = st.pills("Analysis view", views, default="Overview", label_visibility="collapsed", key="insights_analysis_view")

    if view == "Overview":
        a, b = st.columns(2, gap="large")
        med = f["Price_in_Lakhs"].median()
        h = go.Figure(go.Histogram(x=f["Price_in_Lakhs"], nbinsx=40, marker_color=MINT_DEEP, marker_line=dict(color="#0B100F", width=.5)))
        h.add_vline(x=med, line_color=INK, line_dash="dash")
        theme.annotate(h, med, 0, f"Median {med:,.0f} L", ax=60, ay=-60)
        h.update_layout(title="Price distribution (lakhs)", bargap=.04)
        a.plotly_chart(theme.style_fig(h, 360, False), width="stretch")
        with a:
            theme.insight("Price", f"The middle 96% of listings sit between {f['Price_in_Lakhs'].quantile(.02):,.0f} and {f['Price_in_Lakhs'].quantile(.98):,.0f} lakhs, with a median of {med:,.0f} lakhs.")
        s = f.sample(min(len(f), 40000), random_state=1)
        r_ = f["Size_in_SqFt"].corr(f["Price_in_Lakhs"])
        d2 = go.Figure(go.Histogram2d(x=s["Size_in_SqFt"], y=s["Price_in_Lakhs"], nbinsx=30, nbinsy=30, colorscale=theme.MINT_SCALE, showscale=False))
        d2.update_layout(title="Size vs price density", xaxis_title="Size (sq ft)", yaxis_title="Price (lakhs)")
        b.plotly_chart(theme.style_fig(d2, 360, False), width="stretch")
        with b:
            theme.insight("Size and price", f"Correlation between size and price is {r_:+.3f}. In this dataset, a larger home does not systematically mean a higher asking price.")

    elif view == "Opportunity Lab":
        st.markdown('<div class="mut" style="max-width:760px">Set your priorities. The weights are normalised to 100%, then applied to percentile ranks for every city in the filtered data.</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        wg = a.slider("Growth priority", 0, 100, 40)
        wq = b.slider("Investment-quality priority", 0, 100, 35)
        wa = c.slider("Affordability priority", 0, 100, 25)
        if wg + wq + wa == 0:
            st.info("Give at least one priority a non-zero weight.")
        else:
            cs = f.groupby("City", observed=True).agg(
                State=("State", "first"), Tier=("City_Tier", "first"),
                ppsf=("Price_per_SqFt_INR", "median"), growth=("Annual_Growth_Rate", "mean"),
                n=("BHK", "size"), good=("Good_Investment", "mean"),
            ).reset_index()
            cs["growth"] *= 100
            cs["good"] *= 100
            total = wg + wq + wa
            cs["score"] = 100 * (
                (wg / total) * cs["growth"].rank(pct=True)
                + (wq / total) * cs["good"].rank(pct=True)
                + (wa / total) * (1 - cs["ppsf"].rank(pct=True))
            )
            st.plotly_chart(_opportunity_mosaic(cs), width="stretch")
            top = cs.nlargest(3, "score")
            theme.insight("Your market lens", f"With these priorities, {', '.join(top['City'])} lead the opportunity mosaic. Change a weight and both the ranking and city areas update immediately.")

    elif view == "Value & Growth":
        cs = f.groupby("City", observed=True).agg(
            ppsf=("Price_per_SqFt_INR", "median"), growth=("Annual_Growth_Rate", "mean"),
            n=("BHK", "size"), good=("Good_Investment", "mean"),
        ).reset_index()
        cs["growth"] *= 100
        cs["score"] = cs["growth"].rank(pct=True) + (1 - cs["ppsf"].rank(pct=True))
        top = cs.nlargest(3, "score")
        sc = go.Figure(go.Scatter(
            x=cs["ppsf"], y=cs["growth"], mode="markers", text=cs["City"],
            marker=dict(size=np.sqrt(cs["n"]) / 3.2, color=cs["good"] * 100, colorscale=theme.MINT_SCALE,
                        line=dict(color=INK, width=1), showscale=True, colorbar=dict(title="Good %", thickness=10)),
            hovertemplate="%{text}<br>₹/sq ft %{x:,.0f}<br>growth %{y:.2f}%<extra></extra>",
        ))
        t0 = top.iloc[0]
        theme.annotate(sc, t0["ppsf"], t0["growth"], f"Best balance: {t0['City']}", ax=-70, ay=-50)
        sc.update_layout(title="Entry price vs assumed growth by city · bubble = listings, colour = good-investment rate", xaxis_title="Median price per sq ft (INR)", yaxis_title="Avg assumed growth (% / yr)")
        st.plotly_chart(theme.style_fig(sc, 440, False), width="stretch")
        theme.insight("Value vs growth", f"{', '.join(top['City'])} combine lower entry prices with above-average assumed growth in this filtered view.")
        p = f.pivot_table(index="City_Tier", columns="Property_Type", values="Annual_Growth_Rate", aggfunc="mean", observed=True) * 100
        st.plotly_chart(_heat(p.values, list(p.columns), list(p.index), "Assumed growth (% / yr) by city tier and property type", "{:.2f}%", 300), width="stretch")
        bst = p.stack().idxmax()
        theme.insight("Growth model", f"{bst[1]} in {bst[0]} cities carry the highest assumed rate ({p.loc[bst]:.2f}%). These rates come from project assumptions, not observed sales.")

    elif view == "Cities":
        g = f.groupby("City", observed=True)["Good_Investment"].mean().mul(100).sort_values()
        lol = go.Figure()
        for c_, v_ in g.items():
            lol.add_trace(go.Scatter(x=[g.mean(), v_], y=[c_, c_], mode="lines", line=dict(color="rgba(190,215,203,.20)", width=1), showlegend=False, hoverinfo="skip"))
        lol.add_trace(go.Scatter(x=g.values, y=g.index, mode="markers", marker=dict(size=9, color=[MINT if v >= g.mean() else CHROME for v in g.values]), showlegend=False, hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        lol.add_vline(x=g.mean(), line_dash="dot", line_color=GRAPHITE)
        lol.update_layout(title="Good-investment rate by city · dotted line = filtered average", xaxis_title="% of listings that qualify")
        lol.update_xaxes(range=[max(0, g.min() - 1), min(100, g.max() + 1)])
        st.plotly_chart(theme.style_fig(lol, 820, False), width="stretch")
        theme.insight("Cities", f"City rates range from {g.min():.1f}% to {g.max():.1f}%, around an average of {g.mean():.1f}%. Property-level value remains the stronger screen.")
        sp = f.groupby("State", observed=True)["Price_per_SqFt_INR"].mean()
        dev = ((sp / f["Price_per_SqFt_INR"].mean() - 1) * 100).sort_values()
        dv = go.Figure(go.Bar(x=dev.values, y=dev.index, orientation="h", marker_color=[MINT_DEEP if v >= 0 else CHROME for v in dev.values]))
        dv.update_layout(title="Average price per sq ft vs the filtered average (%)", xaxis_title="Deviation (%)")
        dv.add_vline(x=0, line_color=INK, line_width=1)
        st.plotly_chart(theme.style_fig(dv, 520, False), width="stretch")
        theme.insight("States", f"State-level average price per sq ft differs by at most {dev.abs().max():.1f}% from the filtered mean in either direction.")

    elif view == "Property Mix":
        p = f.pivot_table(index="BHK", columns="Price_band", values="Good_Investment", aggfunc="mean", observed=True) * 100
        st.plotly_chart(_heat(p.values, list(p.columns.astype(str)), [f"{i} BHK" for i in p.index], "Good-investment rate (%) by BHK and price band (lakhs)"), width="stretch")
        bs = p.stack().idxmax()
        theme.insight("BHK and budget", f"The strongest segment is {bs[0]} BHK in the {bs[1]} lakh band ({p.loc[bs]:.0f}% qualify). Larger homes at lower prices earn the BHK and value criteria together.")
        bins = pd.cut(f["Age_of_Property"], [-1, 5, 10, 15, 20, 25, 40])
        ga = f.groupby(bins, observed=True)["Annual_Growth_Rate"].mean() * 100
        ag = go.Figure(go.Scatter(x=[f"{int(i.left) + 1}–{int(i.right)}" for i in ga.index], y=ga.values, mode="lines+markers", line=dict(color=MINT_DEEP, width=3), marker=dict(size=9, color=MINT, line=dict(color=INK, width=1.2))))
        ag.update_layout(title="Assumed growth by building age (% / yr)", xaxis_title="Age (years)")
        st.plotly_chart(theme.style_fig(ag, 340, False), width="stretch")
        theme.insight("Age", f"Assumed growth moves from {ga.values[0]:.2f}% for the newest buildings to {ga.values[-1]:.2f}% for the oldest because the growth formula applies an age penalty.")
        base = f["Price_in_Lakhs"].mean()
        rows = []
        for col in ["Furnished_Status", "Facing", "Owner_Type", "Parking_Space", "Security"]:
            for lvl, v in f.groupby(col, observed=True)["Price_in_Lakhs"].mean().items():
                rows.append((f"{col.replace('_', ' ')}: {lvl}", (v / base - 1) * 100))
        pr_ = pd.DataFrame(rows, columns=["k", "v"]).sort_values("v")
        pf = go.Figure(go.Bar(x=pr_["v"], y=pr_["k"], orientation="h", marker_color=[MINT_DEEP if v >= 0 else CHROME for v in pr_["v"]]))
        pf.add_vline(x=0, line_color=INK, line_width=1)
        pf.update_layout(title="Price premium vs average listing (%)", xaxis_title="Premium (%)")
        st.plotly_chart(theme.style_fig(pf, 520, False), width="stretch")
        theme.insight("Features", f"Across these categorical features, the largest absolute average-price difference is {pr_['v'].abs().max():.1f}%.")

    elif view == "Livability":
        st.plotly_chart(_mobility_sankey(f), width="stretch")
        theme.insight("Livability flow", "The Sankey diagram shows how listings move from transport access to overall infrastructure strength and then to the rule-based verdict. It describes associations built into the dataset and label rule, not causation.")
        bands = pd.cut(f["Amenity_Count"], [-1, 1, 3, 5], labels=["0–1 amenities", "2–3 amenities", "4–5 amenities"])
        ag = f.groupby(bands, observed=True).agg(good=("Good_Investment", "mean"), growth=("Annual_Growth_Rate", "mean"), n=("BHK", "size"))
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=ag.index.astype(str), y=ag["good"] * 100, name="Good-investment rate", marker_color=CHROME, text=[f"{v * 100:.1f}%" for v in ag["good"]], textposition="outside"), secondary_y=False)
        fig.add_trace(go.Scatter(x=ag.index.astype(str), y=ag["growth"] * 100, name="Assumed growth / yr", mode="lines+markers", line=dict(color=MINT, width=3), marker=dict(size=9)), secondary_y=True)
        fig.update_yaxes(title_text="Good investments (%)", rangemode="tozero", secondary_y=False)
        fig.update_yaxes(title_text="Growth (% / yr)", showgrid=False, secondary_y=True)
        fig.update_layout(title="Amenity count · investment quality and assumed growth")
        st.plotly_chart(theme.style_fig(fig, 390), width="stretch")
        theme.insight("Amenities", "Amenity count is useful as a neighbourhood summary, but it is not one of the five direct verdict criteria. Read any difference alongside infrastructure, readiness and price positioning.")

    elif view == "Signals":
        num = ["Price_in_Lakhs", "Price_per_SqFt_INR", "Size_in_SqFt", "BHK", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "Infrastructure_Score", "Amenity_Count", "Good_Investment"]
        cm = f[num].astype(float).corr().round(2)
        lab = [c.replace("_", " ").replace("in Lakhs", "") for c in num]
        hm = go.Figure(go.Heatmap(z=cm.values, x=lab, y=lab, colorscale=[[0, "#C86464"], [.5, "#101614"], [1, MINT_DEEP]], zmin=-1, zmax=1, zmid=0, text=cm.values, texttemplate="%{text}", textfont=dict(size=10)))
        hm.update_layout(title="Correlation matrix")
        hm.update_yaxes(autorange="reversed")
        st.plotly_chart(theme.style_fig(hm, 560, False), width="stretch")
        pairs = cm.where(~np.eye(len(cm), dtype=bool)).abs().stack().sort_values(ascending=False)
        a_, b_ = pairs.index[0]
        theme.insight("Signals", f"The strongest link is {a_.replace('_', ' ')} with {b_.replace('_', ' ')} ({cm.loc[a_, b_]:+.2f}). Structural relationships can dominate a correlation matrix, so do not interpret every strong value as a market driver.")

    elif view == "Data Explorer":
        st.markdown('<div class="mut">Preview the filtered evidence or download it for your own analysis. The export respects every filter above.</div>', unsafe_allow_html=True)
        columns = ["State", "City", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_INR", "Age_of_Property", "Availability_Status", "Infrastructure_Score", "Amenity_Count", "Good_Investment", "Annual_Growth_Rate", "Future_Price_5Y_Lakhs"]
        st.dataframe(f[columns].head(2000), hide_index=True, height=480, width="stretch")
        st.caption(f"Previewing up to 2,000 of {len(f):,} filtered rows.")
        st.download_button("Download filtered data (CSV)", f[columns].to_csv(index=False).encode("utf-8"), "estate_iq_filtered_listings.csv", "text/csv")
    theme.footer()


render()
