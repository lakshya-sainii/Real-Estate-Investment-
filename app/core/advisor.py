"""Rule-based 'AI Advisor': understands investment questions (English / Hinglish keywords) and answers
from the project's own data and models. No external API is used."""
import re
import pandas as pd

ALIASES = {"bengaluru": "Bangalore", "bangalore": "Bangalore", "gurugram": "Gurgaon", "vizag": "Vishakhapatnam", "visakhapatnam": "Vishakhapatnam",
           "bombay": "Mumbai", "calcutta": "Kolkata", "madras": "Chennai", "delhi": "New Delhi", "thiruvananthapuram": "Trivandrum",
           "mysuru": "Mysore", "mangaluru": "Mangalore", "cochin": "Kochi", "baroda": "Vadodara"}
TYPES = {"villa": "Villa", "apartment": "Apartment", "flat": "Apartment", "house": "Independent House", "independent": "Independent House"}

SUGGESTIONS = ["Which city is best for investment?", "Compare Pune and Jaipur", "3 BHK under 150 lakh in Hyderabad",
               "How is the 5-year price calculated?", "Ready to move vs under construction", "How accurate is the model?"]


def fmt_price(lakhs: float) -> str:
    return f"₹{lakhs / 100:,.2f} crore" if lakhs >= 100 else f"₹{lakhs:,.1f} lakh"


def parse(q: str, cities: list) -> dict:
    ql = q.lower(); found = []
    for c in sorted(cities, key=len, reverse=True):
        if re.search(rf"\b{re.escape(c.lower())}\b", ql) and c not in found: found.append(c)
    for a, c in ALIASES.items():
        if re.search(rf"\b{a}\b", ql) and c in cities and c not in found: found.append(c)
    # keep order of appearance
    found.sort(key=lambda c: min([ql.find(c.lower())] + [ql.find(a) for a, cc in ALIASES.items() if cc == c and a in ql] if ql.find(c.lower()) >= 0 else [ql.find(a) for a, cc in ALIASES.items() if cc == c and a in ql]))
    bhk = re.search(r"\b([1-5])\s*[- ]?\s*bhk\b", ql)
    budget = re.search(r"(\d+(?:\.\d+)?)\s*(crore|cr|lakhs|lakh|lacs|lac|l)\b", ql)
    val = None
    if budget:
        val = float(budget.group(1)) * (100 if budget.group(2) in ("crore", "cr") else 1)
    mode = "min" if re.search(r"\b(above|over|more than|minimum|se zyada|se upar)\b", ql) else "max"
    ptype = next((v for k, v in TYPES.items() if re.search(rf"\b{k}\b", ql)), None)
    return {"cities": found, "bhk": int(bhk.group(1)) if bhk else None, "budget": val, "budget_mode": mode, "ptype": ptype, "q": ql}


def has(ql, *words):
    return any(w in ql for w in words)


def _city_table(cs: pd.DataFrame) -> str:
    rows = ["| City | Avg price | Median ₹/sq ft | Good-invest. rate | Avg growth / yr |", "|---|---|---|---|---|"]
    for _, r in cs.iterrows():
        rows.append(f"| {r['City']} | {fmt_price(r['Avg price (lakhs)'])} | ₹{r['Median price/sq ft (INR)']:,.0f} | {r['Good-investment rate (%)']:.1f}% | {r['Avg annual growth (%)']:.2f}% |")
    return "\n".join(rows)


def answer(question: str, df: pd.DataFrame, cs: pd.DataFrame, metrics: dict, last_pred: dict | None = None) -> str:
    p = parse(question, cs["City"].tolist()); ql = p["q"]

    if has(ql, "explain", "why", "kyun", "kyu", "reason", "my result", "meri property"):
        if not last_pred:
            return "I can explain a result once you analyse a property. Open the **Predict** page, run an analysis, and ask me again: *“Explain my result”*."
        r = last_pred; met = r["met"]; status = "a **Good Investment**" if r["good"] else "**not a Good Investment**"
        lines = [f"Your {r['inputs']['BHK']} BHK {r['inputs']['Property_Type']} in **{r['inputs']['City']}** is classified as {status} (model confidence {r['confidence'] * 100:.1f}%).",
                 f"It meets **{met} of 5** criteria; a property needs at least 3.", ""]
        lines += [f"- {'Met' if ok else 'Not met'} — {name}: {detail}" for name, ok, detail in r["criteria"]]
        gap = 3 - met
        lines += ["", (f"It is {gap} criterion short of the threshold, so improving the missing items (for example a lower price or a ready-to-move unit) would change the verdict."
                       if gap > 0 else "It clears the threshold, so the verdict is comfortable."),
                  f"Projected value in 5 years: **{fmt_price(r['future'])}** (≈ {r['growth'] * 100:.2f}% a year, {r['tier']} city)."]
        return "\n".join(lines)

    if has(ql, "accura", "model", "algorithm", "xgboost", "reliable", "trust", "mlflow", "kitna sahi"):
        c, g = metrics["clf"], metrics["reg"]; x = c["results"]["XGBoost_Tuned"]; rr = g["results"]["LinearRegression_Ridge"]
        return (f"**Classifier ({c['best_model']}):** accuracy {x['test_accuracy'] * 100:.2f}%, F1 {x['test_f1']:.4f}.  \n"
                f"**Price regressor (Ridge, log target):** RMSE {rr['test_rmse']:.2f} lakhs, MAE {rr['test_mae']:.2f} lakhs.\n\n"
                "Important: both targets come from documented rules (the dataset has no price history), so these scores show the models learned the rules, "
                "not that they can predict real market moves. Treat results as a structured screening aid, not a guarantee.")

    if has(ql, "5 year", "5-year", "5 saal", "five year", "calculated", "calculate", "formula", "growth", "appreciation", "future", "return", "kaise"):
        t = df.groupby("City_Tier", observed=True)["Annual_Growth_Rate"].mean() * 100
        ty = df.groupby("Property_Type", observed=True)["Annual_Growth_Rate"].mean() * 100
        return ("**Future price = Current price × (1 + r)^5**, where the yearly growth rate *r* is feature-based:\n\n"
                "- 6% base, plus city tier (Tier-1 +1.0 pt, Tier-2 +1.5 pt, Tier-3 +0.5 pt)\n- property type (villa +0.5, house +0.25, apartment 0)\n"
                "- under construction +0.5, up to +2.0 for good infrastructure, and a small penalty for older buildings (bounded 4%–14%).\n\n"
                f"Average growth by tier in the data: Tier-1 {t.get('Tier-1', 0):.2f}%, Tier-2 {t.get('Tier-2', 0):.2f}%, Tier-3 {t.get('Tier-3', 0):.2f}%. "
                f"By type: villa {ty.get('Villa', 0):.2f}%, house {ty.get('Independent House', 0):.2f}%, apartment {ty.get('Apartment', 0):.2f}%.\n\n"
                "These rates are assumptions (no historical prices exist in the dataset), so treat the forecast as indicative.")

    if has(ql, "ready", "under construction", "construction"):
        g = df.groupby("Availability_Status", observed=True).agg(n=("Price_in_Lakhs", "size"), good=("Good_Investment", "mean"), growth=("Annual_Growth_Rate", "mean"), price=("Price_in_Lakhs", "median"))
        rm, uc = g.loc["Ready_to_Move"], g.loc["Under_Construction"]
        return (f"**Ready to move:** {rm['good'] * 100:.1f}% are good investments, median price {fmt_price(rm['price'])}, assumed growth {rm['growth'] * 100:.2f}%/yr.  \n"
                f"**Under construction:** {uc['good'] * 100:.1f}% are good investments, median price {fmt_price(uc['price'])}, assumed growth {uc['growth'] * 100:.2f}%/yr.\n\n"
                "Ready-to-move scores higher on the investment rule (no delivery risk); under-construction units get a small growth premium for completion.")

    # ---- budget / property search ----
    if p["budget"] is not None or (p["bhk"] and (has(ql, "budget", "under", "below", "within", "tak"))):
        d = df
        if p["cities"]: d = d[d["City"].isin(p["cities"][:1])]
        if p["bhk"]: d = d[d["BHK"] == p["bhk"]]
        if p["ptype"]: d = d[d["Property_Type"] == p["ptype"]]
        if p["budget"] is not None: d = d[d["Price_in_Lakhs"] <= p["budget"]] if p["budget_mode"] == "max" else d[d["Price_in_Lakhs"] >= p["budget"]]
        label = " ".join(filter(None, [f"{p['bhk']} BHK" if p["bhk"] else None, p["ptype"], f"in {p['cities'][0]}" if p["cities"] else "across India",
                                       (("under " if p["budget_mode"] == "max" else "above ") + fmt_price(p["budget"])) if p["budget"] else None]))
        if d.empty: return f"No listings match **{label}**. Try a higher budget or fewer filters."
        return (f"**{label}**: {len(d):,} matching listings.\n\n"
                f"- Median size {d['Size_in_SqFt'].median():,.0f} sq ft, median price {fmt_price(d['Price_in_Lakhs'].median())} (₹{d['Price_per_SqFt_INR'].median():,.0f}/sq ft)\n"
                f"- **{d['Good_Investment'].mean() * 100:.1f}%** qualify as Good Investments; {d['Is_Ready'] .mean() * 100 if 'Is_Ready' in d else (d['Availability_Status'] == 'Ready_to_Move').mean() * 100:.0f}% are ready to move\n"
                f"- Average assumed growth {d['Annual_Growth_Rate'].mean() * 100:.2f}%/yr → typical 5-year value {fmt_price(d['Future_Price_5Y_Lakhs'].median())}\n\n"
                "Use the **Predict** page to analyse a specific property in this range.")

    # ---- compare ----
    if len(p["cities"]) >= 2 or has(ql, "compare", " vs ", "versus"):
        if len(p["cities"]) < 2: return "Tell me two cities to compare, for example: *Compare Pune and Jaipur*."
        a, b = p["cities"][:2]; t = cs[cs["City"].isin([a, b])].set_index("City").loc[[a, b]].reset_index()
        ga, gb = t.iloc[0], t.iloc[1]
        better_g = a if ga["Avg annual growth (%)"] >= gb["Avg annual growth (%)"] else b
        better_v = a if ga["Good-investment rate (%)"] >= gb["Good-investment rate (%)"] else b
        verdict = (f"**{better_g}** leads on both assumed growth ({max(ga['Avg annual growth (%)'], gb['Avg annual growth (%)']):.2f}%/yr) and good-investment rate. " if better_g == better_v else
                   f"**{better_g}** has the higher assumed growth ({max(ga['Avg annual growth (%)'], gb['Avg annual growth (%)']):.2f}%/yr) and **{better_v}** the higher good-investment rate. ")
        return (_city_table(t) + "\n\n" + verdict +
                "Differences between cities in this dataset are small, so property-level factors (price vs city median, readiness, infrastructure) matter more than the city alone.")

    # ---- best city ----
    if has(ql, "best city", "top city", "which city", "kaun sa city", "kahan", "where should", "best place", "top cities", "invest"):
        s = cs.copy(); s["score"] = s["Avg annual growth (%)"].rank(pct=True) + s["Good-investment rate (%)"].rank(pct=True)
        top = s.sort_values("score", ascending=False).head(5)
        return ("Top 5 cities by combined **assumed growth** and **good-investment rate**:\n\n" + _city_table(top) +
                "\n\nTier-2 cities lead because the growth model gives emerging markets a higher rate. Gaps between cities are small, so compare individual properties too.")

    # ---- single-city info ----
    if p["cities"]:
        c = p["cities"][0]; r = cs[cs["City"] == c].iloc[0]; d = df[df["City"] == c]
        return (f"**{c}** ({r['State']}, {r['Tier']}):\n\n- {int(r['Listings']):,} listings, average price {fmt_price(r['Avg price (lakhs)'])}, median ₹{r['Median price/sq ft (INR)']:,.0f} per sq ft\n"
                f"- {r['Good-investment rate (%)']:.1f}% good-investment rate; assumed growth {r['Avg annual growth (%)']:.2f}%/yr → average 5-year price {fmt_price(r['Avg 5-yr price (lakhs)'])}\n"
                f"- Most common BHK mix is even (≈20% each); {(d['Availability_Status'] == 'Ready_to_Move').mean() * 100:.0f}% ready to move.\n\nTry: *3 BHK under 120 lakh in {c}*.")

    if has(ql, "hello", "hi", "hey", "namaste", "help", "kya kar"):
        return "Hi! I'm your investment advisor for this dataset. Ask about **cities**, **budgets**, **BHK**, **5-year growth**, **ready vs under-construction**, or **explain my result** after a prediction."
    return ("I can answer questions about the project's data and models: city comparisons, budget searches (e.g. *2 BHK under 80 lakh in Pune*), growth forecasts, "
            "readiness, model accuracy, and an explanation of your last prediction. I can't answer unrelated or open-ended questions.")


# ---------------------------------------------------------------------------------------------
from . import business


def reply(question: str, df: pd.DataFrame, cs: pd.DataFrame, metrics: dict, last_pred: dict | None = None) -> dict:
    """Return {'text', 'fig', 'followups'}; business questions come with a chart and follow-up suggestions."""
    ql = question.lower()
    if re.search(r"business question|questions to ask|what should i ask|suggest (a )?question|kya poochu|sawal do", ql):
        return business.listing()
    if re.search(r"explain my|my result|why is my|meri property", ql):
        return {"text": answer(question, df, cs, metrics, last_pred), "fig": None, "followups": ["scenario", "overpriced", "infra"]}
    bid = business.match(ql)
    if bid:
        return business.run(bid, df, cs, metrics, last_pred)
    return {"text": answer(question, df, cs, metrics, last_pred), "fig": None, "followups": ["city_value", "ready_premium", "bhk_budget"]}
