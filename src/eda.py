"""Step 2 - Exploratory Data Analysis (20 questions from the project brief).

Run:  python src/eda.py
Output: reports/figures/eda_*.png, reports/eda_findings.json
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

sns.set_theme(style="whitegrid", palette="deep", font_scale=0.95)
FIG_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(PROCESSED_PATH)
PPSF = "Price_per_SqFt_INR"
F = {}   # findings


def save(name):
    plt.tight_layout(); plt.savefig(FIG_DIR / f"eda_{name}.png", dpi=130); plt.close()


def corr_p(a, b):
    return round(float(df[a].corr(df[b])), 4)


# ---------- Q1-5 Price & size ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["Price_in_Lakhs"], bins=50, ax=ax[0], color="#2a6f97"); ax[0].set_title("Q1  Distribution of Price (₹ lakhs)")
sns.histplot(df["Size_in_SqFt"], bins=50, ax=ax[1], color="#52b788"); ax[1].set_title("Q2  Distribution of Size (sq ft)")
save("q01_02_price_size_distribution")
F["Q1"] = df["Price_in_Lakhs"].describe().round(2).to_dict()
F["Q2"] = df["Size_in_SqFt"].describe().round(2).to_dict()

plt.figure(figsize=(7, 4.5))
sns.boxplot(data=df, x="Property_Type", y=PPSF, showfliers=False); plt.title("Q3  Price per sq ft (₹) by property type")
save("q03_ppsf_by_type")
F["Q3"] = df.groupby("Property_Type")[PPSF].agg(["mean", "median"]).round(0).to_dict()

s = df.sample(20000, random_state=1)
plt.figure(figsize=(7, 4.5))
sns.scatterplot(data=s, x="Size_in_SqFt", y="Price_in_Lakhs", alpha=.25, s=8); plt.title("Q4  Size vs Price")
save("q04_size_vs_price")
F["Q4"] = {"pearson_size_price": corr_p("Size_in_SqFt", "Price_in_Lakhs"),
           "pearson_size_ppsf": corr_p("Size_in_SqFt", PPSF)}

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(x=df[PPSF], ax=ax[0], color="#e07a5f"); ax[0].set_title("Q5  Price per sq ft (₹) - outliers")
sns.boxplot(x=df["Size_in_SqFt"], ax=ax[1], color="#81b29a"); ax[1].set_title("Q5  Size (sq ft) - outliers")
save("q05_outliers")
F["Q5"] = {"ppsf_outliers_iqr": int(df["Outlier_PPSF"].sum()), "ppsf_outlier_pct": round(100 * df["Outlier_PPSF"].mean(), 2),
           "size_outliers": int(df["Outlier_Size"].sum()), "ppsf_p50": round(df[PPSF].median()), "ppsf_p99": round(df[PPSF].quantile(.99)),
           "ppsf_max": round(df[PPSF].max())}

# ---------- Q6-10 Location ----------
st = df.groupby("State")[PPSF].mean().sort_values()
plt.figure(figsize=(8, 6)); st.plot.barh(color="#2a6f97"); plt.title("Q6  Average price per sq ft (₹) by state"); plt.xlabel("₹ / sq ft")
save("q06_ppsf_by_state")
F["Q6"] = {"highest": [st.index[-1], round(st.iloc[-1])], "lowest": [st.index[0], round(st.iloc[0])], "spread_pct": round(100 * (st.max() / st.min() - 1), 2)}

ct = df.groupby("City")["Price_in_Lakhs"].mean().sort_values()
plt.figure(figsize=(8, 9)); ct.plot.barh(color="#52b788"); plt.title("Q7  Average price (₹ lakhs) by city"); plt.xlabel("₹ lakhs")
save("q07_price_by_city")
F["Q7"] = {"highest": [ct.index[-1], round(ct.iloc[-1], 1)], "lowest": [ct.index[0], round(ct.iloc[0], 1)], "spread_pct": round(100 * (ct.max() / ct.min() - 1), 2)}

la = df.groupby("Locality")["Age_of_Property"].median()
plt.figure(figsize=(7, 4)); sns.histplot(la, bins=15, color="#9c6644"); plt.title("Q8  Median property age by locality (500 localities)"); plt.xlabel("Median age (years)")
save("q08_median_age_by_locality")
F["Q8"] = {"min": float(la.min()), "max": float(la.max()), "median": float(la.median()),
           "top5_oldest": la.sort_values(ascending=False).head(5).to_dict()}

bc = pd.crosstab(df["City"], df["BHK"], normalize="index") * 100
plt.figure(figsize=(9, 9)); sns.heatmap(bc, annot=False, cmap="YlGnBu", cbar_kws={"label": "% of listings"}); plt.title("Q9  BHK mix by city (% of city listings)")
save("q09_bhk_by_city")
F["Q9"] = {"overall_bhk_pct": (df["BHK"].value_counts(normalize=True).sort_index() * 100).round(1).to_dict(),
           "max_city_deviation_pp": round(float((bc - bc.mean()).abs().max().max()), 2)}

loc_top = df.groupby("Locality")[PPSF].mean().nlargest(5).index.tolist()
d5 = df[df["Locality"].isin(loc_top)].copy()
d5["Build_Period"] = (d5["Year_Built"] // 5) * 5
tr = d5.groupby(["Build_Period", "Locality"])[PPSF].mean().reset_index()
plt.figure(figsize=(8, 4.5)); sns.lineplot(data=tr, x="Build_Period", y=PPSF, hue="Locality", marker="o")
plt.title("Q10  Price per sq ft trend by construction period - top 5 localities"); plt.xlabel("Year built (5-yr bins)")
save("q10_top5_locality_trend")
F["Q10"] = {"top5_localities": loc_top, "avg_ppsf": df[df["Locality"].isin(loc_top)].groupby("Locality")[PPSF].mean().round(0).to_dict(),
            "top5_vs_overall_gap_pct": round(100 * (df[df["Locality"].isin(loc_top)][PPSF].mean() / df[PPSF].mean() - 1), 2)}

# ---------- Q11-15 Relationships ----------
num = ["Price_in_Lakhs", PPSF, "Size_in_SqFt", "BHK", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "Floor_No", "Total_Floors", "Infrastructure_Score", "Amenity_Count"]
cm = df[num].corr()
plt.figure(figsize=(9, 7.5)); sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7}); plt.title("Q11  Correlation matrix of numeric features")
save("q11_correlation_matrix")
off = cm.where(~np.eye(len(cm), dtype=bool)).abs().stack().sort_values(ascending=False)
F["Q11"] = {"top_pairs": {f"{a} | {b}": round(float(cm.loc[a, b]), 3) for (a, b) in off.index[:6:2]},
            "price_vs_ppsf": round(float(cm.loc["Price_in_Lakhs", PPSF]), 3),
            "size_vs_ppsf": round(float(cm.loc["Size_in_SqFt", PPSF]), 3)}

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(data=df, x="Nearby_Schools", y=PPSF, ax=ax[0], color="#2a6f97", errorbar=None); ax[0].set_title("Q12  Price per sq ft by nearby schools")
sns.barplot(data=df, x="Nearby_Hospitals", y=PPSF, ax=ax[1], color="#e07a5f", errorbar=None); ax[1].set_title("Q13  Price per sq ft by nearby hospitals")
save("q12_13_schools_hospitals_vs_ppsf")
F["Q12"] = {"pearson_schools_ppsf": corr_p("Nearby_Schools", PPSF)}
F["Q13"] = {"pearson_hospitals_ppsf": corr_p("Nearby_Hospitals", PPSF)}

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(data=df, x="Furnished_Status", y="Price_in_Lakhs", ax=ax[0], showfliers=False); ax[0].set_title("Q14  Price (₹ lakhs) by furnished status")
sns.barplot(data=df, x="Facing", y=PPSF, ax=ax[1], errorbar=None, color="#81b29a"); ax[1].set_title("Q15  Price per sq ft (₹) by facing")
save("q14_15_furnishing_facing")
F["Q14"] = df.groupby("Furnished_Status")["Price_in_Lakhs"].agg(["mean", "median"]).round(1).to_dict()
F["Q15"] = df.groupby("Facing")[PPSF].mean().round(0).to_dict()

# ---------- Q16-20 Investment / amenities / ownership ----------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
df["Owner_Type"].value_counts().plot.bar(ax=ax[0], color="#2a6f97", rot=0); ax[0].set_title("Q16  Properties by owner type")
df["Availability_Status"].value_counts().plot.bar(ax=ax[1], color="#52b788", rot=0); ax[1].set_title("Q17  Properties by availability status")
for a in ax:
    for p in a.patches: a.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=8)
save("q16_17_owner_availability")
F["Q16"] = df["Owner_Type"].value_counts().to_dict()
F["Q17"] = df["Availability_Status"].value_counts().to_dict()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(data=df, x="Parking_Space", y="Price_in_Lakhs", ax=ax[0], errorbar=None, color="#9c6644"); ax[0].set_title("Q18  Avg price (₹ lakhs) by parking")
am = pd.DataFrame({a: [df.loc[df[f"Has_{a}"] == 1, PPSF].mean(), df.loc[df[f"Has_{a}"] == 0, PPSF].mean()] for a in ["Gym", "Pool", "Clubhouse", "Playground", "Garden"]}, index=["With", "Without"]).T
am.plot.bar(ax=ax[1], rot=0); ax[1].set_title("Q19  Avg price per sq ft (₹) with vs without amenity")
save("q18_19_parking_amenities")
F["Q18"] = df.groupby("Parking_Space")["Price_in_Lakhs"].mean().round(2).to_dict()
F["Q19"] = {"ppsf_with_vs_without_pct": ((am["With"] / am["Without"] - 1) * 100).round(2).to_dict(),
            "amenity_count_ppsf_corr": corr_p("Amenity_Count", PPSF)}

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(data=df, x="Public_Transport_Accessibility", y=PPSF, order=["Low", "Medium", "High"], ax=ax[0], errorbar=None, color="#2a6f97"); ax[0].set_title("Q20  Price per sq ft (₹) by transport access")
gi = df.groupby("Public_Transport_Accessibility")["Good_Investment"].mean().reindex(["Low", "Medium", "High"]) * 100
gi.plot.bar(ax=ax[1], rot=0, color="#52b788"); ax[1].set_title("Q20  Good-investment rate (%) by transport access")
save("q20_transport")
F["Q20"] = {"ppsf": df.groupby("Public_Transport_Accessibility")[PPSF].mean().round(0).to_dict(), "good_investment_pct": gi.round(2).to_dict()}

# ---------- Brief-listed analyses: city price trend, area vs return, crime proxy, infrastructure vs resale ----------
cg = df.groupby("City_Tier")[["Annual_Growth_Rate"]].mean().mul(100).round(2)
F["Tier_growth_pct"] = cg["Annual_Growth_Rate"].to_dict()
F["GoodInv_rate_by_city_top5"] = (df.groupby("City")["Good_Investment"].mean().nlargest(5) * 100).round(1).to_dict()
F["GoodInv_rate_overall_pct"] = round(100 * df["Good_Investment"].mean(), 2)
F["Area_vs_good_investment_corr"] = corr_p("Size_in_SqFt", "Good_Investment")
F["Infra_vs_future_price_corr"] = corr_p("Infrastructure_Score", "Future_Price_5Y_Lakhs")
F["Infra_vs_ppsf_corr"] = corr_p("Infrastructure_Score", PPSF)
F["Area_vs_growth_corr"] = corr_p("Size_in_SqFt", "Annual_Growth_Rate")
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
(df.groupby("City")["Good_Investment"].mean().sort_values().mul(100)).plot.barh(ax=ax[0], color="#2a6f97", fontsize=6); ax[0].set_title("Good-investment rate (%) by city")
df.assign(b=pd.cut(df["Size_in_SqFt"], 8)).groupby("b", observed=True)["Good_Investment"].mean().mul(100).plot.bar(ax=ax[1], color="#52b788", rot=45, fontsize=7); ax[1].set_title("Good-investment rate (%) by size band")
df.assign(b=pd.cut(df["Infrastructure_Score"], 6)).groupby("b", observed=True)["Future_Price_5Y_Lakhs"].mean().plot.bar(ax=ax[2], color="#e07a5f", rot=45, fontsize=7); ax[2].set_title("Avg 5-yr future price (₹ lakhs) by infrastructure score")
save("investment_drivers")

# Crime-rate: not present in dataset -> documented limitation
F["Crime_rate_note"] = "No crime-rate column exists in india_housing_prices.csv; analysis not possible. Infrastructure proxy used instead."

# Key insight: are features informative about price at all?
X = df[["Size_in_SqFt", "BHK", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "Floor_No", "Amenity_Count", "Transport_Score"]]
F["Max_abs_corr_with_price"] = round(float(X.corrwith(df["Price_in_Lakhs"]).abs().max()), 4)
F["Locality_repeats_across_cities"] = int(df.groupby("Locality")["City"].nunique().median())

(REPORTS_DIR / "eda_findings.json").write_text(json.dumps(F, indent=2, default=str))
print(json.dumps(F, indent=1, default=str)[:6000])
