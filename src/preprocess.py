"""Step 1 - Data preprocessing, feature engineering and target creation.

Run:  python src/preprocess.py
Output: data/processed/india_housing_cleaned.csv  +  reports/data_quality_report.json
"""
import json
import numpy as np
import pandas as pd
from config import *


def city_tier(city: str) -> str:
    return "Tier-1" if city in TIER1 else ("Tier-2" if city in TIER2 else "Tier-3")


def main():
    df = pd.read_csv(RAW_PATH)
    qa = {"raw_rows": len(df), "raw_columns": df.shape[1]}

    # ---------------- 1. Cleaning ----------------
    qa["missing_values_before"] = int(df.isna().sum().sum())
    for c in df.select_dtypes("number"):                       # robust defaults
        df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes(exclude="number"):
        df[c] = df[c].fillna(df[c].mode().iloc[0])
    for c in ["State", "City", "Locality", "Property_Type", "Furnished_Status", "Facing"]:
        df[c] = df[c].str.strip()
    qa["duplicate_rows_removed"] = int(df.duplicated(subset=df.columns.drop("ID")).sum())
    df = df.drop_duplicates(subset=df.columns.drop("ID")).reset_index(drop=True)

    # Age consistency
    age_calc = REFERENCE_YEAR - df["Year_Built"]
    qa["age_inconsistent_rows"] = int((age_calc != df["Age_of_Property"]).sum())
    df["Age_of_Property"] = age_calc

    # Floor logic: Floor_No cannot exceed Total_Floors
    bad_floor = df["Floor_No"] > df["Total_Floors"]
    qa["floor_gt_total_rows"] = int(bad_floor.sum())
    qa["floor_gt_total_pct"] = round(100 * bad_floor.mean(), 2)
    df["Floor_Data_Corrected"] = bad_floor.astype(int)
    df["Total_Floors"] = np.maximum(df["Total_Floors"], df["Floor_No"])

    # Price per sq ft recomputed in INR (source column is rounded to 2 decimals of lakhs)
    df["Price_per_SqFt_INR"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]

    # ---------------- 2. Outlier detection (IQR, flagged not dropped) ----------------
    def iqr_flag(s, k=1.5):
        q1, q3 = s.quantile([.25, .75]); i = q3 - q1
        return ((s < q1 - k * i) | (s > q3 + k * i)).astype(int)
    df["Outlier_PPSF"] = iqr_flag(df["Price_per_SqFt_INR"])
    df["Outlier_Size"] = iqr_flag(df["Size_in_SqFt"])
    df["Outlier_Price"] = iqr_flag(df["Price_in_Lakhs"])
    qa["outliers"] = {c: int(df[c].sum()) for c in ["Outlier_PPSF", "Outlier_Size", "Outlier_Price"]}

    # ---------------- 3. Feature engineering ----------------
    df["City_Tier"] = df["City"].map(city_tier)
    amen = df["Amenities"].fillna("").str.split(r",\s*")
    for a in ["Gym", "Pool", "Clubhouse", "Playground", "Garden"]:
        df[f"Has_{a}"] = amen.apply(lambda x: int(a in x))
    df["Amenity_Count"] = amen.apply(lambda x: len([i for i in x if i]))
    df["Is_Ready_To_Move"] = (df["Availability_Status"] == "Ready_to_Move").astype(int)
    df["Has_Parking"] = (df["Parking_Space"] == "Yes").astype(int)
    df["Has_Security"] = (df["Security"] == "Yes").astype(int)
    df["Transport_Score"] = df["Public_Transport_Accessibility"].map(TRANSPORT_MAP)
    df["School_Density_Score"] = df["Nearby_Schools"] / df["Nearby_Schools"].max()
    df["Hospital_Density_Score"] = df["Nearby_Hospitals"] / df["Nearby_Hospitals"].max()
    df["Infrastructure_Score"] = (df["School_Density_Score"] + df["Hospital_Density_Score"]
                                  + df["Transport_Score"]) / 3
    df["Floor_Ratio"] = df["Floor_No"] / df["Total_Floors"]
    df["SqFt_per_BHK"] = df["Size_in_SqFt"] / df["BHK"]
    df["Log_Price"] = np.log1p(df["Price_in_Lakhs"])
    df["Log_Size"] = np.log1p(df["Size_in_SqFt"])

    # Relative-value features (market context)
    df["City_Median_PPSF"] = df.groupby("City")["Price_per_SqFt_INR"].transform("median")
    df["City_Median_Price"] = df.groupby("City")["Price_in_Lakhs"].transform("median")
    df["PPSF_vs_City_Median"] = df["Price_per_SqFt_INR"] / df["City_Median_PPSF"]

    # ---------------- 4. Target 1: Good_Investment (multi-factor score) ----------------
    # Five equally weighted criteria; a property is a "Good Investment" if it meets >= 3.
    df["Crit_Value_PPSF"] = (df["Price_per_SqFt_INR"] <= df["City_Median_PPSF"]).astype(int)
    df["Crit_Value_Price"] = (df["Price_in_Lakhs"] <= df["City_Median_Price"]).astype(int)
    df["Crit_Family_Size"] = (df["BHK"] >= 3).astype(int)
    df["Crit_Ready"] = df["Is_Ready_To_Move"]
    df["Crit_Infrastructure"] = (df["Infrastructure_Score"] >= 0.5).astype(int)
    crit = [c for c in df.columns if c.startswith("Crit_")]
    df["Investment_Score"] = df[crit].sum(axis=1)
    df["Good_Investment"] = (df["Investment_Score"] >= 3).astype(int)

    # ---------------- 5. Target 2: Future price after 5 years ----------------
    r = (BASE_GROWTH
         + df["City_Tier"].map(TIER_ADJ)
         + df["Property_Type"].map(TYPE_ADJ)
         + df["Availability_Status"].map(STATUS_ADJ)
         + INFRA_MAX_ADJ * df["Infrastructure_Score"]
         + AGE_ADJ_PER_YEAR * df["Age_of_Property"]).clip(GROWTH_MIN, GROWTH_MAX)
    df["Annual_Growth_Rate"] = r
    df["Future_Price_5Y_Lakhs"] = df["Price_in_Lakhs"] * (1 + r) ** FORECAST_YEARS
    df["Fixed8_Future_Price_Lakhs"] = df["Price_in_Lakhs"] * (1 + FIXED_RATE_BASELINE) ** FORECAST_YEARS

    qa["good_investment_rate"] = round(float(df["Good_Investment"].mean()), 4)
    qa["growth_rate_summary"] = df["Annual_Growth_Rate"].describe().round(4).to_dict()
    qa["final_rows"] = len(df); qa["final_columns"] = df.shape[1]

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "data_quality_report.json").write_text(json.dumps(qa, indent=2))
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
