"""Model loading and single-property prediction (same feature logic as src/predict.py)."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ART = Path(__file__).resolve().parents[1] / "artifacts"
REFERENCE_YEAR, YEARS = 2025, 5
TRANSPORT = {"Low": 0.0, "Medium": 0.5, "High": 1.0}


class ModelLoadError(RuntimeError):
    """Raised when model artifacts cannot be deserialised by the local runtime."""


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """Load portable scikit-learn inference artifacts.

    The app deliberately avoids XGBoost deserialisation, preventing the legacy
    booster 'input stream corrupted' failure across XGBoost installations.
    """
    try:
        clf = joblib.load(ART / "good_investment_classifier.joblib")
        reg = joblib.load(ART / "future_price_regressor.joblib")
        lk = json.loads((ART / "lookups.json").read_text(encoding="utf-8"))
        return clf, reg, lk
    except Exception as exc:
        raise ModelLoadError(
            "The saved models are incompatible with the installed scikit-learn runtime. "
            "Reinstall the project requirements, then restart Streamlit."
        ) from exc


def _features(p: dict, lk: dict) -> pd.DataFrame:
    amen = {a.strip() for a in p.get("Amenities", "").split(",") if a.strip()}
    size, price, bhk = p["Size_in_SqFt"], p["Price_in_Lakhs"], p["BHK"]
    floors = max(p["Total_Floors"], p["Floor_No"]); ppsf = price * 1e5 / size
    tr = TRANSPORT[p["Public_Transport_Accessibility"]]; sc = p["Nearby_Schools"] / lk["max_schools"]; ho = p["Nearby_Hospitals"] / lk["max_hospitals"]; c = p["City"]
    row = {"BHK": bhk, "Size_in_SqFt": size, "Price_in_Lakhs": price, "Price_per_SqFt_INR": ppsf, "Age_of_Property": REFERENCE_YEAR - p["Year_Built"],
           "Floor_No": p["Floor_No"], "Total_Floors": floors, "Floor_Ratio": p["Floor_No"] / floors, "Nearby_Schools": p["Nearby_Schools"],
           "Nearby_Hospitals": p["Nearby_Hospitals"], "Transport_Score": tr, "School_Density_Score": sc, "Hospital_Density_Score": ho,
           "Infrastructure_Score": (sc + ho + tr) / 3, "Amenity_Count": len(amen),
           **{f"Has_{a}": int(a in amen) for a in ["Gym", "Pool", "Clubhouse", "Playground", "Garden"]},
           "Has_Parking": int(p["Parking_Space"] == "Yes"), "Has_Security": int(p["Security"] == "Yes"),
           "Is_Ready_To_Move": int(p["Availability_Status"] == "Ready_to_Move"), "SqFt_per_BHK": size / bhk,
           "City_Median_PPSF": lk["city_median_ppsf"][c], "City_Median_Price": lk["city_median_price"][c],
           "PPSF_vs_City_Median": ppsf / lk["city_median_ppsf"][c], "Price_vs_City_Median": price / lk["city_median_price"][c],
           "Log_Price": np.log1p(price), "Log_Size": np.log1p(size),
           "Property_Type": p["Property_Type"], "Furnished_Status": p["Furnished_Status"], "Facing": p["Facing"], "Owner_Type": p["Owner_Type"],
           "City": c, "City_Tier": lk["city_tier"][c]}
    return pd.DataFrame([row])


def predict(p: dict) -> dict:
    clf, reg, lk = load_models()
    X = _features(p, lk); r = X.iloc[0]
    prob = float(clf.predict_proba(X[clf.feature_names_in_])[0, 1])
    future = float(reg.predict(X[reg.feature_names_in_])[0]); price = p["Price_in_Lakhs"]
    growth = (future / price) ** (1 / YEARS) - 1
    criteria = [
        ("Price per sq ft at or below city median", bool(r["PPSF_vs_City_Median"] <= 1), f"INR {r['Price_per_SqFt_INR']:,.0f} vs city median INR {r['City_Median_PPSF']:,.0f}"),
        ("Total price at or below city median", bool(price <= r["City_Median_Price"]), f"{price:,.1f} vs city median {r['City_Median_Price']:,.1f} lakhs"),
        ("3 BHK or larger", bool(p["BHK"] >= 3), f"{p['BHK']} BHK"),
        ("Ready to move", bool(p["Availability_Status"] == "Ready_to_Move"), p["Availability_Status"].replace("_", " ")),
        ("Infrastructure score of 0.5 or more", bool(r["Infrastructure_Score"] >= 0.5), f"score {r['Infrastructure_Score']:.2f} (schools, hospitals, transport)")]
    result = {"good": prob >= 0.5, "prob": prob, "confidence": max(prob, 1 - prob), "price": price, "future": future, "growth": growth,
              "gain": future - price, "criteria": criteria, "met": sum(c[1] for c in criteria), "infra": float(r["Infrastructure_Score"]),
              "ppsf": float(r["Price_per_SqFt_INR"]), "city_ppsf": float(r["City_Median_PPSF"]), "city_price": float(r["City_Median_Price"]),
              "tier": lk["city_tier"][p["City"]], "inputs": p}
    return result
