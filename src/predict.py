"""Inference helper - used from the command line now and by the Streamlit app later.

    from predict import predict_property
    predict_property({...})   # see example in __main__
"""
import json
import joblib
import numpy as np
import pandas as pd
from config import *

_clf = _reg = _lk = None


def _load():
    global _clf, _reg, _lk
    if _clf is None:
        _clf = joblib.load(MODELS_DIR / "good_investment_classifier.joblib")
        _reg = joblib.load(MODELS_DIR / "future_price_regressor.joblib")
        _lk = json.loads((MODELS_DIR / "lookups.json").read_text())
    return _clf, _reg, _lk


def build_features(p: dict, lk: dict) -> pd.DataFrame:
    """Turn one raw listing (dataset-style fields) into the model feature row."""
    amen = set(a.strip() for a in p.get("Amenities", "").split(",") if a.strip())
    size, price, bhk = p["Size_in_SqFt"], p["Price_in_Lakhs"], p["BHK"]
    age = REFERENCE_YEAR - p["Year_Built"]
    floors = max(p["Total_Floors"], p["Floor_No"])
    ppsf = price * 1e5 / size
    transport = TRANSPORT_MAP[p["Public_Transport_Accessibility"]]
    schools = p["Nearby_Schools"] / lk["max_schools"]; hosp = p["Nearby_Hospitals"] / lk["max_hospitals"]
    city = p["City"]
    r = {
        "BHK": bhk, "Size_in_SqFt": size, "Price_in_Lakhs": price, "Price_per_SqFt_INR": ppsf, "Age_of_Property": age,
        "Floor_No": p["Floor_No"], "Total_Floors": floors, "Floor_Ratio": p["Floor_No"] / floors,
        "Nearby_Schools": p["Nearby_Schools"], "Nearby_Hospitals": p["Nearby_Hospitals"], "Transport_Score": transport,
        "School_Density_Score": schools, "Hospital_Density_Score": hosp, "Infrastructure_Score": (schools + hosp + transport) / 3,
        "Amenity_Count": len(amen), **{f"Has_{a}": int(a in amen) for a in ["Gym", "Pool", "Clubhouse", "Playground", "Garden"]},
        "Has_Parking": int(p["Parking_Space"] == "Yes"), "Has_Security": int(p["Security"] == "Yes"),
        "Is_Ready_To_Move": int(p["Availability_Status"] == "Ready_to_Move"), "SqFt_per_BHK": size / bhk,
        "City_Median_PPSF": lk["city_median_ppsf"][city], "City_Median_Price": lk["city_median_price"][city],
        "PPSF_vs_City_Median": ppsf / lk["city_median_ppsf"][city],
        "Log_Price": np.log1p(price), "Log_Size": np.log1p(size),
        "Property_Type": p["Property_Type"], "Furnished_Status": p["Furnished_Status"], "Facing": p["Facing"],
        "Owner_Type": p["Owner_Type"], "City": city, "City_Tier": lk["city_tier"][city]}
    return pd.DataFrame([r])


def predict_property(p: dict) -> dict:
    clf, reg, lk = _load()
    X = build_features(p, lk)
    prob = float(clf.predict_proba(X[clf.feature_names_in_])[0, 1])
    future = float(reg.predict(X[reg.feature_names_in_])[0])
    price = p["Price_in_Lakhs"]
    return {"good_investment": bool(prob >= 0.5), "confidence": prob if prob >= 0.5 else 1 - prob, "good_investment_probability": prob,
            "current_price_lakhs": price, "future_price_5y_lakhs": future,
            "implied_annual_growth_pct": ((future / price) ** (1 / FORECAST_YEARS) - 1) * 100,
            "expected_gain_lakhs": future - price}


if __name__ == "__main__":
    example = {"City": "Pune", "Property_Type": "Apartment", "BHK": 3, "Size_in_SqFt": 1800, "Price_in_Lakhs": 120,
               "Year_Built": 2018, "Furnished_Status": "Semi-furnished", "Floor_No": 8, "Total_Floors": 15,
               "Nearby_Schools": 8, "Nearby_Hospitals": 6, "Public_Transport_Accessibility": "High", "Parking_Space": "Yes",
               "Security": "Yes", "Amenities": "Gym, Pool, Garden", "Facing": "East", "Owner_Type": "Builder",
               "Availability_Status": "Ready_to_Move"}
    print(json.dumps(predict_property(example), indent=2))
