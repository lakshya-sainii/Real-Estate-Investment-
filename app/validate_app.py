"""Run from the project root: python app/validate_app.py"""
import json
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))

import pandas as pd
from core import predictor


def main():
    df = pd.read_parquet(APP / "artifacts" / "listings.parquet")
    assert len(df) == 250000, f"Expected 250,000 listings, found {len(df):,}"
    assert df["City"].nunique() == 42, "Expected 42 cities"
    sample = {
        "City": "Pune", "Property_Type": "Apartment", "BHK": 3,
        "Size_in_SqFt": 1800, "Price_in_Lakhs": 120.0, "Year_Built": 2018,
        "Furnished_Status": "Semi-furnished", "Floor_No": 8, "Total_Floors": 15,
        "Nearby_Schools": 8, "Nearby_Hospitals": 6,
        "Public_Transport_Accessibility": "High", "Parking_Space": "Yes",
        "Security": "Yes", "Amenities": "Gym, Pool, Garden", "Facing": "East",
        "Owner_Type": "Builder", "Availability_Status": "Ready_to_Move",
    }
    result = predictor.predict(sample)
    assert 0 <= result["prob"] <= 1
    assert result["future"] > 0
    assert result["met"] in range(6)
    print(json.dumps({
        "listings": len(df),
        "cities": int(df["City"].nunique()),
        "good_investment": result["good"],
        "probability": round(result["prob"], 4),
        "five_year_price_lakhs": round(result["future"], 2),
        "implied_annual_growth_pct": round(result["growth"] * 100, 2),
        "criteria_met": result["met"],
        "status": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
