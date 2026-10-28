"""Cached data access for the app."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from .geo import CITY_COORDS

ART = Path(__file__).resolve().parents[1] / "artifacts"


@st.cache_data(show_spinner="Loading market data...")
def listings() -> pd.DataFrame:
    return pd.read_parquet(ART / "listings.parquet")


@st.cache_data
def city_stats() -> pd.DataFrame:
    df = listings()
    g = df.groupby("City", observed=True)
    out = pd.DataFrame({
        "State": g["State"].first().astype(str), "Tier": g["City_Tier"].first().astype(str), "Listings": g.size(),
        "Avg price (lakhs)": g["Price_in_Lakhs"].mean(), "Median price/sq ft (INR)": g["Price_per_SqFt_INR"].median(),
        "Good-investment rate (%)": g["Good_Investment"].mean() * 100, "Avg annual growth (%)": g["Annual_Growth_Rate"].mean() * 100,
        "Avg 5-yr price (lakhs)": g["Future_Price_5Y_Lakhs"].mean()}).reset_index()
    out["lat"] = out["City"].map(lambda c: CITY_COORDS[c][0]); out["lon"] = out["City"].map(lambda c: CITY_COORDS[c][1])
    return out


@st.cache_data
def metrics() -> dict:
    return {"clf": json.loads((ART / "classification_metrics.json").read_text()), "reg": json.loads((ART / "regression_metrics.json").read_text()),
            "dq": json.loads((ART / "data_quality_report.json").read_text()), "eda": json.loads((ART / "eda_findings.json").read_text())}


def fig_path(name: str) -> Path:
    return ART / "figures" / name


@st.cache_data(show_spinner=False)
def enriched() -> pd.DataFrame:
    """Listings plus helper columns used by the business questions."""
    df = listings().copy()
    med = df.groupby("City", observed=True)["Price_per_SqFt_INR"].transform("median")
    df["PPSF_ratio"] = df["Price_per_SqFt_INR"] / med
    df["Price_band"] = pd.cut(df["Price_in_Lakhs"], [0, 100, 200, 300, 400, 500], labels=["<100", "100-200", "200-300", "300-400", "400-500"])
    df["Gain_pct"] = (df["Future_Price_5Y_Lakhs"] / df["Price_in_Lakhs"] - 1) * 100
    return df
