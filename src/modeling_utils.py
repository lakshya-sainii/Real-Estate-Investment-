"""Shared helpers for model training: feature sets, splits, pipelines, plots, lookups."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from config import *

NUM_FEATURES = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_INR", "Age_of_Property",
    "Floor_No", "Total_Floors", "Floor_Ratio", "Nearby_Schools", "Nearby_Hospitals",
    "Transport_Score", "School_Density_Score", "Hospital_Density_Score", "Infrastructure_Score",
    "Amenity_Count", "Has_Gym", "Has_Pool", "Has_Clubhouse", "Has_Playground", "Has_Garden",
    "Has_Parking", "Has_Security", "Is_Ready_To_Move", "SqFt_per_BHK",
    "City_Median_PPSF", "City_Median_Price", "PPSF_vs_City_Median"]
CAT_FEATURES = ["Property_Type", "Furnished_Status", "Facing", "Owner_Type", "City", "City_Tier"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
# Market-context columns that mirror the label rules most directly (used in the ablation study)
CONTEXT_FEATURES = ["City_Median_PPSF", "City_Median_Price", "PPSF_vs_City_Median"]

sns.set_theme(style="whitegrid", font_scale=0.95)


def load_data():
    return pd.read_csv(PROCESSED_PATH)


def split_data(df, target, stratify=False, seed=RANDOM_STATE):
    """70 / 15 / 15 train-validation-test split. Validation picks the model; test is reported once."""
    strat = df[target] if stratify else None
    tr, tmp = train_test_split(df, test_size=0.30, random_state=seed, stratify=strat)
    strat2 = tmp[target] if stratify else None
    va, te = train_test_split(tmp, test_size=0.50, random_state=seed, stratify=strat2)
    return tr, va, te


def make_preprocessor(num_cols=None, cat_cols=None):
    num_cols = NUM_FEATURES if num_cols is None else num_cols
    cat_cols = CAT_FEATURES if cat_cols is None else cat_cols
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),                      # feature scaling
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),  # encoding
    ])


def make_pipeline(model, num_cols=None, cat_cols=None):
    return Pipeline([("prep", make_preprocessor(num_cols, cat_cols)), ("model", model)])


def save_lookups(df):
    """Market lookup tables needed to build features for a single new property at inference time."""
    lk = {
        "city_median_ppsf": df.groupby("City")["Price_per_SqFt_INR"].median().to_dict(),
        "city_median_price": df.groupby("City")["Price_in_Lakhs"].median().to_dict(),
        "city_tier": df.groupby("City")["City_Tier"].first().to_dict(),
        "city_state": df.groupby("City")["State"].first().to_dict(),
        "max_schools": int(df["Nearby_Schools"].max()),
        "max_hospitals": int(df["Nearby_Hospitals"].max()),
        "reference_year": REFERENCE_YEAR,
    }
    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "lookups.json").write_text(json.dumps(lk, indent=2))
    return lk


def grouped_permutation_importance(pipe, X, y, scoring, n=15000, repeats=3, seed=RANDOM_STATE):
    """Permutation importance on the original (pre-encoding) columns."""
    idx = X.sample(min(n, len(X)), random_state=seed).index
    r = permutation_importance(pipe, X.loc[idx], y.loc[idx], scoring=scoring, n_repeats=repeats,
                               random_state=seed, n_jobs=1)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


def plot_importance(imp, title, path, top=15):
    s = imp.head(top)[::-1]
    plt.figure(figsize=(7.5, 5.2)); plt.barh(s.index, s.values, color="#2a6f97")
    plt.title(title); plt.xlabel("Permutation importance (drop in score)")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
