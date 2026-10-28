"""Prepare lightweight, self-contained artifacts for the Streamlit app (app/artifacts)."""
import json, shutil
import pandas as pd
from config import *

ART = ROOT / "app" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)
cols = ["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_INR", "Year_Built",
        "Age_of_Property", "Furnished_Status", "Facing", "Owner_Type", "Availability_Status", "Public_Transport_Accessibility",
        "Parking_Space", "Security", "Nearby_Schools", "Nearby_Hospitals", "Amenity_Count", "Infrastructure_Score", "City_Tier",
        "Good_Investment", "Annual_Growth_Rate", "Future_Price_5Y_Lakhs"]
df = pd.read_csv(PROCESSED_PATH, usecols=cols)
for c in ["State", "City", "Locality", "Property_Type", "Furnished_Status", "Facing", "Owner_Type", "Availability_Status",
          "Public_Transport_Accessibility", "Parking_Space", "Security", "City_Tier"]:
    df[c] = df[c].astype("category")
df.to_parquet(ART / "listings.parquet", index=False, compression="zstd")
# The portable classifier is produced by export_portable_classifier.py; do not
# overwrite it with the XGBoost research artifact.
for f in ["future_price_regressor.joblib", "lookups.json"]:
    shutil.copy(MODELS_DIR / f, ART / f)
for f in ["classification_metrics.json", "regression_metrics.json", "data_quality_report.json", "eda_findings.json"]:
    shutil.copy(REPORTS_DIR / f, ART / f)
for f in ["clf_confusion_roc.png", "clf_feature_importance.png", "reg_evaluation.png", "reg_feature_importance.png"]:
    shutil.copy(FIG_DIR / f, ART / "figures" / f)
print("rows", len(df), "| parquet MB", round((ART / "listings.parquet").stat().st_size / 1e6, 1))
