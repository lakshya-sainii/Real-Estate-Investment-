"""Shared configuration for the Real Estate Investment Advisor project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "india_housing_prices.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "india_housing_cleaned.csv"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"
MLFLOW_URI = f"sqlite:///{(ROOT / 'mlflow' / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACTS = (ROOT / "mlflow" / "artifacts").as_posix()

RANDOM_STATE = 42
REFERENCE_YEAR = 2025          # dataset's Age_of_Property = 2025 - Year_Built
FORECAST_YEARS = 5

# ---- Growth-rate assumptions for the 5-year regression target -------------------
# The dataset is a single snapshot (no historical prices), so the 5-year target is
# simulated with a transparent, feature-based annual growth rate r:
#   r = BASE + tier_adj + type_adj + infra_adj + status_adj + age_adj   (clipped)
#   Future_Price = Price * (1 + r) ** 5
BASE_GROWTH = 0.060
TIER1 = ["Mumbai", "New Delhi", "Bangalore", "Chennai", "Hyderabad",
         "Kolkata", "Pune", "Ahmedabad", "Gurgaon", "Noida"]
TIER2 = ["Jaipur", "Lucknow", "Kochi", "Indore", "Surat", "Nagpur", "Bhubaneswar",
         "Coimbatore", "Vishakhapatnam", "Patna", "Bhopal", "Mysore", "Dwarka",
         "Faridabad", "Guwahati", "Trivandrum", "Dehradun", "Raipur", "Vijayawada"]
TIER_ADJ = {"Tier-1": 0.010, "Tier-2": 0.015, "Tier-3": 0.005}
TYPE_ADJ = {"Villa": 0.005, "Independent House": 0.0025, "Apartment": 0.0}
STATUS_ADJ = {"Under_Construction": 0.005, "Ready_to_Move": 0.0}
INFRA_MAX_ADJ = 0.020          # infrastructure score (0-1) adds up to +2.0pp
AGE_ADJ_PER_YEAR = -0.0003     # older stock appreciates a bit slower
GROWTH_MIN, GROWTH_MAX = 0.04, 0.14
FIXED_RATE_BASELINE = 0.08     # brief's "fixed rate" reference (r = 8%)

TRANSPORT_MAP = {"Low": 0.0, "Medium": 0.5, "High": 1.0}
