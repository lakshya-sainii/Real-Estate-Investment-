# Estate IQ — Real Estate Investment Advisor

A complete Indian real-estate analytics project with a reproducible notebook, model-training pipeline, MLflow tracking and a self-contained Streamlit application.

## What is included

- `Real_Estate_Investment_Advisor.ipynb` — preprocessing, feature engineering, 20 EDA questions, classification, regression, MLflow and new-property prediction.
- `app/app.py` — the working Streamlit entry point.
- `app/views/` — Home, Analyse Property, Market Atlas, Insights, AI Advisor, Performance and About.
- `app/artifacts/` — trained models, lookups, metrics, figures and 250,000 compact listing records.
- `src/` — reproducible preprocessing, training, prediction and app-data build scripts.
- `models/`, `reports/`, `mlflow/` — exported models, evaluation output and experiment databases.

## Quick start

Use a fresh virtual environment where possible. From this project root:

### Windows

```bash
py -m pip install -r requirements.txt
py -m streamlit run app/app.py
```

### macOS / Linux

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app/app.py
```

For the lighter app-only installation, use `app/requirements.txt` instead.

## Fixed model-loading error

The previous app could show:

```text
xgboost._c_api.XGBoostError: input stream corrupted
```

This was caused by an XGBoost booster deserialisation incompatibility. The app now uses a portable scikit-learn inference classifier distilled from the same five-criterion investment rule, so it does not load XGBoost at all. The original XGBoost research artifact and held-out metrics remain preserved separately.

After installing the requirements, fully stop and restart Streamlit so the old cached model process is gone.

## New experience

- Premium black/obsidian theme with restrained mint highlights and only subtle glow accents.
- Original home-page composition and Talina/Manrope font system retained.
- Procedural 3D buildings replaced with a premium architectural skyline image.
- One interactive Estate IQ monogram; no emoji-based interface icons.
- Geographic column map replaced by a non-geographic Market Atlas with opportunity ranking, city signatures and radar comparison.
- Insights expanded to eight views, including custom-weight opportunity scores, a connectivity Sankey diagram, amenity analysis and filtered CSV export.

## Retraining

The app runs immediately with the included artifacts. Retraining requires the original raw CSV at:

```text
data/raw/india_housing_prices.csv
```

Then run:

```bash
python src/preprocess.py
python src/eda.py
python src/train_classification.py
python src/train_regression.py
python src/export_portable_classifier.py
python src/build_app_data.py
```

Or use `./run_all.sh` on macOS/Linux.

## Responsible interpretation

The dataset is a single snapshot without historical sale prices. Both targets are rule-derived, and the five-year forecast uses assumed growth rates. High held-out scores demonstrate that the models learned those rules; they are not evidence of real-market forecasting accuracy. Use the app as an analytics and screening demonstration, not as financial advice.
