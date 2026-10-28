"""Export a portable inference classifier for the Streamlit app.

The Good_Investment target is a deterministic majority vote over five screening
criteria. This small scikit-learn model is distilled from evenly sampled rule
examples so the app does not need to deserialise an XGBoost booster. The original
XGBoost research artifact and held-out metrics remain separate.

Run from src/: python export_portable_classifier.py
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["PPSF_vs_City_Median", "Price_vs_City_Median", "BHK", "Is_Ready_To_Move", "Infrastructure_Score"]
RNG = np.random.default_rng(42)


def examples(n):
    X = pd.DataFrame({
        "PPSF_vs_City_Median": RNG.uniform(.25, 1.85, n),
        "Price_vs_City_Median": RNG.uniform(.25, 1.85, n),
        "BHK": RNG.integers(1, 6, n),
        "Is_Ready_To_Move": RNG.integers(0, 2, n),
        "Infrastructure_Score": RNG.uniform(0, 1, n),
    })
    # Add dense threshold neighbourhoods to make boundary behaviour precise.
    boundary = min(n // 5, 20000)
    for col, threshold in [("PPSF_vs_City_Median", 1), ("Price_vs_City_Median", 1), ("Infrastructure_Score", .5)]:
        X.loc[X.index[:boundary], col] = np.clip(threshold + RNG.normal(0, .025, boundary), 0, None)
    criteria = np.column_stack([
        X["PPSF_vs_City_Median"] <= 1,
        X["Price_vs_City_Median"] <= 1,
        X["BHK"] >= 3,
        X["Is_Ready_To_Move"] == 1,
        X["Infrastructure_Score"] >= .5,
    ])
    y = (criteria.sum(axis=1) >= 3).astype(int)
    return X, y


def main():
    X, y = examples(120000)
    Xtest, ytest = examples(30000)
    model = HistGradientBoostingClassifier(
        max_iter=160, learning_rate=.12, max_leaf_nodes=31,
        l2_regularization=.05, early_stopping=False, random_state=42,
    )
    model.fit(X, y)
    prob = model.predict_proba(Xtest)[:, 1]
    pred = (prob >= .5).astype(int)
    metrics = {
        "runtime_model": "HistGradientBoostingClassifier",
        "purpose": "Portable distillation of the deterministic five-criterion investment rule",
        "features": FEATURES,
        "rule_example_validation": {
            "rows": len(ytest),
            "accuracy": float(accuracy_score(ytest, pred)),
            "f1": float(f1_score(ytest, pred)),
            "roc_auc": float(roc_auc_score(ytest, prob)),
        },
        "research_model": "XGBoost_Tuned; see reports/classification_metrics.json",
        "note": "Validation is on held-out rule examples, not a real-market forecasting benchmark.",
    }
    for folder in [ROOT / "models", ROOT / "app" / "artifacts"]:
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, folder / "good_investment_classifier.joblib", compress=3)
    (ROOT / "app" / "artifacts" / "runtime_model.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
