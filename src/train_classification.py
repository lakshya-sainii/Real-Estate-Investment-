"""Step 3 - Classification: Is this property a 'Good Investment'?  (MLflow tracked)

Run:  python src/train_classification.py
Candidates: Logistic Regression, Random Forest, XGBoost (tuned).  Selection on validation F1.
"""
import json, time, warnings
import numpy as np
import pandas as pd
import joblib
import mlflow, mlflow.sklearn
from mlflow.models import infer_signature
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from config import *
from modeling_utils import *

warnings.filterwarnings("ignore")
TARGET = "Good_Investment"


def metrics(pipe, X, y, prefix):
    p = pipe.predict(X); pr = pipe.predict_proba(X)[:, 1]
    return {f"{prefix}_accuracy": accuracy_score(y, p), f"{prefix}_precision": precision_score(y, p),
            f"{prefix}_recall": recall_score(y, p), f"{prefix}_f1": f1_score(y, p), f"{prefix}_roc_auc": roc_auc_score(y, pr)}


def main():
    df = load_data()
    save_lookups(df)
    tr, va, te = split_data(df, TARGET, stratify=True)
    Xtr, ytr, Xva, yva, Xte, yte = tr[ALL_FEATURES], tr[TARGET], va[ALL_FEATURES], va[TARGET], te[ALL_FEATURES], te[TARGET]
    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,} | positive rate {df[TARGET].mean():.3f}")

    mlflow.set_tracking_uri(MLFLOW_URI)
    exp_name = "RealEstate_Classification_GoodInvestment"
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name, artifact_location=f"file://{MLFLOW_ARTIFACTS}/classification")
    mlflow.set_experiment(exp_name)

    # --- XGBoost hyper-parameter search on a 40k subsample (3-fold CV, F1) ---
    sub = tr.sample(40000, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        make_pipeline(XGBClassifier(tree_method="hist", n_jobs=1, random_state=RANDOM_STATE, eval_metric="logloss")),
        {"model__n_estimators": [200, 300, 400], "model__max_depth": [4, 6, 8], "model__learning_rate": [0.05, 0.1, 0.2],
         "model__subsample": [0.8, 1.0], "model__colsample_bytree": [0.6, 0.8, 1.0], "model__min_child_weight": [1, 5]},
        n_iter=6, scoring="f1", cv=3, random_state=RANDOM_STATE, n_jobs=1)
    t0 = time.time(); search.fit(sub[ALL_FEATURES], sub[TARGET])
    best_xgb_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    print("XGB search", round(time.time() - t0), "s ->", best_xgb_params, "cv_f1", round(search.best_score_, 4))

    candidates = {
        "Baseline_MajorityClass": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=500, C=1.0, n_jobs=1),
        "RandomForest": RandomForestClassifier(n_estimators=80, max_depth=14, min_samples_leaf=10, max_features=0.5, n_jobs=1, random_state=RANDOM_STATE),
        "XGBoost_Tuned": XGBClassifier(**best_xgb_params, tree_method="hist", n_jobs=1, random_state=RANDOM_STATE, eval_metric="logloss"),
    }
    results, fitted, run_ids = {}, {}, {}
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            t0 = time.time(); pipe = make_pipeline(model); pipe.fit(Xtr, ytr); fit_s = time.time() - t0
            m = {**metrics(pipe, Xva, yva, "val"), **metrics(pipe, Xte, yte, "test"), "fit_seconds": fit_s}
            mlflow.log_params({"model": name, "n_features": len(ALL_FEATURES), "train_rows": len(tr),
                               **{k: v for k, v in model.get_params().items() if isinstance(v, (int, float, str, bool)) or v is None}})
            mlflow.log_metrics(m); mlflow.set_tag("task", "classification")
            results[name] = m; fitted[name] = pipe; run_ids[name] = run.info.run_id
            print(f"{name:24s} val_F1={m['val_f1']:.4f} test_F1={m['test_f1']:.4f} test_acc={m['test_accuracy']:.4f} AUC={m['test_roc_auc']:.4f} ({fit_s:.0f}s)")

    best = max((k for k in results if not k.startswith("Baseline")), key=lambda k: results[k]["val_f1"])
    pipe = fitted[best]; print("BEST (validation F1):", best)

    # --- Evaluation artefacts for best model ---
    pred = pipe.predict(Xte); proba = pipe.predict_proba(Xte)[:, 1]
    cm = confusion_matrix(yte, pred)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ConfusionMatrixDisplay(cm, display_labels=["Not Good", "Good"]).plot(ax=ax[0], cmap="Blues", colorbar=False, values_format="d")
    ax[0].set_title(f"Confusion matrix - {best} (test)")
    for n_, p_ in fitted.items():
        if n_.startswith("Baseline"): continue
        fpr, tpr, _ = roc_curve(yte, p_.predict_proba(Xte)[:, 1]); ax[1].plot(fpr, tpr, label=f"{n_} (AUC {results[n_]['test_roc_auc']:.3f})")
    ax[1].plot([0, 1], [0, 1], "k--", lw=1); ax[1].set_title("ROC curves (test)"); ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR"); ax[1].legend(loc="lower right")
    plt.tight_layout(); plt.savefig(FIG_DIR / "clf_confusion_roc.png", dpi=130); plt.close()

    imp = grouped_permutation_importance(pipe, Xte, yte, scoring="f1")
    plot_importance(imp, f"Feature importance - {best} (classification)", FIG_DIR / "clf_feature_importance.png")

    # --- Stability: 5-fold CV on a 60k sample for the chosen model ---
    cv = cross_val_score(make_pipeline(candidates[best]), tr.sample(60000, random_state=7)[ALL_FEATURES], tr.sample(60000, random_state=7)[TARGET],
                         cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="f1", n_jobs=1)

    # --- Ablation: remove market-context features that mirror the label rules most directly ---
    abl_num = [c for c in NUM_FEATURES if c not in CONTEXT_FEATURES]
    abl = make_pipeline(XGBClassifier(**best_xgb_params, tree_method="hist", n_jobs=1, random_state=RANDOM_STATE, eval_metric="logloss"), num_cols=abl_num)
    abl.fit(tr[abl_num + CAT_FEATURES], ytr)
    pa = abl.predict(te[abl_num + CAT_FEATURES])
    ablation = {"accuracy": accuracy_score(yte, pa), "f1": f1_score(yte, pa), "roc_auc": roc_auc_score(yte, abl.predict_proba(te[abl_num + CAT_FEATURES])[:, 1])}
    with mlflow.start_run(run_name="Ablation_no_market_context"):
        mlflow.log_metrics({f"test_{k}": v for k, v in ablation.items()}); mlflow.set_tag("task", "classification")

    # --- Log + register champion model ---
    with mlflow.start_run(run_id=run_ids[best]):
        mlflow.log_metrics({"cv5_f1_mean": float(cv.mean()), "cv5_f1_std": float(cv.std())})
        mlflow.log_artifact(str(FIG_DIR / "clf_confusion_roc.png")); mlflow.log_artifact(str(FIG_DIR / "clf_feature_importance.png"))
        sig = infer_signature(Xte.head(100), pipe.predict(Xte.head(100)))
        info = mlflow.sklearn.log_model(pipe, name="model", signature=sig, input_example=Xte.head(3),
                                        registered_model_name="GoodInvestmentClassifier", serialization_format="cloudpickle")
    client = mlflow.MlflowClient()
    mv = client.search_model_versions("name='GoodInvestmentClassifier'")[0]
    client.set_registered_model_alias("GoodInvestmentClassifier", "champion", mv.version)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / "good_investment_classifier.joblib", compress=3)

    out = {"best_model": best, "split": {"train": len(tr), "val": len(va), "test": len(te)}, "xgb_best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in best_xgb_params.items()},
           "results": results, "confusion_matrix_test": cm.tolist(), "cv5_f1_mean": float(cv.mean()), "cv5_f1_std": float(cv.std()),
           "ablation_no_market_context_test": ablation, "top_features": imp.head(10).round(4).to_dict()}
    (REPORTS_DIR / "classification_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ["best_model", "cv5_f1_mean", "ablation_no_market_context_test", "top_features"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
