"""Step 4 - Regression: estimated property price after 5 years (MLflow tracked)

Run:  python src/train_regression.py   (run train_classification.py first - it saves lookups)
Candidates: Linear Regression, Random Forest, XGBoost (tuned). Target modelled in log-space,
predictions returned in lakhs.  Selection on validation RMSE.  Benchmarks: fixed 8% formula
and city-tier average growth formula.
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
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from config import *
from modeling_utils import *

warnings.filterwarnings("ignore")
TARGET = "Future_Price_5Y_Lakhs"
REG_NUM = NUM_FEATURES + ["Log_Price", "Log_Size"]
REG_FEATURES = REG_NUM + CAT_FEATURES


def ttr(model):
    return TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp)


def reg_pipe(model):
    return make_pipeline(ttr(model), num_cols=REG_NUM)


def reg_metrics(y, p, price, true_r, prefix):
    implied_r = (np.asarray(p) / np.asarray(price)) ** (1 / FORECAST_YEARS) - 1
    return {f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y, p))), f"{prefix}_mae": float(mean_absolute_error(y, p)),
            f"{prefix}_r2": float(r2_score(y, p)), f"{prefix}_mape_pct": float(np.mean(np.abs((y - p) / y)) * 100),
            f"{prefix}_growth_rmse_pp": float(np.sqrt(np.mean((implied_r - true_r) ** 2)) * 100)}


def main():
    df = load_data()
    tr, va, te = split_data(df, TARGET)
    Xtr, ytr, Xva, yva, Xte, yte = tr[REG_FEATURES], tr[TARGET], va[REG_FEATURES], va[TARGET], te[REG_FEATURES], te[TARGET]
    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,}")

    mlflow.set_tracking_uri(MLFLOW_URI)
    exp_name = "RealEstate_Regression_FuturePrice5Y"
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name, artifact_location=f"file://{MLFLOW_ARTIFACTS}/regression")
    mlflow.set_experiment(exp_name)

    # --- Benchmarks (no ML) ---
    tier_r = tr.groupby("City_Tier")["Annual_Growth_Rate"].mean()
    bench = {
        "Benchmark_Fixed8pct": te["Fixed8_Future_Price_Lakhs"].values,
        "Benchmark_TierAvgGrowth": (te["Price_in_Lakhs"] * (1 + te["City_Tier"].map(tier_r)) ** FORECAST_YEARS).values}
    results = {}
    for name, p in bench.items():
        with mlflow.start_run(run_name=name):
            m = reg_metrics(yte.values, p, te["Price_in_Lakhs"].values, te["Annual_Growth_Rate"].values, "test")
            mlflow.log_metrics(m); mlflow.set_tag("task", "regression"); results[name] = m
            print(f"{name:24s} test_RMSE={m['test_rmse']:.3f} R2={m['test_r2']:.4f} growth_RMSE={m['test_growth_rmse_pp']:.3f}pp")

    # --- XGBoost search ---
    sub = tr.sample(40000, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(reg_pipe(XGBRegressor(tree_method="hist", n_jobs=1, random_state=RANDOM_STATE)),
        {"model__regressor__n_estimators": [300, 500, 700], "model__regressor__max_depth": [3, 4, 6],
         "model__regressor__learning_rate": [0.05, 0.1, 0.2], "model__regressor__subsample": [0.8, 1.0],
         "model__regressor__colsample_bytree": [0.6, 0.8, 1.0], "model__regressor__min_child_weight": [1, 5]},
        n_iter=6, scoring="r2", cv=3, random_state=RANDOM_STATE, n_jobs=1)
    t0 = time.time(); search.fit(sub[REG_FEATURES], sub[TARGET])
    bp = {k.replace("model__regressor__", ""): v for k, v in search.best_params_.items()}
    print("XGB search", round(time.time() - t0), "s ->", bp)

    candidates = {
        "LinearRegression_Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=60, max_depth=14, min_samples_leaf=10, max_features=0.5, n_jobs=1, random_state=RANDOM_STATE),
        "XGBoost_Tuned": XGBRegressor(**bp, tree_method="hist", n_jobs=1, random_state=RANDOM_STATE)}
    fitted, run_ids = {}, {}
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            t0 = time.time(); pipe = reg_pipe(model); pipe.fit(Xtr, ytr); fit_s = time.time() - t0
            m = {**reg_metrics(yva.values, pipe.predict(Xva), va["Price_in_Lakhs"].values, va["Annual_Growth_Rate"].values, "val"),
                 **reg_metrics(yte.values, pipe.predict(Xte), te["Price_in_Lakhs"].values, te["Annual_Growth_Rate"].values, "test"), "fit_seconds": fit_s}
            mlflow.log_params({"model": name, "target_transform": "log", "n_features": len(REG_FEATURES), "train_rows": len(tr),
                               **{k: v for k, v in model.get_params().items() if isinstance(v, (int, float, str, bool)) or v is None}})
            mlflow.log_metrics(m); mlflow.set_tag("task", "regression")
            results[name] = m; fitted[name] = pipe; run_ids[name] = run.info.run_id
            print(f"{name:24s} val_RMSE={m['val_rmse']:.3f} test_RMSE={m['test_rmse']:.3f} MAE={m['test_mae']:.3f} R2={m['test_r2']:.5f} growth_RMSE={m['test_growth_rmse_pp']:.3f}pp ({fit_s:.0f}s)")

    best = min(candidates, key=lambda k: results[k]["val_rmse"]); pipe = fitted[best]
    print("BEST (validation RMSE):", best)

    pred = pipe.predict(Xte)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    i = np.random.RandomState(0).choice(len(te), 6000, replace=False)
    ax[0].scatter(yte.values[i], pred[i], s=4, alpha=.35, color="#2a6f97"); lim = [0, yte.max()]; ax[0].plot(lim, lim, "r--", lw=1)
    ax[0].set_title(f"Predicted vs actual 5-yr price - {best}"); ax[0].set_xlabel("Actual (₹ lakhs)"); ax[0].set_ylabel("Predicted (₹ lakhs)")
    ax[1].hist(pred - yte.values, bins=60, color="#52b788"); ax[1].set_title("Residuals (₹ lakhs)"); ax[1].set_xlabel("Predicted - actual")
    names = ["Fixed 8%", "Tier-avg growth"] + [k.replace("_", " ") for k in candidates]
    rm = [results["Benchmark_Fixed8pct"]["test_rmse"], results["Benchmark_TierAvgGrowth"]["test_rmse"]] + [results[k]["test_rmse"] for k in candidates]
    ax[2].barh(names[::-1], rm[::-1], color="#e07a5f"); ax[2].set_title("Test RMSE (₹ lakhs) - lower is better"); ax[2].set_xlabel("RMSE")
    plt.tight_layout(); plt.savefig(FIG_DIR / "reg_evaluation.png", dpi=130); plt.close()

    imp = grouped_permutation_importance(pipe, Xte, yte, scoring="r2")
    plot_importance(imp, f"Feature importance - {best} (regression)", FIG_DIR / "reg_feature_importance.png")

    with mlflow.start_run(run_id=run_ids[best]):
        mlflow.log_artifact(str(FIG_DIR / "reg_evaluation.png")); mlflow.log_artifact(str(FIG_DIR / "reg_feature_importance.png"))
        mlflow.sklearn.log_model(pipe, name="model", signature=infer_signature(Xte.head(100), pred[:100]), input_example=Xte.head(3),
                                 registered_model_name="FuturePriceRegressor", serialization_format="cloudpickle")
    client = mlflow.MlflowClient()
    mv = client.search_model_versions("name='FuturePriceRegressor'")[0]
    client.set_registered_model_alias("FuturePriceRegressor", "champion", mv.version)
    joblib.dump(pipe, MODELS_DIR / "future_price_regressor.joblib", compress=3)

    out = {"best_model": best, "xgb_best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in bp.items()}, "results": results,
           "tier_avg_growth_used_in_benchmark": tier_r.round(4).to_dict(), "top_features": imp.head(10).round(4).to_dict(),
           "features_used": REG_FEATURES}
    (REPORTS_DIR / "regression_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ["best_model", "top_features"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
