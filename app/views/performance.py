import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core import theme, data
from core.theme import INK, MINT, MINT_DEEP, CHROME


def render():
    m = data.metrics(); c, g = m["clf"], m["reg"]
    theme.page_title("Performance", "How well the models learned", "Research results on a held-out test set (15% of listings). Experiments are tracked in MLflow; the app uses a portable HGB inference classifier to avoid XGBoost serialisation errors.")
    x = c["results"]["XGBoost_Tuned"]; r = g["results"]["LinearRegression_Ridge"]
    k = st.columns(4)
    k[0].metric("Classifier accuracy", f"{x['test_accuracy'] * 100:.2f}%"); k[1].metric("Classifier F1", f"{x['test_f1']:.4f}")
    k[2].metric("Regression RMSE", f"{r['test_rmse']:.2f} L"); k[3].metric("Regression R²", f"{r['test_r2']:.5f}")
    st.write("")
    t1, t2, t3 = st.tabs(["Classification", "Regression", "Tracking"])
    with t1:
        names = [k_.replace("_", " ") for k_ in c["results"]]
        f1 = go.Figure(go.Bar(x=[c["results"][k_]["test_f1"] for k_ in c["results"]], y=names, orientation="h", marker_color=[CHROME, CHROME, CHROME, MINT_DEEP],
                              text=[f"{c['results'][k_]['test_f1']:.4f}" for k_ in c["results"]], textposition="outside"))
        f1.update_layout(title="Test F1 by model"); f1.update_xaxes(range=[0, 1.15]); f1.update_yaxes(autorange="reversed")
        a, b = st.columns(2, gap="large")
        a.plotly_chart(theme.style_fig(f1, 340, False), width="stretch")
        cm = c["confusion_matrix_test"]
        fig = go.Figure(go.Heatmap(z=cm, x=["Predicted: no", "Predicted: yes"], y=["Actual: no", "Actual: yes"], text=cm, texttemplate="%{text:,}", colorscale=theme.MINT_SCALE, showscale=False))
        fig.update_layout(title="Confusion matrix (test)"); fig.update_yaxes(autorange="reversed"); b.plotly_chart(theme.style_fig(fig, 340, False), width="stretch")
        theme.insight("Classifier", f"XGBoost makes {cm[0][1] + cm[1][0]} errors on {sum(map(sum, cm)):,} test listings. Logistic Regression reaches {c['results']['LogisticRegression']['test_accuracy'] * 100:.0f}% accuracy, so the rule has threshold interactions that tree models capture.")
        fi = pd.Series(c["top_features"]).sort_values().tail(8)
        ff = go.Figure(go.Bar(x=fi.values, y=[i.replace("_", " ") for i in fi.index], orientation="h", marker_color=[MINT_DEEP if v > .05 else CHROME for v in fi.values]))
        ff.update_layout(title="Feature importance (permutation, F1 drop)"); st.plotly_chart(theme.style_fig(ff, 340, False), width="stretch")
        theme.insight("Drivers", "Five features carry the verdict: price, relative price, readiness, infrastructure and BHK. They are exactly the inputs of the labelling rule.")
    with t2:
        order = ["Benchmark_Fixed8pct", "Benchmark_TierAvgGrowth", "RandomForest", "XGBoost_Tuned", "LinearRegression_Ridge"]
        lab = ["Flat 8% growth", "Tier-average growth", "Random Forest", "XGBoost", "Ridge (selected)"]
        fr = go.Figure(go.Bar(x=[g["results"][k_]["test_rmse"] for k_ in order], y=lab, orientation="h", marker_color=[CHROME, CHROME, CHROME, CHROME, MINT_DEEP],
                              text=[f"{g['results'][k_]['test_rmse']:.2f}" for k_ in order], textposition="outside"))
        fr.update_layout(title="Test RMSE in lakhs (lower is better)"); fr.update_yaxes(autorange="reversed"); fr.update_xaxes(range=[0, 17])
        a, b = st.columns(2, gap="large"); a.plotly_chart(theme.style_fig(fr, 340, False), width="stretch")
        fg = go.Figure(go.Bar(x=[g["results"][k_]["test_growth_rmse_pp"] for k_ in order], y=lab, orientation="h", marker_color=[CHROME, CHROME, CHROME, CHROME, MINT_DEEP],
                              text=[f"{g['results'][k_]['test_growth_rmse_pp']:.3f}" for k_ in order], textposition="outside"))
        fg.update_layout(title="Error in implied yearly growth (percentage points)"); fg.update_yaxes(autorange="reversed"); fg.update_xaxes(range=[0, .9])
        b.plotly_chart(theme.style_fig(fg, 340, False), width="stretch")
        theme.insight("Regression", f"Ridge on the log target cuts RMSE from {g['results']['Benchmark_Fixed8pct']['test_rmse']:.1f} lakhs (flat 8%) to {r['test_rmse']:.2f}. The target is almost exactly linear in log space, so the simplest model wins.")
        st.image(str(data.fig_path("reg_evaluation.png")), caption="Predicted vs actual, residuals and RMSE comparison (test set)")
    with t3:
        st.markdown("Every run logs parameters and validation/test metrics to **MLflow** (SQLite backend). The best model of each task is registered as `GoodInvestmentClassifier` and `FuturePriceRegressor` with the alias *champion*.")
        st.code("mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db", language="bash")
    theme.note("Near-perfect scores reflect that both targets are rule-derived (no price history in the data). They show the pipeline learns the rules, not real-market forecasting accuracy.")
    theme.footer()


render()
