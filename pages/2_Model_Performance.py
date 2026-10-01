import pandas as pd
import streamlit as st

from app_utils import load_json_report, figure_path

st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide")
st.title("📊 Model Performance")

modeling = load_json_report("11_modeling_summary.json")
validation = load_json_report("12_model_validation_summary.json")

if not modeling or not validation:
    st.warning("Report JSONs not found under `reports/` — run notebooks 11 and 12 first.")
    st.stop()

st.subheader("Final Model")
st.code(
    f"HistGradientBoostingClassifier  (params: {validation['tuning']['adopted_tuned_params'] and 'tuned' or 'default — tuning gain was within noise'})",
    language=None,
)

st.subheader("Headline Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Test Accuracy", f"{validation['final_model_test']['accuracy']:.2%}")
c2.metric("Test Balanced Accuracy", f"{validation['final_model_test']['balanced_accuracy']:.2%}")
c3.metric("Test ROC-AUC", f"{validation['final_model_test']['roc_auc']:.4f}")
c4.metric("Brier Score (lower=better)", f"{modeling['test']['brier']:.4f}")

st.divider()

st.subheader("Baseline Comparison (held-out test set)")
st.caption(
    "A model is only useful if it beats simple baselines. Here's the honest comparison — "
    "including the strong **persistence** baseline (\"assume next quarter looks like this one\")."
)
rows = [
    {"Model": "Majority class", "Accuracy": modeling["test"]["majority"]["accuracy"],
     "ROC-AUC": modeling["test"]["majority"]["roc_auc"]},
    {"Model": "Persistence baseline", "Accuracy": modeling["test"]["persistence"]["accuracy"],
     "ROC-AUC": modeling["test"]["persistence"]["roc_auc"]},
    {"Model": "Final model (HGB)", "Accuracy": modeling["test"]["model"]["accuracy"],
     "ROC-AUC": modeling["test"]["model"]["roc_auc"]},
]
df_baseline = pd.DataFrame(rows).set_index("Model")
st.dataframe(
    df_baseline.style.format({"Accuracy": "{:.2%}", "ROC-AUC": "{:.4f}"}),
    width='stretch',
)
st.info(
    "The model's edge over persistence is modest on raw accuracy but substantial on "
    "ROC-AUC (ranking/probability quality) — most companies simply don't flip health "
    "state quarter to quarter, so accuracy alone understates the model's value.",
    icon="ℹ️",
)

st.divider()

st.subheader("Cross-Validation (5-fold, grouped by company)")
cv_table = validation["cv"]["table"]
df_cv = pd.DataFrame(cv_table).T
df_cv.columns = ["AUC (mean)", "AUC (std)", "Accuracy (mean)", "Accuracy (std)"]
st.dataframe(
    df_cv.style.format({c: "{:.4f}" for c in df_cv.columns}),
    width='stretch',
)

st.divider()

st.subheader("Diagnostic Charts")
chart_specs = [
    ("modeling_charts", "01_confusion_matrix.png", "Confusion Matrix (test set)"),
    ("modeling_charts", "02_roc_and_calibration.png", "ROC Curve & Calibration"),
    ("model_validation_charts", "01_cv_model_comparison.png", "CV: Model Comparison"),
    ("model_validation_charts", "02_default_vs_tuned.png", "Default vs. Tuned Hyperparameters"),
]
cols = st.columns(2)
for i, (folder, filename, caption) in enumerate(chart_specs):
    path = figure_path(folder, filename)
    with cols[i % 2]:
        if path:
            st.image(str(path), caption=caption, width='stretch')
        else:
            st.caption(f"_(chart not found: {folder}/{filename})_")

st.divider()
with st.expander("Early-warning / recovery signal breakdown"):
    ew = modeling.get("early_warning_currently_healthy", {})
    rec = modeling.get("recovery_currently_atrisk", {})
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Currently Healthy → flags early warning of going At-Risk**")
        st.write(f"Base rate: {ew.get('base_rate', 0):.1%} · "
                 f"Top-decile capture rate: {ew.get('top_decile_rate', 0):.1%} · "
                 f"Lift: {ew.get('lift', 0):.2f}x · AUC: {ew.get('auc', 0):.4f}")
    with e2:
        st.markdown("**Currently At-Risk → flags likely recovery to Healthy**")
        st.write(f"Base rate: {rec.get('base_rate', 0):.1%} · "
                 f"Top-decile capture rate: {rec.get('top_decile_rate', 0):.1%} · "
                 f"Lift: {rec.get('lift', 0):.2f}x · AUC: {rec.get('auc', 0):.4f}")
