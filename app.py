"""
Cross-Industry Corporate Financial Health Classifier — Streamlit Demo
Run with:  streamlit run app.py   (from the project root)
"""
import streamlit as st

from app_utils import load_json_report

st.set_page_config(
    page_title="Financial Health Classifier",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Cross-Industry Corporate Financial Health Classifier")
st.caption(
    "Predicting whether a public company will be **Healthy** or **At-Risk** "
    "next quarter — built end-to-end on SEC's Financial Statement Data Sets."
)

st.divider()

col1, col2, col3, col4 = st.columns(4)
model_summary = load_json_report("12_model_validation_summary.json")
modeling_summary = load_json_report("11_modeling_summary.json")

test_acc = model_summary["final_model_test"]["accuracy"] if model_summary else None
test_auc = model_summary["final_model_test"]["roc_auc"] if model_summary else None
n_companies = (
    (modeling_summary["n_train_companies"] + modeling_summary["n_test_companies"])
    if modeling_summary else None
)
n_rows = (
    (modeling_summary["n_train_rows"] + modeling_summary["n_test_rows"])
    if modeling_summary else None
)

col1.metric("Held-out Test Accuracy", f"{test_acc:.2%}" if test_acc else "—")
col2.metric("Held-out Test ROC-AUC", f"{test_auc:.4f}" if test_auc else "—")
col3.metric("Companies in dataset", f"{n_companies:,}" if n_companies else "—")
col4.metric("Company-quarter rows", f"{n_rows:,}" if n_rows else "—")

st.divider()

left, right = st.columns([3, 2])

with left:
    st.subheader("The Problem")
    st.markdown(
        """
Public financial statements contain everything needed to judge a company's
financial health — but they arrive as thousands of raw, inconsistent regulatory
filings, not a clean table. This project turns 10 quarters of SEC filings
(2024 Q1 – 2026 Q2, ~36 million raw facts) into a single, focused
decision-support tool:

> **Will this company be financially At-Risk or Healthy next quarter,**
> **using only what's in its current filing?**

The target is a single binary classification (`target_next_quarter_health`).
An earlier version of this project also attempted revenue-growth regression
and financial-profile clustering; both were dropped after testing showed the
regression target was mostly an artifact of an annual/quarterly reporting-duration
bug rather than real signal — see the **Limitations & Bias** page for the full story.
        """
    )

    st.subheader("Pipeline (13 stages)")
    st.markdown(
        """
| Stage | What happens |
|---|---|
| 01 – 03 | Load, join, and pivot raw SEC filings; fix currency-mixing & annual/quarterly duration bugs |
| 04 – 06 | Clean, detect outliers, exploratory data analysis |
| 07 – 08 | Engineer 11 financial ratios/growth features, lock company-level train/test split, treat outliers |
| 09 – 10 | Target-relative analysis, leakage-safe feature selection (VIF + permutation importance) |
| 11 – 12 | Train `HistGradientBoostingClassifier`, cross-validate, compare baselines, tune, persist |
| 13 | Inference demo on locked-out test companies |
        """
    )

with right:
    st.subheader("Try it")
    st.markdown(
        "👉 Use the sidebar to navigate:\n\n"
        "- **Live Prediction** — enter a company's ratios and get a real-time "
        "Healthy / At-Risk prediction from the actual trained model\n"
        "- **Model Performance** — full metrics, ROC curve, confusion matrix, "
        "baseline comparisons\n"
        "- **Feature Insights** — which features matter and why (VIF, "
        "permutation importance)\n"
        "- **Limitations & Bias** — honest documentation of where this model "
        "is weaker, and the bugs that were caught and fixed along the way"
    )
    st.info(
        "This model beats a strong **persistence baseline** ('assume next "
        "quarter looks like this one') on ROC-AUC, but only modestly on raw "
        "accuracy — because most companies don't flip health-state quarter "
        "to quarter. See **Model Performance** for the honest comparison.",
        icon="ℹ️",
    )

st.divider()
st.caption(
    "Data source: U.S. SEC Financial Statement Data Sets "
    "(sec.gov/data/financial-statement-data-sets) — public data, no usage restrictions."
)
