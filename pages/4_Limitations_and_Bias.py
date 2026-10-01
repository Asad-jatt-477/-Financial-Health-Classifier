import streamlit as st

from app_utils import load_csv_report

st.set_page_config(page_title="Limitations & Bias", page_icon="⚠️", layout="wide")
st.title("⚠️ Limitations & Bias")
st.caption(
    "Documented honestly, not hidden — this is what a reviewer should know before "
    "trusting this model's predictions in any real scenario."
)

st.subheader("Subgroup Performance (held-out test set)")
df = load_csv_report("11_subgroup_performance.csv")
if df is not None:
    st.dataframe(
        df.style.format({
            "healthy_share": "{:.1%}", "model_acc": "{:.2%}",
            "persist_acc": "{:.2%}", "model_auc": "{:.4f}",
        }),
        width='stretch',
    )
    st.caption(
        "Sectors with small `n` (Agriculture, Construction, Wholesale) have "
        "**statistically noisy** estimates — a couple of companies can swing the "
        "percentage a lot. Don't over-interpret them as 'best/worst sector' claims."
    )
else:
    st.caption("`11_subgroup_performance.csv` not found.")

st.divider()

st.subheader("Known Limitations")
st.markdown(
    """
- **Sector under-representation:** Finance/Real-Estate companies make up a small
  share of the trainable dataset, because banks/REITs don't report a classified
  balance sheet (`current_ratio` is structurally unavailable for them). Predictions
  for this sector should be treated with reduced confidence.
- **Point-in-time simplification:** outlier-treatment percentile bounds are computed
  once from train-company rows (not a strict walk-forward window) — leakage-safe
  with respect to test companies, but not a fully rigorous rolling-time holdout.
- **Small-sample sectors:** Agriculture, Construction, and Wholesale each have well
  under 300 rows in the test set — their subgroup metrics above are high-variance.
    """
)

st.subheader("Bugs Caught and Fixed During Development")
st.markdown(
    """
- **Currency contamination:** an early pipeline version mixed non-USD filings into
  USD figures — fixed by filtering to USD-denominated facts before pivoting.
- **Annual/quarterly duration mixing:** 10-K filings report income-statement figures
  as full-year totals while 10-Q filings report per-quarter — an early version
  silently mixed the two, corrupting ratios like `asset_turnover`/`roa` by 3–4x.
  This is *why the original revenue-growth regression target was dropped*: a naive
  calendar baseline beat the regression model on the contaminated data, proving the
  model was mostly learning the reporting-calendar artifact, not real financial change.
- **Feature-alignment bug at inference time:** one-hot-encoded sector columns could
  silently misalign for small prediction batches — fixed with explicit column
  reindexing, the same pattern this app's `build_feature_row()` uses.
- **Panel-data leakage:** companies appear multiple times across quarters — a random
  train/test split would leak company identity. Fixed with company-grouped splitting,
  locked once and reused through every later stage.
    """
)

st.info(
    "None of this makes the model unusable — it makes the reported numbers "
    "**trustworthy**, because every gap and edge case is documented rather than "
    "papered over.",
    icon="✅",
)
