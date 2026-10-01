import streamlit as st

from app_utils import load_json_report, load_csv_report, figure_path

st.set_page_config(page_title="Feature Insights", page_icon="🔍", layout="wide")
st.title("🔍 Feature Insights")
st.caption("Why these 11 ratio/growth features (out of 13 candidates) made it into the final model.")

selection = load_json_report("10_feature_selection_summary.json")

if selection:
    st.subheader("Selection Summary")
    dropped = selection.get("dropped_units", {})
    if dropped:
        st.markdown("**Dropped features and why:**")
        for feat, reason in dropped.items():
            st.markdown(f"- `{feat}` — {reason}")
    st.markdown(
        f"**Kept:** {', '.join(f'`{f}`' for f in selection.get('selected_ratio_growth_features', []))}"
    )
    st.caption(
        f"Selection was run on {selection['selection_set']['n_companies']:,} train companies "
        f"only ({selection['selection_set']['n_rows']:,} rows) — test companies never touched "
        "this decision, keeping it leakage-safe."
    )

    st.divider()
    cv = selection.get("selection_cv", {})
    if cv:
        c1, c2, c3 = st.columns(3)
        c1.metric("CV AUC — all candidates", f"{cv.get('auc_baseline', 0):.4f}")
        c2.metric("CV AUC — final feature set", f"{cv.get('auc_final', 0):.4f}")
        c3.metric("Fold std (final)", f"{cv.get('auc_fold_std_final', 0):.4f}")
        st.caption(
            "Dropping the redundant/low-relevance features cost essentially nothing — "
            "confirming they weren't adding real signal."
        )

st.divider()
st.subheader("Charts")
chart_specs = [
    ("feature_selection_charts", "02_vif_three_views.png", "Multicollinearity (VIF) — before selection"),
    ("feature_selection_charts", "04_vif_final.png", "Multicollinearity (VIF) — final feature set"),
    ("feature_selection_charts", "03_permutation_importance.png", "Grouped Permutation Importance"),
    ("feature_selection_charts", "01_correlation_pearson_vs_spearman.png", "Correlation: Pearson vs. Spearman"),
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
st.subheader("Detailed Tables")
tab1, tab2, tab3 = st.tabs(["VIF Table", "Permutation Importance", "Redundancy Decisions"])
with tab1:
    df = load_csv_report("10_vif_table.csv")
    st.dataframe(df, width='stretch') if df is not None else st.caption("Not found.")
with tab2:
    df = load_csv_report("10_permutation_importance.csv")
    st.dataframe(df, width='stretch') if df is not None else st.caption("Not found.")
with tab3:
    df = load_csv_report("10_redundancy_decisions.csv")
    st.dataframe(df, width='stretch') if df is not None else st.caption("Not found.")
