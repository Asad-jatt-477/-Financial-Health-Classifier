import streamlit as st

from app_utils import (
    load_model, load_feature_config, load_winsorization_bounds,
    build_feature_row, predict_health,
    PRESET_HEALTHY, PRESET_AT_RISK, PRESET_BORDERLINE,
)

st.set_page_config(page_title="Live Prediction", page_icon="🔮", layout="wide")
st.title("🔮 Live Prediction")
st.caption(
    "Enter a company's financial ratios and get a real prediction from the "
    "**actual trained model** (`models/health_classifier.joblib`) — not a mock-up."
)

clf = load_model()
config = load_feature_config()
bounds = load_winsorization_bounds()

NUMERIC_KEYS = [
    "current_ratio", "cash_ratio", "debt_to_equity", "debt_to_assets",
    "operating_margin", "roa", "roe", "asset_turnover",
    "revenue_growth_qoq", "netincome_growth_qoq", "assets_growth_qoq",
]
SLIDER_DEFAULTS = {  # (lo_fallback, hi_fallback, step) when a bound is missing
    "current_ratio": (0.0, 5.0, 0.01), "cash_ratio": (0.0, 3.0, 0.01),
    "debt_to_equity": (-2.0, 6.0, 0.01), "debt_to_assets": (0.0, 2.0, 0.01),
    "operating_margin": (-1.0, 1.0, 0.01), "roa": (-1.0, 1.0, 0.01),
    "roe": (-2.0, 2.0, 0.01), "asset_turnover": (0.0, 2.0, 0.01),
    "revenue_growth_qoq": (-1.0, 2.0, 0.01), "netincome_growth_qoq": (-2.0, 2.0, 0.01),
    "assets_growth_qoq": (-1.0, 2.0, 0.01),
}


def widget_key(feat: str) -> str:
    return f"sl_{feat}"


def clamp(val, feat):
    b = bounds.get(feat, {})
    lo_fallback, hi_fallback, _ = SLIDER_DEFAULTS[feat]
    lo = float(b["neg_lower"]) if b.get("neg_lower") is not None else lo_fallback
    hi = float(b["pos_upper"]) if b.get("pos_upper") is not None else hi_fallback
    return min(max(float(val), lo), hi)


def apply_preset(preset: dict):
    """Directly writes each widget's OWN session_state key. This must happen
    BEFORE the widgets are instantiated below (Streamlit only honors a
    widget's `value=` argument on its very first creation; after that the
    widget's own session_state[key] wins on every rerun — so updating a
    separate dict does NOT move an already-rendered slider)."""
    for feat in NUMERIC_KEYS:
        st.session_state[widget_key(feat)] = clamp(preset[feat], feat)
    st.session_state["sel_sector"] = preset["industry_sector"]
    st.session_state["radio_filing"] = "Annual (10-K)" if preset["is_annual_filing"] else "Quarterly (10-Q)"
    st.session_state["radio_inventory"] = "Yes" if preset["has_inventory"] else "No"


# Initialize on first load only (so sliders have sensible starting values).
if widget_key("current_ratio") not in st.session_state:
    apply_preset(PRESET_BORDERLINE)

st.subheader("Quick-load an example")
p1, p2, p3, _ = st.columns([1, 1, 1, 3])
if p1.button("✅ Load Healthy example"):
    apply_preset(PRESET_HEALTHY)
    st.rerun()
if p2.button("⚠️ Load At-Risk example"):
    apply_preset(PRESET_AT_RISK)
    st.rerun()
if p3.button("➖ Load Borderline example"):
    apply_preset(PRESET_BORDERLINE)
    st.rerun()
st.caption(
    "These presets are illustrative (not real companies) — they show the tool "
    "responding sensibly to a clearly strong vs. clearly weak balance sheet."
)

st.divider()


def bounded_slider(label, feat, help_text=""):
    b = bounds.get(feat, {})
    lo_fallback, hi_fallback, step = SLIDER_DEFAULTS[feat]
    lo = float(b["neg_lower"]) if b.get("neg_lower") is not None else lo_fallback
    hi = float(b["pos_upper"]) if b.get("pos_upper") is not None else hi_fallback
    return st.slider(label, min_value=round(lo, 2), max_value=round(hi, 2),
                      step=step, help=help_text, key=widget_key(feat))


left, mid, right = st.columns(3)

with left:
    st.markdown("**Liquidity & Leverage**")
    current_ratio = bounded_slider("Current Ratio", "current_ratio",
                                    "Current Assets ÷ Current Liabilities")
    cash_ratio = bounded_slider("Cash Ratio", "cash_ratio", "Cash ÷ Current Liabilities")
    debt_to_equity = bounded_slider("Debt-to-Equity", "debt_to_equity")
    debt_to_assets = bounded_slider("Debt-to-Assets", "debt_to_assets")

with mid:
    st.markdown("**Profitability**")
    operating_margin = bounded_slider("Operating Margin", "operating_margin")
    roa = bounded_slider("Return on Assets (ROA)", "roa")
    roe = bounded_slider("Return on Equity (ROE)", "roe")
    asset_turnover = bounded_slider("Asset Turnover", "asset_turnover")

with right:
    st.markdown("**Growth (quarter-over-quarter)**")
    revenue_growth_qoq = bounded_slider("Revenue Growth QoQ", "revenue_growth_qoq")
    netincome_growth_qoq = bounded_slider("Net Income Growth QoQ", "netincome_growth_qoq")
    assets_growth_qoq = bounded_slider("Assets Growth QoQ", "assets_growth_qoq")

st.divider()
c1, c2, c3 = st.columns(3)
if "sel_sector" not in st.session_state:
    st.session_state["sel_sector"] = "Manufacturing"
if "radio_filing" not in st.session_state:
    st.session_state["radio_filing"] = "Quarterly (10-Q)"
if "radio_inventory" not in st.session_state:
    st.session_state["radio_inventory"] = "Yes"

industry_sector = c1.selectbox("Industry Sector", config["sector_categories"], key="sel_sector")
is_annual_filing = c2.radio("Filing Type", ["Quarterly (10-Q)", "Annual (10-K)"],
                             horizontal=True, key="radio_filing") == "Annual (10-K)"
has_inventory = c3.radio("Reports Inventory?", ["No", "Yes"],
                          horizontal=True, key="radio_inventory") == "Yes"

inputs = {
    "current_ratio": current_ratio, "cash_ratio": cash_ratio,
    "debt_to_equity": debt_to_equity, "debt_to_assets": debt_to_assets,
    "operating_margin": operating_margin, "roa": roa, "roe": roe,
    "asset_turnover": asset_turnover, "revenue_growth_qoq": revenue_growth_qoq,
    "netincome_growth_qoq": netincome_growth_qoq, "assets_growth_qoq": assets_growth_qoq,
    "is_annual_filing": int(is_annual_filing), "has_inventory": int(has_inventory),
    "industry_sector": industry_sector,
}

st.divider()
if st.button("🔎 Predict Financial Health", type="primary", width='stretch'):
    X = build_feature_row(inputs, config)
    label, p_healthy = predict_health(clf, X, config)

    r1, r2 = st.columns([1, 2])
    with r1:
        if label == "Healthy":
            st.success("### ✅ Predicted: Healthy")
        else:
            st.error("### ⚠️ Predicted: At-Risk")
        st.metric("P(Healthy)", f"{p_healthy:.1%}")
        st.progress(p_healthy)
    with r2:
        st.caption(
            f"Model threshold is {config.get('threshold', 0.5):.0%} — probabilities "
            "above this are classified Healthy. This reflects only the ratios entered "
            "above; sector and filing-type context are included in the prediction."
        )
        if industry_sector == "Finance_RealEstate":
            st.warning(
                "Finance/Real-Estate companies are underrepresented in the training "
                "data (banks/REITs don't report a classified balance sheet, so "
                "`current_ratio` is structurally unavailable for many of them). "
                "Treat this sector's predictions with reduced confidence.",
                icon="⚠️",
            )
