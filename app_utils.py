"""
Shared utilities for the Financial Health Classifier Streamlit app.
Loads the trained model + config artifacts ONCE (cached) and exposes
helpers used across every page.
"""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path resolution — works no matter where `streamlit run` is invoked from,
# as long as this file sits at the project root alongside models/ and reports/.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def _require(path: Path, hint: str) -> Path:
    if not path.exists():
        st.error(
            f"**Required file not found:** `{path}`\n\n{hint}\n\n"
            "Make sure you run `streamlit run app.py` from the project root "
            "(the folder that contains `models/` and `reports/`)."
        )
        st.stop()
    return path


@st.cache_resource(show_spinner="Loading trained model…")
def load_model():
    path = _require(
        MODELS_DIR / "health_classifier.joblib",
        "This is the trained HistGradientBoostingClassifier saved by notebook 12.",
    )
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_feature_config() -> dict:
    path = _require(
        MODELS_DIR / "feature_config.json",
        "This defines the exact feature order/columns the model expects, "
        "saved by notebook 12.",
    )
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_winsorization_bounds() -> dict:
    path = _require(
        MODELS_DIR / "winsorization_bounds.json",
        "These are the train-only winsorization bounds from notebook 08 — "
        "used here only to set sensible slider ranges.",
    )
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_json_report(filename: str) -> dict | None:
    path = REPORTS_DIR / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv_report(filename: str) -> pd.DataFrame | None:
    path = REPORTS_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


def figure_path(subfolder: str, filename: str) -> Path | None:
    p = FIGURES_DIR / subfolder / filename
    return p if p.exists() else None


def build_feature_row(inputs: dict, config: dict) -> pd.DataFrame:
    """
    Turns a dict of raw user inputs (ratio/growth values + sector + flags)
    into a single-row DataFrame with EXACTLY the columns/order the model
    expects — same reindex-safety pattern used in notebook 13's inference demo.
    """
    numeric_binary = config["features_numeric_binary"]
    sector_categories = config["sector_categories"]
    feature_columns = config["feature_columns"]

    row = {f: inputs[f] for f in numeric_binary}
    for s in sector_categories:
        row[f"sector_{s}"] = 1 if inputs["industry_sector"] == s else 0

    X = pd.DataFrame([row])
    X = X.reindex(columns=feature_columns, fill_value=0)
    return X


def predict_health(clf, X: pd.DataFrame, config: dict):
    """Returns (label, p_healthy) using the same class-index lookup as notebook 13."""
    classes = list(clf.classes_)
    healthy_idx = classes.index(1)  # 1 = "Healthy" per feature_config.json
    proba = clf.predict_proba(X)[0]
    p_healthy = float(proba[healthy_idx])
    label = "Healthy" if p_healthy >= config.get("threshold", 0.5) else "At-Risk"
    return label, p_healthy


# Illustrative presets for the Live Prediction page's "quick load" buttons.
# These are directionally realistic (not real companies) — meant to show the
# tool responding sensibly to a clearly strong vs. clearly weak balance sheet.
PRESET_HEALTHY = {
    "current_ratio": 2.4, "cash_ratio": 0.9, "debt_to_equity": 0.6,
    "debt_to_assets": 0.30, "operating_margin": 0.15, "roa": 0.09,
    "roe": 0.18, "asset_turnover": 0.9, "revenue_growth_qoq": 0.04,
    "netincome_growth_qoq": 0.05, "assets_growth_qoq": 0.03,
    "is_annual_filing": 0, "has_inventory": 1, "industry_sector": "Manufacturing",
}
PRESET_AT_RISK = {
    "current_ratio": 0.7, "cash_ratio": 0.08, "debt_to_equity": 3.8,
    "debt_to_assets": 0.85, "operating_margin": -0.12, "roa": -0.08,
    "roe": -0.35, "asset_turnover": 0.4, "revenue_growth_qoq": -0.15,
    "netincome_growth_qoq": -0.40, "assets_growth_qoq": -0.05,
    "is_annual_filing": 0, "has_inventory": 1, "industry_sector": "Retail",
}
PRESET_BORDERLINE = {
    "current_ratio": 1.03, "cash_ratio": 0.20, "debt_to_equity": 1.75,
    "debt_to_assets": 0.60, "operating_margin": 0.015, "roa": 0.0,
    "roe": 0.0, "asset_turnover": 0.58, "revenue_growth_qoq": -0.035,
    "netincome_growth_qoq": -0.15, "assets_growth_qoq": -0.005,
    "is_annual_filing": 0, "has_inventory": 1, "industry_sector": "Services",
}  # verified against the real model: P(Healthy) ≈ 0.547, a genuine ~coin-flip case
