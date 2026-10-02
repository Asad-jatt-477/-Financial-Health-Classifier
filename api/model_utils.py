"""
Model loading + prediction helpers for the Vercel serverless function.
Deliberately pandas-free: a plain numpy array in the exact trained column
order gives byte-identical predictions to a DataFrame (verified), and
keeps the deployed function bundle small.
"""
import json
import warnings
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np

# The model was fit on a named DataFrame; predicting from a plain numpy
# array (deliberate — keeps this function pandas-free) triggers a harmless
# "X does not have valid feature names" UserWarning on every call. Verified
# predictions are byte-identical either way, so this warning is suppressed
# rather than paying pandas' bundle-size cost to silence it the other way.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

MODELS_DIR = Path(__file__).resolve().parent / "models"


@lru_cache(maxsize=1)
def get_model():
    return joblib.load(MODELS_DIR / "health_classifier.joblib")


@lru_cache(maxsize=1)
def get_config() -> dict:
    return json.loads((MODELS_DIR / "feature_config.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_bounds() -> dict:
    return json.loads((MODELS_DIR / "winsorization_bounds.json").read_text(encoding="utf-8"))


NUMERIC_FEATURES = [
    "current_ratio", "cash_ratio", "debt_to_equity", "debt_to_assets",
    "operating_margin", "roa", "roe", "asset_turnover",
    "revenue_growth_qoq", "netincome_growth_qoq", "assets_growth_qoq",
]


def build_feature_vector(payload: dict) -> np.ndarray:
    """Builds a single row in EXACTLY the model's trained column order."""
    config = get_config()
    feature_columns = config["feature_columns"]
    sector_categories = config["sector_categories"]

    row = {f: float(payload[f]) for f in NUMERIC_FEATURES}
    row["is_annual_filing"] = 1.0 if payload.get("is_annual_filing") else 0.0
    row["has_inventory"] = 1.0 if payload.get("has_inventory") else 0.0

    sector = payload.get("industry_sector")
    for s in sector_categories:
        row[f"sector_{s}"] = 1.0 if s == sector else 0.0

    # reindex-safety: any column not in row defaults to 0.0 (matches
    # the DataFrame .reindex(fill_value=0) pattern used elsewhere in this project)
    vec = [row.get(c, 0.0) for c in feature_columns]
    return np.array([vec], dtype=float)


def predict(payload: dict) -> dict:
    clf = get_model()
    config = get_config()
    X = build_feature_vector(payload)

    classes = list(clf.classes_)
    healthy_idx = classes.index(1)
    proba = clf.predict_proba(X)[0]
    p_healthy = float(proba[healthy_idx])
    threshold = config.get("threshold", 0.5)
    label = "Healthy" if p_healthy >= threshold else "At-Risk"

    return {
        "label": label,
        "p_healthy": round(p_healthy, 4),
        "threshold": threshold,
    }
