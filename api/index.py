"""
FastAPI backend for the Cross-Industry Corporate Financial Health Classifier.
Deployed on Vercel as a serverless function; served under /api/*.
"""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys

# Vercel loads this file by path, so api/ is NOT on sys.path there (locally,
# local_test_server.py adds it). Add this file's own folder so the sibling
# module model_utils.py can be imported in both environments.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import predict as run_prediction, get_config, get_bounds, NUMERIC_FEATURES  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"

app = FastAPI(title="Financial Health Classifier API")

# Same-origin in production (frontend + API share one Vercel deployment),
# left open for local development against a separately-served frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    current_ratio: float
    cash_ratio: float
    debt_to_equity: float
    debt_to_assets: float
    operating_margin: float
    roa: float
    roe: float
    asset_turnover: float
    revenue_growth_qoq: float
    netincome_growth_qoq: float
    assets_growth_qoq: float
    is_annual_filing: bool = False
    has_inventory: bool = True
    industry_sector: str = Field(..., description="One of the sector_categories from /api/config")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    """Feature list, sector categories, and slider bounds for the frontend form."""
    cfg = get_config()
    bounds = get_bounds()
    return {
        "numeric_features": NUMERIC_FEATURES,
        "sector_categories": cfg["sector_categories"],
        "threshold": cfg.get("threshold", 0.5),
        "bounds": {k: bounds.get(k, {}) for k in NUMERIC_FEATURES},
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    cfg = get_config()
    if req.industry_sector not in cfg["sector_categories"]:
        raise HTTPException(
            status_code=422,
            detail=f"industry_sector must be one of {cfg['sector_categories']}",
        )
    try:
        return run_prediction(req.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/api/metrics")
def metrics():
    """Real, pre-computed model performance numbers from the training pipeline."""
    path = DATA_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found")
    return json.loads(path.read_text(encoding="utf-8"))
