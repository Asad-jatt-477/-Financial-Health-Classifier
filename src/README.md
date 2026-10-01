# Cross-Industry Corporate Financial Analysis

An end-to-end, production-grade data science project built on SEC's **Financial Statement Data Sets** — engineering a clean, forecasting-ready panel dataset from raw regulatory filings to predict whether a public company will be **financially "At-Risk" next quarter**.

**Status: Complete.** All 13 pipeline stages implemented, tested (3-5 sandbox verification rounds each), and validated with grouped cross-validation, a baseline/model comparison, hyperparameter tuning, and a working inference demo on locked-out test companies.

---

## Business Problem

Public financial statements contain the raw signals needed to assess a company's financial health, but they are unstructured, inconsistent across filers, and impractical to analyze at scale by hand. This project turns that raw regulatory data into a single decision-support tool:

**Financial health screening** — predict whether a company will be financially **At-Risk** or **Healthy** *next quarter*, using only fundamentals available in its current filing.

This is a **single binary classification target**: `target_next_quarter_health`. An earlier version of this project also attempted revenue-growth regression and financial-profile clustering; both were removed after testing showed the regression target was largely an artifact of a data-quality bug (see *Known Limitations / Technical Challenges Solved* below) rather than genuine business signal, and the scope was narrowed to the one target that held up under scrutiny.

---

## Data Source

- **Source:** [SEC Financial Statement Data Sets](https://www.sec.gov/data/financial-statement-data-sets) (U.S. SEC, Division of Economic and Risk Analysis)
- **Coverage:** 10 consecutive quarters, 2024 Q1 – 2026 Q2
- **Scale:** ~36 million raw numeric facts, 7,941 companies at intake → 58,243 cleaned company-filings after processing
- **Train/test split:** grouped by company (`cik`), 80/20 — **3,922 train companies (27,840 labelled rows) / 981 test companies (7,051 labelled rows)**. Locked once in stage 07 and reused, unchanged, through every later stage (winsorization bounds, feature selection, tuning, final evaluation) so no test-company information ever leaks into a modeling decision.

---

## Final Results — Financial Health Classification (`HistGradientBoostingClassifier`, tuned)

| Metric | Value |
|---|---|
| Held-out test Accuracy | **87.99%** |
| Held-out test Balanced Accuracy | 86.92% |
| Held-out test ROC-AUC | **0.9391** |
| 5-fold CV ROC-AUC (tuned) | 0.9464 (paired diff over default: +0.0007, within 1 SE — tuning gain is real but small) |
| 5-fold CV ROC-AUC (default params) | 0.9457 |
| 5-fold CV Accuracy (default params) | 89.32% ± 0.59pp |

### Baselines it had to beat

| Baseline | CV ROC-AUC | CV Accuracy | Test Accuracy |
|---|---|---|---|
| Majority class | 0.500 | 65.06% | 65.47% |
| **Persistence** ("next quarter = same as this quarter") | 0.8684 | 88.24% | 87.16% (ROC-AUC 0.857) |
| Logistic Regression | 0.8266 | 75.70% | — |
| **HistGradientBoosting (final model)** | **0.9457–0.9464** | **89.32%** | **87.99% (ROC-AUC 0.939)** |

**Honest read:** the persistence baseline (just assuming nothing changes) is already strong — 88.2% CV accuracy — because most companies don't flip health-state quarter to quarter (At-Risk→At-Risk 91.3% of the time, Healthy→Healthy 83.8% of the time; see `reports/09_transition_matrix.csv`). The model beats it on ROC-AUC (0.946 vs 0.868) but only modestly on raw accuracy, so its real value is in **ranking/probability quality** (who's most at risk), not just a yes/no flip on a mostly-sticky label. This is reported directly rather than only showing the flattering majority-class comparison.

### Feature Selection

Of 13 candidate ratio/growth features, 12 were kept. `net_profit_margin` was dropped as the sole VIF-flagged redundancy (with `operating_margin`) after confirming via drop-one grouped-CV that removing it caused no measurable performance loss (within 1 SE). Final feature set: `current_ratio`, `cash_ratio`, `debt_to_equity`, `debt_to_assets`, `gross_margin`, `operating_margin`, `roa`, `roe`, `asset_turnover`, `revenue_growth_qoq`, `netincome_growth_qoq`, `assets_growth_qoq`, plus `is_annual_filing`, `has_inventory`, and one-hot `industry_sector`.

---

## Known Limitations (documented, not hidden)

- **Sector under-representation:** Finance/Real-Estate companies make up only **8.3%** of the trainable dataset (vs. 48.6% Manufacturing), because banks/REITs don't report a classified balance sheet (`current_ratio` is structurally unavailable for them). **Model predictions for Finance/Real-Estate companies should be treated with reduced confidence** — the inference demo (stage 13) shows a real example of this (a bank with no constructible target).
- **Point-in-time simplification:** Outlier-treatment percentile bounds are computed once from train-company rows only (not a strict walk-forward window) — leakage-safe with respect to test companies, but not a fully rigorous rolling-time holdout.
- **Currency contamination (caught and fixed):** an early pipeline version mixed non-USD filings into USD figures; fixed by filtering to USD-denominated facts before pivoting.
- **Annual/quarterly duration mixing (caught and fixed):** 10-K filings report income-statement figures as full-year totals while 10-Q filings report per-quarter — an early version silently mixed the two, corrupting ratios like `asset_turnover`/`roa` by 3-4x and inflating apparent quarter-to-quarter "growth." This is why the regression target was dropped: a naive calendar baseline beat the regression model on this contaminated data, proving the model was mostly learning the reporting-calendar artifact, not real financial change. Fixed at the source (stage 03) by dividing flow-tag values by their reporting-period length before pivoting; verified the within-company annual/quarterly ratio moved from ~3.8x to ~0.9-1.0x.

---

## Pipeline Architecture

| Stage | Notebook | Description |
|---|---|---|
| 1-3 | `01_load_and_merge.py` → `03_pivot_and_clean.py` | Load, join, pivot raw SEC data; currency-bug fix; annual/quarterly duration fix |
| 4 | `04_data_cleaning.ipynb` | Deduplication (tie-breaker-safe), point-in-time integrity, delinquent/transition-filer flags |
| 5 | `05_outlier_detection.ipynb` | Visual + statistical (sign-split IQR & Modified Z-score) detection; EPS error-correction |
| 6 | `06_EDA.ipynb` | Missingness (industry-driven proof), distributions, seasonality decomposition, cross-industry scale |
| 7 | `07_feature_engineering.ipynb` | 12 ratios/growth features, single forward-shifted leakage-safe target, **company train/test split locked** (`company_split.json`) |
| 8 | `08_ratio_outlier_treatment.ipynb` | Sign-split winsorization on ratios only (not raw dollars — avoids ROA distortion); bounds computed from train companies only |
| 9 | `09_target_relative_eda.ipynb` | Feature-target correlations (train companies only); sample-selection-bias discovery; persistence/transition analysis |
| 10 | `10_feature_selection.ipynb` | VIF + drop-one-CV redundancy check; grouped permutation-importance relevance check |
| 11 | `11_modeling.ipynb` | Classification model training, group-based train/test split, baseline comparisons |
| 12 | `12_model_validation_and_persistence.ipynb` | GroupKFold CV, model comparison (majority/persistence/logistic/HGB), hyperparameter tuning, `joblib` persistence |
| 13 | `13_inference_demo.ipynb` | Live inference demo on locked test-set companies; caught and fixed a real feature-alignment bug |

---

## Key Technical Challenges Solved

- **Currency contamination:** foreign filers reporting in local currency were mixed with USD figures — fixed by filtering to USD-denominated facts.
- **Annual/quarterly duration mixing:** 10-K (annual) vs 10-Q (quarterly) income-statement figures were silently combined at different magnitudes — fixed by dividing flow values by reporting-period length before pivoting; this single fix is what made the eventual single-target scope decision defensible.
- **Memory-constrained joins at 36M-row scale:** resolved via per-quarter chunked processing with parquet predicate pushdown.
- **Sign-mixing outlier-detection bug:** naive IQR wholesale-flagged entire negative-value populations in several columns; fixed with sign-split detection, applied uniformly.
- **Ratio-distortion risk:** proved (with a real JPMorgan example) that capping raw dollar columns before ratio construction inflates ROA by ~18x — treatment moved to the ratio level instead.
- **Panel-data leakage:** companies appear multiple times each across quarters; a random split would leak company identity across train/test — fixed with `GroupShuffleSplit`/`GroupKFold` grouped by company ID, locked once in stage 07 and verified zero-overlap through every later stage.
- **Feature-alignment bug at inference time:** one-hot-encoded industry columns silently misaligned for small prediction batches (missing sector columns for sectors absent from the sample) — caught via testing, fixed with explicit column reindexing (`reindex(columns=EXPECTED_COLS, fill_value=0)`).

---

## Repository Structure

```
cross-industry-financial-analysis_Project/
├── data/
│   ├── raw/              # Original SEC quarterly zips
│   └── processed/        # Pipeline outputs at each stage, incl. company_split.json
├── models/                # Persisted model artifacts (health_classifier.joblib, feature_config.json, winsorization_bounds.json)
├── src/                   # Stage 1-3 pipeline scripts
├── notebooks/             # Stage 4-13 notebooks
├── reports/
│   └── figures/           # All generated charts, organized by stage
├── requirements.txt
└── README.md
```

---

## Tech Stack

Python · pandas · pyarrow (Parquet) · NumPy · scikit-learn · matplotlib · seaborn · scipy · statsmodels · joblib · Jupyter

---

## Data Source Attribution

Data sourced from the U.S. Securities and Exchange Commission's [Financial Statement Data Sets](https://www.sec.gov/data/financial-statement-data-sets), a public dataset with no usage restrictions for research/analysis purposes.
