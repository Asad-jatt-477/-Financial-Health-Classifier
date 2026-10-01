# 📈 Cross-Industry Corporate Financial Health Classifier

**A binary classification system that predicts whether a publicly-traded company will be Healthy or At-Risk next quarter — built end-to-end on 10 quarters of raw SEC regulatory filings, deployed as a live FastAPI-backed web app, using only information genuinely available in the company's current filing.**

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-Live%20API-009688?logo=fastapi&logoColor=white">
  <img alt="Vercel" src="https://img.shields.io/badge/Vercel-Deployed-000000?logo=vercel&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/Status-Deployment--Ready-1E8E5A">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-lightgrey">
</p>

---

## Overview

The **Cross-Industry Corporate Financial Health Classifier** turns 10 quarters (2024 Q1 – 2026 Q2) of raw SEC Financial Statement Data Sets — ~36 million individual regulatory facts across 9 industry sectors — into a single, focused decision-support tool: predicting whether a company will be financially **Healthy** or **At-Risk** next quarter.

Unlike a typical modeling exercise that stops at reporting an accuracy number, this project's central contribution is a **rigorous root-cause investigation** into two data-quality bugs hidden inside the raw regulatory data, and the disciplined decision to drop a regression target once evidence showed it was learning a reporting-calendar artifact rather than real financial signal — rather than reporting an inflated metric.

The project was built as a complete, end-to-end system rather than a notebook exercise: data engineering, leakage-safe feature engineering, model training, statistical evaluation, and a deployed interactive web application are all implemented and working together as a single pipeline.

## Problem Statement

Predicting corporate financial health from regulatory filings faces a recurring set of problems:

- **Raw filings silently mix reporting periods.** 10-K (annual) filings report income-statement figures as full-year totals while 10-Q (quarterly) filings report per-quarter — mixed without correction, this corrupts every ratio built from them and can masquerade as real "growth."
- **Foreign filers can contaminate a "USD" dataset.** Companies reporting in local currency (JPY, KRW, etc.) will silently distort raw dollar columns if not filtered before aggregation.
- **A naive train/test split leaks company identity.** Each company appears many times across quarters — a careless split lets the same company's data appear in both train and test, overstating real-world performance.
- **A model that looks strong can be riding a near-tautological signal without being technically wrong.** If the prediction target is itself defined from two ratios, a model that leans heavily on those same two ratios isn't cheating — but it needs to be explained honestly, not presented as a mysterious multivariate discovery.

A system that reports a headline accuracy without an honest account of *why* it works is not trustworthy — it is a liability waiting to be found out. What is needed is a transparently-diagnosed, defensible answer to *which* signal the data can actually support.

## Our Solution

The project addresses this through a disciplined, evidence-first workflow rather than iterative hyperparameter tuning:

1. **Diagnose before modeling.** A within-company ratio-consistency check (same company, annual vs. quarterly filings) revealed `asset_turnover`/`roa` differing by 3–4x between filing types — the signature of a duration-mixing bug, caught before any model was trained on it.
2. **Fix at the source, verify on synthetic and real data.** Both the currency-contamination and duration-mixing bugs were fixed inside the pivot stage, then stress-tested against synthetic raw files with deliberately injected edge cases, and cross-checked against real filings — zero cell-level discrepancies against ground truth.
3. **Reframe the target only where evidence demanded it.** A naive calendar-based baseline *beat* the original revenue-growth regression model on the (then-contaminated) data — proof the regression target was mostly learning the reporting-calendar artifact. Regression and clustering were dropped; the single classification target that survived scrutiny was kept.
4. **Lock a leakage-safe validation methodology.** A company-grouped `GroupShuffleSplit` (never a random row split) was locked once and reused unchanged through every later stage — winsorization bounds, feature selection, tuning, and final evaluation all respect it.
5. **Select features on evidence, not intuition.** VIF, drop-one grouped-CV, and grouped permutation importance were used to resolve redundancy and confirm relevance, run on train companies only.
6. **Explain honestly, including the target's own construction.** The target is built from `current_ratio` and `NetIncomeLoss` thresholds one quarter forward — which directly explains why those two features dominate permutation importance. This is documented, not hidden.
7. **Deploy it.** The final model, and every number reported about it, is served through a live FastAPI backend and web frontend rather than left in a notebook.

## System Architecture

![Cross-Industry Financial Health Classifier architecture diagram](architecture-diagram.svg)

The diagram above shows the full pipeline the project executes end-to-end, from raw filings to a served prediction. Each stage consumes the previous stage's saved artifact (parquet / JSON), so the pipeline runs reproducibly end-to-end with no manual intervention between steps.

## What Makes This Different

Most "financial ML" portfolio projects fall into one of two categories: a model trained once and reported at face value, or a notebook full of experiments with no account of what was rejected and why. This project is neither.

- **Diagnosis comes before modeling.** The duration-mixing bug was caught by a within-company consistency check *before* any model was trained on the corrupted signal — not discovered after the fact by a suspiciously-good result.
- **Evidence overrides sunk cost.** A regression target the project had already built was dropped entirely once a naive baseline beat it — proof it was learning an artifact, not real signal — rather than being tuned further to paper over the problem.
- **Leakage-safety is a first-class constraint, not an afterthought.** The company-grouped train/test split is locked once and threaded through every later stage (outlier bounds, feature selection, tuning); it was also empirically tested against a naive row-level split to prove the risk was real, not theoretical.
- **The model's own behavior is explained, not just reported.** Feature-importance concentration in two ratios is traced directly to the target's own definition and documented — the same transparency a skeptical reviewer would demand.
- **It is deployed, not just demonstrated.** Every number on the live Performance page is read from the same pipeline artifacts used throughout the project, not hardcoded into a slide.

### Scenarios

| Scenario | A naive approach | This project |
|---|---|---|
| Model leans heavily on 2 features | Reported as an unexplained "key driver" finding | Traced to the target's own construction and documented |
| Regression target underperforms | Tuned further, or quietly dropped without explanation | Root-caused (reporting-duration bug), fixed at the source, then dropped only after re-confirming |
| Train/test split | Random row split | Company-grouped split, locked once, verified against a naive split to quantify the leakage risk |
| Missing values in `current_ratio` | Imputed with mean/median | Left as-is; model's native missing-value handling lets it use the pattern as signal |

## Final Results

| Metric | Value |
|---|---|
| Held-out test Accuracy | **87.62%** |
| Held-out test ROC-AUC | **0.9367** |
| Balanced Accuracy | 86.53% |
| Brier Score | 0.0918 |
| Beats persistence baseline by | +0.092 ROC-AUC |
| 5-fold CV ROC-AUC (GroupKFold) | 0.9427 ± 0.0036 |

Full baseline comparisons, cross-validation results, feature importance, and sector-level subgroup breakdowns are available on the live **Performance** page, rendered directly from `reports/` artifacts.

## Conclusion

The Cross-Industry Corporate Financial Health Classifier demonstrates a complete, evidence-driven approach to applied machine learning: rather than reporting whatever accuracy a model happens to produce, it investigates *why* two ratios dominate the model's decisions, identifies and fixes two distinct data-quality bugs at their source, locks a leakage-safe validation methodology, and reframes the prediction scope around what the data can honestly support. The result — a held-out test ROC-AUC of 0.9367 on a well-posed, transparently-explained binary question, backed by statistical evaluation and a deployed live application — is a defensible, deployment-ready system, which is the standard a genuinely trustworthy predictive model needs to meet.
