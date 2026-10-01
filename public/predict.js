const FIELD_META = {
  current_ratio:       { label: "Current Ratio",        group: "group-liquidity", fallback: [0, 5],    step: 0.01, help: "Current Assets ÷ Current Liabilities" },
  cash_ratio:          { label: "Cash Ratio",            group: "group-liquidity", fallback: [0, 3],    step: 0.01, help: "Cash ÷ Current Liabilities" },
  debt_to_equity:      { label: "Debt-to-Equity",        group: "group-liquidity", fallback: [-2, 6],   step: 0.01 },
  debt_to_assets:      { label: "Debt-to-Assets",        group: "group-liquidity", fallback: [0, 2],    step: 0.01 },
  operating_margin:    { label: "Operating Margin",      group: "group-profit",    fallback: [-1, 1],   step: 0.01 },
  roa:                 { label: "Return on Assets",      group: "group-profit",    fallback: [-1, 1],   step: 0.01 },
  roe:                 { label: "Return on Equity",      group: "group-profit",    fallback: [-2, 2],   step: 0.01 },
  asset_turnover:      { label: "Asset Turnover",        group: "group-profit",    fallback: [0, 2],    step: 0.01 },
  revenue_growth_qoq:    { label: "Revenue Growth QoQ",    group: "group-growth", fallback: [-1, 2],  step: 0.01 },
  netincome_growth_qoq:  { label: "Net Income Growth QoQ", group: "group-growth", fallback: [-2, 2],  step: 0.01 },
  assets_growth_qoq:     { label: "Assets Growth QoQ",     group: "group-growth", fallback: [-1, 2],  step: 0.01 },
};

const PRESETS = {
  healthy: {
    current_ratio: 2.4, cash_ratio: 0.9, debt_to_equity: 0.6, debt_to_assets: 0.30,
    operating_margin: 0.15, roa: 0.09, roe: 0.18, asset_turnover: 0.9,
    revenue_growth_qoq: 0.04, netincome_growth_qoq: 0.05, assets_growth_qoq: 0.03,
    is_annual_filing: false, has_inventory: true, industry_sector: "Manufacturing",
  },
  atrisk: {
    current_ratio: 0.7, cash_ratio: 0.08, debt_to_equity: 3.8, debt_to_assets: 0.85,
    operating_margin: -0.12, roa: -0.08, roe: -0.35, asset_turnover: 0.4,
    revenue_growth_qoq: -0.15, netincome_growth_qoq: -0.40, assets_growth_qoq: -0.05,
    is_annual_filing: false, has_inventory: true, industry_sector: "Retail",
  },
  borderline: {
    current_ratio: 1.03, cash_ratio: 0.20, debt_to_equity: 1.75, debt_to_assets: 0.60,
    operating_margin: 0.015, roa: 0.0, roe: 0.0, asset_turnover: 0.58,
    revenue_growth_qoq: -0.035, netincome_growth_qoq: -0.15, assets_growth_qoq: -0.005,
    is_annual_filing: false, has_inventory: true, industry_sector: "Services",
  },
};

let boundsCache = {};

function makeSlider(key, bounds) {
  const meta = FIELD_META[key];
  const b = bounds[key] || {};
  const lo = (b.neg_lower !== null && b.neg_lower !== undefined) ? b.neg_lower : meta.fallback[0];
  const hi = (b.pos_upper !== null && b.pos_upper !== undefined) ? b.pos_upper : meta.fallback[1];

  const field = document.createElement("div");
  field.className = "field";
  const defaultVal = PRESETS.borderline[key];
  field.innerHTML = `
    <label for="f_${key}">${meta.label}<span class="fv tabular" id="v_${key}">${defaultVal}</span></label>
    <input type="range" id="f_${key}" min="${lo.toFixed(2)}" max="${hi.toFixed(2)}" step="${meta.step}" value="${defaultVal}" title="${meta.help || ""}">
  `;
  return field;
}

async function init() {
  const cfg = await loadJSON("/api/config");
  boundsCache = cfg.bounds;

  for (const key of Object.keys(FIELD_META)) {
    document.getElementById(FIELD_META[key].group).appendChild(makeSlider(key, cfg.bounds));
    const input = document.getElementById(`f_${key}`);
    const out = document.getElementById(`v_${key}`);
    input.addEventListener("input", () => { out.textContent = Number(input.value).toFixed(2); });
  }

  const sectorSel = document.getElementById("sector");
  cfg.sector_categories.forEach(s => {
    const opt = document.createElement("option");
    opt.value = s; opt.textContent = s.replace("_", " / ");
    sectorSel.appendChild(opt);
  });
  sectorSel.value = PRESETS.borderline.industry_sector;
  sectorSel.addEventListener("change", updateSectorWarning);

  document.querySelectorAll("[data-preset]").forEach(btn => {
    btn.addEventListener("click", () => applyPreset(PRESETS[btn.dataset.preset]));
  });

  document.getElementById("predict-btn").addEventListener("click", submitPrediction);
  updateSectorWarning();
}

function applyPreset(preset) {
  for (const key of Object.keys(FIELD_META)) {
    const input = document.getElementById(`f_${key}`);
    const out = document.getElementById(`v_${key}`);
    input.value = preset[key];
    out.textContent = Number(preset[key]).toFixed(2);
  }
  document.getElementById("is_annual_filing").checked = preset.is_annual_filing;
  document.getElementById("has_inventory").checked = preset.has_inventory;
  document.getElementById("sector").value = preset.industry_sector;
  updateSectorWarning();
}

function updateSectorWarning() {
  const sector = document.getElementById("sector").value;
  document.getElementById("sector-warning").style.display = (sector === "Finance_RealEstate") ? "block" : "none";
}

function collectPayload() {
  const payload = {};
  for (const key of Object.keys(FIELD_META)) {
    payload[key] = Number(document.getElementById(`f_${key}`).value);
  }
  payload.is_annual_filing = document.getElementById("is_annual_filing").checked;
  payload.has_inventory = document.getElementById("has_inventory").checked;
  payload.industry_sector = document.getElementById("sector").value;
  return payload;
}

async function submitPrediction() {
  const btn = document.getElementById("predict-btn");
  btn.disabled = true;
  btn.textContent = "Predicting…";
  try {
    const result = await loadJSON("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload()),
    });
    const panel = document.getElementById("result");
    const isHealthy = result.label === "Healthy";
    panel.className = "result-panel show " + (isHealthy ? "healthy" : "atrisk");
    document.getElementById("verdict").textContent = isHealthy ? "Predicted: Healthy" : "Predicted: At-Risk";
    document.getElementById("prob").textContent = `P(Healthy) = ${(result.p_healthy * 100).toFixed(1)}% · threshold ${(result.threshold * 100).toFixed(0)}%`;
    document.getElementById("bar-fill").style.width = (result.p_healthy * 100).toFixed(1) + "%";
  } catch (e) {
    alert("Prediction failed: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict Financial Health";
  }
}

init();
