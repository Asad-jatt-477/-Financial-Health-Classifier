async function loadJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  return res.json();
}

/* ---- Chart primitives (plain SVG, no library) ---- */

const CHART_FONT = "font-family='Inter, sans-serif'";

function niceTicks(max, count = 4) {
  const step = max / count;
  return Array.from({ length: count + 1 }, (_, i) => +(step * i).toFixed(6));
}

/** Vertical column chart. data: [{label, value, color, sub?}] */
function vBarChart(data, { max, format = v => v.toFixed(2), height = 220, title = "" } = {}) {
  const barW = 64, gap = 36, padL = 46, padR = 16, padT = title ? 34 : 18, padB = 34;
  const plotW = data.length * (barW + gap) - gap;
  const width = padL + plotW + padR;
  const plotH = height - padT - padB;
  const m = max ?? Math.max(...data.map(d => d.value)) * 1.15;
  const ticks = niceTicks(m, 4);

  let s = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" ${CHART_FONT}>`;
  if (title) s += `<text x="${padL}" y="18" font-size="12" font-weight="600" fill="#14213D">${title}</text>`;
  ticks.forEach(t => {
    const y = padT + plotH - (t / m) * plotH;
    s += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${width - padR}" y2="${y.toFixed(1)}" stroke="#E3DECF" stroke-width="1"/>`;
    s += `<text x="${padL - 8}" y="${(y + 3.5).toFixed(1)}" text-anchor="end" font-size="10" fill="#4A5468">${format(t)}</text>`;
  });
  s += `<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" stroke="#B8B09A" stroke-width="1"/>`;
  s += `<line x1="${padL}" y1="${padT + plotH}" x2="${width - padR}" y2="${padT + plotH}" stroke="#B8B09A" stroke-width="1"/>`;

  data.forEach((d, i) => {
    const x = padL + i * (barW + gap);
    const h = (d.value / m) * plotH;
    const y = padT + plotH - h;
    s += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW}" height="${h.toFixed(1)}" fill="${d.color}"/>`;
    if (d.errHigh !== undefined) {
      const yErrTop = padT + plotH - (d.errHigh / m) * plotH;
      const yErrBot = padT + plotH - (d.errLow / m) * plotH;
      const cx = x + barW / 2;
      s += `<line x1="${cx}" y1="${yErrTop.toFixed(1)}" x2="${cx}" y2="${yErrBot.toFixed(1)}" stroke="#14213D" stroke-width="1.3"/>`;
      s += `<line x1="${cx-6}" y1="${yErrTop.toFixed(1)}" x2="${cx+6}" y2="${yErrTop.toFixed(1)}" stroke="#14213D" stroke-width="1.3"/>`;
      s += `<line x1="${cx-6}" y1="${yErrBot.toFixed(1)}" x2="${cx+6}" y2="${yErrBot.toFixed(1)}" stroke="#14213D" stroke-width="1.3"/>`;
    }
    s += `<text x="${(x + barW/2).toFixed(1)}" y="${(y - 8).toFixed(1)}" text-anchor="middle" font-size="12" font-weight="700" fill="#14213D">${format(d.value)}</text>`;
    const words = d.label.split(" ");
    words.forEach((w, wi) => {
      s += `<text x="${(x + barW/2).toFixed(1)}" y="${(padT + plotH + 16 + wi*12).toFixed(1)}" text-anchor="middle" font-size="10.5" fill="#4A5468">${w}</text>`;
    });
  });
  s += `</svg>`;
  return s;
}

/** Horizontal bar chart. data: [{label, value}] */
function hBarChart(data, { max, format = v => v.toFixed(3), rowH = 26, color = "#14213D", width = 620 } = {}) {
  const padL = 128, padR = 54, padT = 10, padB = 24;
  const plotW = width - padL - padR;
  const plotH = data.length * rowH;
  const height = padT + plotH + padB;
  const m = max ?? Math.max(...data.map(d => d.value)) * 1.08;
  const ticks = niceTicks(m, 4);

  let s = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" ${CHART_FONT}>`;
  ticks.forEach(t => {
    const x = padL + (t / m) * plotW;
    s += `<line x1="${x.toFixed(1)}" y1="${padT}" x2="${x.toFixed(1)}" y2="${padT + plotH}" stroke="#E3DECF" stroke-width="1"/>`;
    s += `<text x="${x.toFixed(1)}" y="${padT + plotH + 16}" text-anchor="middle" font-size="10" fill="#4A5468">${format(t)}</text>`;
  });
  data.forEach((d, i) => {
    const y = padT + i * rowH;
    const w = (d.value / m) * plotW;
    s += `<text x="${padL - 10}" y="${(y + rowH/2 + 4).toFixed(1)}" text-anchor="end" font-size="11" fill="#14213D">${d.label}</text>`;
    s += `<rect x="${padL}" y="${(y + rowH*0.22).toFixed(1)}" width="${Math.max(w,1.5).toFixed(1)}" height="${(rowH*0.56).toFixed(1)}" fill="${d.color || color}"/>`;
    s += `<text x="${(padL + w + 8).toFixed(1)}" y="${(y + rowH/2 + 4).toFixed(1)}" font-size="10.5" fill="#4A5468">${format(d.value)}</text>`;
  });
  s += `<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT+plotH}" stroke="#B8B09A" stroke-width="1"/>`;
  s += `</svg>`;
  return s;
}
