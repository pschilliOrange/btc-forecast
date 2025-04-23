// js/models/GBM.js  – data fetch & charting for the GBM tab
// ---------------------------------------------------------

import { renderHeatmap }    from "../charts/heatmap.js";
import { renderPdfFan   }   from "../charts/pdfFan.js";
import { renderSummary }    from "../charts/summaryTable.js";
import { renderDiagnostics }from "../charts/contractTable.js";

// -------- helpers (private) -------------------------------------------------
function logNormalPDF(s, m, v) {
  return (1 / (s * v * Math.sqrt(2 * Math.PI))) *
         Math.exp(-Math.pow(Math.log(s) - m, 2) / (2 * v * v));
}

function linspace(a, b, n) {
  const Step = (b - a) / (n - 1);
  return Array.from({ length: n }, (_, i) => a + Step * i);
}

// model‑agnostic hook – future models can expose the same interface
function densityAtPrice(slice, price) {
  // GBM slices carry { m, v }
  return logNormalPDF(price, slice.m, slice.v);
}

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------
export async function init() {
  // 1. pull latest calibration from JSON -------------------------------
  const resp = await fetch("pdfsFromFit/GBMforHTML.json");
  if (!resp.ok) throw new Error(`Failed to load JSON (${resp.status})`);
  const json = await resp.json();
  if (!Array.isArray(json) || !json.length)
    throw new Error("Empty JSON array");

  const latest  = json[json.length - 1];
  const { time_btc_price_pulled, S0, mu_hat, sigma_hat, rss } = latest;
  const slices  = latest.pdf_params;          // [{timestamp,m,v}, …]
  const contracts = latest.contracts;

  // 0. summary banner ---------------------------------------------------
  renderSummary("model-info", {
    timestamp : time_btc_price_pulled,
    S0,
    mu : mu_hat,
    sigma : sigma_hat,
    sde : "dSₜ = μ Sₜ dt + σ Sₜ dWₜ"        // will be swapped for other models
  });

  // 2. shared price grid -----------------------------------------------
  let minP = Infinity, maxP = -Infinity;
  for (const { m, v } of slices) {
    minP = Math.min(minP, Math.exp(m - 4 * v));
    maxP = Math.max(maxP, Math.exp(m + 4 * v));
  }
  const priceGrid = linspace(minP, maxP, 160);

  // 3. heat‑map z‑matrix ------------------------------------------------
  const zMatrix = priceGrid.map(p => slices.map(s => densityAtPrice(s, p)));
  const xDates  = slices.map(s => s.timestamp);

  renderHeatmap("heatmap", {
    xDates,
    yPrices : priceGrid,
    zMatrix,
    title   : "Probability Density over Time (GBM)"
  });

  // 4. pdf‑fan line series ---------------------------------------------
  const SLICE_MOD = Math.ceil(slices.length / 10);
  const fanSeries = slices
    .filter((_, i) => i % SLICE_MOD === 0 || i === slices.length - 1)
    .map(s => ({
      label   : new Date(s.timestamp).toLocaleDateString("en-US",
                 { month: "short", day: "numeric" }),
      density : priceGrid.map(p => densityAtPrice(s, p))
    }));

  renderPdfFan("pdf-fan", {
    priceGrid,
    series : fanSeries,
    title  : "PDF Evolution (GBM)"
  });

  // 5. diagnostics table -----------------------------------------------
  renderDiagnostics("diagnostics", { contracts, rss });
}

// Called by main.js whenever the GBM tab becomes visible
export function resize() {
  ["heatmap", "pdf-fan"].forEach(id => {
    const el = document.getElementById(id);
    if (el && el.data) Plotly.Plots.resize(el);
  });
}
