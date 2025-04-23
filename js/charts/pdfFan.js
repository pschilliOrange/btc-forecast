// js/charts/pdfFan.js  – pure Plotly PDF‑fan
// ----------------------------------------------------------

export function renderPdfFan(containerId, { priceGrid, series, title }) {
    const traces = series.map(({ label, density }) => ({
      x          : priceGrid,
      y          : density,
      mode       : "lines",
      name       : label,
      hovertemplate:
        "Price: $%{x:,.0f}<br>PDF: %{y:.2e}<extra></extra>",
    }));
  
    const layout = {
      title,
      xaxis : { title: "BTC price (USD)", type: "log" },
      yaxis : { title: "Probability density" },
      legend: { orientation: "h" },
      margin: { l: 60, r: 20, t: 30, b: 50 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor : "rgba(0,0,0,0)",
    };
  
    Plotly.newPlot(containerId, traces, layout, { responsive: true });
  }
  