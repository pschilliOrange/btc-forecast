// js/charts/heatmap.js  – pure Plotly heat‑map
// ----------------------------------------------------------

export function renderHeatmap(containerId, { xDates, yPrices, zMatrix, title }) {
    const trace = {
      x         : xDates,
      y         : yPrices,
      z         : zMatrix,
      type      : "heatmap",
      colorscale: "YlGnBu",
      contours  : { coloring: "heatmap" },
    };
  
    const layout = {
      title,
      xaxis : { title: "Date", type: "date", tickformat: "%b %d", zeroline: false },
      yaxis : { title: "BTC price (USD)", type: "log", autorange: true },
      margin: { l: 60, r: 20, t: 30, b: 50 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor : "rgba(0,0,0,0)",
    };
  
    Plotly.newPlot(containerId, [trace], layout, { responsive: true });
  }
  