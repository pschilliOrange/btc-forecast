
export function renderPdfFan(containerId, { priceGrid, series, title }) {
    // Filter series to only include one PDF per day
    const dailySeries = series.filter((s, i) => {
        const date = new Date(s.label);
        const prevDate = i > 0 ? new Date(series[i-1].label) : null;
        return !prevDate || date.getDate() !== prevDate.getDate();
    });

    const traces = dailySeries.map(({ label, density }) => ({
        x: priceGrid,
        y: density,
        mode: "lines",
        name: label,
        hovertemplate: "Price: $%{x:,.0f}<br>PDF: %{y:.2e}<extra></extra>",
    }));

    const layout = {
        title,
        xaxis: { title: "BTC price (USD)", type: "log" },
        yaxis: { title: "Probability density" },
        legend: { orientation: "h", y: -0.2 },
        margin: { l: 60, r: 20, t: 30, b: 80 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
    };

    Plotly.newPlot(containerId, traces, layout, { responsive: true });
}
