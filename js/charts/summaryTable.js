// renders the “last update + μ/σ + S0 + SDE” banner
export function renderSummary(containerId, { timestamp, S0, mu, sigma, sde }) {
    const container = document.getElementById(containerId);
    if (!container) return;
  
    const fmtDate = new Date(timestamp).toLocaleString("en-US", {
      timeZone: "UTC", hour12: false,
    }).replace(",", "");
  
    container.innerHTML = /*html*/`
      <table style="width:100%;border-collapse:collapse;margin-bottom:.5rem">
        <tbody>
          <tr>
            <td><strong>Last update (UTC)</strong></td><td>${fmtDate}</td>
            <td><strong>BTC spot&nbsp;S₀</strong></td><td>$${S0.toLocaleString()}</td>
            <td><strong>μ̂</strong></td><td>${mu.toFixed(4)}</td>
            <td><strong>σ̂</strong></td><td>${sigma.toFixed(4)}</td>
          </tr>
          <tr>
            <td colspan="8" style="padding-top:.4rem;">
              <code style="font-size:90%">${sde}</code>
            </td>
          </tr>
        </tbody>
      </table>`;
  }
  