// renders per‑contract model vs market probabilities (+ RSS)
export function renderDiagnostics(containerId, { contracts, rss }) {
    const container = document.getElementById(containerId);
    if (!container) return;
  
    const rows = contracts.map(c => /*html*/`
      <tr>
        <td>${c.slug}</td>
        <td>${(100*c.P_market).toFixed(2)}%</td>
        <td>${(100*c.P_model ).toFixed(2)}%</td>
        <td>${c.abs_err.toExponential(2)}</td>
      </tr>`).join("");
  
    container.innerHTML = /*html*/`
      <table style="width:100%;border-collapse:collapse;margin-top:.75rem">
        <thead>
          <tr>
            <th>Contract</th><th>P<sub>market</sub></th>
            <th>P<sub>model</sub></th><th>|Error|</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
        <tfoot>
          <tr>
            <td colspan="3" style="text-align:right"><strong>Total&nbsp;RSS</strong></td>
            <td>${rss.toExponential(3)}</td>
          </tr>
        </tfoot>
      </table>`;
  }
  