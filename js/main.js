// js/main.js  – app shell: tabs + bootstrap
// -----------------------------------------

import * as GBM from "./models/GBM.js";

// --------- tab navigation ---------------------------------------------------
function initTabs() {
  const buttons = document.querySelectorAll("nav button[data-target]");

  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      // visual state
      buttons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      const targetId = btn.dataset.target;
      document.querySelectorAll("main section").forEach(sec => {
        sec.classList.toggle("active", sec.id === targetId);
      });

      // let model module handle any resize housekeeping
      if (targetId === "gbm") GBM.resize();
    });
  });
}

// --------- bootstrap --------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
  initTabs();
  GBM.init().catch(err => {
    console.error(err);
    const hm = document.getElementById("heatmap");
    hm.innerHTML = `<p style="color:#ff5555;padding:1rem">${err.message}</p>`;
  });
});
