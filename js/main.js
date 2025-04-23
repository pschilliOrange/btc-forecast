// js/main.js  – app shell: tabs + bootstrap
// -----------------------------------------

import * as GBM from "./models/GBM.js";

// --------- tab navigation ---------------------------------------------------
async function loadContent(page) {
  const content = document.getElementById('content');
  try {
    const response = await fetch(`pages/${page}.html`);
    content.innerHTML = await response.text();
    if (page === 'gbm') {
      await GBM.init();
    }
  } catch (err) {
    console.error(err);
    content.innerHTML = `<p style="color:#ff5555;padding:1rem">Error loading ${page}</p>`;
  }
}

function initTabs() {
  const buttons = document.querySelectorAll("nav button[data-target]");

  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      buttons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      loadContent(btn.dataset.target);
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
