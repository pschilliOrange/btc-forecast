#!/usr/bin/env python3
# fitGBM.py  – helper for calibrating a GBM to binary / barrier BTC markets
# -----------------------------------------------------------------------------
#  New public API
#  --------------
#      fit_gbm_and_export(
#          market_file: str,
#          price_file : str,
#          pdf_timestamps: list[str],
#          market_slugs : list[str] | None = None,
#          out_file: str = "pdfsFromFit/GBMforHTML.json",
#      ) -> dict
#
#      •  pdf_timestamps  – ISO‑8601 UTC instants (“2025‑05‑01T00:00:00Z”, …)
#      •  market_slugs    – restrict the calibration to this subset of markets
#                           (pass None to use *all* contracts in market_file)
#      •  returns the dict that was just written / appended to out_file
#
#  The JSON entry written to *out_file* looks like
#  {
#    "market_slugs": ["will-bitcoin-dip-to-70k-in-april", …],
#    "calibration_time": "2025‑04‑20T15:42:07.123456Z",
#    "S0": 71234.56,
#    "mu_hat":  0.015237,
#    "sigma_hat": 0.6423,
#    "rss": 0.001172,
#    "contracts": [
#        {"slug":"will-bitcoin-dip-to-70k-in-april",
#         "P_model":0.0632,"P_market":0.0598,"abs_err":0.0034},
#        …
#    ],
#    "pdf_params": [
#        {"timestamp":"2025‑05‑01T00:00:00Z",
#         "T_years":0.1077,
#         "m": 11.1702,
#         "v": 0.2137},
#        …
#    ]
#  }
# -----------------------------------------------------------------------------


#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
# GBM calibration (μ, σ) to market‑implied probabilities for BTC – 20 Apr 2025
# ──────────────────────────────────────────────────────────────────────────────
# For each listed contract we have a quoted probability **Pᵐᵏᵗ** that either
#   • the BTC/USD spot Sₜ *touches* a barrier H (one‑touch, up‑ or down‑and‑in),
#   • the spot *exceeds* / *falls below* a level K at expiry (one‑sided binary),
#   • the spot *lies inside* an interval (A, B) at expiry (range binary).
#
# Under geometric Brownian motion
#     dSₜ = μ Sₜ dt + σ Sₜ dWₜ
# we have ln S_T ~ 𝒩(ln S₀ + (μ − ½σ²)T, σ²T).
#
# Closed‑form probabilities
# ─────────────────────────
#   • Barrier touch (FPT)                     P_touch = Φ(z₁) + (S₀/H)^{2α} Φ(z₂)
#   • One‑sided digital                       P_bin   = 1 − Φ(d) (>) or Φ(d) (<)
#   • Range digital                           P_rng   = Φ(u) − Φ(l)
#     where α = μ/σ² − ½ and z₁,z₂,d,u,l follow the derivation in the notebook.
#
# We minimise  Σᵢ (P_modelᵢ − P_marketᵢ)²  over (μ, σ) with σ>0.
# Re‑parametrise σ = exp(θ) so the optimiser is unconstrained in θ.
#
# Output
# ──────
#   • Calibrated (μ̂, σ̂) in annualised units (μ ≈ drift, σ ≈ vol).
#   • A per‑contract line:  model P, market P, absolute error.

numIntervals = 100


import json
import os
from datetime import datetime, timezone
from math import log, sqrt, exp
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.stats import norm
from scipy.optimize import least_squares

# --------------------------------------------------------------------------
# Utility – year fraction ACT/365F
# --------------------------------------------------------------------------
def _yearfrac(t_start: datetime, t_end: datetime) -> float:
    return (t_end - t_start).total_seconds() / (365.0 * 24 * 3600)


# --------------------------------------------------------------------------
#  *** analytic GBM probability helpers (unchanged) ***
# --------------------------------------------------------------------------
def _common_terms(S0, T, mu, sigma):
    m = mu - 0.5 * sigma**2
    sig_sqrt_T = sigma * sqrt(T)
    return m, sig_sqrt_T


def prob_touch(S0, H, T, mu, sigma):
    if T <= 0:
        return 0.0
    m = mu - 0.5 * sigma**2
    v = sigma * sqrt(T)
    alpha = mu / sigma**2 - 0.5

    if H > S0:                                     # up‑barrier
        b = log(H / S0)
        z1 = (b - m * T) / v
        z2 = (-b - m * T) / v
        return norm.cdf(-z1) + (S0 / H) ** (2 * alpha) * norm.cdf(-z2)
    else:                                          # down‑barrier
        b = log(S0 / H)
        z1 = (b + m * T) / v
        z2 = (b - m * T) / v
        return norm.cdf(-z1) + (H / S0) ** (2 * alpha) * norm.cdf(-z2)


def prob_above(S0, K, T, mu, sigma):
    if T <= 0:
        return 1.0 if S0 > K else 0.0
    m, v = _common_terms(S0, T, mu, sigma)
    d = (log(K / S0) - m * T) / v
    return 1.0 - norm.cdf(d)


def prob_below(S0, K, T, mu, sigma):
    return 1.0 - prob_above(S0, K, T, mu, sigma)


def prob_range(S0, A, B, T, mu, sigma):
    if A >= B:
        raise ValueError("Range digital requires A < B")
    m, v = _common_terms(S0, T, mu, sigma)
    u = (log(B / S0) - m * T) / v
    l = (log(A / S0) - m * T) / v
    return norm.cdf(u) - norm.cdf(l)


# --------------------------------------------------------------------------
#  Public API
# --------------------------------------------------------------------------
def fit_gbm_and_export(
    market_file: str,
    price_file: str,
    pdf_timestamps: List[str],
    market_slugs: Optional[List[str]] = None,
    out_file: str = "pdfsFromFit/GBMforHTML.json",
):
    """
    Calibrate a GBM to the chosen market contracts and append results (including
    distribution parameters for the requested pdf_timestamps) to *out_file*.
    """
    # ---- read inputs ------------------------------------------------------
    with open(market_file, "r") as fh:
        all_markets = json.load(fh)
    with open(price_file, "r") as fh:
        pxinfo = json.load(fh)

    S0 = float(pxinfo["price"])
    px_ts = datetime.fromisoformat(pxinfo["timestamp"].replace("Z", "+00:00"))

    # optional contract filtering
    markets = (
        [m for m in all_markets if m["market_slug"] in market_slugs]
        if market_slugs is not None
        else all_markets
    )
    if not markets:
        raise ValueError("No contracts left after filtering by market_slugs")

    # ---- build residuals --------------------------------------------------
    P_market, funcs, labels = [], [], []
    for mkt in markets:
        slug = mkt["market_slug"]
        expiry = datetime.fromisoformat(mkt["expiration"].replace("Z", "+00:00"))
        T = _yearfrac(px_ts, expiry)
        P_mkt = float(mkt["probability"])
        P_market.append(P_mkt)
        labels.append(slug)

        if mkt["option_type"] == "one-touch":
            H = float(mkt["barrier"])
            funcs.append(lambda mu, s, S0=S0, H=H, T=T: prob_touch(S0, H, T, mu, s))
        elif mkt["option_type"] == "binary":
            lo, up = mkt.get("lower"), mkt.get("upper")
            if lo and up:
                A, B = float(lo), float(up)
                funcs.append(lambda mu, s, S0=S0, A=A, B=B, T=T: prob_range(S0, A, B, T, mu, s))
            elif lo:
                K = float(lo)
                funcs.append(lambda mu, s, S0=S0, K=K, T=T: prob_above(S0, K, T, mu, s))
            elif up:
                K = float(up)
                funcs.append(lambda mu, s, S0=S0, K=K, T=T: prob_below(S0, K, T, mu, s))
            else:
                raise ValueError(f"Binary '{slug}' missing strike info")
        else:
            raise ValueError(f"Unknown option_type '{mkt['option_type']}'")

    P_market = np.asarray(P_market)

    def _residuals(x):
        mu, ln_sig = x
        sig = exp(ln_sig)
        model = np.fromiter((f(mu, sig) for f in funcs), dtype=float)
        return model - P_market

    # ---- least squares ----------------------------------------------------
    x0 = np.array([0.0, log(0.65)])
    bnds = ([-np.inf, log(0.01)], [np.inf, log(4.0)])
    res = least_squares(_residuals, x0, bounds=bnds, xtol=1e-12, ftol=1e-12)

    mu_hat, sigma_hat = res.x[0], exp(res.x[1])

    # ---- pdf parameters for requested times -------------------------------
    pdf_params = []
    for ts in pdf_timestamps:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        T = _yearfrac(px_ts, t)
        if T <= 0:
            raise ValueError(f"Requested timestamp {ts} is not after price timestamp")
        m = log(S0) + (mu_hat - 0.5 * sigma_hat**2) * T        # mean of ln S_T
        v = sigma_hat * sqrt(T)                                # st.dev. of ln S_T
        pdf_params.append({"timestamp": ts, "T_years": T, "m": m, "v": v})

    # ---- detailed contract diagnostics ------------------------------------
    contract_info = []
    for slug, f, Pm in zip(labels, funcs, P_market):
        Pmod = float(f(mu_hat, sigma_hat))
        contract_info.append(
            {"slug": slug, "P_model": Pmod, "P_market": Pm, "abs_err": abs(Pmod - Pm)}
        )

    # ---- assemble result object -------------------------------------------
    time_btc_price_pulled = pxinfo["timestamp"]
    result = {
        "market_slugs": labels,
        "time_btc_price_pulled": time_btc_price_pulled,
        "S0": S0,
        "mu_hat": mu_hat,
        "sigma_hat": sigma_hat,
        "rss": float(res.cost),
        "contracts": contract_info,
        "pdf_params": pdf_params,
    }

    # ---- persist to JSON (replace‑or‑append) -------------------------------
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(out_file):
        with open(out_file, "r+") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError:
                data = []
            if not isinstance(data, list):
                raise ValueError(f"{out_file} is not a JSON array")

            # replace existing entry with same market_slugs, else append
            for idx, entry in enumerate(data):
                if entry.get("market_slugs") == labels:
                    data[idx] = result
                    break
            else:
                data.append(result)

            fh.seek(0)
            json.dump(data, fh, indent=2)
            fh.truncate()
    else:
        with open(out_file, "w") as fh:
            json.dump([result], fh, indent=2)


    return result


# --------------------------------------------------------------------------
#  CLI passthrough for backwards compatibility
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, pprint
    from datetime import datetime, timezone
    import json

    mkt_file = sys.argv[1] if len(sys.argv) > 1 else "market_probabilities.json"
    px_file  = sys.argv[2] if len(sys.argv) > 2 else "btc_price.json"

    # load all markets & price info
    with open(mkt_file) as fh:
        markets = json.load(fh)
    with open(px_file) as fh:
        pxinfo = json.load(fh)

    # parse start time and latest expiry
    start  = datetime.fromisoformat(pxinfo["timestamp"].replace("Z", "+00:00"))
    latest = max(
        datetime.fromisoformat(m["expiration"].replace("Z", "+00:00"))
        for m in markets
    )

    # split into numIntervals equal intervals
    pdf_timestamps = [
        (start + (latest - start) * i/numIntervals)
            .astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        for i in range(1, 11)
    ]

    result = fit_gbm_and_export(
        mkt_file,
        px_file,
        pdf_timestamps=pdf_timestamps,
        market_slugs=None
    )

    pprint.pp(result)

