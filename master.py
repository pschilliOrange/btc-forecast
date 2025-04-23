# prepare_gbm_data.py – orchestration script for BTC‑GBM HTML dashboard
# ---------------------------------------------------------------------
# ‑ Purpose ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
#   * Inspects **market_probabilities.json** to discover the furthest‑dated
#     option contract and infers the horizon for the visualisations.
#   * Reads **btc_price.json** for the most recent spot level & timestamp.
#   * Creates an evenly‑spaced set of future ISO‑8601 instants (UTC)
#     on that time range for which probability density parameters are
#     required.
#   * Invokes **fit_gbm_and_export(...)** from *fit_btc_gbm.py* to calibrate
#     a geometric Brownian motion (GBM) and append the resulting PDF params
#     to **pdfsFromFit/GBMforHTML.json** – the single data source consumed by
#     the front‑end dashboard.
#
# ‑ Usage (CLI) ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
#     $ python prepare_gbm_data.py                  # default paths
#     $ python prepare_gbm_data.py  --slices 40     # customise #pdf slices
#     $ python prepare_gbm_data.py  --markets my_markets.json  \
#                                            --price btc_price.json
#
#   The script is safely idempotent – rerunning augments or replaces the
#   matching entry in *GBMforHTML.json* without duplication.
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# local import – assumes script lives next to *fit_btc_gbm.py*
from fitGBM import fit_gbm_and_export
from getBtcPrice import get_btc_price_and_time
from getOptionPrices import get_option_prices

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _iso_z(dt: datetime) -> str:
    """Return an ISO‑8601 string with explicit Z suffix (UTC)."""
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_pdf_schedule(start: datetime, end: datetime, n: int) -> List[str]:
    if end <= start:
        raise ValueError("Latest option expiry must be after price timestamp")
    step = (end - start) / n
    return [_iso_z(start + step * i) for i in range(1, n + 1)]


# ─────────────────────────────────────────────────────────────────────────────
# main routine
# ─────────────────────────────────────────────────────────────────────────────

def main():
    get_btc_price_and_time()
    get_option_prices()

    parser = argparse.ArgumentParser(
        description="Prepare json inputs for the BTC GBM HTML dashboard"
    )
    parser.add_argument(
        "--markets",
        default="market_probabilities.json",
        help="Path to market_probabilities.json",
    )
    parser.add_argument(
        "--price",
        default="btc_price.json",
        help="Path to btc_price.json containing current BTC spot",
    )
    parser.add_argument(
        "--slices",
        type=int,
        default=30,
        help="Number of equally spaced PDF timestamps between now and latest option expiry",
    )
    parser.add_argument(
        "--out",
        default="pdfsFromFit/GBMforHTML.json",
        help="Destination JSON appended/updated by fit_gbm_and_export",
    )

    args = parser.parse_args()

    markets_path = Path(args.markets)
    price_path = Path(args.price)
    out_path = Path(args.out)

    # --- load input files ----------------------------------------------------
    markets = _load_json(markets_path)
    pxinfo = _load_json(price_path)

    # current spot timestamp
    start = datetime.fromisoformat(pxinfo["timestamp"].replace("Z", "+00:00"))

    # furthest contract expiry defines the horizon
    latest_expiry = max(
        datetime.fromisoformat(m["expiration"].replace("Z", "+00:00")) for m in markets
    )

    # build schedule of PDF evaluation times
    pdf_times = _build_pdf_schedule(start, latest_expiry, args.slices)

    # NB: for now we calibrate to *all* contracts; future versions may pass a
    # list of slugs to restrict the fit (via --slugs a.txt or similar)
    print(
        f"[info] Calibrating GBM from {len(markets)} contracts, "
        f"{args.slices} PDF slices ({pdf_times[0]} – {pdf_times[-1]})"
    )

    result = fit_gbm_and_export(
        market_file=str(markets_path),
        price_file=str(price_path),
        pdf_timestamps=pdf_times,
        market_slugs=None,
        out_file=str(out_path),
    )

    print(
        f"[done] Updated {out_path} «μ̂ = {result['mu_hat']:.6f}, "
        f"σ̂ = {result['sigma_hat']:.4f}, rss={result['rss']:.3e}»"
    )


if __name__ == "__main__":
    main()
