#!/usr/bin/env python3
import json
import requests
import sys

# Map event slug to (expiration ISO date, option type)
EVENT_CONFIG = {
    "what-price-will-bitcoin-hit-in-april": (
        "2025-05-01T03:59:00Z",
        "one-touch"
    ),
    "what-price-will-bitcoin-hit-in-2025": (
        "2026-01-01T04:59:00Z",
        "one-touch"
    ),
    "bitcoin-price-on-april-25": (
        "2025-04-25T16:00:00Z",
        "binary"
    ),
    "bitcoin-above-85000-on-april-25": (
        "2025-04-25T16:00:00Z",
        "binary"
    ),
    "bitcoin-up-or-down-in-april": (
        "2025-04-25T16:00:00Z",
        "binary"
    )
}

# Map price-slug to its parameters (now including direction for one‑touchs)
PRICE_CONFIG = {
    # one‑touch markets
    "will-bitcoin-dip-to-70k-in-april":      { "direction": "down-and-in",   "barrier": 70000 },
    "will-bitcoin-reach-90k-in-april":       { "direction": "up-and-in",     "barrier": 90000 },
    "will-bitcoin-reach-95k-in-april":       { "direction": "up-and-in",     "barrier": 95000 },
    "will-bitcoin-reach-100k-in-april":      { "direction": "up-and-in",     "barrier": 100000 },

    # binary markets (single if only one bound, range if both)
    "will-the-price-of-bitcoin-be-less-than-79000-on-apr-25":   { "upper": 79000 },
    "will-the-price-of-bitcoin-be-greater-than-89000-on-apr-25": { "lower": 89000 },
    "will-the-price-of-bitcoin-be-between-79000-and-81000-on-apr-25": { "lower": 79000, "upper": 81000 },
    "will-the-price-of-bitcoin-be-between-81000-and-83000-on-apr-25": { "lower": 81000, "upper": 83000 },
    "will-the-price-of-bitcoin-be-between-83000-and-85000-on-apr-25": { "lower": 83000, "upper": 85000 },
    "will-the-price-of-bitcoin-be-between-85000-and-87000-on-apr-25": { "lower": 85000, "upper": 87000 },
    "will-the-price-of-bitcoin-be-between-87000-and-89000-on-apr-25": { "lower": 87000, "upper": 89000 },

    # these two already exist in PRICE_CONFIG but shown here for clarity
    "bitcoin-above-85000-on-april-25":        { "lower": 85000 },
    "bitcoin-up-or-down-in-april":            { "lower": 82914 },
}



# Base URL for the Polymarket CLOB API
CLOB_ENDPOINT = "https://clob.polymarket.com"
GAMMA_ENDPOINT = "https://gamma-api.polymarket.com"

# How many US dollars we're simulating a market order for
SLIPPAGE_USD = 1000.0

def simulate_market_buy(asks, usd_amount):
    """
    Simulate spending `usd_amount` walking the asks from lowest price up.
    If you run out of asks, assume the rest executes at price = 1.0.
    Returns the average price per share.
    """
    # sort asks by ascending price
    sorted_asks = sorted(asks, key=lambda lvl: float(lvl["price"]))
    
    remaining_usd = usd_amount
    total_shares = 0.0
    total_cost   = 0.0

    for lvl in sorted_asks:
        price = float(lvl["price"])
        size  = float(lvl["size"])       # shares available at this price level
        cap_usd = price * size           # max USD you can spend here

        spend = min(cap_usd, remaining_usd)
        shares = spend / price

        total_shares += shares
        total_cost   += spend
        remaining_usd -= spend

        if remaining_usd <= 0:
            break

    if remaining_usd > 0:
        # no more depth: assume price = 1.0 for all remaining USD
        fallback_price = 1.0
        spend = remaining_usd
        shares = spend / fallback_price

        total_shares += shares
        total_cost   += spend

    # avoid divide‑by‑zero if there were literally no asks
    return (total_cost / total_shares) if total_shares > 0 else 1.0


def simulate_market_sell(bids, usd_amount):
    """
    Simulate raising `usd_amount` walking the bids from highest price down.
    If you run out of bids, assume the rest executes at price = 0.0.
    Returns the average price per share received.
    """
    # sort bids by descending price
    sorted_bids = sorted(bids, key=lambda lvl: float(lvl["price"]), reverse=True)
    
    remaining_usd = usd_amount
    total_shares = 0.0
    total_recv   = 0.0

    for lvl in sorted_bids:
        price = float(lvl["price"])
        size  = float(lvl["size"])
        cap_usd = price * size

        recv = min(cap_usd, remaining_usd)
        shares = recv / price

        total_shares += shares
        total_recv   += recv
        remaining_usd -= recv

        if remaining_usd <= 0:
            break

    if remaining_usd > 0:
        # no more depth: assume price = 0.0 for all remaining USD
        # we model this as "selling" extra shares for zero proceeds
        # so numerator (total_recv) stays the same, denominator ↑→ average ↓
        fallback_shares = remaining_usd  # 1 share “per dollar” as a notional
        total_shares += fallback_shares
        # total_recv += 0 * fallback_shares  # no change
    return (total_recv / total_shares) if total_shares > 0 else 0.0


def get_option_prices():
    try:
        with open("markets_cache.json") as f:
            markets = json.load(f)
    except FileNotFoundError:
        print("Please create `markets_cache.json` with your market list.", file=sys.stderr)
        sys.exit(1)

    results = []
    for m in markets:
        slug = m.get("market_slug")

        # find the “Yes” side token_id
        yes_token = next(
            (t.get("token_id") for t in m.get("tokens", []) if t.get("outcome") == "Yes"),
            None
        )
        if not yes_token:
            print(f"Skipping {slug!r}: no Yes-token", file=sys.stderr)
            continue

        # fetch order book
        resp = requests.get(
            f"{CLOB_ENDPOINT}/book",
            params={"token_id": yes_token}
        )
        resp.raise_for_status()
        book = resp.json()

        # fetch event via Gamma API
        gamma_resp = requests.get(
            f"{GAMMA_ENDPOINT}/markets",
            params={"clob_token_ids": yes_token}
        )
        gamma_resp.raise_for_status()
        gamma_json = gamma_resp.json()
        if isinstance(gamma_json, dict):
            gamma_markets = gamma_json.get("markets", [])
        elif isinstance(gamma_json, list):
            gamma_markets = gamma_json
        else:
            gamma_markets = []

        if gamma_markets:
            gm = gamma_markets[0]
            events = gm.get("events") or []
            if isinstance(events, list) and events:
                ev = events[0]
                event_slug = ev.get("slug") or ev.get("name", "").lower().replace(' ', '-')
            else:
                event_slug = None
        else:
            event_slug = None

        # lookup expiration and option_type
        expiration, option_type = EVENT_CONFIG.get(event_slug, (None, None))

        avg_buy = simulate_market_buy(book.get("asks", []), SLIPPAGE_USD)
        avg_sell = simulate_market_sell(book.get("bids", []), SLIPPAGE_USD)
        probability = (avg_buy + avg_sell) / 2.0

         # **NEW**: attach barrier/range info
        price_info = PRICE_CONFIG.get(slug, {})

        # build result
        result = {
            "market_slug": slug,
            "expiration": expiration,
            "option_type": option_type,
            "probability": probability,
        }

        if option_type == "one-touch":
            # for one-touch: include direction and barrier
            result["direction"] = price_info.get("direction")
            result["barrier"]   = price_info.get("barrier")
        elif option_type == "binary":
            # for binary: include whatever bounds are present
            if "lower" in price_info:
                result["lower"] = price_info["lower"]
            if "upper" in price_info:
                result["upper"] = price_info["upper"]

        results.append(result)

    with open("market_probabilities.json", "w") as f:
        json.dump(results, f, indent=2)



if __name__ == "__main__":
    get_option_prices()
