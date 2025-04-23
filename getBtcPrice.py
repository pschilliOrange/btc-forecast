import requests
import datetime
import json

BASE_URL = "https://api.binance.us"

def get_btc_price_and_time(filename="btc_price.json"):
    # Fetch price
    pr = requests.get(f"{BASE_URL}/api/v3/ticker/price",
                      params={"symbol": "BTCUSDT"}, timeout=10)
    pr.raise_for_status()
    price = float(pr.json()["price"])

    # Fetch server time
    tm = requests.get(f"{BASE_URL}/api/v3/time", timeout=10)
    tm.raise_for_status()
    server_time_ms = tm.json()["serverTime"]

    # Modern, timezone-aware UTC
    ts = datetime.datetime.fromtimestamp(server_time_ms / 1000.0,
                                         tz=datetime.timezone.utc)
    timestamp = ts.isoformat()

    # Write out to JSON
    data = {
        "price": price,
        "timestamp": timestamp
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {price} @ {timestamp} into {filename}")

    return price, timestamp

if __name__ == "__main__":
    get_btc_price_and_time()
