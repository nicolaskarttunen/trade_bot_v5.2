from pathlib import Path
from datetime import datetime, timedelta, timezone
import os
import json
import pickle
import pandas as pd

from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from features import enrich, FEATURES

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env.v5")

cfg = json.loads((BASE / "config.json").read_text())

with open(BASE / "models" / "ensemble.pkl", "rb") as f:
    model_data = pickle.load(f)

client = StockHistoricalDataClient(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY")
)

symbol = "NVDA"

request = StockBarsRequest(
    symbol_or_symbols=[symbol],
    timeframe=TimeFrame(5, TimeFrameUnit.Minute),
    start=datetime.now(timezone.utc) - timedelta(days=2),
    end=datetime.now(timezone.utc),
    feed=DataFeed.IEX
)

bars = client.get_stock_bars(request)

rows = bars.data.get(symbol, [])

if not rows:
    print("EI DATAA: markkina on todennakoisesti kiinni.")
    raise SystemExit(0)

df = pd.DataFrame([
    {
        "timestamp": b.timestamp,
        "open": float(b.open),
        "high": float(b.high),
        "low": float(b.low),
        "close": float(b.close),
        "volume": float(b.volume)
    }
    for b in rows
])

bench_request = StockBarsRequest(
    symbol_or_symbols=["SPY"],
    timeframe=TimeFrame(5, TimeFrameUnit.Minute),
    start=df["timestamp"].min(),
    end=df["timestamp"].max(),
    feed=DataFeed.IEX
)

bench_bars = client.get_stock_bars(bench_request)
bench_rows = bench_bars.data.get("SPY", [])

bench = pd.DataFrame([
    {
        "timestamp": b.timestamp,
        "close": float(b.close),
        "atr_pct": 0.0
    }
    for b in bench_rows
])

if len(df) < 30:
    print("DATAA LIIAN VÄHÄN:", len(df), "bars")
    raise SystemExit(0)

x = enrich(
    df,
    market=bench,
    news=None,
    news_decay_minutes=cfg["news_decay_minutes"],
    news_lookback_minutes=cfg["news_lookback_minutes"]
)

x = x.dropna(subset=FEATURES)

latest = x.iloc[-1]
X = latest[FEATURES].to_numpy(dtype=float).reshape(1, -1)

probs = [
    float(m.predict_proba(X)[0, 1])
    for m in model_data["models"]
]

ml_prob = sum(probs) / len(probs)

print("=" * 60)
print("V5.2 SIGNAL TEST")
print("=" * 60)
print("Symbol:", symbol)
print("Bars:", len(df))
print("Latest:", latest["timestamp"])
print("Close:", latest["close"])
print("ML probabilities:", [round(p, 4) for p in probs])
print("ENSEMBLE ML PROB:", round(ml_prob, 4))
print("News score:", latest["news_score"])
print("ATR %:", round(float(latest["atr_pct"]) * 100, 4))
print("=" * 60)


