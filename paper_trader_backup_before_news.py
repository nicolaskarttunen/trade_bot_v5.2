# V5.2 PAPER TRADING RUNNER
# Alpaca Paper Trading only

import os
import json
import pickle
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from features import enrich, FEATURES


BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env.v5")

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
PAPER = os.getenv("ALPACA_PAPER", "").lower()

if not API_KEY or not SECRET_KEY:
    raise SystemExit("API-tiedot puuttuvat .env.v5-tiedostosta.")

if PAPER not in ("true", "1", "yes"):
    raise SystemExit("TURVAVIRHE: ALPACA_PAPER ei ole true. BOTTI PYSÄYTETTY.")

cfg = json.loads((BASE / "config.json").read_text())

with open(BASE / "models" / "ensemble.pkl", "rb") as f:
    model_data = pickle.load(f)

trading = TradingClient(
    API_KEY,
    SECRET_KEY,
    paper=True
)

data_client = StockHistoricalDataClient(
    API_KEY,
    SECRET_KEY
)

print("=" * 70)
print("V5.2 AI DAY TRADER - PAPER")
print("=" * 70)
print("PAPER MODE: TRUE")
print("SYMBOLS:", ", ".join(cfg["symbols"]))
print("TIMEFRAME:", cfg["timeframe_minutes"], "min")
print("MAX TRADES/DAY:", cfg["max_trades_per_day"])
print("MAX DAILY LOSS:", cfg["max_daily_loss_pct"] * 100, "%")
print("MODEL:", len(model_data["models"]), "models")
print("FEATURES:", len(model_data["features"]))
print("=" * 70)


def market_clock():
    return trading.get_clock()


def get_bars(symbol, days=3):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame(
            cfg["timeframe_minutes"],
            TimeFrameUnit.Minute
        ),
        start=start,
        end=end,
        feed=DataFeed.IEX
    )

    result = data_client.get_stock_bars(request)
    rows = result.data.get(symbol, [])

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([
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


def get_benchmark():
    return get_bars(cfg["benchmark_symbols"][0])


def get_ml_probability(df, benchmark):
    x = enrich(
        df,
        market=benchmark,
        news=None,
        news_decay_minutes=cfg["news_decay_minutes"],
        news_lookback_minutes=cfg["news_lookback_minutes"]
    )

    x = x.dropna(subset=FEATURES)

    if len(x) == 0:
        return None, None

    latest = x.iloc[-1]
    X = latest[FEATURES].to_numpy(dtype=float).reshape(1, -1)

    probabilities = [
        float(model.predict_proba(X)[0, 1])
        for model in model_data["models"]
    ]

    return sum(probabilities) / len(probabilities), latest


def write_trade(row):
    csv = BASE / "trades.csv"

    header = [
        "timestamp",
        "symbol",
        "side",
        "qty",
        "price",
        "order_id",
        "pnl",
        "ml_prob",
        "score",
        "news_score",
        "reason"
    ]

    exists = csv.exists()

    pd.DataFrame([row], columns=header).to_csv(
        csv,
        mode="a",
        header=not exists,
        index=False
    )


def account_status():
    account = trading.get_account()

    return {
        "equity": float(account.equity),
        "cash": float(account.cash),
        "buying_power": float(account.buying_power)
    }


def open_positions():
    return trading.get_all_positions()


def main():
    print()
    print("Paper-botti käynnistetty.")
    print("Oikeaa rahaa EI käytetä.")
    print("Odotetaan markkinaa...")

    while True:
        try:
            clock = market_clock()

            if not clock.is_open:
                print(
                    datetime.now().strftime("%H:%M:%S"),
                    "| Market closed | Next open:",
                    clock.next_open
                )
                time.sleep(60)
                continue

            print()
            print("=" * 70)
            print("MARKET OPEN")
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print("=" * 70)

            status = account_status()

            print(
                "Equity:",
                f"${status['equity']:,.2f}",
                "| Cash:",
                f"${status['cash']:,.2f}"
            )

            for symbol in cfg["symbols"]:
                try:
                    df = get_bars(symbol)

                    if len(df) < 50:
                        continue

                    benchmark = get_benchmark()

                    if len(benchmark) < 50:
                        continue

                    ml_prob, latest = get_ml_probability(
                        df,
                        benchmark
                    )

                    if ml_prob is None:
                        continue

                    print(
                        symbol,
                        "| Close:",
                        round(float(latest.close), 2),
                        "| ML:",
                        round(ml_prob, 4)
                    )

                except Exception as e:
                    print(symbol, "| ERROR:", repr(e))

            time.sleep(300)

        except KeyboardInterrupt:
            print()
            print("V5.2 paper-botti pysäytetty.")
            break

        except Exception as e:
            print("RUNNER ERROR:", repr(e))
            time.sleep(30)


if __name__ == "__main__":
    main()
