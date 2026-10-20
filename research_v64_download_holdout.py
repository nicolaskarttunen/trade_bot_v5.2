#!/usr/bin/env python3
"""Download strictly new, completed 2026-09-01..2026-10-19 IEX data for V6.4.

Read-only Alpaca historical data API. Writes only independent research/data_v64.
Does NOT alter the Oracle live-paper bot, credentials, production data, or orders.
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

PROD = Path("/home/opc/ai_daytrader_v5_1_python314")
DEST = Path("/home/opc/tradebot_v62_research/data_v64")
START = datetime(2026, 9, 1, tzinfo=timezone.utc)
# Exclusive: Oct 20 00:00 UTC; excludes Oct 20 NYSE session entirely.
END = datetime(2026, 10, 20, tzinfo=timezone.utc)


def validate(frame):
    if frame is None or frame.empty or len(frame) < 500:
        return False
    needed = {"timestamp", "open", "high", "low", "close", "volume"}
    if not needed.issubset(frame.columns):
        return False
    dates = pd.to_datetime(frame["timestamp"], utc=True)
    return (
        dates.min() < pd.Timestamp("2026-09-03", tz="UTC")
        and dates.min() >= pd.Timestamp("2026-09-01", tz="UTC")
        and dates.max() >= pd.Timestamp("2026-10-19T13:30:00Z")
        and dates.max() < pd.Timestamp("2026-10-20", tz="UTC")
    )


def main():
    load_dotenv(PROD / ".env.v5")
    if os.getenv("ALPACA_PAPER", "").lower() != "true":
        raise SystemExit("Paper mode not confirmed; refusing to proceed.")
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Alpaca credentials missing; no keys are printed.")
    cfg = json.loads((PROD / "config.json").read_text(encoding="utf-8"))
    symbols = list(dict.fromkeys(cfg["symbols"] + cfg["benchmark_symbols"]))
    if len(symbols) != 102:
        raise SystemExit(f"Expected 102 symbols; got {len(symbols)}")
    client = StockHistoricalDataClient(key, secret)
    DEST.mkdir(parents=True, exist_ok=True)
    failures = []
    print(f"V6.4 held-out completed session data: {START.date()} .. 2026-10-19", flush=True)
    for index, symbol in enumerate(symbols, 1):
        path = DEST / f"{symbol}_5min.csv"
        if path.is_file():
            try:
                if validate(pd.read_csv(path)):
                    print(f"[{index}/102] {symbol}: already complete", flush=True)
                    continue
            except (OSError, ValueError, pd.errors.ParserError):
                pass

        print(f"[{index}/102] {symbol}: downloading", flush=True)
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=START,
                end=END,
                feed=DataFeed.IEX,
                limit=10000,
            )
            for attempt in range(5):
                try:
                    bars = client.get_stock_bars(request).df
                    break
                except Exception as exc:
                    if attempt == 4:
                        raise
                    print(f"  Retry {attempt + 1}/5: {type(exc).__name__}", flush=True)
                    time.sleep(min(60, 5 * 2**attempt))
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.xs(symbol, level=0)
            frame = bars.reset_index()
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame = frame.drop_duplicates(subset="timestamp").sort_values("timestamp")
            if not validate(frame):
                raise ValueError(f"Incomplete bars: n={len(frame)}; start={frame.timestamp.min()}; end={frame.timestamp.max()}")
            temp = path.with_suffix(".partial")
            frame.to_csv(temp, index=False)
            temp.replace(path)
            print(f"  OK {len(frame):,} bars; {frame.timestamp.min()} .. {frame.timestamp.max()}", flush=True)
        except Exception as exc:
            failures.append(symbol)
            print(f"  FAILED {symbol}: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(0.5)

    print(f"\nV6.4 DOWNLOAD RESULT: {102-len(failures)}/102; failed={failures}", flush=True)
    if failures:
        raise SystemExit("Some downloads failed. Re-run to resume; complete files are skipped.")


if __name__ == "__main__":
    main()
