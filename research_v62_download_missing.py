#!/usr/bin/env python3
"""Research-only Alpaca IEX 5-minute history downloader.

Writes only /home/opc/tradebot_v62_research/data.
Uses production settings/keys read-only. No trading API or orders.
"""
import csv
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

PROD = Path("/home/opc/ai_daytrader_v5_1_python314")
RESEARCH = Path("/home/opc/tradebot_v62_research")
OUT = RESEARCH / "data"
START = datetime(2023, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 9, 1, tzinfo=timezone.utc)
CHUNK_DAYS = 45
MAX_RETRIES = 5


def file_looks_complete(path):
    """Never trust an existing truncated or partial CSV only by its size."""
    if not path.is_file() or path.stat().st_size < 500_000:
        return False
    try:
        # Read only the first and last timestamp; do not load millions of rows.
        with path.open(encoding="utf-8-sig", newline="") as fh:
            header = next(csv.reader(fh))
            if "timestamp" not in header:
                return False
            if not {"open", "high", "low", "close", "volume"}.issubset(header):
                return False
            first_line = fh.readline()
            if not first_line:
                return False
            first_date = pd.to_datetime(next(csv.reader([first_line]))[header.index("timestamp")], utc=True)
            last_line = first_line
            for line in fh:
                if line.strip():
                    last_line = line
            last_date = pd.to_datetime(next(csv.reader([last_line]))[header.index("timestamp")], utc=True)
        return first_date < pd.Timestamp("2023-02-01", tz="UTC") and last_date >= pd.Timestamp("2026-08-28", tz="UTC")
    except (OSError, ValueError, StopIteration, IndexError, pd.errors.ParserError):
        return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    load_dotenv(PROD / ".env.v5")
    if os.environ.get("ALPACA_PAPER", "").lower() != "true":
        raise SystemExit("Refusing to run: paper-only environment flag missing.")
    api_key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        raise SystemExit("Missing Alpaca API credentials (not printed).")

    cfg = json.loads((PROD / "config.json").read_text(encoding="utf-8"))
    symbols = list(dict.fromkeys(cfg["symbols"] + cfg["benchmark_symbols"]))
    client = StockHistoricalDataClient(api_key, secret)
    failures = []

    print(f"Research directory: {OUT}", flush=True)
    print(f"Required symbols: {len(symbols)}", flush=True)
    for i, symbol in enumerate(symbols, start=1):
        destination = OUT / f"{symbol}_5min.csv"
        if file_looks_complete(destination):
            print(f"[{i}/{len(symbols)}] {symbol}: already complete; skip", flush=True)
            continue

        print(f"[{i}/{len(symbols)}] {symbol}: downloading", flush=True)
        try:
            batches = []
            cursor = START
            while cursor < END:
                stop = min(cursor + timedelta(days=CHUNK_DAYS), END)
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                    start=cursor,
                    end=stop,
                    feed=DataFeed.IEX,
                    limit=10000,
                )
                for attempt in range(MAX_RETRIES):
                    try:
                        response = client.get_stock_bars(request).df
                        break
                    except Exception as exc:
                        if attempt + 1 == MAX_RETRIES:
                            raise
                        print(f"  API retry {attempt + 1}/{MAX_RETRIES}: {type(exc).__name__}", flush=True)
                        time.sleep(min(60, 5 * (2 ** attempt)))
                if response is not None and not response.empty:
                    if isinstance(response.index, pd.MultiIndex):
                        response = response.xs(symbol, level=0)
                    batches.append(response.reset_index())
                cursor = stop
                time.sleep(0.4)

            if not batches:
                raise RuntimeError("No historical rows returned.")
            frame = pd.concat(batches, ignore_index=True)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame = frame.drop_duplicates(subset="timestamp").sort_values("timestamp")
            if not {"timestamp", "open", "high", "low", "close", "volume"}.issubset(frame.columns):
                raise RuntimeError("Missing required CSV columns.")
            temp_path = destination.with_suffix(".partial")
            frame.to_csv(temp_path, index=False)
            if not file_looks_complete(temp_path):
                raise RuntimeError("Coverage check failed; retaining .partial for inspection.")
            temp_path.replace(destination)
            print(f"  OK: {len(frame):,} bars; {frame['timestamp'].iloc[0]} .. {frame['timestamp'].iloc[-1]}", flush=True)
        except Exception as exc:
            failures.append(symbol)
            print(f"  ERROR {symbol}: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(0.5)

    complete = sum(file_looks_complete(OUT / f"{s}_5min.csv") for s in symbols)
    print(f"\nRESULT: {complete}/{len(symbols)} complete historical files", flush=True)
    print(f"Failed: {', '.join(failures) if failures else 'none'}", flush=True)
    if complete != len(symbols):
        raise SystemExit("Missing data remains. Re-running later will skip complete symbols.")


if __name__ == "__main__":
    main()
