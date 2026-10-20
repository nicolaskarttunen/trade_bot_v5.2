#!/usr/bin/env python3
"""V6.3: exploratory paired diagnostics on V6.2 historical fills.

Tests *fixed, pre-declared* stock opening-range gates, SPY opening trend
gates, and 10/25-minute entry delays on the SAME original selections.
No model refitting and NO alterations to trading, config or production.
This is an exploratory reuse of already inspected historical windows.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
TRADES = BASE / "v6_2_selection_comparison_trades.csv"
DATA = BASE / "data"
WINDOW_OUT = BASE / "v6_3_diagnostics_windows.csv"
SUMMARY_OUT = BASE / "v6_3_diagnostics_summary.csv"
SLIP = (3 + 2) / 10000  # V6.2 slippage + spread, both entry/exit
INITIAL_CASH = 100000.0
EOD = "15:50"

# Fixed thresholds, not chosen using the V6.2 P&L table.
VARIANTS = (
    "baseline",
    "range_le_075",
    "range_le_150",
    "spy_ge_010",
    "spy_ge_020",
    "range150_and_spy010",
    "wait_10m",
    "wait_25m",
)


def load_bars(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["date"] = frame["timestamp"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    frame["clock"] = frame["timestamp"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")
    return frame.sort_values("timestamp").drop_duplicates("timestamp")


def build_day_cache(trades: pd.DataFrame) -> dict:
    date_sets = defaultdict(set)
    for symbol, date in trades[["symbol", "date"]].itertuples(index=False):
        date_sets[str(symbol)].add(str(date))
    date_sets["SPY"].update(map(str, trades["date"].unique()))

    day_cache = {}
    for i, (symbol, dates) in enumerate(sorted(date_sets.items()), 1):
        path = DATA / f"{symbol}_5min.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing historical OHLC file: {path}")
        bars = load_bars(path)
        bars = bars[bars["date"].isin(dates)]
        for date, day in bars.groupby("date", sort=False):
            day_cache[(symbol, date)] = day.reset_index(drop=True)
        print(f"Loaded OHLC {i}/{len(date_sets)}: {symbol}", flush=True)
    return day_cache


def clock_row(day: pd.DataFrame | None, clock: str):
    if day is None:
        return None
    found = day.loc[day["clock"] == clock]
    return None if found.empty else found.iloc[0]


def known_before_entry(row, cache: dict):
    """Only bars completed by the original planned entry time."""
    symbol = str(row["symbol"])
    date = str(row["date"])
    stock_day = cache.get((symbol, date))
    spy_day = cache.get(("SPY", date))
    stock_first = clock_row(stock_day, "09:30")
    spy_first = clock_row(spy_day, "09:30")
    if stock_first is None or spy_first is None:
        return np.nan, np.nan

    # Exactly the first completed NYSE 5-minute candle (09:30-09:35).
    opening_range_pct = (
        100.0 * (float(stock_first["high"]) - float(stock_first["low"]))
        / float(stock_first["open"])
    )

    # At 09:35 only SPY's 09:30 candle is known. At 09:45
    # the 09:40 candle is completed and can be read.
    last_spy_signal = "09:30" if str(row["strategy"]).startswith("0935") else "09:40"
    spy_last = clock_row(spy_day, last_spy_signal)
    if spy_last is None:
        return opening_range_pct, np.nan
    spy_move_pct = (
        100.0 * (float(spy_last["close"]) / float(spy_first["open"]) - 1.0)
    )
    return opening_range_pct, spy_move_pct


def simulate_exit(day: pd.DataFrame, entry_ts: pd.Timestamp, stop: float, target: float):
    future = day.loc[day["timestamp"] >= entry_ts]
    active = future.loc[future["clock"] < EOD]
    for bar in active.itertuples(index=False):
        o, lo, hi = float(bar.open), float(bar.low), float(bar.high)
        if o <= stop:
            return o * (1.0 - SLIP), "stop_gap", bar.timestamp
        if lo <= stop:
            return stop * (1.0 - SLIP), "stop", bar.timestamp
        if o >= target or hi >= target:
            return target * (1.0 - SLIP), "target", bar.timestamp

    eod = future.loc[future["clock"] == EOD]
    if not eod.empty:
        bar = eod.iloc[0]
        return float(bar["open"]) * (1.0 - SLIP), "eod", bar["timestamp"]

    if active.empty:
        return None, "no_exit", None
    bar = active.iloc[-1]
    return float(bar["close"]) * (1.0 - SLIP), "eod_fallback", bar["timestamp"]


def delay_trade(row, cache: dict, delay_minutes: int):
    """Recenter original ATR/floor stop distance; no signal refresh or upsizing."""
    day = cache.get((str(row["symbol"]), str(row["date"])))
    if day is None:
        return None
    wanted = row["entry_ts"] + pd.Timedelta(minutes=delay_minutes)
    later = day.loc[day["timestamp"] == wanted]
    if later.empty:
        return None
    raw_open = float(later.iloc[0]["open"])
    entry = raw_open * (1.0 + SLIP)
    distance = float(row["entry"]) - float(row["stop"])
    original_qty = int(row["qty"])
    # Never allocate more cash or shares than the original trade.
    qty = min(original_qty, int(float(row["entry"]) * original_qty / entry + 1e-9))
    if entry <= 0 or distance <= 0 or qty <= 0:
        return None
    stop = entry - distance
    target = entry + distance * 2.0
    exit_price, reason, exit_ts = simulate_exit(day, wanted, stop, target)
    if exit_price is None:
        return None
    return {"pnl": (exit_price - entry) * qty,
            "exit_reason": reason, "exit_ts": exit_ts, "qty": qty}


def choose_variant(group: pd.DataFrame, name: str):
    vol = group["opening_range_pct"]
    spy = group["spy_move_pct"]
    if name == "baseline" or name.startswith("wait_"):
        return group
    if name == "range_le_075":
        return group.loc[vol <= 0.75]
    if name == "range_le_150":
        return group.loc[vol <= 1.50]
    if name == "spy_ge_010":
        return group.loc[spy >= 0.10]
    if name == "spy_ge_020":
        return group.loc[spy >= 0.20]
    if name == "range150_and_spy010":
        return group.loc[(vol <= 1.50) & (spy >= 0.10)]
    raise ValueError(name)


def metrics(out: pd.DataFrame):
    if out.empty:
        return dict(n=0, pnl=0.0, ret_pct=0.0, pf=0.0, win_pct=0.0, dd_pct=0.0,
                    early_stop_pct=np.nan)
    profits = out.loc[out["pnl"] > 0, "pnl"].sum()
    losses = -out.loc[out["pnl"] < 0, "pnl"].sum()
    pf = float(profits / losses) if losses > 0 else np.inf
    by_day = out.groupby("date", sort=True)["pnl"].sum()
    equity = pd.concat([pd.Series([INITIAL_CASH]), INITIAL_CASH + by_day.cumsum()],
                       ignore_index=True)
    dd = float((equity / equity.cummax() - 1.0).min() * 100)
    stop_early = out["exit_reason"].str.startswith("stop") & (
        (out["exit_ts"] - out["entry_ts"]).dt.total_seconds() <= 1800
    )
    return dict(n=int(len(out)), pnl=float(out["pnl"].sum()),
                ret_pct=float(out["pnl"].sum() / INITIAL_CASH * 100),
                pf=pf, win_pct=float((out["pnl"] > 0).mean() * 100),
                dd_pct=dd, early_stop_pct=float(stop_early.mean() * 100))


def main():
    if not TRADES.is_file():
        raise SystemExit("V6.2 trades CSV missing. Run V6.2 first.")
    df = pd.read_csv(TRADES)
    expected = {"strategy", "window", "symbol", "date", "entry_ts", "exit_ts",
                "entry", "stop", "qty", "pnl", "reason"}
    if not expected.issubset(df.columns):
        raise SystemExit(f"Unexpected V6.2 CSV columns. Missing: {sorted(expected-set(df.columns))}")
    df["date"] = df["date"].astype(str)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)
    print(f"Original V6.2 trades: {len(df):,}; strategies: {df.strategy.nunique()}")
    print("NO orders, model training, config changes or live bot actions.")
    print("Reading historical OHLC for paired gates and fixed waiting scenarios...\n", flush=True)
    cache = build_day_cache(df)

    attrs = [known_before_entry(row, cache) for _, row in df.iterrows()]
    df[["opening_range_pct", "spy_move_pct"]] = pd.DataFrame(attrs, index=df.index)
    for name in ["opening_range_pct", "spy_move_pct"]:
        missing = df[name].isna().sum()
        print(f"Missing ex-ante {name}: {missing}/{len(df)}")
        if missing / len(df) > 0.03:
            raise SystemExit(f"Too much missing price data for reliable analysis: {name}")

    # Validate raw bars still agree with V6.2 simulation for original entries.
    # No silent historical mismatch: stop and target must yield original P&L.
    mismatches = []
    for index, row in df.iterrows():
        day = cache.get((str(row["symbol"]), str(row["date"])))
        if day is None:
            mismatches.append((index, "missing day"))
            continue
        original_entry = day.loc[day["timestamp"] == row["entry_ts"]]
        if original_entry.empty:
            mismatches.append((index, "missing original entry"))
            continue
        implied = float(original_entry.iloc[0]["open"]) * (1.0 + SLIP)
        exit_price, reason, _ = simulate_exit(
            day, row["entry_ts"], float(row["stop"]), float(row["target"])
        )
        if (exit_price is None
            or abs(implied - float(row["entry"])) > 1e-6
            or abs((exit_price - float(row["entry"])) * int(row["qty"])
                   - float(row["pnl"])) > 0.03
            or reason != row["reason"]):
            mismatches.append((index, reason))
    print(f"Baseline fill replay discrepancies: {len(mismatches)} / {len(df)}")
    if mismatches:
        print("First discrepancies:", mismatches[:8])
        raise SystemExit("Original V6.2 bar replay mismatch; no scenario results written.")

    # Prepare once; avoid recomputing each delayed trade for every summary.
    delayed = {}
    for wait in (10, 25):
        records = []
        for _, row in df.iterrows():
            result = delay_trade(row, cache, wait)
            records.append(result)
        delayed[wait] = records
        print(f"Wait +{wait}m: {sum(r is not None for r in records)}/{len(df)} simulated fills")

    results = []
    for (strategy, window), group in df.groupby(["strategy", "window"], sort=True):
        base_metrics = metrics(
            group.assign(exit_reason=group["reason"])
        )
        for variant in VARIANTS:
            filtered = choose_variant(group, variant).copy()
            if variant.startswith("wait_"):
                wait = int(variant.split("_")[1].replace("m", ""))
                # Delayed rows are precomputed by original dataframe index.
                delayed_rows = [(idx, delayed[wait][idx]) for idx in filtered.index
                                if delayed[wait][idx] is not None]
                if delayed_rows:
                    indices = [idx for idx, _ in delayed_rows]
                    filtered = filtered.loc[indices].copy()
                    filtered["pnl"] = [r["pnl"] for _, r in delayed_rows]
                    filtered["exit_reason"] = [r["exit_reason"] for _, r in delayed_rows]
                    filtered["exit_ts"] = pd.to_datetime(
                        [r["exit_ts"] for _, r in delayed_rows], utc=True
                    )
                    filtered["entry_ts"] = filtered["entry_ts"] + pd.Timedelta(minutes=wait)
                else:
                    filtered = filtered.iloc[0:0].copy()
            else:
                filtered["exit_reason"] = filtered["reason"]
            m = metrics(filtered)
            results.append({
                "strategy": strategy, "window": int(window),
                "variant": variant, **m,
                "base_trades": base_metrics["n"],
                "retained_pct": 100.0 * m["n"] / base_metrics["n"] if base_metrics["n"] else 0,
                "delta_vs_baseline_pct": m["ret_pct"] - base_metrics["ret_pct"],
            })

    per_window = pd.DataFrame(results)
    per_window.to_csv(WINDOW_OUT, index=False)
    grouped = per_window.groupby(["strategy", "variant"], sort=False)
    summary = grouped.agg(
        mean_return_pct=("ret_pct", "mean"),
        positive_windows=("ret_pct", lambda a: int((a > 0).sum())),
        better_than_base_windows=("delta_vs_baseline_pct", lambda a: int((a > 0).sum())),
        mean_delta_vs_base_pct=("delta_vs_baseline_pct", "mean"),
        total_trades=("n", "sum"),
        mean_trade_retention_pct=("retained_pct", "mean"),
        mean_pf=("pf", "mean"),
        worst_dd_pct=("dd_pct", "min"),
    ).reset_index()
    summary.to_csv(SUMMARY_OUT, index=False)

    print("\n=== V6.3 EXPLORATORY PAIRED COMPARISON ===")
    for strat in ("0935_first5", "0935_ranked5", "0945_first5", "0945_ranked5"):
        subset = summary.loc[summary["strategy"] == strat].copy()
        print("\n"+strat)
        print(subset.drop(columns=["strategy"]).round(2).to_string(index=False))

    print("\nSaved:", WINDOW_OUT.name, "and", SUMMARY_OUT.name)
    print("IMPORTANT: filters REMOVE original trades without replacing them.")
    print("Waiting keeps stale signals and original risk distance, never increases shares or capital.")
    print("Results are exploratory; five windows were already inspected in V6.2.")
    print("OPG, later scanning, retraining and live paper fills are NOT simulated.")


if __name__ == "__main__":
    main()
