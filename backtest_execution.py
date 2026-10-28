"""Price-path helpers for research backtests. No broker interaction."""
import pandas as pd


def next_entry_and_path(bars, signal_timestamp, max_bars=12):
    """Next *same-symbol* intraday candle and its complete forward price path.

    `bars` must belong to one symbol and be timestamp-sorted. Signal bars
    are start-stamped, so the next timestamp is the earliest tradable bar.
    Restrict to the same New York trading day: no accidental overnight fills.
    """
    if bars is None or bars.empty or max_bars < 1:
        return None, pd.DataFrame()
    signal_ts = pd.Timestamp(signal_timestamp)
    if signal_ts.tzinfo is None:
        raise ValueError("signal_timestamp must be timezone-aware")
    signal_ts = signal_ts.tz_convert("UTC")
    if not bars["timestamp"].is_monotonic_increasing:
        raise ValueError("Price bars must be sorted by timestamp")

    # Using the unfiltered OHLC rows prevents feature-missing bars from
    # disappearing from stop/target simulation.
    same_day = signal_ts.tz_convert("America/New_York").date()
    later = bars.loc[bars["timestamp"] > signal_ts]
    if later.empty:
        return None, pd.DataFrame()
    later = later.loc[
        later["timestamp"].dt.tz_convert("America/New_York").dt.date == same_day
    ].head(max_bars)
    if later.empty:
        return None, pd.DataFrame()
    return later.iloc[0], later


def simulate_long_exit(path, stop, target, slippage_fraction):
    """Return conservatively simulated long exit: price, reason, timestamp."""
    if path is None or path.empty:
        return None, "no_exit", None
    for bar in path.itertuples(index=False):
        opening = float(bar.open)
        if opening <= stop:
            return opening * (1.0 - slippage_fraction), "stop_gap", bar.timestamp
        if opening >= target:
            return target * (1.0 - slippage_fraction), "target_gap", bar.timestamp
        if float(bar.low) <= stop:
            return stop * (1.0 - slippage_fraction), "stop", bar.timestamp
        if opening >= target or float(bar.high) >= target:
            return target * (1.0 - slippage_fraction), "target", bar.timestamp
    end = path.iloc[-1]
    return float(end["close"]) * (1.0 - slippage_fraction), "time_exit", end["timestamp"]
