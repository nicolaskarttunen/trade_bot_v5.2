"""Causal feature contract. Bar timestamps denote five-minute interval START.

Every row is a signal at timestamp + 5 minutes. Stock and benchmark EMA
use exactly the trailing 250 observations, matching the inference input.
Market features are computed on the benchmark's own clock before joining.
"""
import numpy as np
import pandas as pd
from news_features import news_scores

FEATURE_VERSION = "causal_5min_250_v2"
FEATURE_HISTORY_BARS = 250
FEATURES = [
    "ret_1", "ret_3", "ret_12", "range_pct", "close_vs_ema9", "close_vs_ema21",
    "ema9_vs_ema21", "rsi14", "atr_pct", "volume_ratio", "hour_sin", "hour_cos",
    "market_ret_3", "market_ret_12", "market_vs_ema21", "market_atr_pct", "news_score"
]


def normalize_bars(df):
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    if x["timestamp"].isna().any() or x["timestamp"].duplicated().any():
        raise ValueError("Missing or duplicate bar timestamps")
    if (x["timestamp"] != x["timestamp"].dt.floor("5min")).any():
        raise ValueError("Bars must have five-minute start timestamps")
    x = x.sort_values("timestamp").reset_index(drop=True)
    values = x[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values[:, :4] <= 0).any()
            or (values[:, 4] < 0).any()
            or (x["high"] < x[["open", "close", "low"]].max(axis=1)).any()
            or (x["low"] > x[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("Invalid OHLCV data")
    return x


def completed_bars(df, as_of):
    now = pd.Timestamp(as_of)
    if now.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    x = normalize_bars(df)
    return x.loc[x["timestamp"] + pd.Timedelta(minutes=5) <= now].copy()


def bounded_ema(series, span):
    ema = series.ewm(span=span, adjust=False).mean()
    shift = FEATURE_HISTORY_BARS - 1
    decay = (1 - 2 / (span + 1)) ** shift
    return ema + decay * (series.shift(shift) - ema.shift(shift)).fillna(0)


def technical_features(df):
    x = normalize_bars(df)
    c, h, l, v = x.close, x.high, x.low, x.volume
    for periods in (1, 3, 12):
        x[f"ret_{periods}"] = c.pct_change(periods, fill_method=None)
    x["ema9"], x["ema21"] = bounded_ema(c, 9), bounded_ema(c, 21)
    x["close_vs_ema9"], x["close_vs_ema21"] = c / x.ema9 - 1, c / x.ema21 - 1
    x["ema9_vs_ema21"] = x.ema9 / x.ema21 - 1
    d = c.diff()
    gain, loss = d.clip(lower=0).rolling(14).mean(), (-d.clip(upper=0)).rolling(14).mean()
    x["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    x.loc[(loss == 0) & (gain > 0), "rsi14"] = 100.0
    x.loc[(loss == 0) & (gain == 0), "rsi14"] = 50.0
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    x["atr"] = tr.rolling(14).mean()
    x["atr_pct"], x["range_pct"] = x.atr / c, (h-l) / c
    x["volume_ratio"] = v / v.rolling(20).mean().replace(0, np.nan)
    # Retain the existing UTC-based phase, not a strategy parameter change.
    mins = x.timestamp.dt.hour * 60 + x.timestamp.dt.minute
    x["hour_sin"], x["hour_cos"] = np.sin(2*np.pi*mins/390), np.cos(2*np.pi*mins/390)
    x.loc[x.index < FEATURE_HISTORY_BARS - 1, ["ret_1", "ret_3", "ret_12", "close_vs_ema9", "close_vs_ema21", "ema9_vs_ema21", "rsi14", "atr_pct", "range_pct", "volume_ratio", "hour_sin", "hour_cos"]] = np.nan
    return x


def benchmark_features(market):
    # Raw bars are preferred; accepting already-enriched input is safe because
    # technical_features overwrites its derived columns on the original clock.
    m = technical_features(market)
    return m[["timestamp", "ret_3", "ret_12", "close_vs_ema21", "atr_pct"]].rename(columns={
        "ret_3": "market_ret_3", "ret_12": "market_ret_12",
        "close_vs_ema21": "market_vs_ema21", "atr_pct": "market_atr_pct"})


def enrich(df, market=None, news=None, news_decay_minutes=120, news_lookback_minutes=360,
           require_news_received=False):
    x = technical_features(df)
    if market is not None:
        # Exact match: a stale benchmark bar cannot silently be carried forward.
        x = x.merge(benchmark_features(market), on="timestamp", how="left", validate="one_to_one")
    else:
        for name in ("market_ret_3", "market_ret_12", "market_vs_ema21", "market_atr_pct"):
            x[name] = 0.0
    x["signal_available_at"] = x["timestamp"] + pd.Timedelta(minutes=5)
    x["news_score"] = news_scores(x["signal_available_at"], news, news_decay_minutes,
                                  news_lookback_minutes, require_news_received)
    return x


def label(x, horizon=3, threshold=.002):
    """Same-symbol, same-session labels; missing five-minute bars stay unlabeled."""
    if int(horizon) != horizon or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    y = x.copy()
    ts = pd.to_datetime(y["timestamp"], utc=True)
    if not ts.is_monotonic_increasing or ts.duplicated().any():
        raise ValueError("Label one sorted symbol at a time")
    if "symbol" in y and y["symbol"].nunique() > 1:
        raise ValueError("Label one symbol at a time")
    future = y["close"].shift(-int(horizon))
    end = ts.shift(-int(horizon))
    session = ts.dt.tz_convert("America/New_York").dt.date
    valid = (end - ts == pd.Timedelta(minutes=5*int(horizon))) & (session == session.shift(-int(horizon)))
    y["target"] = np.where(valid & future.notna(), (future / y["close"] - 1 >= threshold).astype(float), np.nan)
    y["label_end"] = (end + pd.Timedelta(minutes=5)).where(valid)
    return y
