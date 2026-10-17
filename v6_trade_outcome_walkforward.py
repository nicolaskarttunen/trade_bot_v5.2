import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES as BASE_FEATURES, enrich


BASE = Path(__file__).resolve().parent
CFG = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
DATA_DIR = BASE / "data"
NEWS_DIR = BASE / "news"
NY = "America/New_York"

SYMBOLS = list(CFG["symbols"])
SIGNAL_CLOCK = "09:40"
ENTRY_CLOCK = "09:45"
EOD_EXIT_CLOCK = "15:50"

WINDOWS = [
    ("2023-01-01", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("2023-01-01", "2024-06-30", "2024-07-01", "2025-01-31"),
    ("2023-01-01", "2025-01-31", "2025-02-01", "2025-08-31"),
    ("2023-01-01", "2025-08-31", "2025-09-01", "2026-02-28"),
    ("2023-01-01", "2026-02-28", "2026-03-01", "2026-08-31"),
]

# Locked research risk rules. These do NOT modify config.json or production.
INITIAL_CASH = 100_000.0
RISK_PER_TRADE = 0.01
MAX_POSITION_PCT = 0.10
MAX_TOTAL_EXPOSURE_PCT = 0.50
STOP_ATR_MULT = 1.5
MIN_STOP_PCT = 0.0125
REWARD_RISK = 2.0
SLIPPAGE_BPS = float(CFG.get("slippage_bps", 2.0))
SPREAD_BPS = float(CFG.get("spread_bps", 3.0))
COMMISSION = float(CFG.get("commission_per_trade", 0.0))

ENHANCED_ONLY = [
    "gap_pct",
    "from_open_pct",
    "vwap_dist_pct",
    "opening_range_width_pct",
    "opening_range_pos",
    "rel_volume_0940",
    "prev_day_return",
    "prev_day_range_pct",
    "rs_vs_spy_3",
    "rs_vs_spy_12",
    "qqq_ret_3",
    "qqq_ret_12",
    "rs_vs_qqq_3",
    "rs_vs_qqq_12",
    "momentum_accel",
    "trend_atr",
    "breadth_up",
    "breadth_median_from_open",
    "breadth_median_rs_spy",
]
ENHANCED_FEATURES = list(BASE_FEATURES) + ENHANCED_ONLY

STRATEGIES = [
    {"name": "ridge_base_top3", "model": "ridge_base", "top_n": 3, "positive_only": False},
    {"name": "hgb_base_top3", "model": "hgb_base", "top_n": 3, "positive_only": False},
    {"name": "hgb_enh_top3", "model": "hgb_enh", "top_n": 3, "positive_only": False},
    {"name": "hgb_enh_pos_top3", "model": "hgb_enh", "top_n": 3, "positive_only": True},
    {"name": "hgb_enh_pos_top5", "model": "hgb_enh", "top_n": 5, "positive_only": True},
]


def read_bars(symbol):
    path = DATA_DIR / f"{symbol}_5min.csv"
    if not path.exists():
        return None
    x = pd.read_csv(path)
    need = {"timestamp", "open", "high", "low", "close", "volume"}
    if not need.issubset(x.columns):
        return None
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
    x = x.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    x = x.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["open", "high", "low", "close", "volume"])
    return x


def prepare_market(symbol):
    x = read_bars(symbol)
    if x is None or x.empty:
        raise FileNotFoundError(f"Missing market data for {symbol}")
    pc = x["close"].shift(1)
    tr = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - pc).abs(),
            (x["low"] - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)
    x["atr_pct"] = tr.rolling(14).mean() / x["close"]
    x["ret_3_mkt"] = x["close"].pct_change(3)
    x["ret_12_mkt"] = x["close"].pct_change(12)
    return x


def add_session_features(x):
    x = x.copy()
    x["et"] = x["timestamp"].dt.tz_convert(NY)
    x["trade_date"] = x["et"].dt.date.astype(str)
    x["clock"] = x["et"].dt.strftime("%H:%M")

    reg = x[(x["clock"] >= "09:30") & (x["clock"] <= "16:00")].copy()
    if reg.empty:
        return reg

    g = reg.groupby("trade_date", sort=False)
    reg["day_open"] = g["open"].transform("first")
    reg["session_high"] = g["high"].cummax()
    reg["session_low"] = g["low"].cummin()
    tp = (reg["high"] + reg["low"] + reg["close"]) / 3.0
    reg["cum_pv"] = (tp * reg["volume"]).groupby(reg["trade_date"]).cumsum()
    reg["cum_vol"] = reg["volume"].groupby(reg["trade_date"]).cumsum()
    reg["vwap"] = reg["cum_pv"] / reg["cum_vol"].replace(0, np.nan)

    reg["from_open_pct"] = reg["close"] / reg["day_open"] - 1.0
    reg["vwap_dist_pct"] = reg["close"] / reg["vwap"] - 1.0
    width = reg["session_high"] - reg["session_low"]
    reg["opening_range_width_pct"] = width / reg["day_open"]
    reg["opening_range_pos"] = (reg["close"] - reg["session_low"]) / width.replace(0, np.nan)

    daily = reg.groupby("trade_date", sort=True).agg(
        daily_open=("open", "first"),
        daily_close=("close", "last"),
        daily_high=("high", "max"),
        daily_low=("low", "min"),
    )
    daily["daily_return"] = daily["daily_close"] / daily["daily_open"] - 1.0
    daily["daily_range_pct"] = (daily["daily_high"] - daily["daily_low"]) / daily["daily_open"]
    daily["prev_close"] = daily["daily_close"].shift(1)
    daily["prev_day_return"] = daily["daily_return"].shift(1)
    daily["prev_day_range_pct"] = daily["daily_range_pct"].shift(1)

    reg = reg.merge(
        daily[["prev_close", "prev_day_return", "prev_day_range_pct"]],
        left_on="trade_date",
        right_index=True,
        how="left",
    )
    reg["gap_pct"] = reg["day_open"] / reg["prev_close"] - 1.0
    return reg


def simulate_exit(day, entry_ts, entry, stop, target, sell_slip):
    active = day[day["timestamp"] >= entry_ts].sort_values("timestamp")
    before_eod = active[active["clock"] < EOD_EXIT_CLOCK]

    for _, bar in before_eod.iterrows():
        o = float(bar["open"])
        lo = float(bar["low"])
        hi = float(bar["high"])

        if o <= stop:
            return o * (1.0 - sell_slip), "stop_gap", bar["timestamp"]
        if lo <= stop:
            return stop * (1.0 - sell_slip), "stop", bar["timestamp"]
        if o >= target:
            return target * (1.0 - sell_slip), "target_gap", bar["timestamp"]
        if hi >= target:
            return target * (1.0 - sell_slip), "target", bar["timestamp"]

    eod = active[active["clock"] == EOD_EXIT_CLOCK]
    if not eod.empty:
        bar = eod.iloc[0]
        return float(bar["open"]) * (1.0 - sell_slip), "eod", bar["timestamp"]

    fallback = active[active["clock"] < EOD_EXIT_CLOCK]
    if fallback.empty:
        return None, "no_exit", None
    bar = fallback.iloc[-1]
    return float(bar["close"]) * (1.0 - sell_slip), "eod_fallback", bar["timestamp"]


def build_symbol_samples(symbol, spy_market, qqq_market):
    raw = read_bars(symbol)
    if raw is None or raw.empty:
        return None

    news_path = NEWS_DIR / f"{symbol}_news.csv"
    news = pd.read_csv(news_path) if news_path.exists() else None

    x = enrich(
        raw,
        market=spy_market[["timestamp", "close", "atr_pct"]].copy(),
        news=news,
        news_decay_minutes=CFG.get("news_decay_minutes", 120),
        news_lookback_minutes=CFG.get("news_lookback_minutes", 360),
    )
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    reg = add_session_features(x)
    if reg.empty:
        return None

    q = qqq_market[["timestamp", "ret_3_mkt", "ret_12_mkt"]].copy().rename(
        columns={"ret_3_mkt": "qqq_ret_3", "ret_12_mkt": "qqq_ret_12"}
    )
    reg = pd.merge_asof(
        reg.sort_values("timestamp"),
        q.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    reg["rs_vs_spy_3"] = reg["ret_3"] - reg["market_ret_3"]
    reg["rs_vs_spy_12"] = reg["ret_12"] - reg["market_ret_12"]
    reg["rs_vs_qqq_3"] = reg["ret_3"] - reg["qqq_ret_3"]
    reg["rs_vs_qqq_12"] = reg["ret_12"] - reg["qqq_ret_12"]
    reg["momentum_accel"] = reg["ret_3"] - reg["ret_12"] / 4.0
    reg["trend_atr"] = reg["close_vs_ema21"] / reg["atr_pct"].replace(0, np.nan)

    sig = reg[reg["clock"] == SIGNAL_CLOCK].copy()
    if sig.empty:
        return None

    # Same-time relative volume versus the prior 20 sessions only.
    sig = sig.sort_values("timestamp").reset_index(drop=True)
    sig["vol_med20_0940"] = sig["volume"].shift(1).rolling(20, min_periods=5).median()
    sig["rel_volume_0940"] = sig["volume"] / sig["vol_med20_0940"].replace(0, np.nan)

    buy_slip = (SLIPPAGE_BPS + SPREAD_BPS) / 10000.0
    sell_slip = (SLIPPAGE_BPS + SPREAD_BPS) / 10000.0
    out = []

    by_date = {d: day.copy() for d, day in reg.groupby("trade_date", sort=False)}
    for _, s in sig.iterrows():
        d = str(s["trade_date"])
        day = by_date.get(d)
        if day is None:
            continue
        eb = day[day["clock"] == ENTRY_CLOCK]
        if eb.empty:
            continue
        entry_bar = eb.iloc[0]
        entry = float(entry_bar["open"]) * (1.0 + buy_slip)
        if not np.isfinite(entry) or entry <= 0:
            continue

        atr = float(s["atr"]) if np.isfinite(s["atr"]) else np.nan
        if not np.isfinite(atr) or atr <= 0:
            continue
        stop_distance = max(atr * STOP_ATR_MULT, entry * MIN_STOP_PCT)
        stop = entry - stop_distance
        target = entry + REWARD_RISK * stop_distance

        exit_price, reason, exit_ts = simulate_exit(
            day,
            entry_bar["timestamp"],
            entry,
            stop,
            target,
            sell_slip,
        )
        if exit_price is None:
            continue

        realized_r = (float(exit_price) - entry) / stop_distance
        row = {k: s.get(k, np.nan) for k in BASE_FEATURES}
        for k in ENHANCED_ONLY:
            if k.startswith("breadth_"):
                continue
            row[k] = s.get(k, np.nan)
        row.update(
            {
                "symbol": symbol,
                "timestamp": s["timestamp"],
                "trade_date": d,
                "entry_ts": entry_bar["timestamp"],
                "exit_ts": exit_ts,
                "entry": entry,
                "exit": float(exit_price),
                "stop": stop,
                "target": target,
                "stop_distance": stop_distance,
                "realized_r": realized_r,
                "profitable": float(realized_r > 0),
                "target_hit": float(str(reason).startswith("target")),
                "exit_reason": reason,
            }
        )
        out.append(row)

    if not out:
        return None
    return pd.DataFrame(out)


def add_cross_sectional_breadth(samples):
    x = samples.copy()
    x["breadth_up"] = x.groupby("timestamp")["from_open_pct"].transform(lambda s: (s > 0).mean())
    x["breadth_median_from_open"] = x.groupby("timestamp")["from_open_pct"].transform("median")
    x["breadth_median_rs_spy"] = x.groupby("timestamp")["rs_vs_spy_3"].transform("median")
    return x


def make_models():
    ridge_base = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=10.0),
    )
    hgb_base = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=42,
        ),
    )
    hgb_enh = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=42,
        ),
    )
    return {"ridge_base": ridge_base, "hgb_base": hgb_base, "hgb_enh": hgb_enh}


def period_mask(df, start, end):
    return (df["trade_date"] >= start) & (df["trade_date"] <= end)


def run_portfolio(test_scored, strategy):
    pred_col = f"pred_{strategy['model']}"
    equity = INITIAL_CASH
    trade_rows = []
    daily_rows = []

    for d, day in test_scored.groupby("trade_date", sort=True):
        candidates = day.sort_values(pred_col, ascending=False).copy()
        if strategy["positive_only"]:
            candidates = candidates[candidates[pred_col] > 0.0]
        candidates = candidates.head(int(strategy["top_n"]))

        start_equity = equity
        day_pnl = 0.0
        day_exposure = 0.0
        entries = 0

        for _, r in candidates.iterrows():
            entry = float(r["entry"])
            stop_distance = float(r["stop_distance"])
            if entry <= 0 or stop_distance <= 0:
                continue

            risk_qty = int((start_equity * RISK_PER_TRADE) / stop_distance)
            pos_qty = int((start_equity * MAX_POSITION_PCT) / entry)
            remaining = max(0.0, start_equity * MAX_TOTAL_EXPOSURE_PCT - day_exposure)
            exp_qty = int(remaining / entry)
            qty = min(risk_qty, pos_qty, exp_qty)
            if qty < 1:
                continue

            pnl = (float(r["exit"]) - entry) * qty - COMMISSION
            day_pnl += pnl
            day_exposure += entry * qty
            entries += 1

            trade_rows.append(
                {
                    "strategy": strategy["name"],
                    "date": d,
                    "symbol": r["symbol"],
                    "predicted_r": float(r[pred_col]),
                    "realized_r": float(r["realized_r"]),
                    "entry": entry,
                    "exit": float(r["exit"]),
                    "qty": qty,
                    "pnl": pnl,
                    "exit_reason": r["exit_reason"],
                }
            )

        equity += day_pnl
        daily_rows.append(
            {
                "strategy": strategy["name"],
                "date": d,
                "start_equity": start_equity,
                "pnl": day_pnl,
                "end_equity": equity,
                "entries": entries,
            }
        )

    return equity, pd.DataFrame(trade_rows), pd.DataFrame(daily_rows)


def metrics(final_cash, trades, daily):
    if trades.empty:
        return {
            "return_pct": 0.0,
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_dd_pct": 0.0,
            "avg_r": 0.0,
            "avg_pnl": 0.0,
        }

    gp = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gl = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    pf = gp / gl if gl > 0 else float("inf")
    eq = pd.Series([INITIAL_CASH] + daily["end_equity"].tolist(), dtype=float)
    dd = (eq / eq.cummax() - 1.0).min() * 100.0

    return {
        "return_pct": (final_cash / INITIAL_CASH - 1.0) * 100.0,
        "trades": int(len(trades)),
        "win_rate": float((trades["pnl"] > 0).mean() * 100.0),
        "profit_factor": float(pf),
        "max_dd_pct": float(dd),
        "avg_r": float(trades["realized_r"].mean()),
        "avg_pnl": float(trades["pnl"].mean()),
    }


print("=" * 100)
print("V6 TRADE-OUTCOME WALK-FORWARD")
print("Signal 09:40 ET -> entry 09:45 ET")
print("Target is realized trade R with production-like 1.5 ATR / 1.25% minimum stop and 2R target.")
print("No production files are modified.")
print("=" * 100)

spy = prepare_market("SPY")
qqq = prepare_market("QQQ")
parts = []

for i, symbol in enumerate(SYMBOLS, 1):
    s = build_symbol_samples(symbol, spy, qqq)
    if s is not None and not s.empty:
        parts.append(s)
        print(f"[{i:3}/{len(SYMBOLS)}] {symbol}: {len(s):,} daily samples")
    else:
        print(f"[{i:3}/{len(SYMBOLS)}] {symbol}: no usable samples")

if len(parts) < 10:
    raise RuntimeError("Too few symbols with usable data")

samples = pd.concat(parts, ignore_index=True)
samples = add_cross_sectional_breadth(samples)
samples = samples.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

print(f"\nTotal daily samples: {len(samples):,}")
print(f"Symbols: {samples['symbol'].nunique()}")
print(f"Date range: {samples['trade_date'].min()} .. {samples['trade_date'].max()}")

all_summary = []
all_trades = []

for wno, (train_start, train_end, test_start, test_end) in enumerate(WINDOWS, 1):
    train = samples[period_mask(samples, train_start, train_end)].copy()
    test = samples[period_mask(samples, test_start, test_end)].copy()

    print("\n" + "=" * 100)
    print(f"WINDOW {wno}: train {train_start}..{train_end} | test {test_start}..{test_end}")
    print("=" * 100)
    print(f"Train samples: {len(train):,} | Test samples: {len(test):,}")

    if len(train) < 1000 or len(test) < 100:
        print("Skipping window: insufficient samples")
        continue

    models = make_models()

    models["ridge_base"].fit(train[BASE_FEATURES], train["realized_r"])
    models["hgb_base"].fit(train[BASE_FEATURES], train["realized_r"])
    models["hgb_enh"].fit(train[ENHANCED_FEATURES], train["realized_r"])

    test["pred_ridge_base"] = models["ridge_base"].predict(test[BASE_FEATURES])
    test["pred_hgb_base"] = models["hgb_base"].predict(test[BASE_FEATURES])
    test["pred_hgb_enh"] = models["hgb_enh"].predict(test[ENHANCED_FEATURES])

    for strategy in STRATEGIES:
        final_cash, trades, daily = run_portfolio(test, strategy)
        m = metrics(final_cash, trades, daily)
        row = {
            "window": wno,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "strategy": strategy["name"],
            **m,
        }
        all_summary.append(row)
        if not trades.empty:
            trades["window"] = wno
            all_trades.append(trades)

        print(
            f"{strategy['name']:<20} "
            f"ret={m['return_pct']:7.2f}% | "
            f"trades={m['trades']:4d} | "
            f"WR={m['win_rate']:6.2f}% | "
            f"PF={m['profit_factor']:5.2f} | "
            f"DD={m['max_dd_pct']:7.2f}% | "
            f"avgR={m['avg_r']:+.3f}"
        )

summary = pd.DataFrame(all_summary)
trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

if summary.empty:
    raise RuntimeError("No walk-forward results were produced")

aggregate = summary.groupby("strategy").agg(
    mean_return_pct=("return_pct", "mean"),
    median_return_pct=("return_pct", "median"),
    positive_windows=("return_pct", lambda s: int((s > 0).sum())),
    pf_above_one=("profit_factor", lambda s: int((s > 1).sum())),
    total_trades=("trades", "sum"),
    mean_win_rate=("win_rate", "mean"),
    mean_profit_factor=("profit_factor", "mean"),
    worst_dd_pct=("max_dd_pct", "min"),
    mean_avg_r=("avg_r", "mean"),
).sort_values(["positive_windows", "mean_profit_factor", "mean_return_pct"], ascending=False)

print("\n" + "#" * 100)
print("AGGREGATE WALK-FORWARD COMPARISON")
print("#" * 100)
print(aggregate.to_string(float_format=lambda x: f"{x:.3f}"))

summary.to_csv(BASE / "v6_trade_outcome_walkforward_summary.csv", index=False)
trades.to_csv(BASE / "v6_trade_outcome_walkforward_trades.csv", index=False)
aggregate.to_csv(BASE / "v6_trade_outcome_walkforward_aggregate.csv")

print("\nSaved:")
print("  v6_trade_outcome_walkforward_summary.csv")
print("  v6_trade_outcome_walkforward_trades.csv")
print("  v6_trade_outcome_walkforward_aggregate.csv")
print("\nDo not change production from a single good window. Prefer 4/5+ positive OOS windows, PF > 1, and controlled DD.")
