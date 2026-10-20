"""V6.4 untouched future holdout for previously chosen SPY +0.10% gate.

Trains only through 2026-08-31, tests 2026-09-01..2026-10-19.
No future leakage, no parameter tuning, no production changes.
"""
import json\nfrom pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURES, enrich, label
from ml_models import LogisticGD, RandomLinearEnsemble, CentroidClassifier


BASE = Path(__file__).resolve().parent
CFG = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
# Lock paper reference settings IN MEMORY, not in config.json.
# Local Windows config has differed from the Oracle service.
CFG.update({
    "initial_cash": 100000.0,
    "risk_per_trade_pct": 0.01,
    "max_position_pct": 0.10,
    "max_total_exposure_pct": 0.50,
    "max_open_positions": 5,
    "max_trades_per_day": 5,
    "probability_threshold": 0.55,
    "trade_score_threshold": 55.0,
    "stop_atr_multiplier": 1.5,
    "reward_risk_ratio": 2.0,
    "slippage_bps": 3.0,
    "spread_bps": 2.0,
    "commission_per_trade": 0.0,
    "ml_weight": 0.45,
    "market_weight": 0.50,
    "news_weight": 0.05,
})
DATA_DIR = BASE / "data"
NEWS_DIR = BASE / "news"
SYMBOLS = list(CFG["symbols"])
BENCH = CFG["benchmark_symbols"][0]
NY = "America/New_York"
EXT_DIR = BASE / "data_v64"


def read_extended_csv(symbol):
    """Join baseline 2023-Aug2026 bars and isolated Sep-Oct2026 bars."""
    original = DATA_DIR / f"{symbol}_5min.csv"
    added = EXT_DIR / f"{symbol}_5min.csv"
    if not original.is_file() or not added.is_file():
        raise FileNotFoundError(f"V6.4 missing CSV for {symbol}: {original} / {added}")
    old = pd.read_csv(original)
    new = pd.read_csv(added)
    for frame in (old, new):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if (new.empty or len(new) < 500
        or new["timestamp"].min() >= pd.Timestamp("2026-09-03", tz="UTC")
        or new["timestamp"].min() < pd.Timestamp("2026-09-01", tz="UTC")
        or new["timestamp"].max() < pd.Timestamp("2026-10-19T13:30:00Z")
        or new["timestamp"].max() >= pd.Timestamp("2026-10-20", tz="UTC")):
        raise ValueError(f"V6.4 incomplete Sep-Oct 19 history: {symbol}")
    if old["timestamp"].max() >= new["timestamp"].min():
        raise ValueError(f"Data overlap detected: {symbol}")
    return pd.concat([old, new], ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp")

# Same untouched walk-forward test windows used in the earlier tests.
WINDOWS = [
    ("2023-01-01", "2026-08-31", "2026-09-01", "2026-10-19"),
]

# Fixed production thresholds. We are testing selection/timing, not tuning them.
PROB_THRESHOLD = float(CFG["probability_threshold"])
SCORE_THRESHOLD = float(CFG["trade_score_threshold"])
MIN_STOP_PCT = 0.0125
EOD_EXIT_CLOCK = "15:50"

# Fixed GICS-style sector buckets for the 100-symbol universe.
# This grouping is independent of the OOS results; no retrospective tuning.
SECTOR_GROUPS = {
    "Technology": "NVDA AMD AVGO AAPL MSFT ORCL CRM ADBE QCOM INTC AMAT MU LRCX TXN CSCO".split(),
    "Communication": "META GOOGL GOOG NFLX DIS T VZ CMCSA TMUS CHTR".split(),
    "ConsumerDiscretionary": "AMZN TSLA HD LOW MCD NKE TJX BKNG SBUX GM".split(),
    "ConsumerStaples": "WMT COST PG KO PEP PM MO CL MDLZ KHC".split(),
    "Financials": "JPM BAC WFC C GS MS BLK SCHW AXP USB".split(),
    "HealthCare": "LLY UNH JNJ ABBV MRK PFE TMO ABT ISRG AMGN".split(),
    "Energy": "XOM CVX COP SLB EOG OXY MPC PSX VLO HAL".split(),
    "Industrials": "CAT DE GE HON RTX UNP UPS BA ETN LMT".split(),
    "Materials": "LIN SHW FCX NEM NUE".split(),
    "Utilities": "NEE DUK SO AEP EXC".split(),
    "RealEstate": "PLD AMT O CCI EQIX".split(),
}
SECTOR_BY_SYMBOL = {}
for _sector, _symbols in SECTOR_GROUPS.items():
    for _symbol in _symbols:
        if _symbol in SECTOR_BY_SYMBOL:
            raise RuntimeError(f"Duplicate sector: {_symbol}")
        SECTOR_BY_SYMBOL[_symbol] = _sector
_missing = sorted(set(SYMBOLS) - set(SECTOR_BY_SYMBOL))
if _missing:
    raise RuntimeError(f"Unclassified symbols: {_missing}")

# Five-minute bars are start-stamped. 09:30 OHLC is visible at 09:35.
# The 09:30 OPG auction is NOT modeled: insufficient premarket bars.
STRATEGIES = [
    {"name": "0935_ranked5", "signal_clock": "09:30", "entry_clock": "09:35", "ranked": True, "max_entries": 5, "sector_cap": None},
    {"name": "0935_ranked5_spy_ge010", "signal_clock": "09:30", "entry_clock": "09:35", "ranked": True, "max_entries": 5, "sector_cap": None, "spy_gate": True},
]


def load_market():
    path = DATA_DIR / f"{BENCH}_5min.csv"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark data missing: {path}")

    m = read_extended_csv(BENCH)
    m["timestamp"] = pd.to_datetime(m["timestamp"], utc=True)
    m = m.sort_values("timestamp").drop_duplicates("timestamp")

    pc = m["close"].shift(1)
    tr = pd.concat(
        [
            m["high"] - m["low"],
            (m["high"] - pc).abs(),
            (m["low"] - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)
    m["atr_pct"] = tr.rolling(14).mean() / m["close"]
    return m


def load_symbol(symbol, market):
    path = DATA_DIR / f"{symbol}_5min.csv"
    if not path.exists():
        return None

    # Historical news is available for only a few symbols.
    # Avoid unequal data availability across the universe:
    # hold news neutral for EVERY symbol in this selection-only A/B/C test.
    # This does not reproduce the live news signal, so OOS returns remain proxies.
    news = None

    x = enrich(
        read_extended_csv(symbol),
        market=market,
        news=news,
        news_decay_minutes=CFG["news_decay_minutes"],
        news_lookback_minutes=CFG["news_lookback_minutes"],
    )
    x = label(
        x,
        horizon=CFG["label_horizon_bars"],
        threshold=CFG["label_threshold_pct"],
    )
    x = x.dropna(subset=FEATURES + ["atr", "target"]).copy()
    x["symbol"] = symbol
    x["symbol_order"] = SYMBOLS.index(symbol)
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    x["et"] = x["timestamp"].dt.tz_convert(NY)
    x["trade_date"] = x["et"].dt.date.astype(str)
    x["clock"] = x["et"].dt.strftime("%H:%M")
    return x


def train_ensemble(train):
    X = train[FEATURES].to_numpy(dtype=float)
    y = train["target"].to_numpy().astype(int)
    models = [
        LogisticGD(lr=0.03, epochs=500, l2=0.01),
        RandomLinearEnsemble(n_models=25, seed=42),
        CentroidClassifier(),
    ]
    for model in models:
        model.fit(X, y)
    return models


def predict(models, df):
    X = df[FEATURES].to_numpy(dtype=float)
    probs = [model.predict_proba(X)[:, 1] for model in models]
    return np.mean(probs, axis=0)


def prepare_scores(df, models):
    x = df.copy()
    x["ml_prob"] = predict(models, x)

    # Match PAPER BOT calculate_trade_score(): binary market sign,
    # linear normalized ML score, neutral zero-news = 0.5.
    market_component = (x["market_ret_3"] > 0).astype(float)
    ml_component = ((x["ml_prob"] - 0.5) / 0.5).clip(0.0, 1.0)
    news_component = (x["news_score"].clip(-1.0, 1.0) + 1.0) / 2.0
    x["trade_score"] = 100.0 * (
        float(CFG["ml_weight"]) * ml_component
        + float(CFG["market_weight"]) * market_component
        + float(CFG["news_weight"]) * news_component
    )
    return x


def choose_candidates(scored_by_symbol, strategy):
    rows = []
    signal_clock = strategy["signal_clock"]

    for symbol in SYMBOLS:
        df = scored_by_symbol.get(symbol)
        if df is None or df.empty:
            continue

        sig = df[df["clock"] == signal_clock]
        if sig.empty:
            continue

        sig = sig[
            (sig["ml_prob"] >= PROB_THRESHOLD)
            & (sig["trade_score"] >= SCORE_THRESHOLD)
        ]
        if sig.empty:
            continue

        # One signal per symbol/day at the exact completed signal bar.
        rows.append(sig)

    if not rows:
        return pd.DataFrame()

    candidates = pd.concat(rows, ignore_index=True)

    if strategy["ranked"]:
        candidates = candidates.sort_values(
            ["trade_date", "trade_score", "ml_prob", "symbol"],
            ascending=[True, False, False, True],
        )
    else:
        # Reproduce the production weakness deliberately: first qualifying
        # symbols in the configured symbol order win the slots.
        candidates = candidates.sort_values(
            ["trade_date", "symbol_order"],
            ascending=[True, True],
        )

    return candidates


def get_entry_bar(raw_symbol_df, trade_date, entry_clock):
    rows = raw_symbol_df[
        (raw_symbol_df["trade_date"] == trade_date)
        & (raw_symbol_df["clock"] == entry_clock)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def simulate_exit(raw_symbol_df, trade_date, entry_ts, stop, target, slip):
    day = raw_symbol_df[
        (raw_symbol_df["trade_date"] == trade_date)
        & (raw_symbol_df["timestamp"] >= entry_ts)
    ].sort_values("timestamp")

    # Stop/target can execute before 15:50. At 15:50 we flatten at the
    # opening price of that bar, approximating the production EOD routine.
    active = day[day["clock"] < EOD_EXIT_CLOCK]

    for _, bar in active.iterrows():
        bar_open = float(bar["open"])
        bar_low = float(bar["low"])
        bar_high = float(bar["high"])

        # Conservative gap handling for a sell stop.
        if bar_open <= stop:
            return bar_open * (1.0 - slip), "stop_gap", bar["timestamp"]
        if bar_low <= stop:
            return stop * (1.0 - slip), "stop", bar["timestamp"]

        # Conservative same-bar rule: if both stop and target are touched,
        # stop wins because it is checked first.
        if bar_open >= target:
            return target * (1.0 - slip), "target", bar["timestamp"]
        if bar_high >= target:
            return target * (1.0 - slip), "target", bar["timestamp"]

    eod = day[day["clock"] == EOD_EXIT_CLOCK]
    if not eod.empty:
        bar = eod.iloc[0]
        return float(bar["open"]) * (1.0 - slip), "eod", bar["timestamp"]

    # Fail-safe for occasional missing 15:50 bar: use last regular-session
    # close strictly before 15:50, never an after-hours bar.
    fallback = day[day["clock"] < EOD_EXIT_CLOCK]
    if fallback.empty:
        return None, "no_exit", None
    bar = fallback.iloc[-1]
    return float(bar["close"]) * (1.0 - slip), "eod_fallback", bar["timestamp"]


def run_strategy(scored_by_symbol, raw_by_symbol, strategy):
    slip = (
        float(CFG["slippage_bps"]) + float(CFG["spread_bps"])
    ) / 10000.0

    candidates = choose_candidates(scored_by_symbol, strategy)
    if not candidates.empty and strategy.get("spy_gate"):
        # SPY first 09:30 ET 5-min bar has fully closed by 09:35 ET.
        # Apply gate to ALL originally selected candidates by date;
        # never replace skipped trades with other symbols.
        candidates = candidates[candidates["trade_date"].map(
            lambda d: bool(SPY_GATE.get(str(d), False))
        )].copy()
    if candidates.empty:
        return float(CFG["initial_cash"]), pd.DataFrame(), pd.DataFrame()

    equity = float(CFG["initial_cash"])
    trades = []
    daily_rows = []

    for trade_date, day_candidates in candidates.groupby("trade_date", sort=True):
        day_start_equity = equity
        day_exposure = 0.0
        day_pnl = 0.0
        entries = 0
        sector_counts = {}

        for _, signal in day_candidates.iterrows():
            if entries >= int(strategy["max_entries"]):
                break

            symbol = str(signal["symbol"])
            sector = SECTOR_BY_SYMBOL[symbol]
            sector_cap = strategy.get("sector_cap")
            if sector_cap is not None and sector_counts.get(sector, 0) >= sector_cap:
                continue
            raw = raw_by_symbol[symbol]
            entry_bar = get_entry_bar(raw, trade_date, strategy["entry_clock"])
            if entry_bar is None:
                continue

            entry = float(entry_bar["open"]) * (1.0 + slip)
            if entry <= 0:
                continue

            stop_distance = max(
                float(signal["atr"]) * float(CFG["stop_atr_multiplier"]),
                entry * MIN_STOP_PCT,
            )
            stop = entry - stop_distance
            target = entry + stop_distance * float(CFG["reward_risk_ratio"])

            risk_qty = int(
                (day_start_equity * float(CFG["risk_per_trade_pct"]))
                / stop_distance
            )
            position_qty = int(
                (day_start_equity * float(CFG["max_position_pct"])) / entry
            )
            remaining_exposure = max(
                0.0,
                day_start_equity * float(CFG["max_total_exposure_pct"])
                - day_exposure,
            )
            exposure_qty = int(remaining_exposure / entry)
            qty = min(risk_qty, position_qty, exposure_qty)

            if qty < 1:
                continue

            exit_price, reason, exit_ts = simulate_exit(
                raw,
                trade_date,
                entry_bar["timestamp"],
                stop,
                target,
                slip,
            )
            if exit_price is None:
                continue

            position_value = entry * qty
            pnl = (exit_price - entry) * qty - float(CFG["commission_per_trade"])
            day_exposure += position_value
            day_pnl += pnl
            entries += 1
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

            trades.append(
                {
                    "strategy": strategy["name"],
                    "date": trade_date,
                    "symbol": symbol,
                    "sector": sector,
                    "symbol_order": int(signal["symbol_order"]) + 1,
                    "entry_ts": entry_bar["timestamp"],
                    "exit_ts": exit_ts,
                    "entry": entry,
                    "exit": exit_price,
                    "qty": qty,
                    "pnl": pnl,
                    "ml_prob": float(signal["ml_prob"]),
                    "trade_score": float(signal["trade_score"]),
                    "news_score": float(signal["news_score"]),
                    "stop": stop,
                    "target": target,
                    "reason": reason,
                }
            )

        equity += day_pnl
        daily_rows.append(
            {
                "strategy": strategy["name"],
                "date": trade_date,
                "start_equity": day_start_equity,
                "pnl": day_pnl,
                "end_equity": equity,
                "entries": entries,
            }
        )

    return equity, pd.DataFrame(trades), pd.DataFrame(daily_rows)


def metrics(initial_cash, final_cash, trades, daily):
    if trades.empty:
        return {
            "return_pct": 0.0,
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_dd_pct": 0.0,
            "avg_pnl": 0.0,
        }

    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    eq = pd.Series([initial_cash] + daily["end_equity"].tolist(), dtype=float)
    dd = (eq / eq.cummax() - 1.0).min() * 100.0

    return {
        "return_pct": (final_cash / initial_cash - 1.0) * 100.0,
        "trades": int(len(trades)),
        "win_rate": float((trades["pnl"] > 0).mean() * 100.0),
        "profit_factor": float(pf),
        "max_dd_pct": float(dd),
        "avg_pnl": float(trades["pnl"].mean()),
    }


def slice_period(df, start, end):
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end + " 23:59:59", tz="UTC")
    return df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].copy()


print()
print("=== V6.4 UNTOUCHED OUT-OF-SAMPLE: 09:35 RANKED5 VS SPY +0.10% ===")
print("Locked paper-like score formula and fixed costs/risk across six strategies.")
print("Pre-locked SPY gate at 09:35; 09:30 bar must be complete. Same model and costs.")
print("LIMITATIONS: 09:30 OPG auction and later intraday scans are not replayed.")
print("Walk-forward model retrained on prior historical bars, not today's live model.")
print("2026-09-01..2026-10-19: NEW OOS only. Historical news neutral; fills proxied.")
print("Production, model artifacts, config.json and orders are not modified.")
print()

market = load_market()

# Ex-ante SPY regime: ONLY the completed 09:30-09:35 ET candle is used.
spy_raw = market.copy()
spy_raw["trade_date"] = spy_raw["timestamp"].dt.tz_convert(NY).dt.strftime("%Y-%m-%d")
spy_raw["clock"] = spy_raw["timestamp"].dt.tz_convert(NY).dt.strftime("%H:%M")
spy_opening_bars = spy_raw.loc[spy_raw["clock"] == "09:30"].copy()
spy_opening_bars["opening_move_pct"] = (
    100.0 * (spy_opening_bars["close"] / spy_opening_bars["open"] - 1.0)
)
SPY_GATE = dict(zip(spy_opening_bars["trade_date"], spy_opening_bars["opening_move_pct"] >= 0.10))
print("SPY +0.10% gated days in holdout:", sum(
    allowed for day, allowed in SPY_GATE.items()
    if "2026-09-01" <= day <= "2026-10-19"
))
all_data = {}

for symbol in SYMBOLS:
    x = load_symbol(symbol, market)
    if x is not None and not x.empty:
        all_data[symbol] = x
        print(f"Loaded {symbol}: {len(x):,} rows")

print(f"\nSymbols with data: {len(all_data)}/{len(SYMBOLS)}")
if len(all_data) != len(SYMBOLS):
    raise SystemExit("Missing V6.4 historical stock data; expected all 100 symbols.")

summary = []
all_trade_outputs = []

for wi, (tr_start, tr_end, te_start, te_end) in enumerate(WINDOWS, 1):
    print("\n" + "=" * 78)
    print(f"WINDOW {wi}: train {tr_start}..{tr_end} | test {te_start}..{te_end}")
    print("=" * 78)

    train_frames = []
    test_by_symbol = {}

    for symbol, df in all_data.items():
        train_s = slice_period(df, tr_start, tr_end)
        # Prevent labels at the end of training from reaching beyond training.
        horizon = int(CFG["label_horizon_bars"])
        if len(train_s) > horizon:
            train_s = train_s.iloc[:-horizon].copy()
        else:
            train_s = train_s.iloc[0:0].copy()

        test_s = slice_period(df, te_start, te_end)

        if not train_s.empty:
            train_frames.append(train_s)
        if not test_s.empty:
            test_by_symbol[symbol] = test_s

    if not train_frames or not test_by_symbol:
        print("No data for this window; skipping.")
        continue

    train = pd.concat(train_frames, ignore_index=True).sort_values("timestamp")
    print(f"Training rows: {len(train):,} | test symbols: {len(test_by_symbol)}")

    models = train_ensemble(train)

    scored_by_symbol = {}
    for symbol, test_s in test_by_symbol.items():
        scored_by_symbol[symbol] = prepare_scores(test_s, models)

    for strategy in STRATEGIES:
        final_cash, trades, daily = run_strategy(
            scored_by_symbol,
            test_by_symbol,
            strategy,
        )
        m = metrics(float(CFG["initial_cash"]), final_cash, trades, daily)
        m["top20_pct"] = (
            100.0 * float((trades["symbol_order"] <= 20).mean())
            if not trades.empty else 0.0
        )
        m["technology_pct"] = (
            100.0 * float((trades["sector"] == "Technology").mean())
            if not trades.empty else 0.0
        )
        row = {
            "window": wi,
            "test_start": te_start,
            "test_end": te_end,
            "strategy": strategy["name"],
            **m,
        }
        summary.append(row)
        if not trades.empty:
            trades = trades.copy()
            trades["window"] = wi
            all_trade_outputs.append(trades)

        print(
            f"{strategy['name']:<20} "
            f"ret={m['return_pct']:>7.2f}% | "
            f"trades={m['trades']:>4} | "
            f"WR={m['win_rate']:>6.2f}% | "
            f"PF={m['profit_factor']:>5.2f} | "
            f"DD={m['max_dd_pct']:>7.2f}% | "
            f"avg=${m['avg_pnl']:>7.2f}"
        )

summary_df = pd.DataFrame(summary)
summary_path = BASE / "v6_4_new_oos_summary.csv"
summary_df.to_csv(summary_path, index=False)

if all_trade_outputs:
    trades_df = pd.concat(all_trade_outputs, ignore_index=True)
else:
    trades_df = pd.DataFrame()
trades_path = BASE / "v6_4_new_oos_trades.csv"
trades_df.to_csv(trades_path, index=False)

print("\n=== V6.4 NEW HOLDOUT COMPARISON ===")
if not summary_df.empty:
    agg = (
        summary_df.groupby("strategy")
        .agg(
            mean_return_pct=("return_pct", "mean"),
            median_return_pct=("return_pct", "median"),
            positive_windows=("return_pct", lambda s: int((s > 0).sum())),
            total_trades=("trades", "sum"),
            mean_win_rate=("win_rate", "mean"),
            mean_profit_factor=("profit_factor", "mean"),
            worst_dd_pct=("max_dd_pct", "min"),
            mean_avg_pnl=("avg_pnl", "mean"),
            mean_top20_pct=("top20_pct", "mean"),
            mean_technology_pct=("technology_pct", "mean"),
        )
        .sort_values(["positive_windows", "mean_return_pct"], ascending=[False, False])
    )
    print(agg.to_string(float_format=lambda v: f"{v:.2f}"))

print(f"\nSaved: {summary_path.name}")
print(f"Saved: {trades_path.name}")
print()
print("LIMITATION: OPG 09:30 and repeating intraday scans were not simulated.")
print("This is one short held-out period and is not proof of a durable trading edge.")
print("The SPY filter selects days, not replacement stock candidates, and recalculates sizes independently.")
