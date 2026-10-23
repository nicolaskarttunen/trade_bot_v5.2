"""V6.5 PAPER-LIKE intraday replay, separate from the V6.4 09:35 experiment.

The Sep-Oct 2026 interval has ALREADY BEEN INSPECTED and is development
research, NOT untouched OOS. Simulates ordered 09:35..15:25 scans only:
OPG auction and premarket selection require data we do not possess.
No Alpaca API access or order submission.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURES, enrich
from intraday_replay import session_bounded_label
from ml_models import LogisticGD, RandomLinearEnsemble, CentroidClassifier
from intraday_replay import replay_intraday, as_day_clock


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
DATA_DIR = BASE.parent / "tradebot_v62_research" / "data"
NEWS_DIR = BASE / "news"
SYMBOLS = list(CFG["symbols"])
BENCH = CFG["benchmark_symbols"][0]
NY = "America/New_York"
EXT_DIR = BASE.parent / "tradebot_v62_research" / "data_v64"


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

# Intraday-only scan modes. These follow production SYMBOLS iteration order,
# not V6.4's hindsight-ranked 09:35 cross-sectional selection.
STRATEGIES = [
    {"name": "intraday_order_no_gate", "spy_gate": False},
    {"name": "intraday_order_spy_ge010", "spy_gate": True},
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
    x = session_bounded_label(
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


# Account/event execution lives in pure offline intraday_replay.py.
# It never reads brokerage credentials or submits orders.


def metrics(initial_cash, final_cash, trades, daily, intraday_curve=None):
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

    # Intraday mark-to-market catches drawdown that day-end-only curves miss.
    observations = (
        intraday_curve["equity"].tolist()
        if intraday_curve is not None and not intraday_curve.empty
        else daily["end_equity"].tolist()
    )
    eq = pd.Series([initial_cash] + observations, dtype=float)
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
print("=== V6.5 PAPER-LIKE INTRADAY (NO OPG): ORIGINAL VS SPY +0.10% ===")
print("Every completed 5-minute candle is checked 09:35-15:25 ET in SYMBOL order.")
print("Shared position/exposure/daily risk limits, stop/target and EOD simulation.")
print("OPG / premarket orders are OMITTED. This is NOT a full live-bot replica.")
print("Broker fill timing, spread, early closes and news latency are approximations.")
print("2026-09-01..2026-10-19 was previously inspected: NOT independent OOS.")
print("Corrected research model; historical news is neutral for all symbols.")
print("No Alpaca access. Original V6.4 files and Oracle service are untouched.")
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
all_equity_curves = []

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
    price_by_symbol = {}
    for symbol, test_s in test_by_symbol.items():
        scored_by_symbol[symbol] = prepare_scores(test_s, models)
        # Full ORIGINAL OHLC holdout rows, not rows filtered for ML features.
        raw = pd.read_csv(EXT_DIR / f"{symbol}_5min.csv")
        raw = as_day_clock(raw)
        price_by_symbol[symbol] = raw.loc[
            (raw["date"] >= te_start) & (raw["date"] <= te_end)
        ][["timestamp","open","high","low","close","volume"]].copy()

    for strategy in STRATEGIES:
        final_cash, trades, daily, curve = replay_intraday(
            scored_by_symbol=scored_by_symbol,
            full_prices_by_symbol=price_by_symbol,
            symbols=SYMBOLS,
            sectors=SECTOR_BY_SYMBOL,
            cfg=CFG,
            spy_opening_gate=SPY_GATE,
            require_spy=bool(strategy.get("spy_gate", False)),
        )
        m = metrics(float(CFG["initial_cash"]), final_cash, trades, daily, curve)
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
            trades["strategy"] = strategy["name"]
            all_trade_outputs.append(trades)
        if not curve.empty:
            curve = curve.copy()
            curve["window"] = wi
            curve["strategy"] = strategy["name"]
            all_equity_curves.append(curve)

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
summary_path = BASE / "v6_5_intraday_summary.csv"
summary_df.to_csv(summary_path, index=False)

if all_trade_outputs:
    trades_df = pd.concat(all_trade_outputs, ignore_index=True)
else:
    trades_df = pd.DataFrame()
trades_path = BASE / "v6_5_intraday_trades.csv"
trades_df.to_csv(trades_path, index=False)

print("\n=== V6.5 INTRADAY PAPER-LIKE COMPARISON ===")
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

equity_curve = (
    pd.concat(all_equity_curves, ignore_index=True)
    if all_equity_curves else pd.DataFrame()
)
curve_path = BASE / "v6_5_intraday_equity.csv"
equity_curve.to_csv(curve_path, index=False)
print(f"\nSaved: {summary_path.name}")
print(f"Saved: {trades_path.name}")
print(f"Saved: {curve_path.name}")
print("LIMITATIONS: OPG/premarket not simulated; live news neutralized.")
print("5-minute bar/open execution and EOD are proxies, not Alpaca order fills.")
print("Previously viewed dates are NOT new OOS. No production modifications.")
