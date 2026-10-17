from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
V6_PATH = BASE / "v6_trade_outcome_walkforward.py"
if not V6_PATH.exists():
    raise FileNotFoundError(
        "v6_trade_outcome_walkforward.py is required in the same folder. "
        "Fetch the V6 research script first."
    )

# Reuse V6's data/feature/trade-outcome builders without executing its main backtest.
source = V6_PATH.read_text(encoding="utf-8")
marker = '\nprint("=" * 100)\nprint("V6 TRADE-OUTCOME WALK-FORWARD")'
if marker not in source:
    raise RuntimeError("Could not locate the V6 main-program marker")
shared_source = source.split(marker, 1)[0]
shared = {"__file__": str(V6_PATH), "__name__": "v6_shared_defs"}
exec(compile(shared_source, str(V6_PATH), "exec"), shared)

SYMBOLS = shared["SYMBOLS"]
ENHANCED_FEATURES = shared["ENHANCED_FEATURES"]
WINDOWS = shared["WINDOWS"]
INITIAL_CASH = shared["INITIAL_CASH"]
RISK_PER_TRADE = shared["RISK_PER_TRADE"]
MAX_POSITION_PCT = shared["MAX_POSITION_PCT"]
MAX_TOTAL_EXPOSURE_PCT = shared["MAX_TOTAL_EXPOSURE_PCT"]
COMMISSION = shared["COMMISSION"]
prepare_market = shared["prepare_market"]
build_symbol_samples = shared["build_symbol_samples"]
add_cross_sectional_breadth = shared["add_cross_sectional_breadth"]
make_models = shared["make_models"]
metrics = shared["metrics"]

TRAINING_MODES = {
    "expanding": None,
    "rolling_12m": 12,
    "rolling_18m": 18,
    "rolling_24m": 24,
}

# Small, fixed inner-validation grid. The untouched OOS windows never choose these.
CUTOFF_QUANTILES = [0.50, 0.60, 0.70, 0.80, 0.90]
TOP_N = 3
MIN_VALIDATION_TRADES = 40
MIN_VALIDATION_PF = 1.05
MIN_FIT_SAMPLES = 5_000
MIN_TEST_SAMPLES = 100


def period_mask(df, start, end):
    return (df["trade_date"] >= start) & (df["trade_date"] <= end)


def rolling_start(train_start, train_end, months):
    if months is None:
        return train_start
    end = pd.Timestamp(train_end)
    start = end + pd.Timedelta(days=1) - pd.DateOffset(months=int(months))
    start_s = start.strftime("%Y-%m-%d")
    return max(train_start, start_s)


def split_fit_validation(pool):
    dates = sorted(pool["trade_date"].dropna().unique().tolist())
    if len(dates) < 100:
        return None, None, None

    # Recent validation block: 20% of sessions, bounded to 40..90 sessions.
    val_days = int(round(len(dates) * 0.20))
    val_days = min(90, max(40, val_days))
    if len(dates) - val_days < 60:
        val_days = max(20, len(dates) - 60)
    if val_days <= 0:
        return None, None, None

    val_start = dates[-val_days]
    fit = pool[pool["trade_date"] < val_start].copy()
    val = pool[pool["trade_date"] >= val_start].copy()
    return fit, val, val_start


def run_portfolio_cutoff(scored, cutoff, strategy_name):
    equity = INITIAL_CASH
    trade_rows = []
    daily_rows = []

    for d, day in scored.groupby("trade_date", sort=True):
        candidates = day[np.isfinite(day["predicted_r"])].copy()
        candidates = candidates[candidates["predicted_r"] >= float(cutoff)]
        candidates = candidates.sort_values("predicted_r", ascending=False).head(TOP_N)

        start_equity = equity
        day_pnl = 0.0
        day_exposure = 0.0
        entries = 0

        for _, r in candidates.iterrows():
            entry = float(r["entry"])
            stop_distance = float(r["stop_distance"])
            if not np.isfinite(entry) or not np.isfinite(stop_distance):
                continue
            if entry <= 0 or stop_distance <= 0:
                continue

            risk_qty = int((start_equity * RISK_PER_TRADE) / stop_distance)
            pos_qty = int((start_equity * MAX_POSITION_PCT) / entry)
            remaining = max(
                0.0,
                start_equity * MAX_TOTAL_EXPOSURE_PCT - day_exposure,
            )
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
                    "strategy": strategy_name,
                    "date": d,
                    "symbol": r["symbol"],
                    "predicted_r": float(r["predicted_r"]),
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
                "strategy": strategy_name,
                "date": d,
                "start_equity": start_equity,
                "pnl": day_pnl,
                "end_equity": equity,
                "entries": entries,
            }
        )

    return equity, pd.DataFrame(trade_rows), pd.DataFrame(daily_rows)


def score_metrics(final_cash, trades, daily):
    # V6's metrics() expects the same columns produced above.
    return metrics(final_cash, trades, daily)


def choose_validation_cutoff(model, val, mode_name, window_no):
    scored = val.copy()
    scored["predicted_r"] = model.predict(scored[ENHANCED_FEATURES])
    finite_pred = scored.loc[np.isfinite(scored["predicted_r"]), "predicted_r"]
    if finite_pred.empty:
        return None, []

    raw_candidates = [("positive", 0.0)]
    for q in CUTOFF_QUANTILES:
        cutoff = max(0.0, float(np.quantile(finite_pred.to_numpy(), q)))
        raw_candidates.append((f"q{int(q * 100)}", cutoff))

    # Quantiles can collapse to the same cutoff; test each numeric cutoff only once.
    candidates = []
    seen = set()
    for label, cutoff in raw_candidates:
        key = round(float(cutoff), 8)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((label, float(cutoff)))

    diagnostic_rows = []
    valid = []

    for label, cutoff in candidates:
        name = f"{mode_name}_{label}_val"
        final_cash, trades, daily = run_portfolio_cutoff(scored, cutoff, name)
        m = score_metrics(final_cash, trades, daily)
        passes = (
            m["trades"] >= MIN_VALIDATION_TRADES
            and m["avg_r"] > 0.0
            and m["profit_factor"] > MIN_VALIDATION_PF
            and m["return_pct"] > 0.0
        )
        row = {
            "window": window_no,
            "mode": mode_name,
            "cutoff_label": label,
            "cutoff": cutoff,
            "passes": bool(passes),
            **m,
        }
        diagnostic_rows.append(row)
        if passes:
            valid.append(row)

    if not valid:
        return None, diagnostic_rows

    # Nested selection: maximize validation return only among candidates that
    # already clear minimum trade-count, positive expectancy and PF gates.
    best = sorted(
        valid,
        key=lambda r: (
            r["return_pct"],
            r["profit_factor"],
            r["avg_r"],
            r["trades"],
        ),
        reverse=True,
    )[0]
    return best, diagnostic_rows


print("=" * 100)
print("V6.1 ROLLING + NESTED VALIDATION WALK-FORWARD")
print("Enhanced HGB only; signal 09:40 ET -> entry 09:45 ET; top 3/day.")
print("Compares expanding vs rolling 12/18/24 month training pools.")
print("Predicted-R cutoff is chosen ONLY on the inner validation block.")
print("If validation finds no edge, that mode sits out the OOS window.")
print("No production files or config.json are modified.")
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

summary_rows = []
trade_parts = []
validation_rows = []

for wno, (train_start, train_end, test_start, test_end) in enumerate(WINDOWS, 1):
    test = samples[period_mask(samples, test_start, test_end)].copy()

    print("\n" + "=" * 100)
    print(f"WINDOW {wno}: OOS {test_start}..{test_end}")
    print("=" * 100)

    if len(test) < MIN_TEST_SAMPLES:
        print("Skipping OOS window: insufficient test samples")
        continue

    for mode_name, months in TRAINING_MODES.items():
        effective_start = rolling_start(train_start, train_end, months)
        pool = samples[period_mask(samples, effective_start, train_end)].copy()
        fit, val, val_start = split_fit_validation(pool)

        if fit is None or val is None or len(fit) < MIN_FIT_SAMPLES:
            print(f"{mode_name:<14} insufficient fit/validation data")
            continue

        model = make_models()["hgb_enh"]
        model.fit(fit[ENHANCED_FEATURES], fit["realized_r"])

        chosen, diag = choose_validation_cutoff(model, val, mode_name, wno)
        validation_rows.extend(diag)

        if chosen is None:
            row = {
                "window": wno,
                "mode": mode_name,
                "effective_train_start": effective_start,
                "train_end": train_end,
                "validation_start": val_start,
                "test_start": test_start,
                "test_end": test_end,
                "active": False,
                "cutoff_label": "NONE",
                "cutoff": np.nan,
                "validation_return_pct": np.nan,
                "validation_pf": np.nan,
                "validation_avg_r": np.nan,
                "return_pct": 0.0,
                "trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_dd_pct": 0.0,
                "avg_r": 0.0,
                "avg_pnl": 0.0,
            }
            summary_rows.append(row)
            print(
                f"{mode_name:<14} NO VALIDATION EDGE | "
                f"pool={effective_start}..{train_end} | val={val_start}..{train_end}"
            )
            continue

        scored_test = test.copy()
        scored_test["predicted_r"] = model.predict(scored_test[ENHANCED_FEATURES])
        strategy_name = f"{mode_name}_{chosen['cutoff_label']}"
        final_cash, trades, daily = run_portfolio_cutoff(
            scored_test,
            chosen["cutoff"],
            strategy_name,
        )
        m = score_metrics(final_cash, trades, daily)

        row = {
            "window": wno,
            "mode": mode_name,
            "effective_train_start": effective_start,
            "train_end": train_end,
            "validation_start": val_start,
            "test_start": test_start,
            "test_end": test_end,
            "active": True,
            "cutoff_label": chosen["cutoff_label"],
            "cutoff": chosen["cutoff"],
            "validation_return_pct": chosen["return_pct"],
            "validation_pf": chosen["profit_factor"],
            "validation_avg_r": chosen["avg_r"],
            **m,
        }
        summary_rows.append(row)

        if not trades.empty:
            trades["window"] = wno
            trades["mode"] = mode_name
            trades["cutoff_label"] = chosen["cutoff_label"]
            trades["cutoff"] = chosen["cutoff"]
            trade_parts.append(trades)

        print(
            f"{mode_name:<14} val={chosen['cutoff_label']:<8} "
            f"cut={chosen['cutoff']:+.3f} | "
            f"OOS ret={m['return_pct']:7.2f}% | trades={m['trades']:4d} | "
            f"WR={m['win_rate']:6.2f}% | PF={m['profit_factor']:5.2f} | "
            f"DD={m['max_dd_pct']:7.2f}% | avgR={m['avg_r']:+.3f}"
        )

summary = pd.DataFrame(summary_rows)
validation = pd.DataFrame(validation_rows)
trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()

if summary.empty:
    raise RuntimeError("No V6.1 walk-forward results were produced")

aggregate_rows = []
for mode_name, g in summary.groupby("mode", sort=True):
    active = g[g["active"] == True]
    aggregate_rows.append(
        {
            "mode": mode_name,
            "mean_return_pct": float(g["return_pct"].mean()),
            "median_return_pct": float(g["return_pct"].median()),
            "active_windows": int(g["active"].sum()),
            "positive_windows": int((g["return_pct"] > 0).sum()),
            "pf_above_one": int((active["profit_factor"] > 1.0).sum()) if len(active) else 0,
            "total_trades": int(g["trades"].sum()),
            "mean_active_pf": float(active["profit_factor"].mean()) if len(active) else 0.0,
            "mean_active_avg_r": float(active["avg_r"].mean()) if len(active) else 0.0,
            "worst_dd_pct": float(g["max_dd_pct"].min()),
            "robust_4of5": bool(
                int(g["active"].sum()) >= 4
                and int((g["return_pct"] > 0).sum()) >= 4
                and (int((active["profit_factor"] > 1.0).sum()) if len(active) else 0) >= 4
                and float(g["return_pct"].mean()) > 0.0
            ),
        }
    )

aggregate = pd.DataFrame(aggregate_rows).sort_values(
    ["robust_4of5", "positive_windows", "mean_return_pct", "mean_active_pf"],
    ascending=[False, False, False, False],
)

print("\n" + "#" * 100)
print("V6.1 AGGREGATE WALK-FORWARD COMPARISON")
print("#" * 100)
print(aggregate.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

summary.to_csv(BASE / "v6_1_rolling_validation_summary.csv", index=False)
validation.to_csv(BASE / "v6_1_rolling_validation_choices.csv", index=False)
trades.to_csv(BASE / "v6_1_rolling_validation_trades.csv", index=False)
aggregate.to_csv(BASE / "v6_1_rolling_validation_aggregate.csv", index=False)

print("\nSaved:")
print("  v6_1_rolling_validation_summary.csv")
print("  v6_1_rolling_validation_choices.csv")
print("  v6_1_rolling_validation_trades.csv")
print("  v6_1_rolling_validation_aggregate.csv")
print("\nProduction remains unchanged. A useful pass requires active coverage plus 4/5 positive OOS windows, not just one strong recent period.")