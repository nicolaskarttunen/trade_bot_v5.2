"""Offline ledger consistency checks, not a verification of market fills or alpha.

Only recorded values are checked. Price-path truth, rejected candidates and
intrabar event ordering cannot be reconstructed from a closed-trade ledger.
"""
import numpy as np
import pandas as pd
from intraday_replay import validate_config

NY = "America/New_York"
TOL = 1e-6  # account currency; no percent-relative tolerance for cash balances
TRADE_NUMBERS = ["entry", "exit", "qty", "pnl", "stop", "target", "entry_equity",
                 "planned_stop_loss", "entry_exposure", "post_entry_equity",
                 "symbol_order", "ml_prob", "trade_score", "news_score"]
LIMITATIONS = [
    "Ledger consistency only; OHLC paths and real broker fills not independently verified.",
    "Risk/exposure tests use recorded entry marks; continuous intrabar exposure is not verified.",
    "No calibration claim for ml_prob; no news-effect or profitability validation.",
    "Counts by sector/order describe trades, not capital weights or causal selection bias.",
    "Original files must be retained; manifest hashes detect changes, not authenticate provenance.",
]


def audit_frames(trades, daily, curve, summary, cfg, symbols, sessions, sectors=None):
    """Validate one strategy; do not repair, reorder outputs on disk or tune cfg."""
    validate_config(cfg)
    t, d, c = trades.copy(), daily.copy(), curve.copy()
    issues = []
    def check(ok, code):
        if not bool(np.asarray(ok).all()):
            issues.append(code)
    def close(a, b):
        return np.isclose(a, b, rtol=0, atol=TOL)
    def require(frame, columns, name):
        missing = sorted(set(columns)-set(frame.columns))
        if missing:
            raise ValueError(f"{name}: missing columns {missing}")
    require(d, ["date", "start_equity", "pnl", "end_equity", "entries"], "daily")
    require(c, ["date", "clock", "equity"], "equity")
    require(summary, ["return_pct", "trades", "win_rate", "profit_factor", "max_dd_pct", "avg_pnl"], "summary")
    if len(summary) != 1:
        raise ValueError("Expected exactly one summary row per strategy")
    if not t.empty:
        require(t, [*TRADE_NUMBERS, "date", "symbol", "sector", "entry_ts", "exit_ts", "reason"], "trades")
        for key in TRADE_NUMBERS:
            t[key] = pd.to_numeric(t[key], errors="raise")
        if not np.isfinite(t[TRADE_NUMBERS].to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite trade values")
        for key in ["entry_ts", "exit_ts"]:
            t[key] = pd.to_datetime(t[key], utc=True, errors="raise")
            if t[key].isna().any():
                raise ValueError("Missing trade timestamp")
    for f, keys in [(d, ["start_equity", "pnl", "end_equity", "entries"]), (c, ["equity"])]:
        for key in keys:
            f[key] = pd.to_numeric(f[key], errors="raise")
        if not np.isfinite(f[keys].to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite daily/equity values")
    d = d.sort_values("date")
    check(len(d) == len(sessions) and set(d.date) == set(sessions) and not d.date.duplicated().any(), "session_coverage")
    if d.empty:
        raise ValueError("No daily rows")
    check((d.start_equity > 0) & (d.end_equity > 0), "positive_equity")
    check((d.entries >= 0) & (d.entries == np.floor(d.entries)) & (d.entries <= cfg["max_trades_per_day"]), "daily_entry_limit")
    check(close(d.start_equity.iloc[0], cfg["initial_cash"]), "initial_cash")
    check(close(d.start_equity.iloc[1:].to_numpy(), d.end_equity.iloc[:-1].to_numpy()), "cash_carry")
    check(close(d.end_equity-d.start_equity, d.pnl), "daily_cash_identity")
    if not t.empty:
        check(set(t.date) <= set(sessions), "trade_session_coverage")
        check(set(t.symbol) <= set(symbols), "unknown_symbol")
        check(~t.duplicated(["date", "symbol"]), "repeat_symbol_same_day")
        expected_order = {s: i+1 for i,s in enumerate(symbols)}
        check(t.symbol_order == t.symbol.map(expected_order), "symbol_order")
        if sectors is not None:
            check(t.sector == t.symbol.map(sectors).fillna("Unknown"), "sector_label")
        check((t.qty > 0) & (t.qty == np.floor(t.qty)), "positive_integer_quantity")
        check((t[["entry", "exit", "stop", "target", "entry_equity", "post_entry_equity"]] > 0).all().all(), "positive_trade_values")
        check((t.stop < t.entry) & (t.target > t.entry), "bracket_order")
        check((t.ml_prob >= cfg["probability_threshold"]) & (t.ml_prob <= 1) &
              (t.trade_score >= cfg["trade_score_threshold"]) & (t.trade_score <= 100), "entry_thresholds")
        slip = (cfg["slippage_bps"]+cfg["spread_bps"])/10000
        fee = cfg.get("commission_per_trade", 0)
        check(close((t.exit-t.entry)*t.qty-fee, t.pnl), "trade_pnl_identity")
        check(close((t.entry-t.stop*(1-slip))*t.qty+fee, t.planned_stop_loss), "planned_loss_identity")
        check(close(t.entry_equity-(t.entry-t.entry/(1+slip))*t.qty, t.post_entry_equity), "entry_cost_identity")
        check(t.planned_stop_loss <= t.entry_equity*cfg["risk_per_trade_pct"]+TOL, "planned_risk_limit")
        check(t.entry*t.qty <= t.post_entry_equity*cfg["max_position_pct"]+TOL, "position_limit")
        check((t.entry_exposure >= t.entry*t.qty-TOL) &
              (t.entry_exposure <= t.post_entry_equity*cfg["max_total_exposure_pct"]+TOL), "entry_exposure_limit")
        starts = d.drop_duplicates("date").set_index("date").start_equity
        check(t.entry_equity > t.date.map(starts)*(1-cfg["max_daily_loss_pct"]), "daily_loss_entry_gate")
        entry_et, exit_et = t.entry_ts.dt.tz_convert(NY), t.exit_ts.dt.tz_convert(NY)
        check((entry_et.dt.strftime("%Y-%m-%d") == t.date) &
              (exit_et.dt.strftime("%Y-%m-%d") == t.date) & (t.exit_ts > t.entry_ts), "trade_time_order")
        check((t.entry_ts == t.entry_ts.dt.floor("5min")) & (t.exit_ts == t.exit_ts.dt.floor("5min")), "timestamp_grid")
        flatten = {day:(pd.Timestamp(f"{day} {clock}", tz=NY)-pd.Timedelta(minutes=10)).strftime("%H:%M") for day,clock in sessions.items()}
        entry_clock, exit_clock = entry_et.dt.strftime("%H:%M"), exit_et.dt.strftime("%H:%M")
        check((entry_clock >= "09:35") & (entry_clock < "15:30") &
              (entry_clock < t.date.map(flatten)) & (exit_clock <= t.date.map(flatten)), "session_trade_times")
        check(t.reason.isin(["stop", "stop_gap", "target", "target_gap", "eod"]), "unvalidated_exit_reason")
        check(close(t.loc[t.reason == "stop", "exit"], t.loc[t.reason == "stop", "stop"]*(1-slip)), "stop_fill_identity")
        target = t.reason.isin(["target", "target_gap"])
        check(close(t.loc[target, "exit"], t.loc[target, "target"]*(1-slip)), "target_fill_identity")
        gap = t.reason == "stop_gap"
        check(t.loc[gap, "exit"] <= t.loc[gap, "stop"]*(1-slip)+TOL, "stop_gap_direction")
        check(exit_clock[t.reason == "eod"] == t.loc[t.reason == "eod", "date"].map(flatten), "eod_time")
        # [entry, exit) intervals; five-minute exit times do not resolve
        # within-bar ordering, so this is only recorded concurrency.
        max_open = 0
        for _, day in t.groupby("date"):
            for stamp in day.entry_ts.unique():
                max_open = max(max_open, int(((day.entry_ts <= stamp) & (day.exit_ts > stamp)).sum()))
        check(max_open <= cfg["max_open_positions"], "recorded_open_position_limit")
        pnls, counts = t.groupby("date").pnl.sum(), t.groupby("date").size()
    else:
        max_open, pnls, counts = 0, pd.Series(dtype=float), pd.Series(dtype=float)
    check(close(d.pnl, d.date.map(pnls).fillna(0)), "daily_trade_pnl_reconciliation")
    check(d.entries == d.date.map(counts).fillna(0), "daily_trade_count_reconciliation")
    check(close(d.end_equity.iloc[-1]-cfg["initial_cash"], t.pnl.sum() if not t.empty else 0), "final_pnl_reconciliation")
    # Require every engine mark, including zero-trade sessions.
    expected = [(day, ts.strftime("%H:%M")) for day,clock in sorted(sessions.items())
                for ts in pd.date_range(f"{day} 09:30", pd.Timestamp(f"{day} {clock}")-pd.Timedelta(minutes=10), freq="5min")]
    check(list(zip(c.date, c.clock)) == expected, "equity_grid")
    ends = c.groupby("date", sort=False).equity.last()
    check(close(d.end_equity, d.date.map(ends)), "equity_day_end_reconciliation")
    values = np.r_[float(cfg["initial_cash"]), c.equity.to_numpy()]
    drawdown = float(np.min(values/np.maximum.accumulate(values)-1)*100)
    profit = float(t.loc[t.pnl > 0, "pnl"].sum()) if not t.empty else 0.
    loss = float(-t.loc[t.pnl < 0, "pnl"].sum()) if not t.empty else 0.
    metrics = dict(return_pct=(d.end_equity.iloc[-1]/cfg["initial_cash"]-1)*100,
                   trades=len(t), win_rate=float((t.pnl > 0).mean()*100) if not t.empty else 0.,
                   profit_factor=profit/loss if loss else (float("inf") if len(t) else 0.),
                   max_dd_pct=drawdown, avg_pnl=float(t.pnl.mean()) if not t.empty else 0.)
    for key, value in metrics.items():
        check(close(float(summary.iloc[0][key]), value), "summary_"+key)
    def concentration(key):
        if t.empty:
            return []
        return [dict(name=str(name), trades=len(g), trade_pct=100*len(g)/len(t), pnl=float(g.pnl.sum()))
                for name,g in t.groupby(key, sort=True)]
    return dict(status="passed_ledger_checks" if not issues else "failed_ledger_checks",
                issues=issues, trades=len(t), sessions=len(d), recorded_max_open_positions=max_open,
                top20_trade_pct=100*float((t.symbol_order <= 20).mean()) if not t.empty else 0.,
                sectors=concentration("sector"), symbols=concentration("symbol"),
                positive_days=int((d.pnl > 0).sum()), negative_days=int((d.pnl < 0).sum()),
                zero_days=int((d.pnl == 0).sum()), best_day_pnl=float(d.pnl.max()),
                other_days_pnl=float(d.pnl.sum()-d.pnl.max()), limitations=LIMITATIONS)
