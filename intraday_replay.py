"""Intraday, bar-by-bar PAPER-LIKE replay. Offline research; NEVER places orders.

Models the production *intraday* 09:35-15:30 ET scans in symbol order.
OPG/premarket, news latency, partial fills, broker order timing, exact early
closes and market impact cannot be inferred from five-minute OHLC data.
"""
from features import normalize_bars, label

import numpy as np
import pandas as pd


NY = "America/New_York"


def session_bounded_label(df, horizon=3, threshold=0.002):
    return label(df, horizon=horizon, threshold=threshold)


def as_day_clock(bars):
    """Normalize full OHLC bars to New York date and bar START clock."""
    x = normalize_bars(bars)
    x["date"] = x["timestamp"].dt.tz_convert(NY).dt.strftime("%Y-%m-%d")
    x["clock"] = x["timestamp"].dt.tz_convert(NY).dt.strftime("%H:%M")
    return x


def _book(df):
    return {
        (str(row["date"]), str(row["clock"])): row
        for _, row in as_day_clock(df).iterrows()
    }


def _signal_book(df):
    x = df.copy()
    ts = pd.to_datetime(x["timestamp"], utc=True)
    if ts.isna().any() or ts.duplicated().any() or (ts != ts.dt.floor("5min")).any():
        raise ValueError("Invalid or duplicate signal timestamps")
    # Caller-supplied date/clock columns cannot move a future signal backwards.
    x["trade_date"] = ts.dt.tz_convert(NY).dt.strftime("%Y-%m-%d")
    x["clock"] = ts.dt.tz_convert(NY).dt.strftime("%H:%M")
    if "signal_available_at" in x:
        available = pd.to_datetime(x["signal_available_at"], utc=True)
        if (available != ts + pd.Timedelta(minutes=5)).any():
            raise ValueError("Signal availability must be the bar close")
    return {(str(row["trade_date"]), str(row["clock"])): row for _, row in x.iterrows()}


def validate_config(cfg):
    positive = ["initial_cash", "stop_atr_multiplier", "reward_risk_ratio"]
    fractions = ["max_daily_loss_pct", "max_position_pct", "max_total_exposure_pct", "risk_per_trade_pct"]
    for key in positive + fractions + ["slippage_bps", "spread_bps", "commission_per_trade",
                                        "probability_threshold", "trade_score_threshold"]:
        value = float(cfg.get(key, 0))
        if not np.isfinite(value) or value < 0 or (key in positive and value <= 0):
            raise ValueError(f"Invalid config: {key}")
        if key in fractions and not 0 < value <= 1:
            raise ValueError(f"Invalid fraction: {key}")
    for key in ["max_trades_per_day", "max_open_positions"]:
        value = float(cfg[key])
        if not np.isfinite(value) or int(value) != value or value < 1:
            raise ValueError(f"Invalid count: {key}")
    if float(cfg["slippage_bps"]) + float(cfg["spread_bps"]) >= 10000:
        raise ValueError("Execution costs must be below 100%")
    if float(cfg["probability_threshold"]) > 1 or float(cfg["trade_score_threshold"]) > 100:
        raise ValueError("Invalid decision threshold")


def _previous_clock(clock):
    h, m = map(int, clock.split(":"))
    minutes = h * 60 + m - 5
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def replay_intraday(scored_by_symbol, full_prices_by_symbol, symbols, sectors, cfg,
                    spy_opening_gate=None, require_spy=False, session_closes=None):
    """Return (cash_equity, trades, daily, five_minute_equity).

    Scan at 09:35, 09:40, ... 15:25 ET using only the PREVIOUS completed
    five-minute bar. Market BUY approximated by current bar opening price +
    costs. All symbols are processed in configuration order like production,
    not hindsight-ranked. Positions coexist with open-position and exposure
    caps. A symbol cannot be entered more than once per NY session.

    The SPY filter is available after 09:35; it is NEVER applied to OPG.
    """
    validate_config(cfg)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be nonempty and unique")
    missing = set(symbols) - set(full_prices_by_symbol)
    if missing:
        raise ValueError(f"Missing price coverage: {sorted(missing)}")
    slip = (float(cfg["slippage_bps"]) + float(cfg["spread_bps"])) / 10000.0
    initial = float(cfg["initial_cash"])
    cash = initial
    positions = {}
    trades = []
    daily = []
    curve = []
    prices = {s: _book(full_prices_by_symbol[s]) for s in symbols if s in full_prices_by_symbol}
    signals = {s: _signal_book(scored_by_symbol[s]) for s in symbols if s in scored_by_symbol}
    days = sorted({d for symbol_bars in prices.values() for d, clock in symbol_bars})
    strict = bool(cfg.get("strict_price_coverage", True))

    def equity(mark_prices=None):
        marks = mark_prices or {}
        return cash + sum(
            position["qty"] * marks.get(symbol, position["last_price"])
            for symbol, position in positions.items()
        )

    def close_position(symbol, price, reason, ts):
        nonlocal cash
        position = positions.pop(symbol)
        proceeds = float(price) * position["qty"] - float(cfg.get("commission_per_trade", 0))
        cash += proceeds
        trade = dict(position["signal"])
        trade.update(
            entry_ts=position["entry_ts"], exit_ts=ts,
            entry=position["entry"], exit=float(price),
            qty=position["qty"],
            pnl=(float(price)-position["entry"]) * position["qty"] -
                float(cfg.get("commission_per_trade", 0)),
            stop=position["stop"], target=position["target"], reason=reason,
            entry_equity=position["entry_equity"],
            planned_stop_loss=position["planned_stop_loss"],
            entry_exposure=position["entry_exposure"],
            post_entry_equity=position["post_entry_equity"]
        )
        trades.append(trade)

    for date in days:
        traded_today = set()
        count = 0
        start_equity = equity()
        day_closed = False
        close_clock = "16:00" if session_closes is None else session_closes.get(date)
        if close_clock is None:
            raise ValueError(f"Missing session calendar for {date}")
        session_close = pd.Timestamp(f"{date} {close_clock}", tz=NY)
        if session_close.strftime("%H:%M") not in {"13:00", "16:00"}:
            raise ValueError(f"Unsupported session close: {close_clock}")
        flatten_clock = (session_close - pd.Timedelta(minutes=10)).strftime("%H:%M")
        all_clock = [t.strftime("%H:%M") for t in pd.date_range(
            pd.Timestamp(f"{date} 09:30", tz=NY), session_close - pd.Timedelta(minutes=10), freq="5min")]

        for clock in all_clock:
            current = {s: prices[s].get((date, clock)) for s in symbols if s in prices}
            current = {s: row for s, row in current.items() if row is not None}
            if not current and not positions:
                continue
            mark_open = {s: float(row["open"]) for s, row in current.items()}
            ts = pd.Timestamp(f"{date} {clock}", tz=NY).tz_convert("UTC")
            if strict:
                for held in positions:
                    if held not in current:
                        raise ValueError(f"Unobservable held position {held} at {ts}; replay invalid")

            # Stop/target gaps occur before any new BUY at this same bar open.
            for symbol in list(positions):
                bar = current.get(symbol)
                if bar is None:
                    continue
                pos = positions[symbol]
                opening = float(bar["open"])
                pos["last_price"] = opening
                if opening <= pos["stop"]:
                    close_position(symbol, opening*(1-slip), "stop_gap", bar["timestamp"])
                elif opening >= pos["target"]:
                    close_position(symbol, pos["target"]*(1-slip), "target_gap", bar["timestamp"])

            # 15:50 ET approximation to production start-flattening near EOD.
            if clock == flatten_clock:
                for symbol in list(positions):
                    bar = current.get(symbol)
                    px = float(bar["open"]) if bar is not None else positions[symbol]["last_price"]
                    close_position(symbol, px*(1-slip), "eod" if bar is not None else "eod_missing_bar", ts)
                day_closed = True
                curve.append({"date": date, "clock": clock, "equity": equity()})
                break

            if "09:35" <= clock < min("15:30", flatten_clock):
                gated = require_spy and not bool((spy_opening_gate or {}).get(date, False))
                prev = _previous_clock(clock)
                for order, symbol in enumerate(symbols):
                    if gated or count >= int(cfg["max_trades_per_day"]):
                        break
                    if symbol in positions or symbol in traded_today:
                        continue
                    signal = signals.get(symbol, {}).get((date, prev))
                    bar = current.get(symbol)
                    if signal is None or bar is None:
                        continue
                    ml = float(signal["ml_prob"])
                    if not np.isfinite(float(signal["news_score"])):
                        raise ValueError("Invalid news score")
                    score = float(signal["trade_score"])
                    if not (np.isfinite(ml) and np.isfinite(score)
                            and ml >= float(cfg["probability_threshold"])
                            and score >= float(cfg["trade_score_threshold"])):
                        continue
                    available_equity = equity(mark_open)
                    if available_equity <= start_equity*(1-float(cfg["max_daily_loss_pct"])):
                        break
                    if len(positions) >= int(cfg["max_open_positions"]):
                        continue
                    reference = float(bar["open"])
                    if not np.isfinite(reference) or reference <= 0:
                        continue
                    entry = reference*(1+slip)
                    atr = float(signal["atr"])
                    if not np.isfinite(atr) or atr <= 0:
                        continue
                    dist = max(atr*float(cfg["stop_atr_multiplier"]), reference*0.0125)
                    stop = round(reference-dist, 2)
                    target = round(reference+dist*float(cfg["reward_risk_ratio"]), 2)
                    if stop <= 0 or stop >= entry or target <= entry:
                        continue
                    # Include modeled entry/stop execution costs in the risk
                    # budget. commission_per_trade is one ROUND-TRIP fee,
                    # charged at exit and reserved here.
                    commission = float(cfg.get("commission_per_trade", 0))
                    loss_per_share = entry - stop * (1-slip)
                    risk_qty = int(max(0., available_equity*float(cfg["risk_per_trade_pct"])-commission)/loss_per_share)
                    cost_per_share = entry - reference
                    cap = float(cfg["max_position_pct"])
                    cap_qty = int(available_equity*cap/(entry + cap*cost_per_share))
                    exposure = sum(p["qty"]*p["last_price"] for p in positions.values())
                    cap = float(cfg["max_total_exposure_pct"])
                    exposure_qty = int(max(0., available_equity*cap-exposure)/(entry + cap*cost_per_share))
                    cash_qty = int(max(0., cash-commission*(len(positions)+1))/entry)
                    qty = min(risk_qty, cap_qty, exposure_qty, cash_qty)
                    if qty < 1:
                        continue
                    cash -= entry*qty
                    positions[symbol] = {
                        "entry_ts": bar["timestamp"],
                        "entry_equity": available_equity,
                        "planned_stop_loss": qty*loss_per_share + commission,
                        "entry_exposure": exposure + qty*entry,
                        "post_entry_equity": available_equity - qty*cost_per_share,
                        "entry": entry, "qty": qty, "stop": stop,
                        "target": target, "last_price": reference,
                        "signal": {
                            "date": date, "symbol": symbol,
                            "sector": sectors.get(symbol, "Unknown"),
                            "symbol_order": order+1,
                            "ml_prob": ml, "trade_score": score,
                            "news_score": float(signal["news_score"])
                        }
                    }
                    traded_today.add(symbol)
                    count += 1

            # After all new orders have been decided, apply this bar's
            # high/low to positions. Stops win if both are touched.
            for symbol in list(positions):
                bar = current.get(symbol)
                if bar is None:
                    continue
                pos = positions[symbol]
                if float(bar["low"]) <= pos["stop"]:
                    close_position(symbol, pos["stop"]*(1-slip), "stop", bar["timestamp"] + pd.Timedelta(minutes=5))
                elif float(bar["high"]) >= pos["target"]:
                    close_position(symbol, pos["target"]*(1-slip), "target", bar["timestamp"] + pd.Timedelta(minutes=5))
                else:
                    pos["last_price"] = float(bar["close"])
            mark_close = {s: float(r["close"]) for s, r in current.items()}
            curve.append({"date": date, "clock": clock, "equity": equity(mark_close)})

        # Missing 15:50 bars / shortened sessions: force liquidation at last
        # seen price (NOT a validated early-close fill). Mark explicitly.
        if positions and not day_closed:
            for symbol in list(positions):
                px = positions[symbol]["last_price"]
                close_position(symbol, px*(1-slip), "missing_eod_approx", session_close.tz_convert("UTC"))
        daily.append({
            "date": date, "start_equity": start_equity,
            "pnl": equity()-start_equity, "end_equity": equity(),
            "entries": count,
        })
    return equity(), pd.DataFrame(trades), pd.DataFrame(daily), pd.DataFrame(curve)
