"""Intraday, bar-by-bar PAPER-LIKE replay. Offline research; NEVER places orders.

Models the production *intraday* 09:35-15:30 ET scans in symbol order.
OPG/premarket, news latency, partial fills, broker order timing, exact early
closes and market impact cannot be inferred from five-minute OHLC data.
"""
from collections import defaultdict

import numpy as np
import pandas as pd


NY = "America/New_York"


def as_day_clock(bars):
    """Normalize full OHLC bars to New York date and bar START clock."""
    x = bars.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    x = x.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
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
    if "trade_date" not in x:
        x["trade_date"] = pd.to_datetime(x["timestamp"], utc=True).dt.tz_convert(NY).dt.strftime("%Y-%m-%d")
    if "clock" not in x:
        x["clock"] = pd.to_datetime(x["timestamp"], utc=True).dt.tz_convert(NY).dt.strftime("%H:%M")
    return {
        (str(row["trade_date"]), str(row["clock"])): row
        for _, row in x.iterrows()
    }


def _previous_clock(clock):
    h, m = map(int, clock.split(":"))
    minutes = h * 60 + m - 5
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def replay_intraday(scored_by_symbol, full_prices_by_symbol, symbols, sectors, cfg,
                    spy_opening_gate=None, require_spy=False):
    """Return (cash_equity, trades, daily, five_minute_equity).

    Scan at 09:35, 09:40, ... 15:25 ET using only the PREVIOUS completed
    five-minute bar. Market BUY approximated by current bar opening price +
    costs. All symbols are processed in configuration order like production,
    not hindsight-ranked. Positions coexist with open-position and exposure
    caps. A symbol cannot be entered more than once per NY session.

    The SPY filter is available after 09:35; it is NEVER applied to OPG.
    """
    if not symbols:
        raise ValueError("symbols cannot be empty")
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
    all_clock = [f"{minute//60:02d}:{minute%60:02d}"
                 for minute in range(9*60+30, 15*60+55, 5)]

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
            stop=position["stop"], target=position["target"], reason=reason
        )
        trades.append(trade)

    for date in days:
        traded_today = set()
        count = 0
        start_equity = equity()
        day_closed = False

        for clock in all_clock:
            current = {s: prices[s].get((date, clock)) for s in symbols if s in prices}
            current = {s: row for s, row in current.items() if row is not None}
            if not current and not positions:
                continue
            mark_open = {s: float(row["open"]) for s, row in current.items()}
            ts = next((r["timestamp"] for r in current.values()), pd.Timestamp(date, tz="UTC"))

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
            if clock == "15:50":
                for symbol in list(positions):
                    bar = current.get(symbol)
                    px = float(bar["open"]) if bar is not None else positions[symbol]["last_price"]
                    close_position(symbol, px*(1-slip), "eod" if bar is not None else "eod_missing_bar", ts)
                day_closed = True
                curve.append({"date": date, "clock": clock, "equity": equity()})
                break

            if "09:35" <= clock < "15:30":
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
                    # Live sizing uses the latest trade reference, while the
                    # historical fill uses next-bar OPEN plus spread/slippage.
                    risk_qty = int(available_equity*float(cfg["risk_per_trade_pct"])/dist)
                    cap_qty = int(available_equity*float(cfg["max_position_pct"])/reference)
                    exposure = sum(p["qty"]*p["last_price"] for p in positions.values())
                    exposure_qty = int(max(0.0, available_equity*float(cfg["max_total_exposure_pct"])-exposure)/reference)
                    qty = min(risk_qty, cap_qty, exposure_qty)
                    if qty < 1 or cash < qty*entry:
                        continue
                    cash -= entry*qty
                    positions[symbol] = {
                        "entry_ts": bar["timestamp"],
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
                    close_position(symbol, pos["stop"]*(1-slip), "stop", bar["timestamp"])
                elif float(bar["high"]) >= pos["target"]:
                    close_position(symbol, pos["target"]*(1-slip), "target", bar["timestamp"])
                else:
                    pos["last_price"] = float(bar["close"])
            mark_close = {s: float(r["close"]) for s, r in current.items()}
            curve.append({"date": date, "clock": clock, "equity": equity(mark_close)})

        # Missing 15:50 bars / shortened sessions: force liquidation at last
        # seen price (NOT a validated early-close fill). Mark explicitly.
        if positions and not day_closed:
            for symbol in list(positions):
                px = positions[symbol]["last_price"]
                close_position(symbol, px*(1-slip), "missing_eod_approx", pd.Timestamp(date, tz="UTC"))
        daily.append({
            "date": date, "start_equity": start_equity,
            "pnl": equity()-start_equity, "end_equity": equity(),
            "entries": count,
        })
    return equity(), pd.DataFrame(trades), pd.DataFrame(daily), pd.DataFrame(curve)
