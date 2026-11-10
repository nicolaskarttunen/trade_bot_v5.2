"""Describe historical cost headroom on the validated opening-priority ledgers. No IO writes."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

PERIODS = {"august": 21, "october_november": 14}
ARMS = ("original_no_gate", "original_spy_ge010", "opening_ml_no_gate", "opening_ml_spy_ge010")


def number(value):
    n = float(value)
    if not math.isfinite(n):
        raise ValueError("Non-finite input")
    return n


def describe(trades, daily):
    days = {row["date"]: number(row["pnl"]) for row in daily}
    if not days or len(days) != len(daily):
        raise ValueError("Missing or duplicate dates")
    matched = dict.fromkeys(days, 0.0)
    turnover = 0.0
    for row in trades:
        qty, entry, exit_px = map(number, (row["qty"], row["entry"], row["exit"]))
        if qty < 1 or not qty.is_integer() or min(entry, exit_px) <= 0:
            raise ValueError("Invalid fill")
        matched[row["date"]] += number(row["pnl"])
        turnover += qty * (entry + exit_px)
    if any(abs(matched[d] - pnl) > 1e-6 for d, pnl in days.items()):
        raise ValueError("Daily P&L does not match trades")
    pnl = sum(days.values())
    # An extra fee on each executed notional: no new decisions, fills or sizing.
    return {"trades": len(trades), "net_usd": pnl, "best_day": max(days, key=days.get),
            "other_days_usd": pnl - max(days.values()),
            "extra_break_even_bp_per_side": 10000 * pnl / turnover if turnover else None,
            "net_extra_1bp_usd": pnl - turnover / 10000,
            "net_extra_3bp_usd": pnl - 3 * turnover / 10000}


def run(root):
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "completed_exploratory_trial" or report.get("all_controls_reproduced") is not True:
        raise ValueError("Expected a completed trial with reproduced controls")
    result = []
    for period, count in PERIODS.items():
        tables = []
        for name in ("trades", "daily"):
            rel = f"{period}/{name}.csv"
            raw = (root / rel).read_bytes()
            if hashlib.sha256(raw).hexdigest() != report["outputs"][rel]:
                raise ValueError(f"Hash mismatch: {rel}")
            rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
            if any(r["strategy"] not in ARMS for r in rows):
                raise ValueError("Unexpected strategy")
            tables.append(rows)
        for arm in ARMS:
            t, d = ([r for r in rows if r["strategy"] == arm] for rows in tables)
            if len(d) != count:
                raise ValueError(f"Incomplete period for {arm}")
            result.append(dict(period=period, arm=arm, **describe(t, d)))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    try:
        result = run(parser.parse_args().trial)
    except (KeyError, ValueError, OSError) as exc:
        raise SystemExit(f"Stopped: {exc}")
    print("Fixed historical trades only; USD; extra fee in bp per executed side.")
    print("Period          Arm                    Trades      Net   Ex-best   +1bp net   +3bp net  Breakeven bp")
    for r in result:
        bp = r["extra_break_even_bp_per_side"]
        print(f"{r['period']:15} {r['arm']:21} {r['trades']:6} {r['net_usd']:8.2f} "
              f"{r['other_days_usd']:9.2f} {r['net_extra_1bp_usd']:10.2f} "
              f"{r['net_extra_3bp_usd']:10.2f} {bp if bp is None else round(bp, 2)!s:>12}")
