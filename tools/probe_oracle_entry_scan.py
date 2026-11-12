"""Run the production selection functions with data GETs and submissions denied.

The trading process is never imported or started. Only an explicit AST function
allowlist is compiled; no module-level clients, loops or Telegram threads run.
Reports are private deployment evidence, not a frozen research study.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import pickle
import sys
import threading

os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

FUNCTIONS = ("get_latest_trade_price", "get_bars", "get_ml_probability",
             "calculate_trade_score", "entry_scan_is_current", "run_entry_scan")


class ProbeError(ValueError):
    pass


def deny_submission(*args, **kwargs):
    raise ProbeError("Order submission is disabled in the deployment probe")


def load_functions(path, scope):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    if len(selected) != len(FUNCTIONS):
        raise ProbeError("Production scan function allowlist changed")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return scope


def validate_report(report, symbols):
    if (not report or not report.get("complete") or report.get("status") not in {"complete", "paused"}
            or report.get("scanned_count") != len(symbols) or report.get("universe_count") != len(symbols)
            or len(report.get("signals", [])) != len(symbols)
            or {r["symbol"] for r in report["signals"]} != set(symbols)
            or report.get("submitted_count") != 0 or len(report.get("top5", [])) > 5):
        raise ProbeError("No complete, current, zero-submission universe scan was verified")
    errors = sum("error" in r for r in report["signals"])
    if errors:
        raise ProbeError("Market-data errors occurred; inspect the private probe report before rollout")
    return dict(scanned=report["scanned_count"], universe=len(symbols),
                signal_status_counts=dict(Counter(r["status"] for r in report["signals"])),
                shortlisted=len(report["top5"]), submitted=0, as_of=report["as_of"],
                classification="read_only_production_function_probe")


def make_data_client(pair):
    from alpaca.data.historical import StockHistoricalDataClient
    class GetOnlyData(StockHistoricalDataClient):
        def _one_request(self, method, url, opts, retry):
            if str(method).upper() != "GET":
                raise ProbeError("Non-GET data request blocked")
            return super()._one_request(method, url,
                dict(opts, timeout=(5, 10), allow_redirects=False), 0)
    return GetOnlyData(pair[0], pair[1])


def run_probe(release, output, pair, account, *, data_client=None):
    import pandas as pd
    from alpaca.common.enums import Sort
    from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from features import completed_bars
    from model_pipeline import latest_signal, trade_score, validate_bundle
    from entry_selection import rank_entry_candidates, long_trade_plan, SELECTION_POLICY

    release, output = Path(release), Path(output)
    cfg = json.loads((release / "config.json").read_text())
    with (release / "models/ensemble.pkl").open("rb") as model:
        bundle = pickle.load(model)
    if validate_bundle(bundle, cfg, live=True) != "neutral":
        raise ProbeError("This cutover probe requires the verified neutral-news candidate")
    equity = float(account["equity"])
    if not math.isfinite(equity) or equity <= 0:
        raise ProbeError("Invalid paper equity for the scan")
    state_file = release / "paper_live_state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    today = pd.Timestamp.now(tz="America/New_York").date().isoformat()
    if state.get("date") != today:
        state = dict(date=today, trades=0, start_equity=equity, traded_symbols=[])
    if (not isinstance(state.get("trades"), int) or state["trades"] < 0
            or not math.isfinite(float(state.get("start_equity", 0))) or float(state["start_equity"]) <= 0):
        raise ProbeError("Invalid daily risk state for the scan")
    # The cutover gate requires a flat account and no previous journal for this
    # legacy migration. No broker maintenance or local state reset is performed.
    client = data_client if data_client is not None else make_data_client(pair)
    output.mkdir(parents=True, exist_ok=True)
    def atomic(path, report):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(path)
    scope = dict(pd=pd, datetime=datetime, timedelta=timedelta, timezone=timezone, math=math,
        CFG=cfg, MODEL=bundle, SYMBOLS=cfg["symbols"], BENCHMARK=cfg["benchmark_symbols"][0],
        data_client=client, StockBarsRequest=StockBarsRequest, StockLatestTradeRequest=StockLatestTradeRequest,
        TimeFrame=TimeFrame, TimeFrameUnit=TimeFrameUnit, Sort=Sort, completed_bars=completed_bars,
        latest_signal=latest_signal, trade_score=trade_score, rank_entry_candidates=rank_entry_candidates,
        long_trade_plan=long_trade_plan, SELECTION_POLICY=SELECTION_POLICY,
        ENTRY_SCAN_FILE=output / "entry_scan.json", atomic_json=atomic,
        get_daily_state=lambda: dict(state),
        daily_loss_reached=lambda s: (equity <= float(s["start_equity"]) * (1-float(cfg["max_daily_loss_pct"])),
                                     equity, float(s["start_equity"])),
        get_portfolio_entry_usage=lambda: (set(), 0.), maintain_order_lifecycle=lambda: None,
        submit_paper_long=deny_submission, telegram_trading_enabled=False,
        news_lock=threading.Lock(), news_items=[], telegram_lock=threading.Lock(), telegram_signals=[])
    load_functions(release / "paper_trader_integrated.py", scope)
    report = scope["run_entry_scan"]()
    summary = validate_report(report, cfg["symbols"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    try:
        if args.release_dir.resolve() != BASE:
            raise ProbeError("Run the probe from the exact prepared release")
        from dotenv import dotenv_values
        from tools.check_oracle_rollout import paper_gets
        values = dotenv_values(BASE / ".paper-service.env", interpolate=False)
        pair = (values["ALPACA_API_KEY"], values["ALPACA_SECRET_KEY"])
        broker = paper_gets(pair)
        if broker["positions"] or broker["orders"]:
            raise ProbeError("Paper account is not flat for the scan probe")
        run_probe(BASE, args.output_dir, pair, broker["account"])
    except Exception as error:
        print("PROBE_STOPPED", type(error).__name__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
