"""Fixed August 2026 development replay. Offline, no credentials or old models.

Default: validate inputs and report coverage/size. --run additionally trains
the shared ensemble and replays both fixed SPY arms. Never repairs missing bars.
"""
import argparse
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import shutil
import sys
from unittest.mock import patch

# Bound worker stacks and avoid BLAS thread startup per small disk block.
# These variables affect only this CLI process and its children.
if "--low-memory" in sys.argv:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import v6_5_intraday_replay as reference
from features import FEATURES, enrich, label, normalize_bars
from model_pipeline import fit_bundle, predict_bundle, trade_score, training_rows
from intraday_replay import replay_intraday, as_day_clock
from research_validation import input_manifest, build_spy_gate, paired_daily_comparison
from disk_training import TrainingStore, fit_disk_bundle, RECORD, BLOCK_ROWS

WINDOW = ("2023-01-01", "2026-07-31", "2026-08-01", "2026-08-31")
CALENDAR = BASE / "research_calendars/nyse_2026_08.csv"
NY = "America/New_York"
TRAIN_START = pd.Timestamp(WINDOW[0], tz=NY)
CUTOFF = pd.Timestamp("2026-08-01", tz=NY)
END = pd.Timestamp("2026-09-01", tz=NY)
OHLCV = ["timestamp", "open", "high", "low", "close", "volume"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sessions():
    sessions = pd.read_csv(CALENDAR, dtype=str)
    expected = pd.bdate_range("2026-08-03", "2026-08-31").strftime("%Y-%m-%d").tolist()
    # August 2026 has no scheduled NYSE holidays or early closes. This is a
    # fixed verified month, NOT a general weekday-based exchange calendar.
    if sessions.date.tolist() != expected or not sessions.close.eq("16:00").all():
        raise ValueError("Fixed August calendar does not match the reviewed schedule")
    return dict(zip(sessions.date, sessions.close))


def read_source(path, expected_hash=None):
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected_hash is not None and sha != expected_hash:
        raise ValueError(f"Source changed during run: {Path(path).name}")
    frame = pd.read_csv(io.BytesIO(raw), usecols=OHLCV)
    ts = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
    if ts.isna().any() or not ts.is_monotonic_increasing or ts.duplicated().any():
        raise ValueError(f"Invalid/duplicate/unordered timestamps: {Path(path).name}")
    frame["timestamp"] = ts
    frame = normalize_bars(frame)
    # Do not let later history enter features, labels, training or inference.
    return frame.loc[(ts >= TRAIN_START) & (ts < END)].reset_index(drop=True), sha


def coverage(frame, sessions, symbol):
    test = as_day_clock(frame.loc[frame.timestamp >= CUTOFF])
    regular = test.loc[test.clock.between("09:30", "15:55")]
    actual_days = set(regular.date)
    if actual_days != set(sessions):
        raise ValueError(f"{symbol}: missing/unexpected August sessions: "
                         f"{sorted(actual_days ^ set(sessions))}")
    expected = pd.DatetimeIndex([ts for day in sessions for ts in pd.date_range(
        f"{day} 09:30", f"{day} 15:55", freq="5min", tz=NY)]).tz_convert("UTC")
    missing = expected.difference(pd.DatetimeIndex(regular.timestamp))
    return {"symbol": symbol, "regular_bars": len(regular),
            "expected_regular_bars": len(expected),
            "missing_bars": [t.isoformat() for t in missing],
            "raw_training_rows": int((frame.timestamp < CUTOFF).sum())}


def prepare(frame, market, cfg):
    x = enrich(frame, market, news=None,
               news_decay_minutes=cfg["news_decay_minutes"],
               news_lookback_minutes=cfg["news_lookback_minutes"])
    x = label(x, cfg["label_horizon_bars"], cfg["label_threshold_pct"])
    train = training_rows(x, TRAIN_START, CUTOFF)
    # Retain compact numeric training columns instead of all OHLC/metadata.
    train = train[["timestamp", *FEATURES, "target"]].copy()
    test = x.loc[(x.timestamp >= CUTOFF) & (x.timestamp < END)].copy()
    test = test.loc[np.isfinite(test[FEATURES + ["atr"]].to_numpy(dtype=float)).all(axis=1)]
    if train.empty or test.empty:
        raise ValueError("Every stock must have usable training and August features")
    return train, test


def replay_by_session(scored, prices, symbols, sectors, cfg, *, spy_opening_gate,
                      require_spy, session_closes):
    """Same engine, one flat-to-flat session at a time; carry actual cash."""
    cash = float(cfg["initial_cash"])
    trades, daily, curves = [], [], []
    for day in sorted(session_closes):
        start = pd.Timestamp(day, tz=NY)
        end = start + pd.DateOffset(days=1)
        def select(frames):
            return {s: x.loc[(x.timestamp >= start) & (x.timestamp < end)].copy()
                    for s, x in frames.items()}
        cash, t, d, c = replay_intraday(
            select(scored), select(prices), symbols, sectors,
            dict(cfg, initial_cash=cash), spy_opening_gate=spy_opening_gate,
            require_spy=require_spy, session_closes={day: session_closes[day]})
        if len(d) != 1 or str(d.iloc[0]["date"]) != day:
            raise ValueError(f"Missing replay session {day}")
        if not t.empty:
            trades.append(t)
        daily.append(d)
        if not c.empty:
            curves.append(c)
    def combine(frames):
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return cash, combine(trades), combine(daily), combine(curves)


def run(data_dir, output_dir, execute=False, low_memory=False):
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output_dir == data_dir or data_dir in output_dir.parents:
        raise ValueError("Output must be outside original data directory")
    if output_dir.exists():
        raise FileExistsError("Use a new output directory")
    symbols, bench = list(reference.SYMBOLS), reference.BENCH
    cfg = dict(reference.CFG, feed="sip", strict_price_coverage=True)
    paths = {s: data_dir / f"{s}_5min.csv" for s in symbols + [bench]}
    absent = [str(p) for p in paths.values() if not p.is_file()]
    if absent:
        raise FileNotFoundError(f"Missing {len(absent)} input files; first: {absent[:3]}")
    sessions = load_sessions()
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "checking_inputs", "classification": "august_development_only",
              "independent_evidence": False, "profitability_validated": False,
              "data_kind": "user_supplied_history_not_independently_verified",
              "feed_assumption": "sip", "full_feed_adjustment_provenance_verified": False,
              "corporate_actions_verified": False, "point_in_time_universe_verified": False,
              "network_blocked": True, "broker_access": False, "news_mode": "neutral",
              "OPG_included": False, "parameter_search": False, "strict_price_coverage": True,
              "windows": [WINDOW], "symbols": symbols, "benchmark": bench,
              "sessions": len(sessions), "coverage": [], "input_sha256": {},
              "low_memory": low_memory, "training_block_rows": BLOCK_ROWS if low_memory else None}
    report_path = output_dir / "run_report.json"
    write_json(report_path, report)
    try:
        with ExitStack() as stack:
            for obj, attr in [(socket.socket, "connect"), (socket.socket, "connect_ex"),
                              (socket, "create_connection")]:
                stack.enter_context(patch.object(obj, attr, side_effect=RuntimeError("Network forbidden")))
            for i, (symbol, path) in enumerate(paths.items(), 1):
                frame, sha = read_source(path)
                row = coverage(frame, sessions, symbol)
                report["input_sha256"][str(path)] = sha
                report["coverage"].append(row)
                print(f"Checked {i}/{len(paths)} {symbol}: {len(row['missing_bars'])} August gaps", flush=True)
                if symbol == bench and row["missing_bars"]:
                    raise ValueError("Benchmark must cover every scheduled August regular-hours bar")
            del frame
            report["raw_training_rows_upper_bound"] = sum(
                r["raw_training_rows"] for r in report["coverage"] if r["symbol"] in symbols)
            report["single_feature_matrix_gib_upper_bound"] = (
                report["raw_training_rows_upper_bound"] * len(FEATURES) * 8 / 1024**3)
            report["memory_note"] = "Full training needs several matrices plus copies; this is NOT peak RAM"
            manifest = input_manifest([], cfg, [WINDOW])
            manifest["input_sha256"] = dict(report["input_sha256"])
            manifest["input_sha256"][str(CALENDAR)] = digest(CALENDAR)
            manifest["source_sha256"]["tools/run_august_simulation.py"] = digest(Path(__file__))
            manifest["source_sha256"]["disk_training.py"] = digest(BASE / "disk_training.py")
            manifest.update(classification=report["classification"], feed_assumption="sip",
                            full_feed_adjustment_provenance_verified=False, status="inputs_checked",
                            low_memory=low_memory)
            write_json(output_dir / "manifest.json", manifest)
            write_json(report_path, report)
            print(f"Raw training rows (upper bound): {report['raw_training_rows_upper_bound']:,}", flush=True)
            print(f"One feature matrix upper bound: {report['single_feature_matrix_gib_upper_bound']:.2f} GiB; peak RAM is larger", flush=True)
            if execute:
                if low_memory:
                    # Conservative allocation allowance: shards + merged data +
                    # standardized cache, plus headroom. Never delete user inputs.
                    required = report["raw_training_rows_upper_bound"]*RECORD.itemsize*3 + 1024**3
                    available = shutil.disk_usage(output_dir).free
                    report["required_free_disk_bytes"] = required
                    if available < required:
                        raise ValueError(f"Low-memory run needs {required/1024**3:.1f} GiB free disk; available {available/1024**3:.1f} GiB")
                    print("Low-memory mode: disk training, full-batch updates, daily replay. No rows or models removed.", flush=True)
                report["status"] = "preparing_features"
                write_json(report_path, report)
                market, _ = read_source(paths[bench], report["input_sha256"][str(paths[bench])])
                gate = build_spy_gate(market)
                train_parts, test_frames, prices = [], {}, {}
                store = TrainingStore(output_dir / "training_cache") if low_memory else None
                for i, symbol in enumerate(symbols, 1):
                    raw, _ = read_source(paths[symbol], report["input_sha256"][str(paths[symbol])])
                    train_s, test_s = prepare(raw, market, cfg)
                    if low_memory:
                        store.append(train_s)
                    else:
                        train_parts.append(train_s)
                    test_frames[symbol] = test_s
                    prices[symbol] = raw.loc[raw.timestamp >= CUTOFF, OHLCV].copy()
                    print(f"Features {i}/{len(symbols)} {symbol}: train={len(train_s):,}, test={len(test_s):,}", flush=True)
                if low_memory:
                    print("Merging training shards in stable timestamp order on disk...", flush=True)
                    store.finish()
                    rows, last = store.rows, pd.Timestamp(store.last_timestamp, tz="UTC")
                else:
                    train = pd.concat(train_parts, ignore_index=True).sort_values("timestamp", kind="stable")
                    train_parts.clear()
                    rows, last = len(train), train.timestamp.max()
                del train_s, raw
                report.update(status="training", training_rows=rows,
                              training_last_bar=str(last))
                write_json(report_path, report)
                print("Training fixed ensemble; this can take a long time. No parameter search.", flush=True)
                if low_memory:
                    bundle = fit_disk_bundle(store, cfg, "neutral", metadata={"windows": [WINDOW]})
                    report["training_cache_sha256"] = digest(store.path)
                    report["training_cache_directory_removed"] = store.cleanup()
                else:
                    bundle = fit_bundle(train, cfg, "neutral", metadata={"windows": [WINDOW]})
                    del train
                scored = {}
                for symbol, test_s in test_frames.items():
                    x = test_s[["timestamp", "signal_available_at", "atr", "news_score"]].copy()
                    x["ml_prob"] = predict_bundle(bundle, test_s, cfg)
                    x["trade_score"] = trade_score(x.ml_prob, test_s.market_ret_3, x.news_score, cfg)
                    scored[symbol] = x
                test_frames.clear()
                daily_outputs, trade_outputs, curve_outputs, summary = [], [], [], []
                for strategy in reference.STRATEGIES:
                    report.update(status="replaying", active_arm=strategy["name"])
                    write_json(report_path, report)
                    print(f"Replaying {strategy['name']}", flush=True)
                    replay = replay_by_session if low_memory else replay_intraday
                    cash, trades, daily, curve = replay(
                        scored, prices, symbols, reference.SECTOR_BY_SYMBOL, cfg,
                        spy_opening_gate=gate, require_spy=strategy["spy_gate"], session_closes=sessions)
                    if set(daily.date) != set(sessions) or len(daily) != len(sessions):
                        raise ValueError("Replay did not preserve every session")
                    summary.append(dict(strategy=strategy["name"],
                                        **reference.metrics(cfg["initial_cash"], cash, trades, daily, curve)))
                    for frame, collection in [(trades, trade_outputs), (daily, daily_outputs), (curve, curve_outputs)]:
                        collection.append(frame.assign(strategy=strategy["name"]))
                # Do not publish partial-arm performance when either arm fails.
                if any(digest(p) != report["input_sha256"][str(p)] for p in paths.values()):
                    raise ValueError("Input changed during replay")
                for name, frames in [("trades", trade_outputs), ("daily", daily_outputs), ("equity", curve_outputs)]:
                    pd.concat(frames, ignore_index=True).to_csv(output_dir / f"{name}.csv", index=False)
                pd.DataFrame(summary).to_csv(output_dir / "summary.csv", index=False)
                comparison = paired_daily_comparison(daily_outputs[0], daily_outputs[1])
                comparison.update(classification="august_development_only", independent_evidence=False)
                comparison["gate_allowed_sessions"] = sum(bool(gate[d]) for d in sessions)
                write_json(output_dir / "spy_comparison.json", comparison)
                manifest["feature_contract"] = bundle["feature_contract"]
                report["status"] = "completed_development_comparison"
            else:
                report["status"] = "preflight_complete_no_backtest"
            report["source_hashes_unchanged_at_end"] = all(
                digest(p) == report["input_sha256"][str(p)] for p in paths.values())
            if not report["source_hashes_unchanged_at_end"]:
                raise ValueError("Input changed during validation")
            write_json(report_path, report)
            manifest["status"] = report["status"]
            manifest["outputs"] = {p.name: digest(p) for p in output_dir.iterdir()
                                   if p.is_file() and p.name != "manifest.json"}
            write_json(output_dir / "manifest.json", manifest)
    except BaseException as exc:
        report.update(status="failed_not_validated", error_type=type(exc).__name__, error=str(exc))
        write_json(report_path, report)
        if (output_dir / "manifest.json").is_file():
            manifest["status"] = "failed_not_validated"
            write_json(output_dir / "manifest.json", manifest)
        raise
    print("Status:", report["status"], flush=True)
    print("Results:", output_dir, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Train/replay after input checks; default only validates")
    parser.add_argument("--low-memory", action="store_true", help="Use disk-backed full-batch training and per-session replay")
    args = parser.parse_args()
    print("AUGUST 2026 DEVELOPMENT CHECK / REPLAY; no credentials, orders or production changes.", flush=True)
    try:
        run(args.data_dir, args.output_dir, args.run, args.low_memory)
    except Exception as exc:
        raise SystemExit(f"Stopped ({type(exc).__name__}): {exc}. No validated result.")


if __name__ == "__main__":
    main()
