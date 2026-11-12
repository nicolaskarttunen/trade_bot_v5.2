"""Download a complete universe and train a separate, low-memory paper candidate.

No trading client, service changes, deployed-model writes or performance claims.
Credentials are read on the server only. Downloads and the completed training
matrix are resumable; an interrupted model fit restarts from that matrix.
API pagination: https://docs.alpaca.markets/us/reference/stockbars
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

# Keep the existing paper service's CPU and memory available.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
import requests
from dotenv import dotenv_values

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from disk_training import TrainingStore, fit_disk_bundle, RECORD
from features import FEATURES, FEATURE_HISTORY_BARS, enrich, label, normalize_bars
from model_pipeline import feature_contract, predict_bundle, training_rows, validate_bundle

OHLCV = ["timestamp", "open", "high", "low", "close", "volume"]
ENDPOINT = "https://data.alpaca.markets/v2/stocks/bars"
SOURCE_NAMES = ["features.py", "news_features.py", "model_pipeline.py",
                "ml_models.py", "disk_training.py", "tools/prepare_oracle_candidate.py"]


class CandidateError(ValueError):
    """Actionable text constructed here, never an HTTP body or credential."""


def config_snapshot(value):
    if isinstance(value, dict):
        return {key: ("<redacted>" if any(word in key.lower().replace("-", "_")
                    for word in ("secret", "token", "password", "credential", "api_key", "apikey", "authorization"))
                    else config_snapshot(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [config_snapshot(item) for item in value]
    return value


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, writer, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb" if binary else "w") as target:
            writer(target)
            target.flush()
            os.fsync(target.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def json_write(path, value):
    atomic(path, lambda target: json.dump(value, target, indent=2, allow_nan=False))


def bounds(cfg):
    def local(value):
        result = pd.Timestamp(value)
        return result.tz_localize("America/New_York") if result.tzinfo is None else result
    start, end = local(cfg["history_start"]).tz_convert("UTC"), local(cfg["history_end"]).tz_convert("UTC")
    if start >= end or end > pd.Timestamp.now(tz="UTC"):
        raise CandidateError("History bounds must be ordered, completed timestamps")
    if int(cfg["timeframe_minutes"]) != 5 or cfg["feed"] not in {"iex", "sip"}:
        raise CandidateError("Only the configured five-minute IEX or SIP feed is supported")
    return start, end


def protocol(cfg):
    start, end = bounds(cfg)
    symbols = list(cfg["symbols"])
    if not symbols or len(symbols) != len(set(symbols)):
        raise CandidateError("Universe must be nonempty and unique")
    universe = list(dict.fromkeys(symbols + [cfg["benchmark_symbols"][0]]))
    if any(not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", s) for s in universe):
        raise CandidateError("Unsafe or unsupported symbol")
    return dict(schema=1, config=config_snapshot(cfg),
                config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                symbols=universe, start=str(start), end_exclusive=str(end),
                adjustment="raw", asof=end.tz_convert("America/New_York").date().isoformat(),
                news_mode="neutral", feature_contract=feature_contract(cfg, "neutral"),
                source_sha256={n: digest(BASE / n) for n in SOURCE_NAMES},
                python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                alpaca_py=version("alpaca-py"))


def service_credentials(source):
    """Read only the existing same-user bot's pair; never export its environment."""
    try:
        pid = int(subprocess.check_output(["systemctl", "show", "v52-paper-bot.service",
                  "-p", "MainPID", "--value"], text=True, stderr=subprocess.DEVNULL).strip())
        if pid <= 0:
            return None
        process = Path("/proc") / str(pid)
        if process.stat().st_uid != os.geteuid():
            return None
        arguments = (process / "cmdline").read_bytes().split(b"\0")
        if os.fsencode(str(source / "paper_trader_integrated.py")) not in arguments:
            return None
        values = {}
        for entry in (process / "environ").read_bytes().split(b"\0"):
            name, _, value = entry.partition(b"=")
            if name in {b"ALPACA_API_KEY", b"ALPACA_SECRET_KEY"}:
                values[os.fsdecode(name)] = os.fsdecode(value)
        key, secret = values.get("ALPACA_API_KEY"), values.get("ALPACA_SECRET_KEY")
        return (key, secret) if key and secret else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def credentials(source):
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    for path in (source / ".env.v5", source / ".env", Path("/etc/v52-paper-bot.env")):
        if key and secret:
            break
        try:
            values = dotenv_values(path, interpolate=False) if path.is_file() else {}
        except PermissionError:
            continue
        # Never combine half a credential pair from different files.
        if values.get("ALPACA_API_KEY") and values.get("ALPACA_SECRET_KEY"):
            key, secret = values["ALPACA_API_KEY"], values["ALPACA_SECRET_KEY"]
    if not key or not secret:
        existing = service_credentials(source)
        if existing:
            return existing
        raise CandidateError("Market-data credentials unavailable on server")
    return key, secret


@contextmanager
def exclusive(root):
    # Oracle/Linux only. Prevent two user launches from sharing a training cache.
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CandidateError("This candidate already has an active process") from None
        yield


def get_page(session, params, sleeper=time.sleep):
    for attempt in range(6):
        try:
            response = session.get(ENDPOINT, params=params, timeout=(10, 60), allow_redirects=False)
            if response.status_code in {429, 500, 502, 503, 504}:
                if attempt == 5:
                    raise CandidateError(f"Historical-data retry limit: HTTP {response.status_code}")
                sleeper(min(2 ** attempt, 30))
                continue
            if response.status_code != 200:
                raise CandidateError(f"Historical-data request refused: HTTP {response.status_code}")
            return response.json()
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 5:
                raise CandidateError("Historical-data connection retry limit") from None
            sleeper(min(2 ** attempt, 30))
    raise AssertionError("Unreachable retry state")


def download_symbol(session, symbol, cfg, start, end, asof, fetch=None):
    fetch = fetch or (lambda params: get_page(session, params))
    params = dict(symbols=symbol, timeframe="5Min", start=start.isoformat(),
                  end=(end - pd.Timedelta(microseconds=1)).isoformat(), limit=10000,
                  feed=cfg["feed"], adjustment="raw", sort="asc", asof=asof)
    records, seen = [], set()
    while True:
        body = fetch(dict(params))
        bars = body.get("bars", {}).get(symbol, [])
        records.extend({"timestamp": b["t"], "open": b["o"], "high": b["h"],
                        "low": b["l"], "close": b["c"], "volume": b["v"]} for b in bars)
        token = body.get("next_page_token")
        if not token:
            break
        if token in seen:
            raise CandidateError("Repeated historical-data page token")
        seen.add(token)
        params["page_token"] = token
    if not records:
        raise CandidateError(f"No historical bars for {symbol}")
    frame = normalize_bars(pd.DataFrame(records, columns=OHLCV))
    if (frame.timestamp < start).any() or (frame.timestamp >= end).any():
        raise CandidateError(f"Historical-data bounds mismatch: {symbol}")
    if len(frame) <= FEATURE_HISTORY_BARS + int(cfg["label_horizon_bars"]):
        raise CandidateError(f"Insufficient historical bars: {symbol}")
    return frame


def checked_frame(path, metadata):
    if digest(path) != metadata["sha256"]:
        raise CandidateError(f"Historical input changed: {path.name}")
    frame = normalize_bars(pd.read_csv(path, usecols=OHLCV))
    if len(frame) != metadata["rows"]:
        raise CandidateError(f"Historical input row count changed: {path.name}")
    return frame


def run(source, output, *, fetch=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise CandidateError("Candidate output must be outside the running bot directory")
    cfg = json.loads((source / "config.json").read_text())
    expected = protocol(cfg)
    start, end = bounds(cfg)
    with exclusive(output):
        status_path = output / "status.json"
        if status_path.exists():
            report = json.loads(status_path.read_text())
            if report["protocol"] != expected:
                raise CandidateError("Candidate configuration, source or environment changed; use a new output")
        else:
            if any(p.name != "run.lock" for p in output.iterdir()):
                raise CandidateError("Unrecognized output files; use a new output directory")
            report = dict(protocol=expected, status="starting", data={},
                          classification="paper_candidate_not_validated", profitability_validated=False,
                          deployment_performed=False, stock_universe=len(cfg["symbols"]))

        def progress(status, **details):
            report.update(status=status, updated_at=datetime.now(timezone.utc).isoformat(), **details)
            json_write(status_path, report)
            print(status, json.dumps(details, allow_nan=False), flush=True)

        try:
            progress("checking_downloads")
            pending = []
            for symbol in expected["symbols"]:
                path = output / "data" / f"{symbol}_5min.csv"
                if symbol in report["data"]:
                    if digest(path) != report["data"][symbol]["sha256"]:
                        raise CandidateError(f"Saved data changed: {symbol}")
                else:
                    pending.append(symbol)
            session = None
            if pending and fetch is None:
                key, secret = credentials(source)
                session = requests.Session()
                session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
            try:
                for symbol in pending:
                    progress("downloading", active_symbol=symbol, completed_downloads=len(report["data"]),
                             total_downloads=len(expected["symbols"]))
                    frame = download_symbol(session, symbol, cfg, start, end, expected["asof"], fetch)
                    path = output / "data" / f"{symbol}_5min.csv"
                    atomic(path, lambda target: frame.to_csv(target, index=False))
                    report["data"][symbol] = dict(sha256=digest(path), rows=len(frame),
                        first=str(frame.timestamp.iloc[0]), last=str(frame.timestamp.iloc[-1]))
                    progress("download_saved", completed_downloads=len(report["data"]),
                             total_downloads=len(expected["symbols"]))
                    del frame
            finally:
                if session is not None:
                    session.close()

            candidate = output / "candidate.pkl"
            if report.get("candidate_sha256"):
                if digest(candidate) != report["candidate_sha256"]:
                    raise CandidateError("Saved candidate model changed")
                with candidate.open("rb") as model_file:
                    validate_bundle(pickle.load(model_file), cfg, live=True)
                progress("candidate_ready_not_deployed", candidate_path=str(candidate))
                return report
            if candidate.exists():
                raise CandidateError("Unregistered candidate file; use a new output")

            if report.get("training_store"):
                saved = report["training_store"]
                cache = output / saved["directory"]
                path = cache / "training.bin"
                if digest(path) != saved["sha256"] or path.stat().st_size != saved["rows"] * RECORD.itemsize:
                    raise CandidateError("Saved training matrix changed")
                store = SimpleNamespace(root=cache, path=path, rows=saved["rows"])
            else:
                rows_upper = sum(report["data"][s]["rows"] for s in cfg["symbols"])
                required = rows_upper * RECORD.itemsize * 3 + 1024 ** 3
                if shutil.disk_usage(output).free < required:
                    raise CandidateError(f"Training needs at least {required / 1024**3:.1f} GiB free disk")
                cache = Path(tempfile.mkdtemp(prefix="training-", dir=output)) / "store"
                store = TrainingStore(cache)
                market_symbol = cfg["benchmark_symbols"][0]
                market = checked_frame(output / "data" / f"{market_symbol}_5min.csv", report["data"][market_symbol])
                anchors = []
                for index, symbol in enumerate(cfg["symbols"], 1):
                    raw = checked_frame(output / "data" / f"{symbol}_5min.csv", report["data"][symbol])
                    prepared = label(enrich(raw, market, None, cfg["news_decay_minutes"], cfg["news_lookback_minutes"]),
                                     cfg["label_horizon_bars"], cfg["label_threshold_pct"])
                    train = training_rows(prepared, start, end)
                    if train.empty:
                        raise CandidateError(f"No usable training rows: {symbol}")
                    store.append(train[["timestamp", *FEATURES, "target"]])
                    anchors.append(train.iloc[-1][FEATURES].to_numpy(dtype=float).tolist())
                    progress("preparing_features", active_symbol=symbol, completed_stocks=index,
                             total_stocks=len(cfg["symbols"]), training_rows=store.rows)
                    del raw, prepared, train
                del market
                progress("merging_training_matrix", training_rows=store.rows)
                store.finish()
                report["training_store"] = dict(directory=str(cache.relative_to(output)), rows=store.rows,
                                                sha256=digest(store.path))
                report["prediction_anchors"] = anchors
                progress("training_matrix_saved", training_rows=store.rows)

            metadata = dict(classification="paper_candidate_not_validated", config=config_snapshot(cfg),
                            train_start=str(start), train_end_exclusive=str(end),
                            data_sha256={s: d["sha256"] for s, d in report["data"].items()},
                            target=dict(horizon_bars=cfg["label_horizon_bars"],
                                        threshold_pct=cfg["label_threshold_pct"], expected_profit_model=False))
            progress("training", training_rows=store.rows)
            bundle = fit_disk_bundle(store, cfg, "neutral", metadata=metadata,
                                     progress=lambda message: progress("training", training_progress=message))
            validate_bundle(bundle, cfg, live=True)
            anchors = pd.DataFrame(report["prediction_anchors"], columns=FEATURES)
            predictions = predict_bundle(bundle, anchors, cfg)
            if len(predictions) != len(cfg["symbols"]) or digest(store.path) != report["training_store"]["sha256"]:
                raise CandidateError("Candidate prediction or training-integrity check failed")
            for symbol, data in report["data"].items():
                if digest(output / "data" / f"{symbol}_5min.csv") != data["sha256"]:
                    raise CandidateError(f"Data changed during training: {symbol}")
            atomic(candidate, lambda target: pickle.dump(bundle, target), binary=True)
            report.update(candidate_sha256=digest(candidate), prediction_checks=len(predictions))
            progress("candidate_ready_not_deployed", candidate_path=str(candidate))
            return report
        except BaseException as error:
            # Never include an HTTP body, headers, credentials or exception text.
            progress("failed", error_type=type(error).__name__,
                     error_detail=str(error) if isinstance(error, CandidateError) else "Model or data check failed")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        os.nice(10)
        run(args.source_dir, args.output_dir)
    except Exception as error:
        detail = str(error) if isinstance(error, CandidateError) else "Inspect status.json and candidate inputs"
        print(f"FAILED: {type(error).__name__}: {detail}; no deployment performed", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
