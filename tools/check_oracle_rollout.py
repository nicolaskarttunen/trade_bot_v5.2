"""Inspect a completed Oracle candidate and existing paper state; never deploy.

No trading-process import, order SDK/client, broker writes or local state writes.
Paper API GETs use server-side credentials and fixed endpoints only.
"""
import argparse
import ast
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import sqlite3
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))


class CheckError(ValueError):
    """Only messages authored here may be printed; never HTTP bodies/secrets."""


def read(path):
    return json.loads(path.read_text())


def verify_candidate(source, output):
    # Import only offline modules. Do not construct the trading process/client.
    from tools import prepare_oracle_candidate as preparation
    from features import FEATURES
    from model_pipeline import predict_bundle, validate_bundle
    import pandas as pd

    cfg, report = read(source / "config.json"), read(output / "status.json")
    expected = preparation.protocol(cfg)
    if report.get("status") != "candidate_ready_not_deployed":
        raise CheckError("Candidate preparation is not complete")
    if report.get("protocol") != expected:
        raise CheckError("Candidate source, configuration or environment changed")
    if set(report.get("data", {})) != set(expected["symbols"]):
        raise CheckError("Candidate historical universe is incomplete")
    if report.get("prediction_checks") != len(cfg["symbols"]):
        raise CheckError("Candidate prediction record is incomplete")
    for symbol, record in report["data"].items():
        if preparation.digest(output / "data" / f"{symbol}_5min.csv") != record["sha256"]:
            raise CheckError("Saved candidate data changed")
    store = report["training_store"]
    matrix = (output / store["directory"] / "training.bin").resolve()
    if not matrix.is_relative_to(output) or preparation.digest(matrix) != store["sha256"]:
        raise CheckError("Saved candidate training matrix changed")
    candidate = output / "candidate.pkl"
    if preparation.digest(candidate) != report.get("candidate_sha256"):
        raise CheckError("Candidate model hash changed")
    # Only load the known local candidate after its recorded hash was verified.
    with candidate.open("rb") as handle:
        bundle = pickle.load(handle)
    mode = validate_bundle(bundle, cfg, live=True)
    predictions = predict_bundle(bundle, pd.DataFrame(report["prediction_anchors"], columns=FEATURES), cfg)
    if len(predictions) != len(cfg["symbols"]):
        raise CheckError("Candidate prediction count changed")
    return cfg, dict(verified=True, stocks=len(cfg["symbols"]),
                     saved_datasets=len(report["data"]), predictions_checked=len(predictions),
                     news_mode=mode, prepared_at=report["updated_at"],
                     candidate_sha256=report["candidate_sha256"])


def paper_gets(pair):
    import requests
    with requests.Session() as session:
        session.trust_env = False
        session.headers.update({"APCA-API-KEY-ID": pair[0], "APCA-API-SECRET-KEY": pair[1]})
        result = {}
        for name, suffix in (("account", "account"), ("positions", "positions"),
                             ("orders", "orders?status=open&nested=true&limit=500")):
            try:
                response = session.get("https://paper-api.alpaca.markets/v2/" + suffix,
                                       timeout=(5, 15), allow_redirects=False)
                if response.status_code != 200:
                    raise CheckError(f"Paper GET {name}: HTTP {response.status_code}")
                result[name] = response.json()
            except requests.RequestException:
                raise CheckError(f"Paper GET {name} failed; response details withheld") from None
    return result


def journal_summary(path, account_id):
    if not path.exists():
        return dict(present=False, intents=0, unresolved=0, account_matches=None), set()
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)) as db:
        db.execute("PRAGMA query_only=ON")
        metadata = dict(db.execute("SELECT key,value FROM metadata"))
        if metadata.get("schema_version") != "1":
            raise CheckError("Existing journal schema is unsupported")
        rows = db.execute("SELECT broker_id,state FROM intents").fetchall()
    return dict(present=True, intents=len(rows), unresolved=sum(s != "observed" for _, s in rows),
                account_matches=metadata.get("account_id") == str(account_id)), {b for b, _ in rows if b}


def account_state(source, broker):
    account, positions, orders = (broker[name] for name in ("account", "positions", "orders"))
    if (not isinstance(account, dict) or not account.get("id")
            or not isinstance(positions, list) or not isinstance(orders, list)
            or any(not isinstance(x, dict) for x in positions + orders)):
        raise CheckError("Unexpected paper account response shape")
    if len(orders) >= 500:
        raise CheckError("Paper open-order response may be truncated")
    journal, known = journal_summary(source / "paper_order_intents.sqlite3", account["id"])
    queue_path = source / "opening_orders.json"
    queue = read(queue_path) if queue_path.exists() else {}
    if not isinstance(queue, dict) or any(not isinstance(x, dict) for x in queue.values()):
        raise CheckError("Existing opening queue is invalid")
    legacy = sum(x.get("status") == "submitted" and x.get("order_id") not in known for x in queue.values())
    state_path = source / "paper_live_state.json"
    state = read(state_path) if state_path.exists() else None
    if state is not None and not isinstance(state, dict):
        raise CheckError("Existing daily state is invalid")
    findings = []
    if not journal["intents"] and (positions or orders):
        findings.append("Fresh journal with broker positions/open orders: reconcile before rollout")
    if journal["account_matches"] is False:
        findings.append("Existing journal belongs to a different account")
    if journal["unresolved"]:
        findings.append("Existing unresolved order intents require reconciliation")
    if legacy:
        findings.append("Legacy submitted queue records require reconciliation")
    if journal["intents"] and (not state or state.get("date") is None):
        findings.append("Daily baseline is absent despite existing journal history")
    if account.get("status") != "ACTIVE" or account.get("trading_blocked") is not False:
        findings.append("Paper account is not confirmed active and unblocked")
    return dict(account_status=account.get("status"), positions=len(positions), open_parent_orders=len(orders),
                position_symbols=[x.get("symbol") for x in positions],
                open_order_symbols=[x.get("symbol") for x in orders],
                journal=journal, opening_queue_records=len(queue), legacy_submitted_records=legacy,
                opening_queue_status_counts=dict(Counter(str(x.get("status", "missing")) for x in queue.values())),
                daily_state={k: state.get(k) for k in ("date", "trades", "opening_scan_done")} if state else None,
                reconciliation_findings=findings)


def service_info():
    properties = ("ActiveState", "SubState", "MainPID", "User", "WorkingDirectory")
    result = subprocess.check_output(["systemctl", "show", "v52-paper-bot.service",
                  *(arg for key in properties for arg in ("-p", key)), "--no-pager"],
                  text=True, stderr=subprocess.DEVNULL)
    return dict(line.split("=", 1) for line in result.splitlines() if "=" in line)


def service_startup(source, service, proc_root=Path("/proc")):
    """Verify process identity before reading selected initial environment keys."""
    if service.get("ActiveState") != "active":
        raise CheckError("Existing paper service is not active")
    try:
        pid = int(service["MainPID"])
        if pid <= 0:
            raise CheckError("Existing paper service has no running PID")
        process = proc_root / str(pid)
        if process.stat().st_uid != os.geteuid():
            raise CheckError("Existing service is not owned by the current user")
        arguments = (process / "cmdline").read_bytes().split(b"\0")
        if os.fsencode(str(source / "paper_trader_integrated.py")) not in arguments:
            raise CheckError("Existing service does not run the expected source file")
        if (process / "cwd").resolve() != source:
            raise CheckError("Existing service working directory differs from the source")
        values = {}
        names = {b"ALPACA_API_KEY", b"ALPACA_SECRET_KEY", b"PYTHON_DOTENV_DISABLED"}
        for entry in (process / "environ").read_bytes().split(b"\0"):
            name, _, value = entry.partition(b"=")
            if name in names:
                values[os.fsdecode(name)] = os.fsdecode(value)
        # Field 22 is index 19 after pid and the parenthesized process name.
        ticks = int((process / "stat").read_text().rsplit(")", 1)[1].split()[19])
        boot = next(int(line.split()[1]) for line in (proc_root / "stat").read_text().splitlines()
                    if line.startswith("btime "))
        started_at = boot + ticks / os.sysconf("SC_CLK_TCK")
        return values, started_at
    except CheckError:
        raise
    except (OSError, KeyError, ValueError, IndexError, StopIteration):
        raise CheckError("Cannot inspect the existing same-user service startup on this host") from None


def running_service_pair(source, service):
    values, started_at = service_startup(source, service)
    names = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")
    key, secret = (values.get(name) for name in names)
    if key and secret:
        return (key, secret), "verified_service_initial_environment"
    if any(name in values for name in names):
        raise CheckError("Partial/empty service credential pair; inspect locally rather than mixing sources")
    if str(values.get("PYTHON_DOTENV_DISABLED", "")).lower() in {"1", "true", "t", "yes", "y"}:
        raise CheckError("Service dotenv loading was disabled at startup")
    script, env_file = source / "paper_trader_integrated.py", source / ".env.v5"
    try:
        # The process's /proc environ need not include keys loaded by dotenv later.
        # Reconstruct only this explicitly supported startup path, not arbitrary files.
        for path in (script, env_file):
            info = path.stat()
            if max(info.st_mtime, info.st_ctime) > started_at:
                raise CheckError("Service source or .env.v5 changed since startup; account binding needs local confirmation")
        tree = ast.parse(script.read_text())
        loads = [node.value for node in tree.body if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                 and node.value.func.id == "load_dotenv"]
        if (len(loads) != 1 or loads[0].keywords or len(loads[0].args) != 1
                or not isinstance(loads[0].args[0], ast.Constant) or loads[0].args[0].value != ".env.v5"):
            raise CheckError("Service dotenv startup is not the supported literal .env.v5 load")
        from dotenv import dotenv_values
        settings = dotenv_values(env_file, interpolate=False)
        pair = tuple(settings.get(name) for name in names)
        if not all(pair) or any("$" in value for value in pair):
            raise CheckError("Service .env.v5 must contain one complete literal credential pair")
        # Detect concurrent edits while reading; never silently use a changed pair.
        if any(max(path.stat().st_mtime, path.stat().st_ctime) > started_at for path in (script, env_file)):
            raise CheckError("Service startup inputs changed during inspection")
        return pair, "verified_service_startup_dotenv_file"
    except (OSError, SyntaxError):
        raise CheckError("Cannot read the service's startup .env.v5 safely on this host") from None


def inspect(source, output, *, get_broker=None):
    import fcntl  # Oracle/Linux only; do not create or change the existing lock.
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise CheckError("Use the separate existing candidate output")
    with (output / "run.lock").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CheckError("Candidate preparation still holds its lock; retry after it exits") from None
        cfg, candidate = verify_candidate(source, output)
        service = service_info()
        credential_source = "synthetic_test_broker"
        if get_broker is None:
            pair, credential_source = running_service_pair(source, service)
            broker = paper_gets(pair)
        else:
            broker = get_broker()
        state = account_state(source, broker)
        if service != service_info():
            raise CheckError("Existing service changed during inspection; repeat the check")
        return dict(checked_at=datetime.now(timezone.utc).isoformat(), candidate=candidate,
                    service=service, credential_source=credential_source, paper_state=state,
                    configured_risk={k: cfg[k] for k in ("risk_per_trade_pct", "max_position_pct",
                        "max_total_exposure_pct", "max_open_positions", "max_daily_loss_pct", "max_trades_per_day")},
                    assessment="inspection_only_repeat_at_cutover", deployment_performed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(inspect(args.source_dir, args.candidate_dir), indent=2, allow_nan=False))
    except Exception as error:
        detail = str(error) if isinstance(error, CheckError) else "Check local candidate/state inputs privately"
        print(f"CHECK_STOPPED {type(error).__name__}: {detail}; no deployment performed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
