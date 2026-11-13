"""Close one explicitly authorized paper position once, then restore its old service.

Run from a separate recovery worktree, never by importing a trading process.
The retired release, journal and backups remain intact. A claimed POST is never
retried, including after an exception, 404, interruption or a repeated launch.
"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid

SERVICE = "v52-paper-bot.service"
API = "https://paper-api.alpaca.markets/v2"
TERMINAL = {"filled", "canceled", "expired", "rejected"}
RUNTIME = ("paper_live_state.json", "opening_orders.json", "paper_order_entries.json",
           "paper_order_notifications.json", "news_live_history.json", "trades.csv")


class RecoveryError(ValueError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        if hasattr(os, "O_DIRECTORY"):
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def progress(output, phase, **fields):
    value = dict(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(), **fields)
    save(output / "status.json", value)
    print("RECOVERY", json.dumps(value), flush=True)


class Paper:
    def __init__(self, pair):
        import requests
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))
        self.session.headers.update({"APCA-API-KEY-ID": pair[0], "APCA-API-SECRET-KEY": pair[1]})

    def request(self, method, path, *, params=None, payload=None):
        import requests
        try:
            response = self.session.request(method, API + path, params=params, json=payload,
                timeout=(5, 15), allow_redirects=False)
        except requests.RequestException:
            raise RecoveryError("Paper request outcome unavailable; no automatic resend") from None
        if response.status_code not in {200, 201}:
            raise RecoveryError(f"Paper {method}: HTTP {response.status_code}; no automatic resend")
        return response.json()

    def account(self):
        return self.request("GET", "/account")

    def positions(self):
        return self.request("GET", "/positions")

    def orders(self):
        value = self.request("GET", "/orders", params=dict(status="open", nested="false", limit=500))
        if not isinstance(value, list) or len(value) >= 500:
            raise RecoveryError("Open-order response is invalid or may be truncated")
        return value

    def clock(self):
        return self.request("GET", "/clock")

    def parent(self, order_id):
        return self.request("GET", "/orders/" + str(uuid.UUID(order_id)), params=dict(nested="true"))

    def lookup(self, cid):
        return self.request("GET", "/orders:by_client_order_id", params=dict(client_order_id=cid))

    def submit(self, payload):
        return self.request("POST", "/orders", payload=payload)


def require_account(broker, account_id):
    account = broker.account()
    if (str(account.get("id")) != account_id or account.get("status") != "ACTIVE"
            or account.get("trading_blocked") is not False):
        raise RecoveryError("Expected active paper account was not verified")


def require_position(positions, symbol, quantity):
    if (len(positions) != 1 or positions[0].get("symbol") != symbol
            or positions[0].get("side") != "long"
            or Decimal(str(positions[0]["qty"])) != quantity):
        raise RecoveryError("Current position differs from the one authorized for closing")


def validate_close(order, intent):
    if (order.get("client_order_id") != intent["client_order_id"]
            or order.get("symbol") != intent["symbol"] or order.get("side") != "sell"
            or order.get("type", order.get("order_type")) != "market"
            or order.get("order_class") not in {None, "", "simple"}
            or Decimal(str(order.get("qty", 0))) != Decimal(intent["qty"])
            or Decimal(str(order.get("filled_qty", 0))) > Decimal(intent["qty"])):
        raise RecoveryError("Close-order identity/quantity mismatch")


def close_once(broker, intent_path, account_id, entry, symbol, quantity, *, attempts=20, sleeper=time.sleep):
    """The caller has stopped and excluded all trading writers before this call."""
    quantity = Decimal(str(quantity))
    if quantity <= 0 or quantity != quantity.to_integral_value():
        raise RecoveryError("An exact positive whole-share quantity is required")
    binding = dict(account_id=account_id, entry_client_id=entry["client_id"],
                   symbol=symbol, qty=str(quantity))
    cid = "legacy-close-" + hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()[:30]
    expected = dict(binding, client_order_id=cid)
    if intent_path.exists():
        intent = read(intent_path)
        if any(intent.get(key) != value for key, value in expected.items()):
            raise RecoveryError("Existing close intent belongs to another position/account")
        require_account(broker, account_id)
        # A missing/unknown order is never permission to POST again.
        order = broker.lookup(cid)
    else:
        require_account(broker, account_id)
        positions, orders = broker.positions(), broker.orders()
        parent = broker.parent(entry["broker_id"])
        if (parent.get("client_order_id") != entry["client_id"]
                or parent.get("symbol") != symbol or parent.get("side") != "buy"
                or parent.get("status") not in TERMINAL
                or Decimal(str(parent.get("filled_qty", 0))) != quantity):
            raise RecoveryError("Original BUY is not terminal with the authorized filled quantity")
        if not positions and not orders:
            return dict(already_flat=True, sold_by_recovery=False)
        require_position(positions, symbol, quantity)
        if orders:
            raise RecoveryError("Open orders require reconciliation before the authorized close")
        legs = parent.get("legs") or []
        if (len(legs) != 2 or any(leg.get("symbol") != symbol or leg.get("side") != "sell"
                or leg.get("status") not in TERMINAL or Decimal(str(leg.get("filled_qty", 0))) != 0
                for leg in legs)):
            raise RecoveryError("Original bracket exits are not confirmed terminal and unfilled")
        clock = broker.clock()
        timestamp = datetime.fromisoformat(str(clock.get("timestamp")))
        if (clock.get("is_open") is not True or timestamp.tzinfo is None
                or abs((datetime.now(timezone.utc) - timestamp).total_seconds()) > 60):
            raise RecoveryError("Market is closed; the recovery will not queue an overnight SELL")
        # Repeat immediately before the durable claim; no orders are canceled.
        require_position(broker.positions(), symbol, quantity)
        if broker.orders():
            raise RecoveryError("Open-order state changed before closing")
        intent = dict(expected, claimed=True)
        save(intent_path, intent)  # durable BEFORE the only possible POST
        payload = dict(symbol=symbol, qty=str(quantity), side="sell", type="market",
                       order_class="simple", time_in_force="day", client_order_id=cid,
                       extended_hours=False)
        try:
            order = broker.submit(payload)
        except Exception:
            save(intent_path, dict(intent, outcome="unknown_do_not_resend"))
            raise
    for attempt in range(attempts):
        validate_close(order, intent)
        save(intent_path, dict(intent, last_order=order))
        require_account(broker, account_id)
        positions, orders = broker.positions(), broker.orders()
        if (order.get("status") == "filled" and Decimal(str(order["filled_qty"])) == quantity
                and not positions and not orders):
            result = dict(sold_by_recovery=True, symbol=symbol, filled_qty=str(quantity),
                          filled_avg_price=order.get("filled_avg_price"), order_id=order.get("id"))
            save(intent_path, dict(intent, complete=True, result=result))
            return result
        if order.get("status") in TERMINAL and order.get("status") != "filled":
            raise RecoveryError("Close order ended without a confirmed full close; no resend")
        if attempt + 1 < attempts:
            sleeper(2)
            order = broker.lookup(cid)
    raise RecoveryError("Close is not yet confirmed flat; inspect the existing order, never resend")


def merge_daily(current, old, symbols, max_trades):
    if (not isinstance(current.get("date"), str) or not isinstance(current.get("trades"), int)
            or current["trades"] < len(symbols) or current["trades"] < 0
            or Decimal(str(current.get("start_equity", 0))) <= 0):
        raise RecoveryError("Current daily risk state is invalid")
    result = dict(current)
    used = set(current.get("traded_symbols", [])) | set(symbols)
    if old.get("date") == current["date"]:
        if Decimal(str(old.get("start_equity", 0))) != Decimal(str(current["start_equity"])):
            raise RecoveryError("Same-day equity baselines disagree")
        result["trades"] = max(current["trades"], int(old["trades"]))
        used |= set(old.get("traded_symbols", []))
        result["opening_scan_done"] = bool(current.get("opening_scan_done") or old.get("opening_scan_done"))
    elif old.get("date") and old["date"] > current["date"]:
        raise RecoveryError("Refusing to replace a newer daily state")
    result["trades"] = max(result["trades"], len(used))
    result["traded_symbols"] = sorted(used)
    # Already-consumed attempts can exceed a configured cap; retain them.
    if max_trades <= 0:
        raise RecoveryError("Invalid existing trade cap")
    return result


class Host:
    def __init__(self, output):
        self.output = output

    def run(self, *argv):
        result = subprocess.run(list(map(str, argv)), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
        if result.returncode:
            save(self.output / "host-error.json", dict(command=list(map(str, argv)),
                 code=result.returncode, stderr=result.stderr.decode(errors="replace")))
            raise RecoveryError("Host command failed; inspect private host-error.json")
        return result.stdout

    def info(self):
        raw = self.run("systemctl", "show", SERVICE, "-p", "ActiveState", "-p", "SubState",
                       "-p", "MainPID", "-p", "WorkingDirectory", "-p", "User", "--no-pager").decode()
        return dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)

    def sudo(self, *args):
        return self.run("sudo", "-n", *args)


def verify_old_files(source, release, backup):
    names = ("paper_trader_integrated.py", "features.py", "ml_models.py", "models/ensemble.pkl",
             "config.json", ".env.v5")
    with tarfile.open(backup / "original-tree.tar", "r:") as archive:
        for name in names:
            with archive.extractfile("original/" + name) as saved:
                archived = hashlib.sha256(saved.read()).digest()
            if hashlib.sha256((source / name).read_bytes()).digest() != archived:
                raise RecoveryError("Original code/model/configuration changed since its deployment backup")
    if (source / "config.json").read_bytes() != (release / "config.json").read_bytes():
        raise RecoveryError("Old and replacement configurations differ; inspect before restoring")


def capture_old_runtime(host, release, source, manifest):
    info = host.info()
    if info.get("WorkingDirectory") != str(release) or info.get("ActiveState") != "active":
        raise RecoveryError("The expected replacement service is not active")
    pid = int(info["MainPID"])
    proc = Path("/proc") / str(pid)
    if (pid <= 0 or proc.stat().st_uid != os.geteuid() or (proc / "cwd").resolve() != release
            or os.fsencode(str(release / "paper_trader_integrated.py")) not in (proc / "cmdline").read_bytes().split(b"\0")):
        raise RecoveryError("Service process identity is not the expected same-user writer")
    allowed = {"HOME", "PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "LD_PRELOAD", "VIRTUAL_ENV",
               "PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONNOUSERSITE", "PYTHONSAFEPATH",
               "PYTHONUTF8", "PYTHONPLATLIBDIR"}
    environment = {}
    for entry in (proc / "environ").read_bytes().split(b"\0"):
        name, _, value = entry.partition(b"=")
        if os.fsdecode(name) in allowed:
            environment[os.fsdecode(name)] = os.fsdecode(value)
    base = Path(manifest["base_interpreter"]).resolve()
    if base != (proc / "exe").resolve():
        raise RecoveryError("Original base interpreter differs from the running service")
    code = "import numpy,pandas,requests,websocket,dotenv,alpaca,pickle;pickle.load(open('models/ensemble.pkl','rb'))"
    result = subprocess.run([str(base), "-c", code], cwd=source, env=environment,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
    if result.returncode:
        raise RecoveryError("Original interpreter/model preflight failed; inspect the old runtime privately")


def assert_no_writer(source, release):
    targets = {os.fsencode(str(p / "paper_trader_integrated.py")) for p in (source, release)}
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != os.geteuid():
                continue
            args = (process / "cmdline").read_bytes().split(b"\0")
            if targets.intersection(args) or (any(a in args for a in (b"paper_trader_integrated.py", b"./paper_trader_integrated.py"))
                    and (process / "cwd").resolve() in {source, release}):
                raise RecoveryError("A trading writer remains after service stop")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


def owned_entry(release, symbol):
    with closing(sqlite3.connect((release / "paper_order_intents.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        metadata = dict(db.execute("SELECT key,value FROM metadata"))
        entries = [dict(r) for r in db.execute("SELECT * FROM intents WHERE role='entry' AND symbol=? ORDER BY created_at DESC LIMIT 1", (symbol,))]
        if not entries or metadata.get("schema_version") != "1" or not metadata.get("account_id"):
            raise RecoveryError("Expected owned entry and journal schema are absent")
        entry = entries[0]
        if db.execute("SELECT 1 FROM intents WHERE state != 'observed' LIMIT 1").fetchone():
            raise RecoveryError("Unresolved broker intents require reconciliation first")
        used = {r[0] for r in db.execute("SELECT DISTINCT symbol FROM intents WHERE role='entry' AND session=?", (entry["session"],))}
        return metadata["account_id"], entry, used


def start_original(host, override, source, backup, output, broker, account_id, close, *, sleeper=time.sleep):
    require_account(broker, account_id)
    if broker.positions() or broker.orders():
        raise RecoveryError("Account changed before original selection; service remains stopped")
    saved_override = output / "replacement-override.conf"
    if override.exists():
        if not saved_override.exists():
            saved_override.write_bytes(override.read_bytes()); os.chmod(saved_override, 0o600)
        host.sudo("rm", "--", override)
    elif not saved_override.exists():
        raise RecoveryError("Expected deployment override is absent without a saved recovery copy")
    host.sudo("systemctl", "daemon-reload")
    if host.run("systemctl", "cat", SERVICE, "--no-pager") != (backup / "original-unit.txt").read_bytes():
        raise RecoveryError("Selected original unit differs from its deployment backup; service remains stopped")
    info = host.info()
    if info.get("WorkingDirectory") != str(source) or info.get("MainPID") != "0":
        raise RecoveryError("Systemd did not select the stopped original application")
    require_account(broker, account_id)
    if broker.positions() or broker.orders():
        raise RecoveryError("Account changed before original startup; service remains stopped")
    save(output / "old-start-claimed.json", dict(source=str(source), close=close))
    host.sudo("systemctl", "start", SERVICE)
    previous = None
    for _ in range(5):
        sleeper(1)
        info = host.info()
        if (info.get("ActiveState") != "active" or info.get("WorkingDirectory") != str(source)
                or int(info.get("MainPID", "0")) <= 0 or previous is not None and previous != info["MainPID"]):
            raise RecoveryError("Original startup was not stable; inspect retained state, do not repeat the close")
        previous = info["MainPID"]
    return info


def execute(release, backup, output, symbol, quantity):
    from dotenv import dotenv_values
    from io import StringIO
    release, backup, output = (Path(p).resolve() for p in (release, backup, output))
    manifest = read(backup / "manifest.json")
    source = Path(manifest["source"]).resolve()
    override = Path(manifest["override"])
    if (manifest["release"] != str(release) or source == release
            or override.parent != Path("/etc/systemd/system/v52-paper-bot.service.d")
            or override.name != "zz-oracle-top5-" + release.name + ".conf"
            or output.parent != backup):
        raise RecoveryError("Recovery paths do not match the existing deployment manifest")
    host = Host(output)
    info = host.info()
    if (output / "restored.json").exists() and info.get("WorkingDirectory") == str(source) and info.get("ActiveState") == "active":
        progress(output, "already_restored", service=info)
        return
    verify_old_files(source, release, backup)
    env = dotenv_values(release / ".paper-service.env", interpolate=False)
    old_env = dotenv_values(source / ".env.v5", interpolate=False)
    names = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")
    if not all(env.get(key) and old_env.get(key) == env[key] for key in names):
        raise RecoveryError("Original dotenv credential pair differs from the verified paper release")
    root_env = dotenv_values(stream=StringIO(host.sudo("cat", "/etc/v52-paper-bot.env").decode()), interpolate=False)
    if any(key in root_env and root_env[key] != env[key] for key in names):
        raise RecoveryError("Original service environment credential override differs from the verified account")
    broker = Paper((env["ALPACA_API_KEY"], env["ALPACA_SECRET_KEY"]))
    if (output / "writer-stopped.json").exists():
        if info.get("ActiveState") != "inactive" or info.get("MainPID") != "0":
            raise RecoveryError("A previous recovery stopped its writer; inspect any newly started service before resuming")
    else:
        capture_old_runtime(host, release, source, manifest)
        host.sudo("true")
        host.sudo("systemctl", "stop", SERVICE)
        info = host.info()
        if info.get("ActiveState") != "inactive" or info.get("MainPID") != "0":
            raise RecoveryError("Replacement writer did not stop cleanly")
        assert_no_writer(source, release)
        save(output / "writer-stopped.json", dict(release=str(release), source=str(source)))
        progress(output, "writer_stopped")
    assert_no_writer(source, release)
    # Read the final journal only after stopping its writer. No attempt arriving
    # between preflight and stop may disappear from recovery or the daily cap.
    account_id, entry, used = owned_entry(release, symbol)
    require_account(broker, account_id)
    # Back up both runtime sets and SQLite consistently before any broker write.
    if not (output / "runtime-backup-complete.json").exists():
        tag = uuid.uuid4().hex[:12]
        archive_path = output / ("runtime-before-recovery-" + tag + ".tar")
        journal_path = output / ("journal-before-recovery-" + tag + ".sqlite3")
        with archive_path.open("xb") as target:
            os.chmod(archive_path, 0o600)
            with tarfile.open(fileobj=target, mode="w") as archive:
                for label, root in (("release", release), ("original", source)):
                    for name in RUNTIME:
                        if (root / name).exists():
                            archive.add(root / name, arcname=label + "/" + name)
        with closing(sqlite3.connect((release / "paper_order_intents.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
            with closing(sqlite3.connect(journal_path)) as destination:
                os.chmod(journal_path, 0o600)
                db.backup(destination)
        save(output / "runtime-backup-complete.json", dict(complete=True,
             runtime_archive=str(archive_path), journal=str(journal_path)))
    close = close_once(broker, output / "close-intent.json", account_id, entry, symbol, quantity)
    progress(output, "close_confirmed", close=close)
    require_account(broker, account_id)
    if broker.positions() or broker.orders():
        raise RecoveryError("Account is not flat with no open orders; original service remains stopped")
    current = read(release / "paper_live_state.json")
    if current["date"] != entry["session"]:
        raise RecoveryError("Trading day changed; inspect daily state before restoration")
    for root in (source, release):
        queue = read(root / "opening_orders.json") if (root / "opening_orders.json").exists() else {}
        if queue:
            raise RecoveryError("Opening queue requires reconciliation before restoring the old writer")
    old = read(source / "paper_live_state.json")
    state = merge_daily(current, old, used, int(read(source / "config.json")["max_trades_per_day"]))
    save(output / "restored-daily-state.json", state)
    for name in RUNTIME:
        if name == "paper_live_state.json":
            save(source / name, state)
        elif (release / name).exists():
            # JSON is validated before any replacement. CSV is retained verbatim.
            if name.endswith(".json"):
                read(release / name)
            descriptor, temporary = tempfile.mkstemp(dir=source, prefix=name + ".")
            os.close(descriptor)
            try:
                shutil.copyfile(release / name, temporary)
                os.chmod(temporary, 0o600)
                os.replace(temporary, source / name)
            finally:
                Path(temporary).unlink(missing_ok=True)
    verify_old_files(source, release, backup)
    assert_no_writer(source, release)
    info = start_original(host, override, source, backup, output, broker, account_id, close)
    result = dict(service=info, close=close, retained_daily_trades=state["trades"],
                  retired_release=str(release), audit_directory=str(output))
    save(output / "restored.json", result)
    progress(output, "legacy_restored", **result)


def main():
    import fcntl
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--quantity", type=int, required=True)
    parser.add_argument("--execute-authorized-close-and-restore", action="store_true", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    with (args.output_dir / "recovery.lock").open("ab") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("RECOVERY_ALREADY_RUNNING", flush=True)
            return 2
        try:
            progress(args.output_dir, "starting_recovery", symbol=args.symbol, authorized_quantity=args.quantity)
            execute(args.release_dir, args.backup_dir, args.output_dir, args.symbol, args.quantity)
        except Exception as error:
            detail = str(error) if isinstance(error, RecoveryError) else "Inspect private recovery inputs; no blind retry"
            progress(args.output_dir, "needs_inspection", error_type=type(error).__name__, detail=detail)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
