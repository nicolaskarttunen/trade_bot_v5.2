"""Deploy the verified, flat legacy Oracle paper account into a separate release.

Run as the existing service user. Uses existing sudo privileges only. It never
submits/cancels/closes broker orders, edits original files or retrains a model.
The new paper service can trade normally after a successful switch.
"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid

os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from tools import check_oracle_rollout as check
from tools.prepare_oracle_candidate import digest

SERVICE = "v52-paper-bot.service"
RUNTIME = (".env.v5", "paper_live_state.json", "opening_orders.json", "paper_order_entries.json",
           "paper_order_notifications.json", "news_live_history.json", "trades.csv")


class DeployError(ValueError):
    pass


def private_write(path, value):
    with path.open("x", encoding="utf-8") as target:
        os.chmod(path, 0o600)
        target.write(value)


def save_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.chmod(temp, 0o600)
    temp.replace(path)


def require_flat(source, snapshot, account_id=None):
    state = check.account_state(source, snapshot)
    if account_id is not None and snapshot["account"]["id"] != account_id:
        raise DeployError("Paper account identity changed")
    if (state["positions"] or state["open_parent_orders"] or state["reconciliation_findings"]
            or state["journal"]["present"] or state["opening_queue_records"]):
        raise DeployError("Legacy migration requires the verified empty queue, absent journal and flat paper account")
    if any((source / ("paper_order_intents.sqlite3" + suffix)).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise DeployError("Original journal sidecars require inspection before a fresh journal is initialized")
    return state


def require_closed_window(clock, now=None):
    import pandas as pd
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    stamp = pd.Timestamp(clock["timestamp"])
    next_open = pd.Timestamp(clock["next_open"])
    if (stamp.tzinfo is None or next_open.tzinfo is None or abs((now-stamp).total_seconds()) > 60
            or clock.get("is_open") is not False or (next_open-now).total_seconds() < 1800):
        raise DeployError("Switch requires a fresh closed-market clock and at least 30 minutes before the open")
    local = now.tz_convert("America/New_York")
    if local.hour == 9 and 15 <= local.minute < 30:
        raise DeployError("Switch is outside the opening-order window only")


def service_env(pair):
    if any(not re.fullmatch(r"[A-Za-z0-9_+=/.\-]+", value) for value in pair):
        raise DeployError("Credential encoding requires private host review")
    return f"ALPACA_API_KEY={pair[0]}\nALPACA_SECRET_KEY={pair[1]}\nALPACA_PAPER=true\n"


def override_text(release, interpreter, environment_file, base_interpreter):
    if any(not re.fullmatch(r"/[A-Za-z0-9_/ .\-]+", str(p)) or " " in str(p)
           for p in (release, interpreter, environment_file, base_interpreter)):
        raise DeployError("Use ordinary absolute deployment paths without spaces")
    return (f"[Service]\nWorkingDirectory={release}\nExecStart=\n"
            f"ExecStart=@{base_interpreter} {interpreter} -E -s -u {release}/paper_trader_integrated.py\n"
            f"EnvironmentFile={environment_file}\nStandardOutput=journal\nStandardError=journal\n")


def service_settings(unit_text):
    """Copy effective Service fragments, without Unit relationships or Install hooks."""
    lines = ["[Service]\n"]
    selected = False
    for line in unit_text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            selected = stripped == "[Service]"
        elif selected:
            lines.append(line)
    return "".join(lines) + "\n"


class Host:
    def __init__(self, override):
        self.override = override
        self.environment_file = Path("/etc") / ("v52-paper-" + override.stem + ".env")
        self.base_interpreter = Path(sys._base_executable).resolve()
        self.backup = None
        self.original_settings = None

    def command(self, argv, timeout=90):
        try:
            return subprocess.check_output(argv, stderr=subprocess.PIPE, timeout=timeout)
        except subprocess.CalledProcessError as error:
            if self.backup is not None:
                save_json(self.backup / "failed-host-command.json", dict(
                    command=list(map(str, argv)), returncode=error.returncode,
                    stderr=(error.stderr or b"").decode(errors="replace")))
            raise

    def sudo(self, *args):
        return self.command(["sudo", "-n", *map(str, args)])

    def info(self):
        return check.service_info()

    def prepare(self, backup, source, service):
        self.backup = backup
        self.sudo("true")
        if self.override.exists():
            raise DeployError("Deployment override path already exists")
        hooks = ["ExecStartPre", "ExecStartPost", "ExecCondition", "ExecStop", "ExecStopPost"]
        directories = ["RuntimeDirectory", "StateDirectory", "CacheDirectory", "LogsDirectory", "ConfigurationDirectory"]
        properties = ["Type", "KillMode", *hooks, *directories, "RootDirectory", "RootImage",
                      "ExecStart", "FragmentPath", "DropInPaths", "EnvironmentFiles"]
        raw = self.command(["systemctl", "show", SERVICE,
                           *(a for p in properties for a in ("-p", p)), "--no-pager"]).decode()
        values = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        if (values.get("Type") not in {"simple", "exec"} or values.get("KillMode") != "control-group"
                or any(values.get(p) for p in (*hooks, *directories, "RootDirectory", "RootImage"))):
            raise DeployError("Existing service hooks or isolation require a host-specific switch")
        original = self.command(["systemctl", "cat", SERVICE]).decode()
        private_write(backup / "original-unit.txt", original)
        self.original_settings = service_settings(original)
        original_python = verify_original_runtime(source, service, values)
        if original_python.resolve() != self.base_interpreter:
            raise DeployError("The staging venv is not based on the verified original Python binary")

    def restore_context(self, path):
        command = shutil.which("restorecon")
        if command is None and Path("/usr/sbin/restorecon").is_file():
            command = "/usr/sbin/restorecon"
        if command is not None:
            self.sudo(command, path)  # Default label on this one newly installed file only.

    def verify_runtime(self, release, interpreter):
        """Ask PID1 to exercise the exact launch before stopping the original writer."""
        if self.original_settings is None or self.backup is None:
            raise DeployError("Original service settings must be verified first")
        # A unique root-only file in /etc is readable by the service manager;
        # the original environment file and all host security policies stay intact.
        self.sudo("test", "!", "-e", self.environment_file)
        self.sudo("test", "!", "-L", self.environment_file)
        self.sudo("install", "-T", "-m", "0600", "-o", "root", "-g", "root",
                  release / ".paper-service.env", self.environment_file)
        self.restore_context(self.environment_file)
        mode = self.sudo("stat", "-c", "%u %g %a", self.environment_file).decode().strip()
        if mode != "0 0 600":
            raise DeployError("Dedicated system environment file is not root-owned mode 0600")
        probe_name = self.override.stem + "-runtime-check.service"
        probe_path = Path("/etc/systemd/system") / probe_name
        self.sudo("test", "!", "-e", probe_path)
        self.sudo("test", "!", "-L", probe_path)
        proof = release / ".service-runtime-proof.json"
        if proof.exists():
            raise DeployError("Runtime proof path already exists")
        unit = (self.original_settings + "\n[Service]\nType=oneshot\nRestart=no\nRemainAfterExit=no\n"
                "TimeoutStartSec=45\nExecStart=\nExecReload=\n"
                f"WorkingDirectory={release}\nEnvironmentFile={self.environment_file}\n"
                "StandardOutput=journal\nStandardError=journal\n"
                f"ExecStart=@{self.base_interpreter} {interpreter} -E -s -u "
                f"{release}/tools/check_oracle_service_runtime.py --release-dir {release} "
                f"--expected-uid {os.geteuid()} --base-interpreter {self.base_interpreter}\n")
        local_unit = self.backup / "runtime-check.service"
        private_write(local_unit, unit)
        installed = False
        loaded = False
        try:
            installed = True
            self.sudo("install", "-T", "-m", "0600", "-o", "root", "-g", "root", local_unit, probe_path)
            self.restore_context(probe_path)
            loaded = True
            self.sudo("systemctl", "daemon-reload")
            self.sudo("systemctl", "start", probe_name)
            state = self.command(["systemctl", "show", probe_name, "-p", "Result", "-p", "ExecMainStatus", "--no-pager"]).decode()
            values = dict(line.split("=", 1) for line in state.splitlines() if "=" in line)
            evidence = check.read(proof)
            if (values.get("Result") != "success" or values.get("ExecMainStatus") != "0"
                    or evidence.get("uid") != os.geteuid() or evidence.get("prefix") != str(release / ".venv")
                    or evidence.get("base_interpreter") != str(self.base_interpreter)
                    or evidence.get("model_sha256") != digest(release / "models/ensemble.pkl")
                    or evidence.get("config_sha256") != digest(release / "config.json")
                    or not all(evidence.get(k) is True for k in ("environment_matches", "paper", "write_test", "network_blocked"))):
                raise DeployError("Systemd did not confirm the isolated, private paper runtime")
            save_json(self.backup / "runtime-check.json", evidence)
            return evidence
        except Exception:
            try:
                log = self.sudo("journalctl", "-u", probe_name, "-n", "60", "--no-pager")
                private_write(self.backup / "runtime-check-log.txt", log.decode(errors="replace"))
            except Exception:
                pass
            raise DeployError(f"Systemd runtime preflight failed before stopping the original; inspect {self.backup}") from None
        finally:
            if installed:
                errors = []
                commands = [("rm", "-f", probe_path), ("systemctl", "daemon-reload")]
                if loaded:
                    commands.insert(0, ("systemctl", "stop", probe_name))
                for command in commands:
                    try:
                        self.sudo(*command)
                    except Exception:
                        errors.append(command[0])
                if errors:
                    raise DeployError("Runtime probe cleanup was not confirmed; original service has not been stopped")

    def stop(self):
        self.sudo("systemctl", "stop", SERVICE)
        info = self.info()
        if info.get("ActiveState") != "inactive" or info.get("MainPID") != "0":
            raise DeployError("Existing writer did not stop cleanly")

    def install(self, path):
        self.sudo("install", "-D", "-m", "0644", path, self.override)
        self.restore_context(self.override)
        self.sudo("systemctl", "daemon-reload")

    def verify_unit(self, release, interpreter):
        raw = self.command(["systemctl", "show", SERVICE, "-p", "ExecStart", "-p", "EnvironmentFiles",
                            "-p", "WorkingDirectory", "-p", "StandardOutput", "-p", "StandardError", "--no-pager"]).decode()
        values = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        expected = f"argv[]={interpreter} -E -s -u {release}/paper_trader_integrated.py ;"
        if (values.get("WorkingDirectory") != str(release) or expected not in values.get("ExecStart", "")
                or f"path={self.base_interpreter} ;" not in values.get("ExecStart", "")
                or values["ExecStart"].count("argv[]=") != 1
                or values.get("StandardOutput") != "journal" or values.get("StandardError") != "journal"
                or not values.get("EnvironmentFiles", "").endswith(f"{self.environment_file} (ignore_errors=no)")):
            raise DeployError("Systemd did not select the exact release, interpreter and final paper environment")

    def start(self):
        self.sudo("systemctl", "start", SERVICE)

    def remove_override(self):
        self.sudo("rm", "-f", self.override)
        self.sudo("systemctl", "daemon-reload")

    def healthy(self, release, account_id):
        previous = None
        for _ in range(10):
            time.sleep(1)
            info = self.info()
            if (info.get("ActiveState") != "active" or info.get("WorkingDirectory") != str(release)
                    or int(info.get("MainPID", "0")) <= 0):
                raise DeployError("Replacement service did not stay active")
            if previous is not None and info.get("MainPID") != previous:
                raise DeployError("Replacement service restarted during validation")
            previous = info.get("MainPID")
        summary, _ = check.journal_summary(release / "paper_order_intents.sqlite3", account_id)
        if not summary["present"] or summary["account_matches"] is not True or summary["intents"]:
            raise DeployError("Replacement journal startup was not confirmed empty and account-bound")
        check.service_startup(release, info)
        return info

    def restored(self, source):
        previous = None
        for _ in range(3):
            time.sleep(1)
            info = self.info()
            if (info.get("ActiveState") != "active" or info.get("WorkingDirectory") != str(source)
                    or int(info.get("MainPID", "0")) <= 0
                    or previous is not None and previous != info["MainPID"]):
                return False
            previous = info["MainPID"]
        return True


def verify_original_runtime(source, service, unit):
    # Reproduce only Python/import-related startup variables, never dump the
    # process environment or run/import the old trading process itself.
    _, started = check.service_startup(source, service)
    env_specs = unit.get("EnvironmentFiles", "")
    pattern = r"(/[^ ]+) \(ignore_errors=(?:yes|no)\)"
    if re.sub(pattern, "", env_specs).strip():
        raise DeployError("Original environment-file layout needs host-specific recovery validation")
    inputs = [unit.get("FragmentPath", ""), *unit.get("DropInPaths", "").split(), *re.findall(pattern, env_specs)]
    for name in filter(None, inputs):
        info = Path(name).stat()
        if max(info.st_mtime, info.st_ctime) > started:
            raise DeployError("Original systemd startup inputs changed since launch; verify recovery on host")
    command = unit.get("ExecStart", "")
    paths = re.findall(r"\bpath=([^ ;]+)", command)
    if (len(paths) != 1 or not Path(paths[0]).name.startswith("python")
            or f"argv[]={paths[0]} -u {source}/paper_trader_integrated.py ;" not in command):
        raise DeployError("Original service command needs host-specific recovery validation")
    allowed = {b"HOME", b"PATH", b"LANG", b"LC_ALL", b"LD_LIBRARY_PATH", b"LD_PRELOAD", b"VIRTUAL_ENV",
               b"PYTHONPATH", b"PYTHONHOME", b"PYTHONUSERBASE", b"PYTHONNOUSERSITE",
               b"PYTHONSAFEPATH", b"PYTHONUTF8", b"PYTHONPLATLIBDIR"}
    environment = {}
    for entry in (Path("/proc") / service["MainPID"] / "environ").read_bytes().split(b"\0"):
        name, _, value = entry.partition(b"=")
        if name in allowed:
            environment[os.fsdecode(name)] = os.fsdecode(value)
    code = "import numpy,pandas,requests,websocket,dotenv,alpaca,pickle;pickle.load(open('models/ensemble.pkl','rb'))"
    try:
        subprocess.run([paths[0], "-c", code], cwd=source, env=environment, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
    except (OSError, subprocess.SubprocessError):
        raise DeployError("Original interpreter/model recovery check failed; original service remains running") from None
    return Path(paths[0])


def assert_no_source_writer(source):
    expected = os.fsencode(str(source / "paper_trader_integrated.py"))
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid == os.geteuid():
                arguments = (process / "cmdline").read_bytes().split(b"\0")
                if (expected in arguments or
                        (any(a in arguments for a in (b"paper_trader_integrated.py", b"./paper_trader_integrated.py"))
                         and (process / "cwd").resolve() == source)):
                    raise DeployError("A source writer remains outside the stopped service")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue


def prepare_release(stage, source, candidate, release, interpreter, pair):
    release.mkdir(mode=0o700)
    # Archive a committed tree only; local candidate data/secrets are not copied from Git.
    with tempfile.TemporaryFile() as stream:
        subprocess.run(["git", "-C", str(stage), "archive", "--format=tar", "HEAD"], stdout=stream, check=True)
        stream.seek(0)
        with tarfile.open(fileobj=stream) as archive:
            archive.extractall(release, filter="data")
    for name in ("features.py", "news_features.py", "model_pipeline.py", "ml_models.py", "disk_training.py"):
        if digest(release / name) != digest(stage / name):
            raise DeployError("Release pipeline differs from the verified candidate environment")
    shutil.copy2(source / "config.json", release / "config.json")
    shutil.copy2(candidate / "candidate.pkl", release / "models/ensemble.pkl")
    os.chmod(release / "models/ensemble.pkl", 0o600)
    private_write(release / ".paper-service.env", service_env(pair))
    # Rebuild an independent venv on this same host, using the installed wheel
    # contents, not a relocated venv or a network dependency upgrade.
    runtime = release / ".venv"
    subprocess.run([str(interpreter), "-E", "-s", "-m", "venv", str(runtime)],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    query = "import sysconfig;print(sysconfig.get_path('purelib'))"
    old_lib = Path(subprocess.check_output([str(interpreter), "-E", "-s", "-c", query], text=True).strip())
    new_python = runtime / "bin/python"
    new_lib = Path(subprocess.check_output([str(new_python), "-E", "-s", "-c", query], text=True).strip())
    if not old_lib.is_relative_to(interpreter.parent.parent) or not new_lib.is_relative_to(runtime):
        raise DeployError("Python package paths are not inside their isolated environments")
    shutil.copytree(old_lib, new_lib, dirs_exist_ok=True, symlinks=True)
    # Every runtime module must resolve from the new environment. Repeat the
    # complete model contract and saved prediction checks with that interpreter.
    validation = """import importlib, pathlib, sys
release, source, candidate = map(pathlib.Path, sys.argv[1:])
prefix = release / '.venv'
assert pathlib.Path(sys.prefix).resolve() == prefix.resolve()
for name in ['numpy','pandas','requests','websocket','dotenv','alpaca']:
    module = importlib.import_module(name)
    assert pathlib.Path(module.__file__).resolve().is_relative_to(prefix.resolve())
sys.path.insert(0, str(release))
from tools.check_oracle_rollout import verify_candidate
from tools.prepare_oracle_candidate import digest
_, verified = verify_candidate(source, candidate)
assert digest(release / 'models/ensemble.pkl') == verified['candidate_sha256']
"""
    subprocess.run([str(new_python), "-E", "-s", "-c", validation, str(release), str(source), str(candidate)],
                   executable=str(Path(sys._base_executable).resolve()),
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    return new_python


def release_identity(release):
    names = ("paper_trader_integrated.py", "features.py", "news_features.py", "model_pipeline.py",
             "ml_models.py", "entry_selection.py", "order_journal.py", "order_lifecycle.py",
             "paper_order_adapter.py", "tools/check_oracle_service_runtime.py",
             "tools/probe_oracle_entry_scan.py", "config.json", "models/ensemble.pkl")
    return {name: digest(release / name) for name in names}


def backup_and_copy_state(source, release, backup):
    # Original tree is untouched. Service is stopped; its full Git/worktree state,
    # config, locally modified ignore file, secrets and untracked reports remain here.
    archive_path = backup / "original-tree.tar"
    with archive_path.open("xb") as stream:
        os.chmod(archive_path, 0o600)
        with tarfile.open(fileobj=stream, mode="w") as archive:
            archive.add(source, arcname="original", recursive=True)
    for name in RUNTIME:
        path = source / name
        if path.exists():
            shutil.copy2(path, release / name)
            os.chmod(release / name, 0o600)
    if (source / "config.json").read_bytes() != (release / "config.json").read_bytes():
        raise DeployError("Source configuration changed before cutover")


def probe_release(release, backup, interpreter):
    # Avoid starting the 100-stock request loop just before a bucket expires.
    remaining = 300 - time.time() % 300
    if remaining < 240:
        print(f"DEPLOY_PHASE waiting_for_fresh_scan_bucket seconds={int(remaining)+2}", flush=True)
        deadline = time.monotonic() + remaining + 2
        while time.monotonic() < deadline:
            time.sleep(min(5., deadline-time.monotonic()))
    output = backup / "scan"
    with (backup / "scan.log").open("xb") as log:
        os.chmod(backup / "scan.log", 0o600)
        subprocess.run([str(interpreter), "-E", "-s", "-u", str(release / "tools/probe_oracle_entry_scan.py"),
                        "--release-dir", str(release), "--output-dir", str(output)],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=360)
    return check.read(output / "summary.json")


def cutover(source, release, backup, interpreter, pair, host, broker_get, *, probe=probe_release,
            expected_service=None, identity=None):
    stopped = installed = start_attempted = False
    account_id = None
    try:
        snapshot = broker_get()
        require_flat(source, snapshot); require_closed_window(snapshot["clock"])
        account_id = snapshot["account"]["id"]
        if expected_service is not None and host.info() != expected_service:
            raise DeployError("Original service changed while the release was prepared")
        stopped = True
        host.stop()
        print("DEPLOY_PHASE original_service_stopped", flush=True)
        assert_no_source_writer(source)
        snapshot = broker_get()
        require_flat(source, snapshot, account_id); require_closed_window(snapshot["clock"])
        backup_and_copy_state(source, release, backup)
        print("DEPLOY_PHASE original_backup_complete", flush=True)
        scan = probe(release, backup, interpreter)
        print("DEPLOY_PHASE zero_order_scan_verified", flush=True)
        snapshot = broker_get()
        require_flat(source, snapshot, account_id); require_closed_window(snapshot["clock"])
        override = backup / "new-override.conf"
        private_write(override, override_text(release, interpreter, host.environment_file, host.base_interpreter))
        installed = True  # Also clean up a partially failed install of this unique path.
        host.install(override)
        print("DEPLOY_PHASE override_installed", flush=True)
        host.verify_unit(release, interpreter)
        print("DEPLOY_PHASE unit_selection_verified", flush=True)
        snapshot = broker_get()
        require_flat(source, snapshot, account_id); require_closed_window(snapshot["clock"])
        if (source / "config.json").read_bytes() != (release / "config.json").read_bytes():
            raise DeployError("Original configuration changed during rollout")
        if identity is not None and identity != release_identity(release):
            raise DeployError("Verified release identity changed during rollout")
        start_attempted = True
        print("DEPLOY_PHASE replacement_start_requested", flush=True)
        host.start()
        info = host.healthy(release, account_id)
        snapshot = broker_get()
        if snapshot["account"]["id"] != account_id or snapshot["positions"] or snapshot["orders"]:
            raise DeployError("Paper state changed during replacement startup")
        return dict(deployment_performed=True, service=info, scan_before_first_entry=scan,
                    release_dir=str(release), backup_dir=str(backup),
                    original_tree_unchanged=True, next_step="Verify the first scheduled paper scan and order journal")
    except BaseException:
        if stopped:
            try:
                if start_attempted:
                    host.stop()
                    snapshot = broker_get()
                    journal, _ = check.journal_summary(release / "paper_order_intents.sqlite3", account_id)
                    if (snapshot["account"]["id"] != account_id
                            or snapshot["positions"] or snapshot["orders"] or journal["intents"]):
                        raise DeployError("New order state exists; preserve replacement state and reconcile manually")
                if installed:
                    host.remove_override()
                host.start()
                if not host.restored(source):
                    raise DeployError("Original service restoration was not confirmed")
                save_json(backup / "failure-state.json", dict(original_service_restored=True))
            except BaseException:
                save_json(backup / "failure-state.json", dict(original_service_restored=False,
                          action="Inspect service, retained release journal and broker state on host"))
                raise DeployError("Automatic restoration was not confirmed; inspect retained state on host") from None
        raise


def deploy(source, candidate, expected_model):
    source, candidate = Path(source).resolve(), Path(candidate).resolve()
    if source == BASE or source in BASE.parents or BASE in source.parents:
        raise DeployError("Run from the existing separate staging worktree")
    interpreter = Path(sys.executable).absolute()
    subprocess.run(["git", "-C", str(BASE), "diff", "--quiet", "HEAD", "--"], check=True)
    cfg, verified = check.verify_candidate(source, candidate)
    if verified["stocks"] != 100 or verified["candidate_sha256"] != expected_model:
        raise DeployError("Expected 100-stock candidate identity differs")
    service = check.service_info()
    if service.get("WorkingDirectory") != str(source):
        raise DeployError("Existing service no longer points to the supplied original source")
    pair, _ = check.running_service_pair(source, service)
    snapshot = check.paper_gets(pair, include_clock=True)
    require_flat(source, snapshot); require_closed_window(snapshot["clock"])
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    release_parent, backup_parent = source.parent / "tradebot-releases", source.parent / "tradebot-backups"
    release_parent.mkdir(mode=0o700, exist_ok=True); backup_parent.mkdir(mode=0o700, exist_ok=True)
    release, backup = release_parent / tag, backup_parent / tag
    backup.mkdir(mode=0o700)
    print("DEPLOY_PREPARING " + json.dumps(dict(release_dir=str(release), backup_dir=str(backup))), flush=True)
    original_size = sum(path.lstat().st_size + 1024 for path in source.rglob("*"))
    required = original_size + 2 * 1024**3
    if shutil.disk_usage(source.parent).free < required:
        raise DeployError("Insufficient free disk for the complete original-tree backup and release")
    host = Host(Path("/etc/systemd/system") / (SERVICE + ".d") / ("zz-oracle-top5-" + tag + ".conf"))
    host.prepare(backup, source, service)
    interpreter = prepare_release(BASE, source, candidate, release, interpreter, pair)
    commit = subprocess.check_output(["git", "-C", str(BASE), "rev-parse", "HEAD"], text=True).strip()
    save_json(backup / "manifest.json", dict(code_commit=commit, candidate=verified,
              source=str(source), release=str(release), interpreter=str(interpreter),
              base_interpreter=str(host.base_interpreter), environment_file=str(host.environment_file),
              override=str(host.override), configured_risk={k: cfg[k] for k in (
                  "max_position_pct", "max_total_exposure_pct", "risk_per_trade_pct", "max_daily_loss_pct")},
              config_sha256=digest(source / "config.json")))
    host.verify_runtime(release, interpreter)
    print("DEPLOY_PHASE systemd_runtime_verified_original_still_running", flush=True)
    result = cutover(source, release, backup, interpreter, pair, host,
                     lambda: check.paper_gets(pair, include_clock=True), expected_service=service,
                     identity=release_identity(release))
    result["code_commit"] = commit
    result["candidate_sha256"] = expected_model
    save_json(backup / "result.json", result)
    return result


def main():
    import fcntl
    os.umask(0o077)
    def interrupted(signum, frame):
        signal.signal(signum, signal.SIG_IGN)
        raise DeployError("Deployment interrupted; inspect the retained result or recovery state")
    for name in ("SIGHUP", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), interrupted)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", required=True, type=Path)
    p.add_argument("--candidate-dir", required=True, type=Path)
    p.add_argument("--expected-model-sha256", required=True)
    args = p.parse_args()
    try:
        with (args.candidate_dir / "run.lock").open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(json.dumps(deploy(args.source_dir, args.candidate_dir, args.expected_model_sha256), indent=2))
    except Exception as error:
        detail = str(error) if isinstance(error, (DeployError, check.CheckError)) else "Inspect the private deployment backup and failure state"
        print(f"DEPLOY_STOPPED {type(error).__name__}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
