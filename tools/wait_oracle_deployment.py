"""Start one detached, bounded waiter for the existing closed-market cutover.

No relaxed market gate, trading-process changes or broker writes. The original
service stays running while waiting. The existing deployment command executes
once after the eligible close; a failed cutover is never blindly retried.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from tools import check_oracle_rollout as check
from tools import deploy_oracle_paper as deploy
from tools.prepare_oracle_candidate import digest


class WaitError(ValueError):
    pass


@contextmanager
def lock(path, create=True):
    import fcntl
    with path.open("a+b" if create else "rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def fingerprint(account):
    return hashlib.sha256(str(account["id"]).encode()).hexdigest()


def head():
    return subprocess.check_output(["git", "-C", str(BASE), "rev-parse", "HEAD"], text=True).strip()


def clean():
    subprocess.run(["git", "-C", str(BASE), "diff", "--quiet", "HEAD", "--"], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def clock_time(clock):
    import pandas as pd
    now = pd.Timestamp.now(tz="UTC")
    timestamp = pd.Timestamp(clock["timestamp"])
    if timestamp.tzinfo is None or abs((now-timestamp).total_seconds()) > 60:
        raise WaitError("Broker/server clock is not fresh; inspect clock synchronization before scheduling")
    return now


def earliest_attempt(clock):
    import pandas as pd
    now = clock_time(clock)
    try:
        deploy.require_closed_window(clock, now)
        return now
    except deploy.DeployError:
        close = pd.Timestamp(clock["next_close"])
        if close.tzinfo is None or close <= now:
            raise WaitError("Broker next-close timestamp is unavailable")
        return close.tz_convert("UTC") + pd.Timedelta(minutes=15)


def ready(clock, not_before):
    import pandas as pd
    now = clock_time(clock)
    if now < pd.Timestamp(not_before):
        return False
    try:
        deploy.require_closed_window(clock, now)
    except deploy.DeployError:
        return False
    return True


def poll_snapshot(source):
    service = check.service_info()
    if service.get("WorkingDirectory") != str(source):
        raise WaitError("Service moved away from the intended original source; do not repeat a cutover")
    pair, _ = check.running_service_pair(source, service)
    return check.paper_gets(pair, include_clock=True)


def guard(intent):
    clean()
    if head() != intent["code_commit"]:
        raise WaitError("Staging code changed while waiting")
    if digest(Path(intent["source_dir"]) / "config.json") != intent["config_sha256"]:
        raise WaitError("Source configuration changed while waiting")
    if digest(Path(intent["candidate_dir"]) / "candidate.pkl") != intent["candidate_sha256"]:
        raise WaitError("Candidate model changed while waiting")


def save_status(output, intent, phase, **details):
    value = dict(phase=phase, pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(),
                 earliest_attempt_utc=intent["not_before"], code_commit=intent["code_commit"],
                 candidate_sha256=intent["candidate_sha256"], **details)
    deploy.save_json(output / "status.json", value)
    with (output / "history.jsonl").open("a") as log:
        log.write(json.dumps(value) + "\n")
    return value


def execute(intent, output):
    argv = [sys.executable, "-E", "-s", "-u", str(BASE / "tools/deploy_oracle_paper.py"),
            "--source-dir", intent["source_dir"], "--candidate-dir", intent["candidate_dir"],
            "--expected-model-sha256", intent["candidate_sha256"]]
    with (output / "deployment.log").open("ab") as log:
        result = subprocess.run(argv, cwd=BASE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode


def supervise(intent, output, *, snapshot=poll_snapshot, deploy_once=execute, sleeper=time.sleep,
              now=lambda: time.time()):
    source = Path(intent["source_dir"])
    while now() < intent["deadline_epoch"]:
        guard(intent)
        try:
            broker = snapshot(source)
        except check.CheckError as error:
            # These messages are authored by paper_gets, without bodies/headers.
            message = str(error)
            if any(code in message for code in ("HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504")) or "response details withheld" in message:
                save_status(output, intent, "waiting_after_transient_read_error", error_type=type(error).__name__)
                sleeper(60)
                continue
            raise
        if fingerprint(broker["account"]) != intent["account_sha256"]:
            raise WaitError("Paper account identity changed while waiting")
        if ready(broker["clock"], intent["not_before"]):
            if broker["positions"] or broker["orders"]:
                save_status(output, intent, "waiting_for_flat_account", positions=len(broker["positions"]),
                            open_parent_orders=len(broker["orders"]))
                sleeper(60)
                continue
            # The existing cutover repeats every flat-account/runtime/model check.
            save_status(output, intent, "deploying_once")
            code = deploy_once(intent, output)
            return save_status(output, intent, "deployment_command_completed" if code == 0 else "deployment_command_failed",
                               deployment_exit_code=code, log_file=str(output / "deployment.log"))
        save_status(output, intent, "waiting_for_safe_window", market_is_open=broker["clock"]["is_open"],
                    positions=len(broker["positions"]), open_parent_orders=len(broker["orders"]))
        sleeper(60)
    raise WaitError("Wait deadline reached; original service has not been stopped by this waiter")


def worker(output, intent_file):
    try:
        with lock(output / "worker.lock"):
            intent = check.read(intent_file)
            try:
                save_status(output, intent, "starting_waiter")
                result = supervise(intent, output)
                return 0 if result["phase"] == "deployment_command_completed" else 1
            except Exception as error:
                detail = str(error) if isinstance(error, (WaitError, check.CheckError)) else "Inspect the host inputs privately"
                save_status(output, intent, "waiter_failed", error_type=type(error).__name__, error_detail=detail)
                return 1
    except BlockingIOError:
        return 2  # Never overwrite the active waiter's state.
    return 0


def active_status(output):
    try:
        with lock(output / "worker.lock"):
            return None
    except BlockingIOError:
        status = check.read(output / "status.json")
        process = Path("/proc") / str(int(status["pid"]))
        if (process.stat().st_uid != os.geteuid() or
                os.fsencode(str(BASE / "tools/wait_oracle_deployment.py")) not in (process / "cmdline").read_bytes().split(b"\0")):
            raise WaitError("Waiter lock belongs to an unverified process")
        return status


def start(source, candidate, output, expected_model):
    import pandas as pd
    source, candidate, output = map(lambda p: Path(p).resolve(), (source, candidate, output))
    if output == source or source in output.parents or output == candidate or candidate in output.parents:
        raise WaitError("Use a separate staging research directory for the waiter")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    with lock(output / "launch.lock"):
        active = active_status(output)
        if active:
            existing = check.read(output / "intent.json")
            if (existing["source_dir"] != str(source) or existing["candidate_sha256"] != expected_model
                    or existing["candidate_dir"] != str(candidate) or existing["code_commit"] != head()):
                raise WaitError("An active waiter belongs to another immutable deployment intent")
            return dict(already_running=True, status=active, state_file=str(output / "status.json"))
        previous = check.read(output / "status.json") if (output / "status.json").exists() else {}
        if previous.get("phase") == "deployment_command_completed":
            return dict(previous_attempt_completed=True, service_now=check.service_info(),
                        state_file=str(output / "status.json"), deployment_log=str(output / "deployment.log"))
        clean()
        with lock(candidate / "run.lock", create=False):
            _, verified = check.verify_candidate(source, candidate)
        if verified["candidate_sha256"] != expected_model or verified["stocks"] != 100:
            raise WaitError("Intended 100-stock candidate differs")
        service = check.service_info()
        if service.get("WorkingDirectory") != str(source):
            raise WaitError("The original service is no longer selected; inspect the existing deployment result")
        pair, _ = check.running_service_pair(source, service)
        broker = check.paper_gets(pair, include_clock=True)
        not_before = earliest_attempt(broker["clock"])
        if broker["account"].get("status") != "ACTIVE" or broker["account"].get("trading_blocked") is not False:
            raise WaitError("Existing paper account is not active and unblocked")
        tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        readiness = output / ("readiness-" + tag); readiness.mkdir(mode=0o700)
        host = deploy.Host(Path("/etc/systemd/system") / (deploy.SERVICE + ".d") / ("zz-waiter-readiness-" + tag + ".conf"))
        # Read-only service/runtime preparation: no stop, install, start or broker write.
        host.prepare(readiness, source, service)
        intent = dict(code_commit=head(), source_dir=str(source), candidate_dir=str(candidate),
                      candidate_sha256=expected_model, config_sha256=digest(source / "config.json"),
                      account_sha256=fingerprint(broker["account"]), not_before=not_before.isoformat(),
                      deadline_epoch=time.time()+24*3600)
        deploy.save_json(output / "intent.json", intent)
        intent_file = readiness / "intent.json"
        deploy.save_json(intent_file, intent)
        if (output / "status.json").exists():
            (output / "status.json").replace(readiness / "previous-status.json")
        with (output / "waiter.log").open("ab") as log:
            process = subprocess.Popen([sys.executable, "-E", "-s", "-u", str(BASE / "tools/wait_oracle_deployment.py"),
                    "run", "--output-dir", str(output), "--intent-file", str(intent_file)], cwd=BASE, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status = None
        for _ in range(30):
            time.sleep(.5)
            try:
                status = active_status(output)
            except (FileNotFoundError, ProcessLookupError):
                status = None  # Child acquired its lock but is still writing initial state.
            if status is not None or process.poll() is not None:
                break
        if status is None:
            state = check.read(output / "status.json") if (output / "status.json").exists() else {}
            if state.get("phase") == "deployment_command_completed":
                return dict(running=False, status=state, state_file=str(output / "status.json"))
            raise WaitError("Detached waiter did not remain active; inspect waiter.log and status.json")
        return dict(running=True, pid=status["pid"], status=status["phase"],
                    earliest_attempt_helsinki=pd.Timestamp(intent["not_before"]).tz_convert("Europe/Helsinki").isoformat(),
                    state_file=str(output / "status.json"), deployment_log=str(output / "deployment.log"))


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["start", "run"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--intent-file", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "run":
            if args.intent_file is None:
                raise WaitError("Worker requires its immutable per-attempt intent file")
            return worker(args.output_dir, args.intent_file)
        if not all((args.source_dir, args.candidate_dir, args.expected_model_sha256)):
            raise WaitError("Start requires the original source, candidate and verified model hash")
        print(json.dumps(start(args.source_dir, args.candidate_dir, args.output_dir, args.expected_model_sha256), indent=2))
    except Exception as error:
        detail = str(error) if isinstance(error, (WaitError, check.CheckError, deploy.DeployError)) else "Inspect readiness inputs on the host"
        print(f"WAITER_STOPPED {type(error).__name__}: {detail}; no clock gate relaxed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
