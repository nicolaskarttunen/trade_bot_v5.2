# Oracle rollout gates

This is a preparation runbook, not a record of deployment. The candidate job
started from `08bb220c0e79b865ba254bd2d9082d3d68792f27` in the isolated worktree.
Do not update that worktree, its Python environment or the source configuration
while preparation is running. Do not start another fit on this 1-OCPU/6-GB host.

## Inspect the existing job without restarting it

Run in the existing `opc` SSH session. This block reads selected service fields,
the original PID and the candidate status; it neither imports the trading
process nor connects to the broker. A PID row alone does not establish that the
original job is still running: PIDs can be reused. Use status timestamps and the
matching command line, and inspect the existing log if needed.

```bash
bash <<'SH'
set -eu
stage=/home/opc/tradebot-top5-check-20261112T002134546405Z
systemctl show v52-paper-bot.service \
  -p ActiveState -p SubState -p MainPID -p User -p WorkingDirectory --no-pager
"$stage/.venv-oracle/bin/python" - "$stage" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

stage = Path(sys.argv[1])
output = stage / "research_runs/oracle-candidate-v1"
path = output / "status.json"
if not path.is_file():
    raise SystemExit("Candidate status is unavailable; no progress inferred.")
report = json.loads(path.read_text())
keys = ("status", "updated_at", "stock_universe", "active_symbol",
        "completed_downloads", "total_downloads", "completed_stocks",
        "training_rows", "training_progress", "prediction_checks",
        "deployment_performed", "error_type")
summary = {key: report[key] for key in keys if key in report}
summary["saved_download_records"] = len(report.get("data", {}))
process = Path("/proc/3618759")
try:
    arguments = (process / "cmdline").read_bytes().split(b"\0")
    expected_args = [b"-u", b"tools/prepare_oracle_candidate.py",
                     b"--source-dir", b"/home/opc/ai_daytrader_v5_1_python314",
                     b"--output-dir", os.fsencode(str(output))]
    summary["original_pid_matches_job"] = (
        process.stat().st_uid == os.geteuid()
        and (process / "cwd").resolve() == stage.resolve()
        and arguments[1:7] == expected_args
    )
except OSError:
    summary["original_pid_matches_job"] = False
candidate = output / "candidate.pkl"
expected = report.get("candidate_sha256")
if expected and candidate.is_file():
    digest = hashlib.sha256()
    with candidate.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    summary["candidate_file_hash_matches"] = digest.hexdigest() == expected
print(json.dumps(summary, indent=2))
PY
SH
```

The process check prints only a boolean, never an environment or command line.
It is a point-in-time identity check, not proof of future progress.

`downloading` identifies a current attempt; `data` contains the saved symbol
records. Only `candidate_ready_not_deployed`, all 101 expected symbol records,
100 prediction checks and the matching candidate hash establish completion of
preparation. These are integrity/compatibility checks, not profitability checks.
`completed_stocks` and `training_progress` may describe an earlier phase; read
them together with `status` and `updated_at`. The original startup output proves
only that NVDA downloading began, not that it finished.

Do not infer failure from an unchanged timestamp alone: a download or model fit
can take time. If the process has exited, inspect the existing status and log
before deciding to resume the same command. A failed fitting phase restarts the
fit from its verified completed matrix. Never delete checkpoints to unblock it.

## Candidate and runtime checks before a service switch

1. Verify the preparation protocol against the current source `config.json`,
   the staged feature/training source hashes and the exact preparation Python,
   NumPy and pandas versions. Check every saved data hash and the model hash.
   Validate the trusted candidate with `validate_bundle(..., live=True)` in its
   compatible environment. Preserve IEX, the 100 symbols, the label definition
   and every user risk limit. A changed config requires resolving the mismatch,
   not accepting a legacy bundle or editing the recorded hash.
2. Determine the effective running service environment and its import paths
   privately. `/usr/bin/python3.14` failing to import NumPy interactively did
   not establish how the already-running process obtained its dependencies.
   `ExecStart` using the same binary is insufficient. Select an explicit,
   verified environment for the replacement service, including its current
   environment-file behavior and `ALPACA_PAPER=true`. Keep all credentials on
   the host; do not dump `/proc/.../environ`, environment files or `systemctl`
   environment values into reports.
3. Read the dedicated paper account's positions, nested open orders and order
   identities without POST/DELETE calls. Reconcile them with the existing
   opening queue, durable journal and daily state. New code refuses a fresh
   journal with any broker position or open order. It also rejects legacy
   submitted queue records unknown to that journal. Do not create a fake empty
   journal, erase queues, reset daily counters, or cancel/close anything merely
   to make startup pass. Either verify a flat account with no outstanding
   orders and resolve historic queue records, or prepare a separately reviewed
   reconciliation for the actual state.
4. Keep the current service running until replacement and rollback steps are
   concrete. Immediately before switching, stop its sole writer for a
   consistent backup and repeat account/order checks. Retain the original
   checkout/ref, service definition and overrides, compatible interpreter,
   model, locally edited `.gitignore`/`config.json`, all untracked reports/logs,
   `opening_orders.json`, `paper_live_state.json`, trade and news history, and
   all order/notification state. Back up SQLite consistently, including any
   WAL state; do not copy only a live database's main file. Protect backups
   containing credentials with existing host permissions and never publish
   them. Do not run old and new writers from separate worktrees: their local
   process locks do not coordinate across directories.
5. Deploy only the verified candidate/code/config/environment combination.
   Preparation is not the service directory. Do not execute an unrestricted
   `git pull` over the modified original checkout, overwrite it with a clean
   checkout, or point the service at a fresh worktree with missing account
   state. Preserve a reproducible rollback combination. After the first new
   order, reconcile new intents and broker state before any rollback; restoring
   an older journal alone would lose order knowledge.

No stop/restart, model copy or service-edit command is supplied here: its exact
form depends on the runtime and account/state checks above, which have not yet
been observed. The original paper service has not been changed by preparation.

### Inspect a completed candidate and the existing paper account

After completion, use the tested `tools/check_oracle_rollout.py` from this
branch, with the candidate's unchanged preparation modules and environment:

```bash
stage=/home/opc/tradebot-top5-check-20261112T002134546405Z
cd "$stage"
.venv-oracle/bin/python tools/check_oracle_rollout.py \
  --source-dir /home/opc/ai_daytrader_v5_1_python314 \
  --candidate-dir "$stage/research_runs/oracle-candidate-v1"
```

The check verifies config/source/environment and all recorded data, matrix and
model hashes, validates the model and repeats each stock's saved prediction
probe. It acquires the existing preparation lock without creating files or
starting training. It then uses only the verified same-user running service's
Alpaca credential pair for GETs to the fixed **paper** account, positions and
nested open-orders endpoints. It never constructs a trading client, imports the
trading process, prints credentials, creates a journal or changes broker/local
state. A full 500-parent open-order response stops rather than assuming it is
complete. SQLite is opened with `mode=ro` and `query_only=ON`.

Output includes the current service, configured risk limits, account/queue/
journal counts and reconciliation findings. `assessment` is always
`inspection_only_repeat_at_cutover`, even for a flat account. The legacy bot
continues running during this inspection, so its observations do not replace
the consistent backup, fresh reconciliation or first-scan verification at the
actual switch. A service PID/state change during the check stops it.

The command requires the original service to still expose a verifiable same-user
credential pair. It deliberately does not silently fall back to a different
environment/file pair and thereby inspect a different paper account. If access
is unavailable, inspect the running account on the host; never export keys.

## Verify selection and pre-open behavior

Before the first replacement-version entry, verify the 100-stock barrier and
all shortlisted reviews with the offline entry-selection/order-boundary tests
in the replacement environment. Do not import `paper_trader_integrated.py` as a
read-only smoke check: its module body loads the model and constructs clients,
and its running loop can submit orders. There is no documented read-only
live-scan CLI in this revision.

For the first actual paper scan, retain `entry_scan.json` and the order metadata.
Require `universe_count=100`, `scanned_count=100`, a `signals` record for each
configured symbol, `complete=true`, a current `as_of`, and every shortlisted
review before any submission. A completed scan counts unavailable/error records
too; separately report their number. It does not imply 100 usable signals or
five eligible trades. Zero eligible candidates is a valid outcome. Expired or
partially scanned decisions must not submit entries.

The current opening window is 09:15 through before 09:28 America/New_York.
Each scan must also remain within its five-minute clock bucket. Schedule and
assess this in New York time rather than hard-coding a Finnish offset across
daylight-saving changes. Pre-open analysis never removes the final price,
freshness, sizing, daily-loss, exposure and pending-order checks at submission.
Record the actual 100-stock scan duration on this host; synthetic tests cannot
establish that the live API completes before expiry.

Shortlist target/stop dollar amounts are cost scenarios. The candidate's current
15-minute/+0.20% event target is not a calibrated whole-day net-profit forecast.
The separate frozen forward/ML-pair experiments are not tests of this new rule.

See [candidate preparation](ORACLE_CANDIDATE.md),
[selection behavior](TOP5_ENTRY_SELECTION.md) and
[frozen-study migration requirements](WINDOWS_STUDY_MIGRATION.md).
