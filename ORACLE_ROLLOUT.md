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

Preparation and inspection do not change the service. After the flat legacy
account has been verified, the separate cutover command below can make the
explicit service switch with backups, a zero-order scan and startup checks.

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
starting training. It then resolves the verified same-user running service's
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

The command verifies the running PID's user, source argument and working
directory before reading selected startup environment keys. A complete initial
pair takes precedence. The legacy bot also calls `load_dotenv(".env.v5")` after
process creation; those later-added keys need not appear in `/proc/PID/environ`.
When neither initial key is present, the check supports that exact literal
startup load from the verified service directory. Both source and dotenv file
mtime/ctime must predate process startup, derived from Linux boot/start ticks.
The dotenv pair must be complete and literal; changed files, disabled loading,
interpolation, partial initial pairs, different working directories and
unreadable process metadata stop the check for local account confirmation.
`credential_source` reports only the selected source type, never values.
Timestamp checks are consistency evidence, not an authenticated historical
snapshot; the account still needs reconciliation at cutover.

The invoking shell's own keys, generic `.env` files and other accounts are not
fallback sources for this check. If startup inputs have changed or access is
unavailable, inspect the running account on the host; never export keys.

### Switch the verified flat legacy paper account

`tools/deploy_oracle_paper.py` is an explicitly mutating command. Run it as the
existing service user from the staging worktree, with its verified interpreter
and the exact model hash from successful inspection:

```bash
stage=/home/opc/tradebot-top5-check-20261112T002134546405Z
cd "$stage"
nice -n 10 .venv-oracle/bin/python -E -s -u tools/deploy_oracle_paper.py \
  --source-dir /home/opc/ai_daytrader_v5_1_python314 \
  --candidate-dir "$stage/research_runs/oracle-candidate-v1" \
  --expected-model-sha256 VERIFIED_CANDIDATE_HASH
```

This narrow legacy migration requires 100 configured stocks, an empty opening
queue, no existing journal or journal sidecars, a flat active paper account,
supported systemd startup settings and existing password-free sudo rights. It
does not grant sudo/IAM privileges. The broker clock must be fresh, closed and
at least 30 minutes before the next open, outside the opening-order window.

The command creates private `tradebot-releases/TIMESTAMP-ID` and
`tradebot-backups/TIMESTAMP-ID` directories beside the original application.
It first checks that the original interpreter can re-import dependencies and
load the original model with the service's selected Python startup variables,
without importing the trading process or printing its environment. Changed
systemd startup inputs or an unverified old runtime block the switch while the
original service stays running. This addresses the earlier interactive NumPy
import failure rather than assuming the same binary is sufficient for recovery.
It archives the committed code, preserves the original config byte-for-byte,
copies the verified candidate, and builds an independent venv on the same host
using the existing installed Python packages, with no network upgrade. Imports
must resolve inside that venv; the full model protocol and prediction probes
must pass there before the old service is stopped. The new systemd command
executes the verified base Python binary with the release venv path as argv[0],
using systemd's documented `@` prefix and `-E -s -u`. The exact launch must
resolve `sys.prefix` and dependency origins inside that release's independent
venv. This avoids executing a home-directory binary or symlink directly.
A unique `/etc/v52-paper-zz-oracle-top5-TAG.env`, owned by root with mode 0600,
pins the verified existing credential pair and `ALPACA_PAPER=true`. The existing
`/etc/v52-paper-bot.env` stays untouched. Credentials never appear in command
arguments or reports. The private release copy remains available for its
data-only probes and the retained backup; neither copy is made world-readable.
New stdout/stderr go to the same systemd journal unit instead of appending any
legacy file destination inside the original tree.

Before stopping the old process, a separate oneshot system unit tests the exact
base-binary/venv launch and root-owned environment file. It inherits the original
Service settings, including its user and security restrictions, but no Unit
relationships or Install hooks. Unsupported lifecycle hooks or managed directory
settings stop preparation. The test imports dependencies, validates the model,
compares credential values privately, checks paper mode and temporary write
access, and writes a private proof. It denies socket connections and never
imports the trading module body or creates a journal, client, order or thread.
Its temporary unit is stopped and removed; the original service remains running
until this proof succeeds. A failed proof leaves private `runtime-check-log.txt`
and command diagnostics in the new backup. The dedicated root environment file
is retained with the release and recorded in the manifest, including on rollback.

This addresses the observed 12 November manager error, "Failed to load
environment files: Permission denied", before a replacement process could start.
Default SELinux file labels are restored only on the newly installed files when
`restorecon` exists. No policy, sudo/IAM rights or host security mode is changed.
See the primary [systemd environment-file documentation](https://www.freedesktop.org/software/systemd/man/250/systemd.exec.html#EnvironmentFile=)
and [argv[0] launch syntax](https://www.freedesktop.org/software/systemd/man/250/systemd.service.html#ExecStart=).

After stopping the original writer, it repeats the flat-account check, archives
the complete original tree including Git metadata, local config/ignore edits,
secrets, logs and untracked reports, then copies only the needed runtime files
into the release. The original tree is never edited or cleaned. Daily counters
are copied without manual reset. Insufficient disk space, changed config or
release identity, a remaining source writer or any new position/order blocks
cutover.

`tools/probe_oracle_entry_scan.py` compiles an explicit AST allowlist of the
actual production selection/data functions. It never imports the process's
module body, starts threads, constructs a trading client or initializes a
journal. Its flat-account inputs are read-only; all submission calls are denied
and its data client accepts GETs only. It requires every configured symbol to
be visited in one current five-minute bucket, zero submissions and no per-stock
API errors. Unavailable signals remain counted and are reported separately;
100 visits do not mean 100 usable signals or five eligible trades. A partial or
expired probe restores the unchanged original service.

The probe explicitly calls `run_entry_scan(research_only=True)`. This mode can
collect and review signals when the real day's trade budget or loss limit is
already reached, then returns before the entire submission phase. It reports
the actual daily risk inputs, keeps the stored counters/config byte-for-byte,
and has independent submission and HTTP-write denial. Ordinary trading calls
retain their original early stop at either daily limit. This fixes the observed
12 November cutover failure at five trades out of five; resetting the daily
state or increasing its limit is not part of the remedy. A failed attempt's
backup/release/logs must remain available when starting a separate new attempt.

Only then does the tool install a new uniquely named systemd drop-in, verify the
selected executable/directory/environment file, start the release and check a
stable process, its account-bound empty journal and the still-flat paper account.
Once the switch succeeds, the new service may submit normal paper orders in its
configured trading windows. `deployment_performed=true` and `result.json` refer
to this verified service switch, not proof of future scan timing or profitability.

Before a new service has started, failures resume the unchanged original. After
a start attempt, automatic restoration requires a flat broker snapshot and no
new journal intents. Otherwise it retains the new state, stops the replacement
and reports that restoration was not confirmed; reconcile on the host rather
than overwriting the new journal. No old positions are closed and no broker
orders are canceled as part of deployment. Backups and failed releases remain
available for inspection, with private `failure-state.json` if recovery failed.

This command does not transfer, re-register or change the frozen Windows studies.

### Wait on Oracle when the market is already open

The closed-market gate is intentional. `tools/wait_oracle_deployment.py start`
can start one detached, bounded waiter without changing the trading process or
relaxing that gate. It validates the model, account binding, existing privileges
and old-runtime recovery on the host first. If the original cutover window is
already available it can proceed immediately; otherwise its earliest attempt is
15 minutes after the broker's reported next close, including early-close days.
No hard-coded market calendar or Finnish UTC offset is used.

```bash
stage=/home/opc/tradebot-top5-check-20261112T002134546405Z
cd "$stage"
nice -n 10 .venv-oracle/bin/python -E -s -u tools/wait_oracle_deployment.py start \
  --source-dir /home/opc/ai_daytrader_v5_1_python314 \
  --candidate-dir "$stage/research_runs/oracle-candidate-v1" \
  --output-dir "$stage/research_runs/oracle-cutover-wait-v1" \
  --expected-model-sha256 VERIFIED_CANDIDATE_HASH
```

The worker starts in its own session with disconnected stdin and private log
files. Confirm its returned PID/status; it then continues without the SSH/PC or
ChatGPT session. Duplicate launches use locks rather than starting a second
worker. Code commit, source config, model hash and the paper account identity
are bound to the intent. Changing those while it waits stops that intent; use a
separate worktree for other development. It polls read-only data once a minute,
keeps the original service running, waits for a flat account after the permitted
window and retries transient read errors within a 24-hour maximum wait. It
never closes positions or cancels orders to create a flat account.

At eligibility it calls the unchanged cutover command exactly once. All fresh
model/account/clock/runtime checks, backups, scan and recovery rules still apply.
An actual failed cutover is recorded instead of blindly retried. A previously
completed waiter does not schedule another legacy migration. Waiting is not a
deployment success: inspect `status.json` and `deployment.log` under the waiter
output and require the deployment command's real result and selected service.
The logs/intent contain no API credential values. A VM shutdown interrupts this
process; an SSH or local-PC shutdown does not.

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

Opening-order maintenance must never cancel a partially filled intraday bracket.
Canceling a bracket parent can also cancel its protective children. Only an entry
explicitly recorded with `mode=opening` uses the cancel-then-OCO flow; intraday
partial fills continue blocking new entries until their broker state is resolved.
This guard prevents a new cancellation, but does not repair an already canceled
bracket. For an existing partial position, verify the final parent fill, all exit
legs and the current position before deciding its recovery or switching versions.
Keep that position's journal and consumed daily trade slot intact.

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
