# Authorized paper close and legacy restoration

`tools/recover_oracle_legacy.py` is a separate Linux/Oracle recovery command.
Use it only after the user authorizes the exact paper symbol and whole-share
quantity to close and the return to the retained original version.

Run it from a separate, committed recovery worktree with the existing preparation
venv. Supply the actual replacement release, its original deployment backup,
a new child output directory under that backup, the authorized symbol and share
quantity, and `--execute-authorized-close-and-restore`.

The command verifies the original code, model, config and dotenv against the
retained original-tree archive. It checks the credential pair against the
replacement and the original root-managed environment file without printing
credentials. It also tests the original interpreter/model import path before
stopping the current service. All existing original configuration values remain.

After stopping and excluding the current writer, it reads the final account-bound
journal and backs up both runtime file sets plus a consistent SQLite backup.
It permits one simple market SELL only when the dedicated paper account holds
exactly the authorized long quantity, has no open orders, and the owned original
BUY plus its exits are terminal. The paper broker clock must be fresh and open.
No orders are canceled and no other position is closed.

The exact close intent is fsynced before its only possible POST. An ambiguous
response, restart or 404 cannot authorize another submission. Repeated runs of
the same operation inspect the existing client order ID. A partial close stays
pending until its fill and the flat account are verified; it is never replaced
or automatically resent. Keep the entire output directory intact.

Only after confirmed flat positions and no open orders does recovery carry the
latest runtime files back to the original directory. Same-day consumed entry
slots and traded symbols never decrease; conflicting equity baselines stop
restoration. The retired journal and recovery close evidence remain available.

Recovery removes only the specific replacement override recorded by its own
deployment manifest. The selected original systemd unit must match the private
original-unit backup exactly. Its startup must stay active on a stable PID in
the original working directory. The old strategy may then trade under its own
rules with the retained daily count.

`status.json` records `starting_recovery`, `writer_stopped`, `close_confirmed`
and `legacy_restored`. On failure, `needs_inspection` is explicit. A failure
after stopping the writer leaves it stopped, preserving the outstanding close
and all evidence. Inspect the existing order/state; do not erase the intent or
submit a replacement close. No automatic switch back to the retired version
occurs on an ambiguous broker or startup outcome.

The new tests exercise a single exact SELL, lost acknowledgements, a timeout
before acceptance, pending partial closes, duplicate launches, identity and
quantity mismatches, order-state changes, original model drift, daily risk
preservation, and the backup/close/original-start sequence with fake brokers
and a fake service manager. Actual host restoration requires the recorded
Oracle result; offline tests do not prove a deployment.
