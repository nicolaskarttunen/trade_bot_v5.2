# Preserve the registered Windows studies during migration

Migration is pending. The Oracle candidate is a new IEX paper model; it must not
replace either frozen study's SIP data, numeric model or registration. Passing
Linux CI does not establish that the user's registered experiments moved or
that their original inputs match this checkout.

## Required source evidence

Use the existing Windows repository
`C:\Users\nigez\Downloads\tradebot-simulaatio-20261026-225346` and preserve both
complete state directories:

| Directory under `research_runs` | Registered observation period | Destination behavior |
| --- | --- | --- |
| `forward-20261110` | 10 November–8 December 2026 | Continue the same four arms and saved data chain |
| `ml-pair-20261111` | 11 November–9 December 2026 | Read the same 19 shared snapshots and own only the final session |

Before packaging, wait for any existing writer to finish and prevent the next
specific Windows task launch during the consistent copy. Record the existing
task definitions and last run status. Do not stop unrelated processes, delete
state, initialize a replacement registration or train a replacement model.
Keep those study tasks quiescent through destination validation and cutover;
if validation fails, resume only the original source copy and record that the
proposed destination is not accepted. Never run both copies as collectors.

Inventory the original checkout commit and dirty source-file hashes, exact
Python/NumPy/pandas versions, dependency versions, forward registration and
`ready.json` hashes, numeric `model.json`, ML-pair registration, every existing
snapshot ZIP/manifest and accepted daily result/manifest. Preserve all other
state files too, including last-attempt status. Compare each source file named
in `registration.json` to its actual bytes. Include package byte sizes and
SHA-256 hashes outside the two study trees, and verify them after transfer.
Exclude credential files, private keys and unrelated production directories.

The original research Python was reported as 3.12.10; its actual frozen NumPy
and pandas versions must be read from the registrations and checked against
the interpreter. Oracle's candidate Python 3.14.7/pandas 3.0.0 environment is
not an interchangeable research environment. An architecture or OS change
also needs numerical replay/anchor evidence even when version strings match.

## Existing portability barriers in the code

`tools/watch_forward.py:protocol()` hashes its declared source files and records
Python, NumPy and pandas versions. `load_ready()` compares that exact protocol
and the saved registration/model hashes. Updating the study's frozen modules
or choosing a different dependency version causes a deliberate rejection.

`tools/ml_forward_pair.py:plan(source)` additionally stores the absolute
`source_path`. `registration()` compares the entire plan and its hash. Moving
the files from a Windows path to `/home/opc/...` therefore fails unchanged code,
even if every model and snapshot byte is identical. Editing `source_path` in
the registration or recalculating its plan hash would rewrite the original
registration and is not an acceptable migration.

Before implementing a portability adapter, inspect the actual source
registrations and transferred state. Any adapter must be outside the hashed
research modules, keep the original registrations byte-for-byte, and record a
separate migration attestation binding the original Windows path, destination
path, exact runtime, source/model/snapshot/result hashes and transfer time.
Path translation can be the only semantic exception; settings, sessions, model,
arms, costs and source-version checks must still match. Test rejection of wrong
destinations, changed registrations, sources, models and saved prices. Do not
monkeypatch away protocol checking or reuse a fabricated registration.

No portability adapter is claimed to be validated here. The actual source
inventory is still needed to implement and verify it against this user's
registered pair.

## Acceptance before switching collection away from Windows

1. Keep the verified archive and original state unchanged. Work from a verified
   destination copy, in a separate research directory and exact research
   environment. Do not reuse the candidate training output or start another
   ensemble fit on the small Oracle host.
2. Validate existing registrations, numeric model decoding and anchors, snapshot
   hashes and daily ledger chain offline. On disposable copies, compare a
   replay of an already-saved session across the two environments, with the
   same frozen code and inputs. Record any numerical differences and resolve
   them before accepting the destination. Do not overwrite accepted results.
3. Verify read-only access to the same SIP/raw/5Min historical data with the
   same as-of date. IEX access for the candidate does not demonstrate SIP
   entitlement. Never substitute IEX or upgrade a subscription to make these
   experiments run.
4. Use one sequential destination runner with a lock covering both research
   jobs. Keep the current after-close policy (not before 30 minutes after the
   actual session close), original date cohorts and late-session reporting.
   The Windows task time was 21:45 UTC; preserve its intent while checking the
   host timezone. A timer definition alone is not proof that a run succeeded.
5. Verify a destination invocation and its state/last-attempt output, then
   disable only `TradeBot-Simulation-Forward-20261110` and
   `TradeBot-Simulation-MLPair-20261111` on Windows if still enabled. Ensure no
   old process remains writing. Retain the old results and task definitions
   for audit. Do not allow two independent collectors to evolve different
   copies of the registered experiment.

Missed sessions stay in the original cohort. A later first fetch must remain
marked late; never backdate a registration, replace a missing day, or label a
later fetch as real-time observation. Do not publish interim model-selection
conclusions or change the original December result reminders.

These steps define acceptance; none of the transfer, path-adapter validation,
destination scheduling or Windows task disabling has yet been confirmed.
