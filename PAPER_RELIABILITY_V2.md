# Paper candidate: reliability work, 17 November 2026

This is a review candidate, not an installed Oracle release or a validated profitable strategy. The running service remains the restored legacy bot until a separately verified cutover. No production configuration, account, position, service, model or frozen Windows study was modified by this work.

## Changes

- Includes the previously reviewed intraday partial-bracket fix: opening-order maintenance must not cancel an intraday bracket's partially filled parent. Its original broker-managed exits stay attached. This does not retroactively repair an orphaned position.
- Replaces repeated bulk EOD cancellation/closing with durable per-order cancellation claims and per-symbol closing intents. First stop new entries persistently, reconcile known IDs, cancel known non-close orders, observe terminal parents/children, reconcile bought minus sold quantities against positions, then submit one simple DAY market SELL for the remaining integer quantity.
- A partial/pending close is neither canceled nor sent again. Unknown acknowledgments, 404s, position lag, negative/fractional positions, foreign orders and inconsistent cumulative fills cannot authorize a new SELL. Confirmed terminal exits plus observed flat positions and an empty order book are required for success.
- Nested exit fills must not regress, disappear or change identity. Alpaca's automatic sibling quantity reduction after a partial take-profit is supported.
- Journal v1 upgrades transactionally to v2, preserving intents and events. Account/version checks run before DDL. Old writers reject v2. Copying a journal back to old code is not a supported rollback. Retain a consistent pre-upgrade backup and reconcile runtime state before any recovery.
- Model validation binds the recorded price-event horizon and return threshold to configuration. The target remains a short-horizon price event, not a calibrated probability that a complete bracket trade will make money. Numeric research artifacts use version 2 to retain this target. Frozen version-1 studies must continue with their own original source and artifacts; no silent conversion.

## Material limits and verification gates

Exactly-once intent reservation prioritizes avoiding duplicate SELLs. If a process dies after reservation but before the HTTP request, or a close terminates with a residual quantity, automatic continuation may be blocked. Preserve the journal, inspect the known broker ID and positions, and perform an explicitly reviewed recovery. Do not delete a marker or journal to force progress. EOD maintenance does not promise liquidation during a broker outage or after the session closes.

Broker bracket exits activate only after the entry completely fills. Keeping a partial bracket attached does not prove that the partial position already has an active stop. Additional entries stay blocked while it is partial. Opening standalone entries have a fill-to-protection interval too. This residual execution risk still needs Oracle paper observation before rollout; unit tests cannot eliminate it.

The old Oracle candidate's price data remain useful, but the strict source manifest will reject blindly reusing that candidate with changed training sources. Do not bypass the source guard or overwrite its protocol. A new candidate must retain verified data provenance and the matching training/feature contract.

Tests use deterministic fake brokers, real SDK requests and actual SQLite transactions; network access is blocked. They verify behavior, not market profitability. Local validation uses Python 3.12 and pandas 2.2.3/3.0.0; Oracle Python 3.14 still requires its own test run and service-runtime proof.

Before a strategy replacement, compare on the same time-ordered held-out sessions with all 100 symbols, costs, partial-fill assumptions, scheduled/early closes, flat-to-flat accounting and zero-trade days. Evaluate net P&L, drawdown and probability calibration separately. Do not optimize stops/thresholds against the two losing days. Compare predeclared 20% per-position/100% total and 10%/50% research profiles; no risk settings were changed here. Smaller exposure reduces absolute risk but cannot turn a negative edge into a positive one.

The existing original-order replay does not reproduce full-universe top-five ranking or opening-order execution. It must not be labeled as a backtest of the new candidate. Existing frozen November studies remain separate and unchanged.

## Oracle verification, no deployment

Fetch this review branch into the existing staging repository, create an independent detached worktree, and run `tools/run_audit_tests.py` using the existing staging virtual environment. Save results outside the running service directory. This exercises new code without loading the old live model, changing the service or contacting Alpaca. Keep the exact tested commit and output directory for the next review.

Reference API behavior: [Alpaca orders](https://docs.alpaca.markets/us/docs/orders-at-alpaca), [paper fills](https://docs.alpaca.markets/us/docs/paper-trading).
