# Audit hardening v1 — research-only / NOT deployed

Base: `v5.3-experimental` (100-symbol research branch).
This branch contains reviewable changes; it has not been deployed to the
Oracle paper-trading service, and its historical profitability has not been
revalidated. Do not overwrite the running service or live model artifacts.

## Implemented
1. `ml_models.py`: CentroidClassifier now estimates its margin scale only
   from training samples, never from a prediction batch. Old pickled centroid
   models have no `margin_scale_` and raise a descriptive error. **Retraining
   the entire ensemble is mandatory before using a new model artifact.**
2. `features.py`: rows without a qualifying news item have news_score = 0,
   not NaN, even when a separate historic news table exists.
3. `backtest_execution.py` + `backtest.py`: entries use the next bar for
   the SAME signal symbol and SAME New York session; all OHLC bars, not only
   bars with valid ML features, drive exits. The simulated long stop fills at
   the bar open when the market gaps through the stop. The legacy 20-option
   threshold search is prominently labeled development selection, NOT an
   unbiased out-of-sample return.
4. `paper_trader_integrated.py`: on failure to fetch positions, raise and
   block processing instead of returning an apparently flat portfolio.
   **Only the development copy was changed, not Oracle production.**
5. `test_audit_hardening.py`: offline regressions for batch-independent
   predictions, old model compatibility error, missing news, cross-symbol
   entry, overnight entry, gap stops, conservative same-bar exit, and failed
   broker position lookup (mocked, no broker connection).

## Verify locally (no Alpaca keys required)
Run from this branch's root:
```powershell
python -m py_compile ml_models.py features.py backtest.py backtest_execution.py paper_trader_integrated.py test_audit_hardening.py
python -m unittest -v test_audit_hardening
```

## NOT yet addressed / prerequisites for future live deployment
- Compare GitHub development branch against **the exact Oracle service file**
  and launch configuration. The main branch audit examined a seven-symbol
  version; Oracle paper uses a different 100-symbol setup.
- Backtest.py is legacy development research. Other legacy modules
  (walk_forward.py, nested_sensitivity_test.py) still need a systematic
  same-symbol execution and train/validation-boundary review.
- Align historical news computation, live news computation, and data
  publication/receipt times in a single definition. Neutral NaNs alone
  do not ensure feature parity.
- Broker fills vs submission records: reconcile Alpaca order and fill
  history with IDs, partial fills, fees, stops, and live paper equity.
- Deterministic client_order_id plus robust pending order reservation and
  reconciliation should be assessed against all order paths.
- Explicitly verify completed signal bar freshness, data feed paging, and
  early-close protections on the exact currently-running Oracle process.
- Retrain models with corrected centroid/news and re-run realistic OOS
  tests, including the locked V6.4 holdout. Results from before these
  corrections are **not** directly comparable.
- Perform unit tests and dry-run/paper-only rollout. Never enable real
  Alpaca trading as part of these changes.

No evidence of improved expectancy or positive live P&L is claimed.
