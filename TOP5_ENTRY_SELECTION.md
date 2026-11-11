# Full-universe entry selection

The paper process now completes a scan of the configured universe before any
new entry decision. With the current configuration that universe has 100 stocks.
Opening selection and intraday selection both use this process.

1. Fix one decision timestamp and retrieve SPY once. Score every stock against
   that timestamp and benchmark. Record unavailable data and individual errors.
2. Exclude occupied symbols, pending buys and symbols already traded that day.
   Keep the existing ML and trade-score thresholds. Rank eligible candidates by
   descending trade score, then descending ML output, then symbol for exact ties.
3. Review up to five candidates using fresh trade prices, ATR, stock and SPY
   momentum, RSI, relative volume and the receipt-aware news feature. The model
   already consumes the technical features. The review records them explicitly.
4. Calculate each candidate's share count, stop, target and conditional dollar
   outcomes using the paper process's existing risk settings. Modeled slippage,
   spread and commission are charged on both sides. A target scenario must cover
   those costs. Review all five before attempting the first order.
5. Recheck price, position/exposure limits, daily risk, pause state and freshness
   at submission. Recalculate the reported plan for the actual allowed quantity.
   An ambiguous submission stops the remaining attempts through the existing
   durable order lifecycle.

The review uses market data and the bot's existing news feature. Missing prices
can leave fewer than five executable
candidates. Exact score ties use symbol order so configuration-list order cannot
silently determine priority.

## Report

`entry_scan.json` records the most recent scan, its timestamp, coverage, all stock
statuses, the five reviews, conditional net target/stop P&L and submission status.
The console prints `ENTRY SCAN` and `TOP 1` through `TOP 5`. Telegram `/signals 5`
shows the latest shortlist with share counts, bracket levels and cost scenarios.
Selection evidence is also retained in the durable order metadata and opening
queue. Previously queued candidates without full-universe selection evidence are
marked `selection_required` and need a fresh scan.

`expected_pnl_usd` is null. The existing classifier estimates the event defined by
`label_horizon_bars` and `label_threshold_pct`, currently a price increase of at
least 0.20% over 15 minutes. That output is not a calibrated probability of
hitting the take-profit before the stop. Target and stop P&L are conditional
scenarios, and end-of-day exits, gaps and actual execution costs can differ.

## Timing and rollout

A scan that crosses a five-minute boundary expires before orders are attempted.
Opening scans must also complete before 09:28 New York time. Order maintenance
continues between stock evaluations. An expired opening scan can be retried in
the remaining opening window.

This development branch is based on `audit-hardening-v1`. Its paper process needs
the model contract and order journal required by that branch. It is not a drop-in
update for legacy model artifacts. Deploying it requires checking the running
Oracle checkout, its model and its existing order state first.

The ongoing four-arm and ML-pair studies execute their frozen research modules
with the previously registered arms. Profitability of this new
selection rule has not been measured on the user's 100-stock history. The earlier
opening-probability ranking experiment performed worse on both inspected periods,
so higher-ranked classifier output alone is not evidence of better returns.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest test_entry_selection test_order_lifecycle test_paper_boundaries
```

Tests use generated signals, fake broker responses and no credentials. They cover
the 100-stock barrier, all five reviews before orders, shared inputs, stale scans,
paused trading, API errors, old queues, ambiguous acknowledgements, sizing and
both-side costs. These tests establish behavior and do not measure market returns.
