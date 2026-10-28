# V6.5 intraday research replay (NO OPG)

**Status:** code proposed on audit-hardening-v1. No Oracle service changes.
The 2026-09-01..2026-10-19 data was already examined; this is a
retrospective comparison, **NOT** a fresh holdout or a profitable strategy claim.

## What differs from V6.4?
- V6.4 selected up to five *ranked* 09:35 entries from a single scan.
- V6.5 uses the production-like SYMBOLS sequence and repeatedly scans every
  five minutes from **09:35 to 15:25 ET**, using **only the prior completed**
  5-minute candle for the signal and the **next bar opening price** as fill proxy.
- Training labels are computed ONLY from future candles in the same New York
  session; near-close observations without enough future bars are not labeled.
- The two study arms are (a) no SPY gate and (b) fixed +0.10% threshold on
  the completed first SPY 09:30 candle. No parameter optimization.
- One shared paper-like portfolio enforces max open positions (5), gross
  exposure <= 50%, per-position cap <= 10%, per-trade risk budget <= 1%,
  at most five entries per day, maximum one trade per symbol/day,
  daily loss entry block, ATR/1.25% stops, 2R targets and approximate 15:50 EOD.
- Simulated positions can overlap. Stop gaps fill at bar OPEN minus costs;
  when target and stop both occur within one bar the stop wins. Equity and
  5-minute drawdown include mark-to-market (but OHLC cannot reconstruct the
  intrabar equity path).

## Critical limitations
- **OPG / premarket scan NOT replayed.** A signal selected at ~09:15 would
  require historical premarket bars and any contemporaneous news. The SPY
  09:30-09:35 gate is not knowable before the 09:30 open and cannot gate OPG.
- NEWS is neutral. Live Alpaca news and historical news are still different.
- Bar opening fills, stops, costs, EOD, early closes, pending order
  reconciliations and partial fills are approximations, not broker execution.
- Research retrains a corrected ML model through 2026-08-31; **this is not
  necessarily the exact model artifact deployed by the Oracle paper service**.
- Paper scans run near each 5-minute boundary, not necessarily simultaneously
  across 100 symbols, and fill prices may differ from next bar OPEN.
- The +0.10% SPY filter has previously been selected on old tests and inspected
  on this interval: no new independent validation until subsequent dates.

## Current verified workflow

See [AUDIT_REVIEW_2026-10-23.md](AUDIT_REVIEW_2026-10-23.md) for the updated
causal feature contract, strict price coverage, calendar inputs and exact
commands. See [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) for the fixed
SPY comparison and news evaluation. Original limitations above are historical;
the new driver requires a supplied exchange calendar for early-close timing.

```powershell
python tools/run_audit_tests.py --output-dir audit_evidence/local-check
python v6_5_intraday_replay.py --help
```

Do not cherry-pick only three replay files from the previous instructions:
the current replay requires the shared feature, model, news and validation
modules. Use a separate full checkout of `audit-hardening-v1`, never the
running Oracle directory. Existing model artifacts require retraining.

Outputs now go to a NEW `--output-dir` and include daily results and a manifest.
Old +0.65% or other V6.4 values are not comparable validation evidence.

## Next milestones
1. Review **actual Oracle launched file** and model artifact metadata, not
   merely GitHub branch labels.
2. Confirm exact completed candle timing and correct the live feature
   transformation to match training (especially intraday market features).
3. Broker-order reconciliation, partial-fill protection and alerting.
4. Historical premarket/news data and a separate realistic OPG simulation.
5. More later uninspected paper observation days.

## Windows offline demonstration

For a separate checkout and credential-free synthetic demonstration, see
[SIMULATION_WINDOWS.md](SIMULATION_WINDOWS.md). This runs the actual replay
engine with generated two-stock data and the offline fault suite. It does
not replace the strict 100-stock historical data requirements above.
