# V6.5 intraday research replay (NO OPG)

**Status:** code proposed on audit-hardening-v1. No Oracle service changes.
The 2026-09-01..2026-10-19 data was already examined; this is a
retrospective comparison, **NOT** a fresh holdout or a profitable strategy claim.

## What differs from V6.4?
- V6.4 selected up to five *ranked* 09:35 entries from a single scan.
- V6.5 uses the production-like SYMBOLS sequence and repeatedly scans every
  five minutes from **09:35 to 15:25 ET**, using **only the prior completed**
  5-minute candle for the signal and the **next bar opening price** as fill proxy.
- Training labels are computed ONLY from future candles in the same New York\n  session; near-close observations without enough future bars are not labeled.\n- The two study arms are (a) no SPY gate and (b) fixed +0.10% threshold on
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

## Run from Windows (research only)
Project folders expected under Documents, next to one another:
- `tradebot_audit_hardening` (this repository)
- `tradebot_v62_research/data` and `tradebot_v62_research/data_v64` (102 CSVs each)

```powershell
cd "$HOME\Documents\tradebot_audit_hardening"
git fetch origin audit-hardening-v1
git checkout origin/audit-hardening-v1 -- intraday_replay.py v6_5_intraday_replay.py test_intraday_replay.py
python -m py_compile intraday_replay.py v6_5_intraday_replay.py test_intraday_replay.py
python -m unittest -v test_audit_hardening test_intraday_replay
python -u v6_5_intraday_replay.py 2>&1 | Tee-Object -FilePath v6_5_intraday_run.log
```

Outputs: `v6_5_intraday_summary.csv`, `v6_5_intraday_trades.csv`,
`v6_5_intraday_equity.csv`.

Comparison with V6.4's +0.65% is **not like-for-like**: ranking and timing
are different, and V6.5 now admits later intraday entries and overlapping
positions. Report these caveats with any result.

## Next milestones
1. Review **actual Oracle launched file** and model artifact metadata, not
   merely GitHub branch labels.
2. Confirm exact completed candle timing and correct the live feature
   transformation to match training (especially intraday market features).
3. Broker-order reconciliation, partial-fill protection and alerting.
4. Historical premarket/news data and a separate realistic OPG simulation.
5. More later uninspected paper observation days.
