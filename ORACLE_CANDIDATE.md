# Oracle paper candidate preparation

`tools/prepare_oracle_candidate.py` downloads the configured stock universe and
primary benchmark, then trains a separate compatible model using the existing
disk-backed full-batch trainer. It never imports the trading process, submits
orders, changes a service, or writes to the running bot's directory.

```bash
.venv-oracle/bin/python -u tools/prepare_oracle_candidate.py \
  --source-dir /home/opc/ai_daytrader_v5_1_python314 \
  --output-dir /home/opc/tradebot-top5-check-20261112T002134546405Z/research_runs/oracle-candidate-v1
```

The source config is snapshotted and checked on every continuation. Feed, symbol
list, risk settings, label horizon and training dates are retained. For the
current config the feed is IEX, the universe has 100 stocks plus SPY, and the
history ends at the exclusive `history_end` boundary. Date-only history bounds
are interpreted at midnight New York time. All bars are freshly requested with
raw adjustment, matching the paper process's default request policy; every API
page is consumed. Original data files are neither assumed complete nor changed.
Missing IEX bars remain missing: labels require consecutive same-session bars
and benchmark alignment is exact. This does not establish point-in-time universe
selection or independently verify corporate actions.

Credentials come from the process environment or one complete pair in the
source `.env.v5`, source `.env`, or `/etc/v52-paper-bot.env`. They are never
written into reports. Only the historical market-data endpoint is contacted.
Authentication/subscription errors stop the job; it never upgrades a plan or
switches feeds. Transient GET failures have bounded retries.

One process lock prevents concurrent runs sharing an output. Successful symbol
downloads and the completed training matrix have hashes and can be reused by
running the same command again. An interrupted model fit restarts from that
matrix, not from partial model weights. Interrupted feature preparation starts
a new cache; old partial caches are retained. Source/config/environment changes
or changed saved data stop continuation. Disk space is checked before feature
preparation. Numerical libraries use one thread, and the CLI lowers its own CPU
priority. This does not guarantee a machine's available memory or completion
time.

`status.json` records the current phase. A successful job ends at
`candidate_ready_not_deployed`, with `candidate.pkl`, a feature-contract check
and finite prediction checks for every stock. Training uses neutral news,
because complete receipt-timestamped historical news is not supplied. The model
still learns the configured short-horizon price event; it is not a whole-day net
profit predictor. No profitability result or deployment is inferred from its
completion. The frozen forward/ML-pair studies and their models are untouched.

After preparation, deployment still needs the source config, model, process
environment, broker positions, pending orders and journal checked, followed by
an explicit service switch and verification of the 100-stock selection barrier.
