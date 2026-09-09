# AI Day Trader V5.2 — ML + Alpaca News (research only)

V5.2 is a research/backtesting upgrade for Python 3.14.

What changed:
- Real historical Alpaca news is downloaded from the Alpaca News API.
- News is converted to a transparent sentiment score and time-decayed.
- The ML feature set now includes `news_score`.
- Backtest uses only news published at or before each 5-minute bar (no intentional look-ahead).
- Backtest sweeps multiple probability/score thresholds and reports trades, return, win rate, profit factor and max drawdown.
- Still NO order execution.

Important:
- This is not a guaranteed profitable strategy.
- The current sentiment model is a lightweight local lexicon, not an LLM.
- IEX stock data is a limited market-data feed; a higher-quality SIP dataset would make validation stronger.
- The news history is downloaded separately because it is much larger/different data than stock bars.

Run order:
1. Existing `.env` with ALPACA_API_KEY and ALPACA_SECRET_KEY must be present.
2. `download_data.py`
3. `download_news.py`
4. `train_models.py`
5. `backtest.py`

Do not connect this version to live trading yet. Validate out-of-sample and walk-forward performance first.
