# AI Trading Research System

A Python-based machine-learning and market-news research project for testing systematic trading ideas with historical data and Alpaca paper trading.

This project is built for research and experimentation. It is not presented as a profitable strategy and is not configured for live-money trading.

## What it includes

- Historical stock market data ingestion through Alpaca
- Historical and live news ingestion
- Technical and market-regime features
- Time-decayed news sentiment features
- Custom machine-learning ensemble models
- Backtesting with configurable execution assumptions
- Walk-forward and sensitivity testing
- Stress testing and Monte Carlo research
- Risk controls
- Alpaca paper-trading integration

## Research approach

The system combines several signal groups:

- Short-term price returns
- EMA relationships
- RSI
- ATR and volatility
- Volume
- Broader market movement
- News sentiment

News is only applied when its publication timestamp is at or before the market bar being evaluated, helping avoid intentional look-ahead bias.

The repository also contains several validation scripts used to test the strategy under different thresholds and assumptions.

## Tech stack

- Python
- pandas
- NumPy
- alpaca-py
- python-dotenv

## Safety

The paper-trading runner requires paper mode to be explicitly enabled.

API credentials are loaded from local environment files and should never be committed to Git.

The repository includes only placeholder credentials in `.env.example`.

## Basic setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your local environment configuration from the example file and add Alpaca paper-trading credentials.

The main research workflow includes:

```text
download_data.py
download_news.py
train_models.py
backtest.py
```

Additional scripts cover walk-forward validation, sensitivity testing, stress tests and paper-trading experiments.

## Important note

Backtest results are research outputs, not guarantees of future performance. Transaction costs, data quality, market regime changes and execution differences can materially affect real-world results.

## About

Built as an independent Python and machine-learning project by Nicolas Karttunen.
