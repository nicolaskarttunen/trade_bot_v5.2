from pathlib import Path
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env.v5")

client = StockHistoricalDataClient(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY")
)

request = StockBarsRequest(
    symbol_or_symbols=["NVDA"],
    timeframe=TimeFrame.Minute,
    start=datetime.now(timezone.utc) - timedelta(days=2),
    end=datetime.now(timezone.utc),
    feed=DataFeed.IEX
)

bars = client.get_stock_bars(request)

print("ALPACA DATA OK")
print("NVDA bars:", len(bars.data.get("NVDA", [])))


