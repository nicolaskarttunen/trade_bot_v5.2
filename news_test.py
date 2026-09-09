from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest
import os, json
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

load_dotenv(".env.v5")

client = NewsClient(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY")
)

symbols = ["NVDA","TSLA","AMD","AAPL","MSFT","META","AMZN"]

for symbol in symbols:
    q = NewsRequest(
        symbols=symbol,
        start=datetime.now(timezone.utc) - timedelta(hours=24),
        end=datetime.now(timezone.utc),
        limit=20
    )

    r = client.get_news(q)
    news = r.data.get("news", [])

    print(f"\n{symbol}: {len(news)} uutista")

    for n in news[:5]:
        print(
            n.created_at,
            "|",
            n.source,
            "|",
            n.headline
        )
