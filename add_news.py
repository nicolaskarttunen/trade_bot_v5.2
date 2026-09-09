from pathlib import Path
p = Path("paper_trader.py")
s = p.read_text(encoding="utf-8")
s = s.replace(
    "from alpaca.data.historical import StockHistoricalDataClient",
    "from alpaca.data.historical import StockHistoricalDataClient, NewsClient"
)
s = s.replace(
    "from alpaca.data.requests import StockBarsRequest",
    "from alpaca.data.requests import StockBarsRequest, NewsRequest"
)
s = s.replace(
    "data_client = StockHistoricalDataClient(",
    "news_client = NewsClient(API_KEY, SECRET_KEY)\n\ndata_client = StockHistoricalDataClient("
)
marker = "def get_bars(symbol, days=3):"
news_func = '''def get_live_news(symbol, hours=24):
    now = datetime.now(timezone.utc)

    request = NewsRequest(
        symbols=symbol,
        start=now - timedelta(hours=hours),
        end=now,
        limit=50
    )

    result = news_client.get_news(request)
    return result.data.get("news", [])


def get_news_score(symbol):
    news = get_live_news(symbol)
    now = datetime.now(timezone.utc)

    recent = []
    previous = []

    for n in news:
        try:
            age = (now - n.created_at).total_seconds() / 60

            if age < 0 or age > 360:
                continue

            text = f"{n.headline} {getattr(n, 'summary', '') or ''}".lower()

            positive = [
                "beat", "beats", "growth", "strong", "surge",
                "bullish", "upgrade", "upgraded", "profit",
                "record", "positive", "raises", "raised",
                "outperform", "partnership", "approval",
                "approved", "expands", "demand", "revenue"
            ]

            negative = [
                "miss", "misses", "weak", "drop", "drops",
                "bearish", "downgrade", "downgraded", "sell",
                "loss", "warning", "warns", "cut", "cuts",
                "negative", "lawsuit", "investigation", "delay",
                "risk", "risks", "crash", "recall", "decline"
            ]

            pos = sum(text.count(w) for w in positive)
            neg = sum(text.count(w) for w in negative)

            total = pos + neg

            sentiment = 0.0 if total == 0 else (pos - neg) / total
            sentiment = max(-1.0, min(1.0, sentiment))

            weight = math.exp(-age / 120)

            if age <= 30:
                recent.append((sentiment, weight))
            else:
                previous.append((sentiment, weight))

        except Exception:
            continue

    def weighted(values):
        if not values:
            return 0.0

        total_weight = sum(w for _, w in values)

        if total_weight == 0:
            return 0.0

        return sum(s * w for s, w in values) / total_weight

    recent_score = weighted(recent)
    previous_score = weighted(previous)
    change = recent_score - previous_score

    score = 50 + recent_score * 30 + change * 20
    score = max(0, min(100, score))

    return score, recent_score, previous_score, len(recent)


'''
s = s.replace(marker, news_func + marker)
s = s.replace(
    'print("=" * 70)\n\n\ndef market_clock():',
    'print("=" * 70)\n\n\ndef market_clock():'
)
p.write_text(s, encoding="utf-8")
print("NEWS Kytketty paper_trader.py:hin")
