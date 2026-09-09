from pathlib import Path
import json
import re
import math
import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env.v5")

client = NewsClient(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY")
)

SYMBOLS = ["NVDA","TSLA","AMD","AAPL","MSFT","META","AMZN"]
HISTORY_FILE = BASE / "news_live_history.json"

POSITIVE = {
    "beat":3,"beats":3,"growth":2,"strong":2,"surge":3,"surges":3,
    "bullish":3,"upgrade":3,"upgraded":3,"buy":2,"profit":2,"profits":2,
    "record":3,"positive":2,"raises":3,"raised":3,"outperform":3,
    "partnership":2,"deal":2,"approval":3,"approved":3,"expands":2,
    "expansion":2,"demand":2,"revenue":2,"accelerate":2,"accelerates":2,
    "optimistic":2
}

NEGATIVE = {
    "miss":3,"misses":3,"weak":2,"drop":3,"drops":3,"bearish":3,
    "downgrade":3,"downgraded":3,"sell":2,"loss":3,"losses":3,
    "warning":2,"warns":2,"cut":3,"cuts":3,"negative":2,"lawsuit":3,
    "investigation":3,"delay":2,"delays":2,"risk":2,"risks":2,
    "crash":4,"crashes":4,"recall":3,"recalls":3,"decline":2,
    "declines":2,"concern":2,"concerns":2,"slump":3
}

HIGH_IMPACT = {
    "earnings","guidance","forecast","acquisition","merger","lawsuit",
    "investigation","approval","recall","contract","partnership",
    "tariff","regulation","fda","ceo","bankruptcy","offering",
    "buyback","dividend","outlook"
}

def analyze(text):
    words = set(re.findall(r"[a-z]+", text.lower()))

    pos = sum(POSITIVE.get(w, 0) for w in words)
    neg = sum(NEGATIVE.get(w, 0) for w in words)

    total = pos + neg

    if total == 0:
        sentiment = 0.0
    else:
        sentiment = max(-1.0, min(1.0, (pos - neg) / total))

    impact = 1.0

    if words & HIGH_IMPACT:
        impact += 0.5

    if abs(sentiment) >= 0.75:
        impact += 0.25

    return sentiment, impact

def load_history():
    if not HISTORY_FILE.exists():
        return []

    try:
        return json.loads(
            HISTORY_FILE.read_text(encoding="utf-8")
        )
    except Exception:
        return []

def save_history(history):
    HISTORY_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def fetch_news(symbol):
    now = datetime.now(timezone.utc)

    request = NewsRequest(
        symbols=symbol,
        start=now - timedelta(hours=24),
        end=now,
        limit=50
    )

    result = client.get_news(request)
    return result.data.get("news", [])

def calculate_score(symbol, history):
    now = datetime.now(timezone.utc)

    recent = []
    previous = []

    for item in history:
        if symbol not in item.get("symbols", []):
            continue

        try:
            t = datetime.fromisoformat(item["created_at"])
            age = (now - t).total_seconds() / 60

            if age < 0 or age > 360:
                continue

            s = float(item.get("sentiment", 0))
            impact = float(item.get("impact", 1))

            weight = math.exp(-age / 120) * impact

            if age <= 30:
                recent.append((s, weight))
            else:
                previous.append((s, weight))

        except Exception:
            continue

    def weighted(values):
        if not values:
            return 0.0

        total = sum(w for _, w in values)

        if total == 0:
            return 0.0

        return sum(s * w for s, w in values) / total

    recent_score = weighted(recent)
    previous_score = weighted(previous)
    change = recent_score - previous_score

    score = 50 + recent_score * 30 + change * 20

    return (
        max(0, min(100, score)),
        recent_score,
        previous_score,
        change,
        len(recent)
    )

def main():
    history = load_history()
    existing_ids = {str(x["id"]) for x in history}

    new_count = 0

    print("=" * 75)
    print("V5.2 LIVE NEWS ENGINE")
    print("=" * 75)

    for symbol in SYMBOLS:
        try:
            news = fetch_news(symbol)

            for n in news:
                news_id = str(n.id)

                if news_id in existing_ids:
                    continue

                headline = n.headline or ""
                summary = getattr(n, "summary", "") or ""

                text = f"{headline} {summary}"

                s, impact = analyze(text)

                symbols = list(getattr(n, "symbols", []) or [])

                history.append({
                    "id": news_id,
                    "created_at": n.created_at.isoformat(),
                    "symbol": symbol,
                    "symbols": symbols,
                    "source": str(n.source),
                    "headline": headline,
                    "summary": summary,
                    "sentiment": round(s, 4),
                    "impact": round(impact, 3)
                })

                existing_ids.add(news_id)
                new_count += 1

        except Exception as e:
            print(symbol, "ERROR:", repr(e))

    save_history(history)

    print("Uusia uutisia:", new_count)
    print()

    for symbol in SYMBOLS:
        score, recent, previous, change, count = calculate_score(
            symbol, history
        )

        print(
            f"{symbol}: "
            f"SCORE={score:5.1f} | "
            f"30min={recent:+.3f} | "
            f"6h={previous:+.3f} | "
            f"MUUTOS={change:+.3f} | "
            f"TUOREITA={count}"
        )

    print()
    print("Historia päivitetty.")
    print("=" * 75)

if __name__ == "__main__":
    main()
