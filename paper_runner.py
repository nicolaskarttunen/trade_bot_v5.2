import os
import time
import json
import pickle
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

BASE = Path(__file__).resolve().parent

load_dotenv(BASE / ".env.v5")

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
PAPER = os.getenv("ALPACA_PAPER", "").lower()

if not API_KEY or not SECRET_KEY:
    raise SystemExit("ALPACA_API_KEY / ALPACA_SECRET_KEY puuttuu .env.v5-tiedostosta.")

if PAPER not in ("true", "1", "yes"):
    raise SystemExit("TURVAVIRHE: ALPACA_PAPER ei ole true. Runner pysäytetty.")

cfg = json.loads((BASE / "config.json").read_text())

with open(BASE / "models" / "ensemble.pkl", "rb") as f:
    model_data = pickle.load(f)

client = TradingClient(API_KEY, SECRET_KEY, paper=True)

print("=" * 60)
print("V5.2 PAPER RUNNER")
print("=" * 60)
print("MODE: ALPACA PAPER ONLY")
print("SYMBOLS:", ", ".join(cfg["symbols"]))
print("TIMEFRAME:", cfg["timeframe_minutes"], "min")
print("MAX TRADES/DAY:", cfg["max_trades_per_day"])
print("MAX DAILY LOSS:", cfg["max_daily_loss_pct"] * 100, "%")
print("MODEL:", len(model_data["models"]), "models")
print("FEATURES:", len(model_data["features"]))
print("=" * 60)

clock = client.get_clock()

print("MARKET OPEN:", clock.is_open)
print("NEXT OPEN:", clock.next_open)
print("NEXT CLOSE:", clock.next_close)

if not clock.is_open:
    print()
    print("Markkina on kiinni.")
    print("Runner jää odottamaan markkinoiden avausta...")
    print("Pysäytä tarvittaessa: Ctrl+C")

while True:
    try:
        clock = client.get_clock()

        if clock.is_open:
            print()
            print("=" * 60)
            print("MARKKINA ON AUKI")
            print("V5.2 PAPER RUNNER VALMIS")
            print(datetime.now(timezone.utc).isoformat())
            print("=" * 60)
            break

        print(
            datetime.now().strftime("%H:%M:%S"),
            "| Market closed | Next open:",
            clock.next_open
        )

        time.sleep(60)

    except KeyboardInterrupt:
        print()
        print("Runner pysäytetty käyttäjän toimesta.")
        raise SystemExit(0)

    except Exception as e:
        print("Clock error:", repr(e))
        time.sleep(30)
