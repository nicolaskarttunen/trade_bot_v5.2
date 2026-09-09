import os
import json
import websocket
from dotenv import load_dotenv

load_dotenv(".env.v5")

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

URL = "wss://stream.data.alpaca.markets/v1beta1/news"

def on_open(ws):
    print("NEWS WEBSOCKET CONNECTED")

    ws.send(json.dumps({
        "action": "auth",
        "key": API_KEY,
        "secret": SECRET_KEY
    }))

    ws.send(json.dumps({
        "action": "subscribe",
        "news": ["NVDA", "TSLA", "AMD", "AAPL", "MSFT", "META", "AMZN"]
    }))

def on_message(ws, message):
    print("NEWS:", message)

def on_error(ws, error):
    print("ERROR:", error)

def on_close(ws, code, msg):
    print("CLOSED:", code, msg)

ws = websocket.WebSocketApp(
    URL,
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close
)

ws.run_forever()
