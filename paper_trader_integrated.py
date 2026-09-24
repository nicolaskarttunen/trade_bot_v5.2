import os
import json
import math
import time
import pickle
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import websocket
import requests
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from features import enrich, FEATURES


# =====================================================# V5.2 INTEGRATED PAPER TRADER
# Realtime news + ML + 5min bars + Alpaca Paper orders
# =====================================================
BASE = Path(__file__).resolve().parent
load_dotenv(".env.v5")

CFG = json.loads((BASE / "config.json").read_text(encoding="utf-8"))

with open(BASE / "models" / "ensemble.pkl", "rb") as f:
    MODEL = pickle.load(f)

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise SystemExit("ALPACA API -avaimet puuttuvat .env.v5-tiedostosta.")

if str(os.getenv("ALPACA_PAPER", "")).lower() != "true":
    raise SystemExit(
        "TURVAVIRHE: ALPACA_PAPER ei ole true. "
        "Tämä ohjelma sallii vain Paper Tradingin."
    )

trading = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

SYMBOLS = CFG["symbols"]
BENCHMARK = CFG["benchmark_symbols"][0]

NEWS_FILE = BASE / "news_live_history.json"
TRADE_FILE = BASE / "trades.csv"
STATE_FILE = BASE / "paper_live_state.json"
ORDER_NOTIFICATION_FILE = BASE / "paper_order_notifications.json"

news_lock = threading.Lock()
news_items = []

# TELEGRAM CONTROL
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
telegram_trading_enabled = True
telegram_signals = []
telegram_offset = None
telegram_lock = threading.Lock()


# ------------------------------------------------------------
# NEWS
# ------------------------------------------------------------

POSITIVE = {
    "beats", "beat", "surge", "surges", "rises", "rise", "record",
    "strong", "growth", "upgrade", "upgraded", "bullish", "profit",
    "profits", "partnership", "approval", "approved", "launch",
    "launches", "wins", "win", "buyback", "outperform", "demand",
    "raises", "raised", "guidance raised", "positive"
}

NEGATIVE = {
    "miss", "misses", "falls", "fall", "drop", "drops", "weak",
    "decline", "downgrade", "downgraded", "bearish", "loss",
    "losses", "lawsuit", "probe", "investigation", "warning",
    "cuts", "cut", "recall", "recalls", "delay", "delays",
    "layoffs", "tariff", "ban", "banned", "negative", "disappointing"
}


def load_news_history():
    if not NEWS_FILE.exists():
        return []

    try:
        return json.loads(NEWS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_news_history(items):
    tmp = NEWS_FILE.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(items[-5000:], ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    tmp.replace(NEWS_FILE)


def calculate_sentiment(text):
    text = (text or "").lower()

    pos = sum(text.count(word) for word in POSITIVE)
    neg = sum(text.count(word) for word in NEGATIVE)

    total = pos + neg

    if total == 0:
        return 0.0

    return max(-1.0, min(1.0, (pos - neg) / total))


def process_news(raw):
    try:
        created_at = raw.get("created_at") or raw.get("updated_at")
        if not created_at:
            return

        headline = raw.get("headline", "") or ""
        summary = raw.get("summary", "") or ""

        symbols = raw.get("symbols") or []
        if isinstance(symbols, str):
            symbols = [symbols]

        unique_id = f"{raw.get('id', '')}|{created_at}"

        item = {
            "id": unique_id,
            "created_at": str(created_at),
            "symbols": list(symbols),
            "headline": headline,
            "summary": summary,
            "source": raw.get("source", ""),
            "sentiment": calculate_sentiment(headline + " " + summary),
            "impact": 1.0,
        }

        with news_lock:
            global news_items

            if any(x.get("id") == unique_id for x in news_items):
                return

            news_items.append(item)
            save_news_history(news_items)

        print(
            f"{datetime.now():%H:%M:%S} | NEWS | "
            f"{','.join(symbols) or '-'} | {headline[:110]}"
        )

    except Exception as exc:
        print("NEWS parse error:", repr(exc))


def get_news_signal(symbol):
    """
    Returns:
        recent_score: -1..+1
        change_score: -1..+1

    Same general scale as the historical news_score feature.
    """

    now = datetime.now(timezone.utc)

    recent = []
    previous = []

    with news_lock:
        items = list(news_items)

    for item in items:
        if symbol not in (item.get("symbols") or []):
            continue

        try:
            timestamp = pd.Timestamp(
                item["created_at"]
            ).tz_convert("UTC").to_pydatetime()
        except Exception:
            continue

        age_minutes = (now - timestamp).total_seconds() / 60.0

        if age_minutes < 0 or age_minutes > 360:
            continue

        weight = math.exp(
            -age_minutes / max(float(CFG["news_decay_minutes"]), 1.0)
        )

        value = (
            float(item.get("sentiment", 0.0))
            * float(item.get("impact", 1.0))
            * weight
        )

        if age_minutes <= 30:
            recent.append((value, weight))
        else:
            previous.append((value, weight))

    def weighted(values):
        if not values:
            return 0.0

        numerator = sum(value for value, weight in values)
        denominator = sum(weight for value, weight in values)

        return numerator / denominator if denominator else 0.0

    recent_score = weighted(recent)
    previous_score = weighted(previous)

    change = recent_score - previous_score

    return (
        max(-1.0, min(1.0, recent_score)),
        max(-1.0, min(1.0, change)),
    )


def news_stream_loop():
    """
    Permanent realtime news listener.
    It runs in a background thread and never places orders.
    """

    url = "wss://stream.data.alpaca.markets/v1beta1/news"

    def on_open(ws):
        print("NEWS STREAM | connected")

        ws.send(json.dumps({
            "action": "auth",
            "key": API_KEY,
            "secret": SECRET_KEY
        }))

        ws.send(json.dumps({
            "action": "subscribe",
            "news": SYMBOLS
        }))

    def on_message(ws, message):
        try:
            messages = json.loads(message)

            for msg in messages:
                msg_type = msg.get("T")

                if msg_type == "n":
                    process_news(msg)

                elif msg_type == "error":
                    print("NEWS STREAM ERROR:", msg)

                elif msg_type in ("success", "subscription"):
                    print("NEWS STREAM:", msg)

        except Exception as exc:
            print("NEWS stream parse error:", repr(exc))

    def on_error(ws, error):
        print("NEWS STREAM connection error:", error)

    def on_close(ws, code, message):
        print("NEWS STREAM closed:", code, message)

    while True:
        try:
            ws = websocket.WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )

            ws.run_forever(
                ping_interval=20,
                ping_timeout=10
            )

        except Exception as exc:
            print("NEWS STREAM restart:", repr(exc))

        time.sleep(5)


# ------------------------------------------------------------
# MARKET DATA / ML
# ------------------------------------------------------------

def get_latest_trade_price(symbol):
    """Return Alpaca's latest trade price for order pricing."""
    try:
        request = StockLatestTradeRequest(
            symbol_or_symbols=symbol,
            feed=CFG["feed"],
        )

        trades = data_client.get_stock_latest_trade(request)
        trade = trades[symbol]

        price = float(trade.price)

        if price <= 0:
            raise ValueError(f"invalid latest trade price: {price}")

        return price

    except Exception as exc:
        print(f"LATEST PRICE ERROR | {symbol} | {exc!r}")
        return None


def get_bars(symbol, limit=250):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=5)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start,
        end=end,
        limit=limit,
        feed=CFG["feed"],
    )

    result = data_client.get_stock_bars(request)
    df = result.df

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.index, pd.MultiIndex):
        try:
            df = df.xs(symbol, level="symbol").reset_index()
        except Exception:
            df = df.reset_index()
    else:
        df = df.reset_index()

    df.columns = [str(column).lower() for column in df.columns]

    if "timestamp" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "timestamp"})

    needed = ["timestamp", "open", "high", "low", "close", "volume"]

    if not all(column in df.columns for column in needed):
        return pd.DataFrame()

    return df[needed].copy()


def get_benchmark():
    benchmark = get_bars(BENCHMARK, 250)
    if benchmark.empty:
        return benchmark
    benchmark = enrich(benchmark)
    return benchmark[['timestamp', 'close', 'atr_pct']].copy()


def get_ml_probability(symbol):
    df = get_bars(symbol)

    if len(df) < 60:
        return None, None

    benchmark = get_benchmark()

    recent_news, _ = get_news_signal(symbol)

    enriched = enrich(
        df,
        market=benchmark,
        news=None,
        news_decay_minutes=CFG["news_decay_minutes"],
        news_lookback_minutes=CFG["news_lookback_minutes"],
    )

    # Live news signal is kept on the same -1..1 scale as the
    # historical news feature used during model training.
    enriched["news_score"] = recent_news

    enriched = enriched.dropna(subset=FEATURES)

    if enriched.empty:
        return None, None

    row = enriched.iloc[-1]

    X = row[FEATURES].to_numpy(dtype=float).reshape(1, -1)

    probabilities = []

    for model in MODEL["models"]:
        probabilities.append(
            float(model.predict_proba(X)[:, 1][0])
        )

    return float(np.mean(probabilities)), row


# ------------------------------------------------------------
# TELEGRAM
# ------------------------------------------------------------

def telegram_send(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        r=requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id":TELEGRAM_CHAT_ID,"text":message},
            timeout=10
        )
        return bool(r.ok and r.json().get("ok"))
    except Exception as exc:
        print("TELEGRAM SEND ERROR:", repr(exc))
        return False


def telegram_loop():
    global telegram_offset, telegram_trading_enabled
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("TELEGRAM | asetukset puuttuvat")
        return

    telegram_send("🟢 V5.3 Telegram-yhteys käynnissä.")

    while True:
        try:
            params={"timeout":25,"limit":50}
            if telegram_offset is not None:
                params["offset"]=telegram_offset

            r=requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                params=params,
                timeout=35
            )

            if not r.ok:
                print("TELEGRAM POLL HTTP:", r.status_code)
                time.sleep(5)
                continue

            data=r.json()

            if not data.get("ok"):
                print("TELEGRAM POLL ERROR:", data)
                time.sleep(5)
                continue

            for update in data.get("result", []):
                telegram_offset=update["update_id"]+1

                message=update.get("message")
                if not message:
                    continue

                chat_id=str((message.get("chat") or {}).get("id",""))
                if chat_id != str(TELEGRAM_CHAT_ID):
                    continue

                text=str(message.get("text","")).strip()
                if not text:
                    continue

                parts = text.split()
                command=parts[0].lower().split("@")[0]
                args=parts[1:]

                if command == "/start":
                    telegram_trading_enabled=True
                    telegram_send("🟢 UUSIEN TREIDIEN TEKO ON PÄÄLLÄ.")

                elif command == "/stop":
                    telegram_trading_enabled=False
                    telegram_send("🔴 UUSIEN TREIDIEN TEKO POIS PÄÄLTÄ. Avoimia positioita ei suljeta.")

                elif command == "/status":
                    telegram_send(
                        f"🤖 V5.3 PAPER BOT\n"
                        f"Uudet treidit: {'🟢 ON' if telegram_trading_enabled else '🔴 OFF'}\n"
                        f"Avoimet positiot: {len(get_positions())}"
                    )

                elif command == "/positions":
                    positions=get_positions()
                    if not positions:
                        telegram_send("📋 Ei avoimia positioita.")
                    else:
                        lines=["📋 AVOIMET POSITIOT",""]
                        for symbol,p in positions.items():
                            lines.append(
                                f"{symbol} | qty {getattr(p,'qty','?')} | "
                                f"entry {getattr(p,'avg_entry_price','?')} | "
                                f"P/L {getattr(p,'unrealized_pl','?')}"
                            )
                        telegram_send("\n".join(lines))

                elif command == "/signals":
                    try:
                        limit = int(args[0]) if args else 10
                    except ValueError:
                        limit = 10

                    limit = max(1, min(50, limit))

                    with telegram_lock:
                        items = list(telegram_signals[-limit:])

                    if not items:
                        telegram_send("\U0001F4CA Ei signaaleja viel\u00e4.")
                    else:
                        lines = [
                            f"\U0001F4CA VIIMEISET {len(items)} SIGNAALIA",
                            ""
                        ]

                        for item in reversed(items):
                            decision = item["decision"]

                            if decision == "BUY":
                                icon = "\U0001F7E2"
                                decision_text = "BUY"
                            else:
                                icon = "\U0001F7E1"
                                decision_text = "NO TRADE"

                            lines.extend([
                                f"{icon} {item['symbol']}",
                                f"\U0001F552 {item['time']}",
                                f"ML: {item['ml'] * 100:.1f}%",
                                f"Score: {item['score']:.1f}/100",
                                f"News: {item['news']:+.2f}",
                                f"\u27A1\uFE0F {decision_text}",
                                ""
                            ])

                        telegram_send("\n".join(lines))

                elif command == "/help":
                    telegram_send(
                        "🤖 V5.3 KOMENNOT\n\n"
                        "/start – salli uudet treidit\n"
                        "/stop – estä uudet treidit\n"
                        "/status – botin tila\n"
                        "/positions – avoimet positiot\n"
                        "/help – komennot"
                    )

        except Exception as exc:
            print("TELEGRAM LOOP ERROR:", repr(exc))
            time.sleep(10)

# ------------------------------------------------------------
# RISK / STATE
# ------------------------------------------------------------

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(
                STATE_FILE.read_text(encoding="utf-8")
            )
        except Exception:
            pass

    return {
        "date": None,
        "start_equity": None,
        "trades": 0,
    }


def save_state(state):
    STATE_FILE.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8"
    )


def load_notified_order_ids():
    if ORDER_NOTIFICATION_FILE.exists():
        try:
            data = json.loads(
                ORDER_NOTIFICATION_FILE.read_text(encoding="utf-8")
            )
            if isinstance(data, list):
                return set(str(order_id) for order_id in data)
        except Exception:
            pass

    return set()


def save_notified_order_ids(order_ids):
    ORDER_NOTIFICATION_FILE.write_text(
        json.dumps(sorted(order_ids), indent=2),
        encoding="utf-8"
    )


def get_daily_state():
    state = load_state()

    today = datetime.now().date().isoformat()

    account = trading.get_account()
    equity = float(account.equity)

    if state.get("date") != today:
        state = {
            "date": today,
            "start_equity": equity,
            "trades": 0,
        }

        save_state(state)

    return state


def daily_loss_reached(state):
    account = trading.get_account()

    start_equity = float(state["start_equity"])
    current_equity = float(account.equity)

    limit = float(CFG["max_daily_loss_pct"])

    reached = current_equity <= start_equity * (1.0 - limit)

    return reached, current_equity, start_equity


def get_positions():
    try:
        return {
            position.symbol: position
            for position in trading.get_all_positions()
        }
    except Exception:
        return {}


def has_open_buy_order(symbol):
    """Return True if an active BUY order already exists for this symbol."""
    try:
        orders = trading.get_orders()
        active_statuses = {
            "new",
            "accepted",
            "pending_new",
            "partially_filled",
            "pending_replace",
            "accepted_for_bidding",
        }

        for order in orders:
            if (
                order.symbol == symbol
                and str(order.side).lower().endswith("buy")
                and str(order.status).lower() in active_statuses
            ):
                return True

        return False
    except Exception as e:
        print(f"ORDER CHECK ERROR | {symbol} | {e}")
        # Fail closed: if we cannot verify orders, do not send a new BUY.
        return True


def write_trade(
    timestamp,
    symbol,
    side,
    quantity,
    price,
    order_id,
    pnl,
    ml_probability,
    score,
    news_score,
    reason,
):
    exists = TRADE_FILE.exists()

    with TRADE_FILE.open("a", encoding="utf-8") as file:
        if not exists:
            file.write(
                "timestamp,symbol,side,qty,price,order_id,"
                "pnl,ml_prob,score,news_score,reason\n"
            )

        file.write(
            f"{timestamp},{symbol},{side},{quantity},{price},"
            f"{order_id},{pnl},{ml_probability},{score},"
            f"{news_score},{reason}\n"
        )


# ------------------------------------------------------------
# SIGNAL / ORDER
# ------------------------------------------------------------

def calculate_trade_score(probability, row, news_score):
    market_return = float(row.get("market_ret_3", 0.0))

    market_component = 1.0 if market_return > 0 else 0.0

    ml_component = max(
        0.0,
        min(1.0, (probability - 0.5) / 0.5)
    )

    news_component = (news_score + 1.0) / 2.0

    score = 100.0 * (
        float(CFG["ml_weight"]) * ml_component
        + float(CFG["market_weight"]) * market_component
        + float(CFG["news_weight"]) * news_component
    )

    return score


def submit_paper_long(symbol, reference_price, probability, score, news_score, row):
    state = get_daily_state()

    if state["trades"] >= int(CFG["max_trades_per_day"]):
        return

    reached, equity, start_equity = daily_loss_reached(state)

    if reached:
        print(
            f"RISK STOP | daily loss limit reached | "
            f"equity={equity:.2f} | start={start_equity:.2f}"
        )
        return

    if symbol in get_positions():
        return

    # Use a fresh market price for actual order pricing.
    # The 5-minute row is still used for ML features and ATR.
    signal_price = reference_price
    latest_price = get_latest_trade_price(symbol)

    if latest_price is None:
        print(
            f"ORDER BLOCKED | {symbol} | "
            f"latest trade price unavailable"
        )
        return

    reference_price = latest_price

    print(
        f"PRICE SYNC | {symbol} | "
        f"signal={signal_price:.4f} | "
        f"latest={reference_price:.4f} | "
        f"diff_pct={(reference_price / signal_price - 1.0) * 100:+.2f}%"
    )

    # Risk budget.
    risk_dollars = equity * float(CFG["risk_per_trade_pct"])

    # ATR-based stop distance.
    # ATR comes from the signal bar, but SL/TP are anchored
    # to the fresh market price.
    stop_distance = max(
        float(row["atr"]) * float(CFG["stop_atr_multiplier"]),
        reference_price * 0.003
    )

    quantity_by_risk = int(
        risk_dollars / stop_distance
    )

    # Hard position-size cap.
    max_position_value = equity * float(CFG["max_position_pct"])

    quantity_by_cap = int(
        max_position_value / reference_price
    )

    quantity = min(
        quantity_by_risk,
        quantity_by_cap
    )

    if quantity < 1:
        print(
            f"{symbol} | ei kauppaa | laskettu määrä < 1"
        )
        return

    stop_price = round(
        reference_price - stop_distance,
        2
    )

    target_price = round(
        reference_price
        + stop_distance * float(CFG["reward_risk_ratio"]),
        2
    )

    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=quantity,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(
            limit_price=target_price
        ),
        stop_loss=StopLossRequest(
            stop_price=stop_price
        ),
    )

    # Prevent duplicate BUY orders when the position has not appeared yet.
    if symbol in get_positions() or has_open_buy_order(symbol):
        print(
            f"ORDER BLOCKED | {symbol} | "
            f"existing position or active BUY order"
        )
        return

    print(f"ORDER DEBUG | {symbol} | ref={reference_price:.4f} | ATR={float(row['atr']):.4f} | stop_dist={stop_distance:.4f} | SL={stop_price:.2f} | TP={target_price:.2f}")
    order = trading.submit_order(order_request)

    state["trades"] += 1
    save_state(state)

    reason = (
        f"ML={probability:.3f}; "
        f"SCORE={score:.1f}; "
        f"NEWS={news_score:+.3f}; "
        f"SL={stop_price}; "
        f"TP={target_price}"
    )

    write_trade(
        datetime.now(timezone.utc).isoformat(),
        symbol,
        "BUY",
        quantity,
        reference_price,
        str(order.id),
        0,
        probability,
        score,
        news_score,
        reason,
    )

    print(
        f"ORDER | PAPER BUY {symbol} x{quantity} | "
        f"ref={reference_price:.2f} | "
        f"SL={stop_price:.2f} | "
        f"TP={target_price:.2f} | "
        f"order_id={order.id}"
    )


def flatten_end_of_day():
    try:
        current_positions = get_positions()

        if current_positions:
            trading.close_all_positions(cancel_orders=True)
            print("EOD | kaikki paper-positiot suljettu.")

    except Exception as exc:
        print("EOD flatten error:", repr(exc))



# ------------------------------------------------------------
# TRADE EXECUTION NOTIFICATIONS
# ------------------------------------------------------------

notified_order_ids = load_notified_order_ids()
order_monitor_started_at = datetime.now(timezone.utc)


def _order_value(order):
    try:
        qty = float(order.filled_qty or order.qty or 0)
        price = float(order.filled_avg_price or 0)
        return qty * price
    except Exception:
        return 0.0


def _bracket_prices(order):
    stop_price = None
    target_price = None

    try:
        for leg in (order.legs or []):
            if getattr(leg, "stop_price", None) is not None:
                stop_price = float(leg.stop_price)

            if getattr(leg, "limit_price", None) is not None:
                target_price = float(leg.limit_price)
    except Exception:
        pass

    return stop_price, target_price


def _send_buy_fill_notification(order):
    symbol = str(order.symbol)
    qty = float(order.filled_qty or order.qty or 0)
    price = float(order.filled_avg_price or 0)
    value = qty * price

    stop_price, target_price = _bracket_prices(order)

    lines = [
        "\U0001F7E2 OSTO TOTEUTETTU",
        "",
        symbol,
        f"M\u00e4\u00e4r\u00e4: {qty:g} kpl",
        f"Hinta: ${price:.2f}",
        f"Arvo: ${value:,.2f}",
    ]

    if stop_price is not None:
        lines.append(f"\U0001F6D1 Stop: ${stop_price:.2f}")

    if target_price is not None:
        lines.append(f"\U0001F3AF Target: ${target_price:.2f}")

    try:
        entry_file = BASE / "paper_order_entries.json"
        entries = {}

        if entry_file.exists():
            entries = json.loads(
                entry_file.read_text(encoding="utf-8")
            )

        entries[symbol] = price

        entry_file.write_text(
            json.dumps(entries, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        print("ORDER ENTRY SAVE ERROR:", repr(exc))

    return telegram_send("\n".join(lines))


def _send_sell_fill_notification(order, entry_price=None):
    symbol = str(order.symbol)
    qty = float(order.filled_qty or order.qty or 0)
    price = float(order.filled_avg_price or 0)

    reason = "MYYNTI"
    icon = "\U0001F534"

    try:
        order_type = str(order.order_type).lower()

        if "limit" in order_type:
            reason = "TAKE PROFIT"
            icon = "\U0001F3AF"
        elif "stop" in order_type:
            reason = "STOP LOSS"
            icon = "\U0001F6D1"
    except Exception:
        pass

    lines = [
        f"{icon} MYYNTI TOTEUTETTU",
        "",
        symbol,
        f"M\u00e4\u00e4r\u00e4: {qty:g} kpl",
        f"Exit: ${price:.2f}",
        "",
        f"{icon} {reason}",
    ]

    if entry_price is not None:
        pnl = (price - entry_price) * qty
        sign = "+" if pnl >= 0 else "-"
        lines.append(f"P/L: {sign}${abs(pnl):,.2f}")

    return telegram_send("\n".join(lines))


def check_filled_orders():
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        orders = trading.get_orders(
            filter=GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                limit=100,
                nested=True,
            )
        )

        for order in orders:
            order_id = str(order.id)

            if order_id in notified_order_ids:
                continue

            status = str(order.status).lower()

            if "filled" not in status:
                continue
            filled_at = getattr(order, "filled_at", None)

            if filled_at is not None and filled_at < order_monitor_started_at:
                continue

            side = str(order.side).lower()

            if "buy" in side:
                sent = _send_buy_fill_notification(order)

            elif "sell" in side:
                entry_price = None

                try:
                    entry_file = BASE / "paper_order_entries.json"

                    if entry_file.exists():
                        entries = json.loads(
                            entry_file.read_text(encoding="utf-8")
                        )
                        entry_price = entries.get(str(order.symbol))
                        if entry_price is not None:
                            entry_price = float(entry_price)
                except Exception:
                    entry_price = None

                sent = _send_sell_fill_notification(
                    order,
                    entry_price=entry_price,
                )

            else:
                continue

            if not sent:
                continue

            # Poistetaan SELL-entry vasta onnistuneen Telegram-ilmoituksen jälkeen.
            if "sell" in side:
                try:
                    entry_file = BASE / "paper_order_entries.json"

                    if entry_file.exists():
                        entries = json.loads(
                            entry_file.read_text(encoding="utf-8")
                        )
                        entries.pop(str(order.symbol), None)
                        entry_file.write_text(
                            json.dumps(entries, indent=2),
                            encoding="utf-8",
                        )
                except Exception:
                    pass

            notified_order_ids.add(order_id)
            save_notified_order_ids(notified_order_ids)

    except Exception as exc:
        print("ORDER NOTIFICATION ERROR:", repr(exc))


# ------------------------------------------------------------
# MAIN MARKET LOOP
# ------------------------------------------------------------

def market_loop():
    print()
    print("=" * 65)
    print("V5.2 INTEGRATED PAPER TRADER")
    print("PAPER ONLY | REALTIME NEWS | ML | 5-MIN BARS | ORDERS ENABLED")
    print("=" * 65)
    print("Symbols:", ", ".join(SYMBOLS))
    print()

    while True:
        try:
            clock = trading.get_clock()

            if not clock.is_open:
                print(
                    f"{datetime.now():%H:%M:%S} | Market closed | "
                    f"Next open: {clock.next_open}"
                )
                time.sleep(60)
                continue

            now_et = pd.Timestamp.now(
                tz="America/New_York"
            )

            # Close positions near the end of the regular session.
            if now_et.hour == 15 and now_et.minute >= 55:
                flatten_end_of_day()
                time.sleep(300)
                continue

            check_filled_orders()

            state = get_daily_state()

            reached, equity, start_equity = daily_loss_reached(state)

            if reached:
                print(
                    f"{datetime.now():%H:%M:%S} | "
                    f"DAILY RISK STOP | "
                    f"equity={equity:.2f} | "
                    f"start={start_equity:.2f}"
                )
                time.sleep(300)
                continue

            current_positions = get_positions()

            for symbol in SYMBOLS:
                if symbol in current_positions:
                    continue

                probability, row = get_ml_probability(symbol)

                if probability is None:
                    continue

                recent_news, news_change = get_news_signal(symbol)

                trade_score = calculate_trade_score(
                    probability,
                    row,
                    recent_news
                )

                print(
                    f"{datetime.now():%H:%M:%S} | "
                    f"{symbol} | "
                    f"close={float(row.close):.2f} | "
                    f"ML={probability:.3f} | "
                    f"NEWS={recent_news:+.3f} | "
                    f"SCORE={trade_score:.1f}"
                )


                signal_decision = (
                    "BUY"
                    if (
                        telegram_trading_enabled
                        and probability >= float(CFG["probability_threshold"])
                        and trade_score >= float(CFG["trade_score_threshold"])
                    )
                    else "NO TRADE"
                )

                with telegram_lock:
                    telegram_signals.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "symbol": symbol,
                        "ml": float(probability),
                        "score": float(trade_score),
                        "news": float(recent_news),
                        "decision": signal_decision,
                    })

                    if len(telegram_signals) > 100:
                        del telegram_signals[:-100]
                if (
                    probability
                    >= float(CFG["probability_threshold"])
                    and trade_score
                    >= float(CFG["trade_score_threshold"])
                ):
                    submit_paper_long(
                        symbol,
                        float(row.close),
                        probability,
                        trade_score,
                        recent_news,
                        row,
                    )

            # Avoid repeatedly processing the same completed bar.
            time.sleep(300)

        except KeyboardInterrupt:
            print()
            print("V5.3 paper-botti pysäytetty.")
            break

        except Exception as exc:
            print(
                "RUNNER ERROR:",
                repr(exc)
            )
            time.sleep(30)


if __name__ == "__main__":
    news_items = load_news_history()

    news_thread = threading.Thread(
        target=news_stream_loop,
        daemon=True
    )
    news_thread.start()

    telegram_thread = threading.Thread(target=telegram_loop, daemon=True)
    telegram_thread.start()

    market_loop()
