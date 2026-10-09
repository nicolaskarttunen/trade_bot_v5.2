#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
BOT = ROOT / "paper_trader_integrated.py"

text = BOT.read_text(encoding="utf-8")
original = text

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = ROOT / f"paper_trader_integrated_BEFORE_STALE_DATA_FIX_{stamp}.py"
backup.write_text(text, encoding="utf-8")

old_request = '''        limit=limit,\n        feed=CFG["feed"],\n'''
new_request = '''        # Request enough history that Alpaca cannot truncate us to\n        # the oldest 250 bars in the lookback window. We trim to the\n        # caller-requested number only after sorting by timestamp.\n        limit=max(int(limit), 1000),\n        feed=CFG["feed"],\n'''
if old_request not in text:
    raise SystemExit("Expected StockBarsRequest limit block not found")
text = text.replace(old_request, new_request, 1)

old_return = '''    if not all(column in df.columns for column in needed):\n        return pd.DataFrame()\n\n    return df[needed].copy()\n\n\ndef get_benchmark():\n'''
new_return = '''    if not all(column in df.columns for column in needed):\n        return pd.DataFrame()\n\n    clean = df[needed].copy()\n    clean["timestamp"] = pd.to_datetime(\n        clean["timestamp"],\n        utc=True,\n        errors="coerce",\n    )\n    clean = clean.dropna(subset=["timestamp"])\n    clean = clean.sort_values("timestamp")\n    clean = clean.drop_duplicates(\n        subset=["timestamp"],\n        keep="last",\n    )\n\n    # Always return the newest bars, never the oldest bars from the\n    # requested lookback window.\n    return clean.tail(int(limit)).reset_index(drop=True)\n\n\ndef is_fresh_intraday_row(symbol, row, max_age_minutes=12.0):\n    """Fail closed when an intraday ML signal is based on stale bars."""\n    try:\n        timestamp = pd.Timestamp(row["timestamp"])\n        if timestamp.tzinfo is None:\n            timestamp = timestamp.tz_localize("UTC")\n        else:\n            timestamp = timestamp.tz_convert("UTC")\n\n        now_utc = pd.Timestamp.now(tz="UTC")\n        age_minutes = (now_utc - timestamp).total_seconds() / 60.0\n\n        if age_minutes < -1.0 or age_minutes > float(max_age_minutes):\n            print(\n                f"STALE DATA BLOCK | {symbol} | "\n                f"bar={timestamp.isoformat()} | "\n                f"age_min={age_minutes:.1f}"\n            )\n            return False\n\n        return True\n\n    except Exception as exc:\n        print(\n            f"STALE DATA BLOCK | {symbol} | "\n            f"timestamp check failed | {exc!r}"\n        )\n        return False\n\n\ndef get_benchmark():\n'''
if old_return not in text:
    raise SystemExit("Expected get_bars return block not found")
text = text.replace(old_return, new_return, 1)

needle = '''                probability, row = get_ml_probability(symbol)\n\n                if probability is None:\n                    continue\n\n                recent_news, news_change = get_news_signal(symbol)\n'''
replacement = '''                probability, row = get_ml_probability(symbol)\n\n                if probability is None:\n                    continue\n\n                # Intraday entries must use a recent 5-minute bar.\n                # If Alpaca returns stale history, fail closed instead\n                # of placing an order from an old signal.\n                if not is_fresh_intraday_row(symbol, row):\n                    continue\n\n                recent_news, news_change = get_news_signal(symbol)\n'''

# There are two similar probability blocks: opening scan and intraday scan.
# This exact indentation/signature is used by both, so patch the LAST one only.
pos = text.rfind(needle)
if pos == -1:
    raise SystemExit("Expected intraday probability block not found")
text = text[:pos] + text[pos:].replace(needle, replacement, 1)

if text == original:
    raise SystemExit("No changes applied")

BOT.write_text(text, encoding="utf-8")
print("STALE INTRADAY DATA FIX APPLIED")
print(f"Backup: {backup.name}")
print("Safety: newest bars + 12-minute intraday freshness gate")
print("Run: python3.14 -m py_compile paper_trader_integrated.py")
