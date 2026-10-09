from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "paper_trader_integrated.py"

text = TARGET.read_text(encoding="utf-8")

old = '''                    state = get_daily_state()
                    state["trades"] += 1
                    traded_symbols = state.setdefault("traded_symbols", [])
                    if symbol not in traded_symbols:
                        traded_symbols.append(symbol)
                    save_state(state)
'''

new = '''                    state = get_daily_state()
                    state["trades"] += 1
                    traded_symbols = state.setdefault("traded_symbols", [])
                    filled_symbol = str(order.symbol)
                    if filled_symbol not in traded_symbols:
                        traded_symbols.append(filled_symbol)
                    save_state(state)
'''

if old not in text:
    raise SystemExit("Expected opening trade-state block not found; no changes made.")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = TARGET.with_name(f"paper_trader_integrated_BEFORE_OPENING_STATE_FIX_{stamp}.py")
backup.write_text(text, encoding="utf-8")

updated = text.replace(old, new, 1)
TARGET.write_text(updated, encoding="utf-8")

print("OPENING TRADE STATE FIX APPLIED")
print(f"Backup: {backup.name}")
print("Run: python3.14 -m py_compile paper_trader_integrated.py")
