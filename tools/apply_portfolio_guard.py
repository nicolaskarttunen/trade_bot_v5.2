from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOT = ROOT / "paper_trader_integrated.py"
CFG = ROOT / "config.json"

HELPERS = r'''

def _active_buy_orders():
    """Return active BUY orders. Fail closed by raising on API errors."""
    active_statuses = {
        "new",
        "accepted",
        "pending_new",
        "partially_filled",
        "pending_replace",
        "accepted_for_bidding",
    }

    orders = trading.get_orders()
    return [
        order
        for order in orders
        if str(order.side).lower().endswith("buy")
        and str(order.status).lower() in active_statuses
    ]


def get_portfolio_entry_usage(opening_orders=None):
    """
    Return (occupied_symbols, gross_exposure_dollars).

    Exposure includes live positions, remaining quantity on active BUY
    orders, and opening orders already submitted in the current in-memory
    opening queue. Any API/pricing failure raises so callers can fail closed.
    """
    positions = get_positions()
    occupied = set(positions)
    exposure = 0.0

    for symbol, position in positions.items():
        market_value = float(getattr(position, "market_value", 0) or 0)
        if market_value == 0:
            qty = abs(float(getattr(position, "qty", 0) or 0))
            price = float(
                getattr(position, "current_price", 0)
                or getattr(position, "avg_entry_price", 0)
                or 0
            )
            market_value = qty * price
        exposure += abs(market_value)

    active_buy_symbols = set()

    for order in _active_buy_orders():
        symbol = str(order.symbol)
        active_buy_symbols.add(symbol)
        occupied.add(symbol)

        qty = float(getattr(order, "qty", 0) or 0)
        filled_qty = float(getattr(order, "filled_qty", 0) or 0)
        remaining_qty = max(0.0, qty - filled_qty)

        if remaining_qty <= 0:
            continue

        price = float(getattr(order, "limit_price", 0) or 0)
        if price <= 0:
            price = float(getattr(order, "filled_avg_price", 0) or 0)
        if price <= 0:
            latest = get_latest_trade_price(symbol)
            if latest is None or latest <= 0:
                raise RuntimeError(
                    f"cannot price active BUY reservation for {symbol}"
                )
            price = float(latest)

        exposure += remaining_qty * price

    # Alpaca can take a moment to expose a just-submitted market order.
    # Count submitted opening orders from our own queue as reservations too,
    # but avoid double counting symbols already visible through Alpaca.
    if isinstance(opening_orders, dict):
        for symbol, data in opening_orders.items():
            if data.get("status") != "submitted":
                continue
            if symbol in positions or symbol in active_buy_symbols:
                continue

            qty = float(data.get("quantity", 0) or 0)
            price = float(
                data.get("market_reference_price", 0)
                or data.get("reference_price", 0)
                or 0
            )

            if qty <= 0 or price <= 0:
                raise RuntimeError(
                    f"invalid opening reservation for {symbol}"
                )

            occupied.add(symbol)
            exposure += qty * price

    return occupied, exposure


def enforce_entry_limits(
    symbol,
    reference_price,
    requested_quantity,
    equity,
    opening_orders=None,
):
    """
    Enforce portfolio-wide entry limits immediately before each BUY.

    Hard limits:
      - max_open_positions unique position/pending-BUY symbols
      - max_total_exposure_pct gross long exposure including pending BUYs

    Returns the maximum allowed integer quantity. Failures block the entry.
    """
    try:
        requested_quantity = int(requested_quantity)
        reference_price = float(reference_price)
        equity = float(equity)

        if requested_quantity < 1 or reference_price <= 0 or equity <= 0:
            return 0

        occupied, exposure = get_portfolio_entry_usage(
            opening_orders=opening_orders
        )

        max_open = int(CFG.get("max_open_positions", 5))
        max_exposure_pct = float(
            CFG.get("max_total_exposure_pct", 0.50)
        )

        if symbol not in occupied and len(occupied) >= max_open:
            print(
                f"PORTFOLIO BLOCK | {symbol} | "
                f"open_or_pending={len(occupied)}/{max_open}"
            )
            return 0

        max_exposure = equity * max_exposure_pct
        remaining_exposure = max(0.0, max_exposure - exposure)
        qty_by_exposure = int(remaining_exposure / reference_price)
        allowed_quantity = min(requested_quantity, qty_by_exposure)

        if allowed_quantity < 1:
            print(
                f"PORTFOLIO BLOCK | {symbol} | "
                f"exposure=${exposure:,.2f}/${max_exposure:,.2f} | "
                f"open_or_pending={len(occupied)}/{max_open}"
            )
            return 0

        if allowed_quantity < requested_quantity:
            print(
                f"PORTFOLIO CAP | {symbol} | "
                f"qty={requested_quantity}->{allowed_quantity} | "
                f"exposure=${exposure:,.2f}/${max_exposure:,.2f}"
            )

        return allowed_quantity

    except Exception as exc:
        print(
            f"PORTFOLIO GUARD ERROR | {symbol} | {exc!r} | "
            f"entry blocked"
        )
        return 0
'''


def patch_bot(text: str) -> str:
    # Preserve the validated 1.25% minimum stop if this branch is older.
    text = text.replace(
        "reference_price * 0.003",
        "reference_price * 0.0125",
    )

    if "def enforce_entry_limits(" not in text:
        marker = "\ndef write_trade("
        if marker not in text:
            raise RuntimeError("could not locate write_trade insertion point")
        text = text.replace(marker, HELPERS + marker, 1)

    # Add portfolio guard to normal/intraday entries.
    submit_start = text.index("def submit_paper_long(")
    submit_end = text.index("\ndef flatten_end_of_day(", submit_start)
    submit = text[submit_start:submit_end]

    if "opening_orders=None," not in submit:
        pattern = re.compile(
            r"(\n    quantity = min\(\n"
            r"        quantity_by_risk,\n"
            r"        quantity_by_cap\n"
            r"    \)\n)"
        )
        replacement = r'''\1
    quantity = enforce_entry_limits(
        symbol=symbol,
        reference_price=reference_price,
        requested_quantity=quantity,
        equity=equity,
        opening_orders=None,
    )
'''
        submit, count = pattern.subn(replacement, submit, count=1)
        if count != 1:
            raise RuntimeError("could not patch submit_paper_long quantity guard")

    text = text[:submit_start] + submit + text[submit_end:]

    # Add portfolio guard to queued opening-market entries. Pass the in-memory
    # queue so earlier submissions in the same loop reserve slots/exposure even
    # if Alpaca has not surfaced them through get_orders() yet.
    opening_start = text.index("def submit_queued_opening_orders(")
    opening_end = text.index("\ndef submit_opening_oco(", opening_start)
    opening = text[opening_start:opening_end]

    if "opening_orders=opening_orders," not in opening:
        pattern = re.compile(
            r"(\n        quantity = min\(\n"
            r"            planned_quantity,\n"
            r"            quantity_by_cap,\n"
            r"        \)\n)"
        )
        replacement = r'''\1
        quantity = enforce_entry_limits(
            symbol=symbol,
            reference_price=latest_price,
            requested_quantity=quantity,
            equity=equity,
            opening_orders=opening_orders,
        )
'''
        opening, count = pattern.subn(replacement, opening, count=1)
        if count != 1:
            raise RuntimeError(
                "could not patch submit_queued_opening_orders quantity guard"
            )

    text = text[:opening_start] + opening + text[opening_end:]

    return text


def patch_config(data: dict) -> dict:
    data["max_open_positions"] = 5
    data["max_total_exposure_pct"] = 0.50
    return data


def main() -> None:
    if not BOT.exists() or not CFG.exists():
        raise SystemExit("Run this script from the trade_bot_v5.2 repository.")

    bot_text = BOT.read_text(encoding="utf-8")
    cfg_data = json.loads(CFG.read_text(encoding="utf-8"))

    new_bot = patch_bot(bot_text)
    new_cfg = patch_config(cfg_data)

    if new_bot == bot_text and new_cfg == cfg_data:
        print("PORTFOLIO GUARD ALREADY APPLIED")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BOT.with_name(f"paper_trader_integrated_BEFORE_PORTFOLIO_GUARD_{stamp}.py")
    shutil.copy2(BOT, backup)

    BOT.write_text(new_bot, encoding="utf-8")
    CFG.write_text(
        json.dumps(new_cfg, indent=2) + "\n",
        encoding="utf-8",
    )

    print("PORTFOLIO GUARD APPLIED")
    print(f"Backup: {backup.name}")
    print("Limits: max_open_positions=5, max_total_exposure_pct=0.50")
    print("Run: python -m py_compile paper_trader_integrated.py")


if __name__ == "__main__":
    main()
