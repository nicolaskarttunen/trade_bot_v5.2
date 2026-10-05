from pathlib import Path

BOT = Path(__file__).resolve().parents[1] / "paper_trader_integrated.py"


def main():
    text = BOT.read_text(encoding="utf-8")

    old_active = '''def _active_buy_orders():
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
'''

    new_active = '''def _enum_text(value):
    return str(getattr(value, "value", value)).lower()


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

    try:
        orders = trading.get_orders()
    except Exception as exc:
        raise RuntimeError(f"active order lookup failed: {exc!r}") from exc

    return [
        order
        for order in orders
        if _enum_text(order.side) == "buy"
        and _enum_text(order.status) in active_statuses
    ]
'''

    old_positions = '''    positions = get_positions()
    occupied = set(positions)
    exposure = 0.0
'''

    new_positions = '''    try:
        positions = {
            position.symbol: position
            for position in trading.get_all_positions()
        }
    except Exception as exc:
        raise RuntimeError(f"position lookup failed: {exc!r}") from exc

    occupied = set(positions)
    exposure = 0.0
'''

    changed = False

    if old_active in text:
        text = text.replace(old_active, new_active, 1)
        changed = True
    elif "def _enum_text(value):" not in text:
        raise RuntimeError("Could not find _active_buy_orders block")

    if old_positions in text:
        text = text.replace(old_positions, new_positions, 1)
        changed = True
    elif "position lookup failed" not in text:
        raise RuntimeError("Could not find portfolio position lookup block")

    if changed:
        BOT.write_text(text, encoding="utf-8")
        print("PORTFOLIO GUARD HARDENED")
    else:
        print("PORTFOLIO GUARD ALREADY HARDENED")


if __name__ == "__main__":
    main()
