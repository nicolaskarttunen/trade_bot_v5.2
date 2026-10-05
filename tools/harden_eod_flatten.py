from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOT = ROOT / "paper_trader_integrated.py"

NEW_FLATTEN = r'''
# BROKER-CLOCK EOD FLATTEN
eod_critical_alert_date = None


def _send_eod_critical_once(message):
    """Send at most one EOD critical Telegram alert per NY trading date."""
    global eod_critical_alert_date

    today = pd.Timestamp.now(
        tz="America/New_York"
    ).date().isoformat()

    if eod_critical_alert_date == today:
        return

    eod_critical_alert_date = today

    try:
        telegram_send(message)
    except Exception as exc:
        print("EOD TELEGRAM ERROR:", repr(exc))


def flatten_end_of_day(seconds_to_close=None):
    """
    Repeated-safe end-of-day flatten.

    Uses a strict Alpaca position lookup so API failures never look like an
    empty portfolio. The market loop calls this repeatedly during the last
    10 minutes of the broker-reported session, including early-close days.
    """
    try:
        positions = list(trading.get_all_positions())
    except Exception as exc:
        print("EOD POSITION CHECK ERROR:", repr(exc))

        if seconds_to_close is not None and seconds_to_close <= 60:
            _send_eod_critical_once(
                "🚨 EOD CRITICAL\n\n"
                "Position check failed inside the final minute. "
                "Verify Alpaca positions manually."
            )

        return False

    if not positions:
        print("EOD VERIFIED | portfolio flat")
        return True

    symbols = [str(position.symbol) for position in positions]

    print(
        f"EOD FLATTEN | remaining={len(symbols)} | "
        f"symbols={','.join(symbols)} | "
        f"seconds_to_close={seconds_to_close}"
    )

    try:
        # Cancel bracket/OCO exits and submit market closes for every
        # remaining position. This is safe to call again if verification
        # shows that a position did not close on the first attempt.
        trading.close_all_positions(cancel_orders=True)
    except Exception as exc:
        print("EOD CLOSE ERROR:", repr(exc))

        if seconds_to_close is not None and seconds_to_close <= 60:
            _send_eod_critical_once(
                "🚨 EOD CRITICAL\n\n"
                f"Close request failed with {len(symbols)} position(s) "
                "remaining. Verify Alpaca manually."
            )

        return False

    # Give market orders a short moment to fill, then verify from Alpaca.
    time.sleep(3)

    try:
        remaining = list(trading.get_all_positions())
    except Exception as exc:
        print("EOD VERIFY ERROR:", repr(exc))

        if seconds_to_close is not None and seconds_to_close <= 60:
            _send_eod_critical_once(
                "🚨 EOD CRITICAL\n\n"
                "Could not verify that end-of-day closes filled. "
                "Check Alpaca manually."
            )

        return False

    if not remaining:
        print("EOD VERIFIED | all paper positions closed")
        return True

    remaining_symbols = [str(position.symbol) for position in remaining]

    print(
        f"EOD RETRY NEEDED | remaining={len(remaining_symbols)} | "
        f"symbols={','.join(remaining_symbols)}"
    )

    if seconds_to_close is not None and seconds_to_close <= 60:
        _send_eod_critical_once(
            "🚨 EOD CRITICAL\n\n"
            f"{len(remaining_symbols)} position(s) still open inside the "
            f"final minute: {', '.join(remaining_symbols)}"
        )

    return False
'''

NEW_MARKET_BLOCK = r'''            # Start flattening 10 minutes before Alpaca's scheduled close.
            # Using broker clock handles normal sessions and early-close days.
            close_et = pd.Timestamp(clock.next_close)

            if close_et.tzinfo is None:
                close_et = close_et.tz_localize("UTC")

            close_et = close_et.tz_convert("America/New_York")
            seconds_to_close = max(
                0.0,
                (close_et - now_et).total_seconds(),
            )

            if 0 < seconds_to_close <= 600:
                is_flat = flatten_end_of_day(
                    seconds_to_close=seconds_to_close
                )

                # Retry quickly when anything is still open; once flat,
                # keep verifying periodically until the session closes.
                time.sleep(30 if is_flat else 10)
                continue
'''


def patch_bot(text: str) -> str:
    if "# BROKER-CLOCK EOD FLATTEN" in text:
        print("EOD HARDENING ALREADY APPLIED")
        return text

    flatten_start = text.index("def flatten_end_of_day():")
    flatten_end_marker = (
        "\n\n# ------------------------------------------------------------\n"
        "# TRADE EXECUTION NOTIFICATIONS"
    )
    flatten_end = text.index(flatten_end_marker, flatten_start)

    text = (
        text[:flatten_start]
        + NEW_FLATTEN.rstrip()
        + text[flatten_end:]
    )

    old_market_block = '''            # Close positions near the end of the regular session.\n            if now_et.hour == 15 and now_et.minute >= 55:\n                flatten_end_of_day()\n                time.sleep(300)\n                continue\n'''

    if old_market_block not in text:
        raise RuntimeError("could not locate old EOD market-loop block")

    text = text.replace(old_market_block, NEW_MARKET_BLOCK, 1)
    return text


def main() -> None:
    if not BOT.exists():
        raise SystemExit("Run this script from the trade bot repository.")

    old_text = BOT.read_text(encoding="utf-8")
    new_text = patch_bot(old_text)

    if new_text == old_text:
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BOT.with_name(
        f"paper_trader_integrated_BEFORE_EOD_HARDEN_{stamp}.py"
    )
    shutil.copy2(BOT, backup)
    BOT.write_text(new_text, encoding="utf-8")

    print("EOD FLATTEN HARDENED")
    print(f"Backup: {backup.name}")
    print("Behavior: broker-clock close, retry+verify, early-close safe")
    print("Run: python3.14 -m py_compile paper_trader_integrated.py")


if __name__ == "__main__":
    main()
