"""Paper order lifecycle with an injected broker. Tested without network IO.

No automatic resend, protective-order retry or emergency market SELL after an
ambiguous response. Unknown state blocks new entries until a matching broker
snapshot is observed. Opening partials: cancel remainder, observe terminal
parent, compare position, then protect only the final filled quantity.
"""
from decimal import Decimal
from order_journal import (OrderBlocked, SubmissionUnknown, client_id, enum_value,
                           decimal_value, TERMINAL)


def field(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def snapshot(order):
    out = {name: str(field(order, name)) if field(order, name) is not None else None
           for name in ["id", "client_order_id", "symbol", "qty", "filled_qty", "filled_avg_price", "stop_price", "limit_price"]}
    for name in ["status", "side", "order_class"]:
        out[name] = enum_value(field(order, name))
    out["order_type"] = enum_value(field(order, "order_type", field(order, "type")))
    out["legs"] = [snapshot(leg) for leg in (field(order, "legs", []) or [])]
    return out


def family(order):
    result = {}
    def add(node):
        if node.get("id"):
            result[node["id"]] = node
        for child in node.get("legs", []):
            add(child)
    add(order)
    return list(result.values())


def exit_state(entry, protective, positions, expected_stop):
    """Require exact position/fill agreement and a visible active stop.

    This is a conservative snapshot check, not a guarantee against later broker
    races, slippage or both exit legs filling. Such mismatches block entries.
    """
    if not protective:
        return False, "Protective order not yet identified"
    exits = [r for r in family(protective) if r.get("side") == "sell"]
    if any(r.get("symbol") != entry["symbol"] for r in exits):
        return False, "Exit symbol mismatch"
    if not exits or any(r.get("status") not in TERMINAL | {"new", "accepted", "partially_filled", "held"} for r in exits):
        return False, "Missing/unsupported exit state"
    sold = sum((decimal_value(r.get("filled_qty")) for r in exits), Decimal(0))
    filled = decimal_value(entry["filled_qty"])
    if sold > filled:
        return False, "Exit fills exceed bought quantity; possible short position"
    actual = decimal_value(field(positions.get(entry["symbol"]), "qty", 0))
    expected = filled-sold
    if actual != expected:
        return False, "Position and cumulative fills disagree"
    if not actual:
        if any(r["status"] not in TERMINAL for r in exits):
            return False, "Flat position still has active exits"
        return True, "Closed and reconciled"
    if any(r["status"] not in TERMINAL and
           decimal_value(r["qty"])-decimal_value(r["filled_qty"]) > actual for r in exits):
        return False, "Active exit quantity exceeds observed position"
    for leg in exits:
        if (leg["order_type"] == "stop" and leg["status"] in {"new", "accepted", "partially_filled"}
                and decimal_value(leg["qty"])-decimal_value(leg["filled_qty"]) == actual
                and decimal_value(leg.get("stop_price"), positive=True) == decimal_value(expected_stop, positive=True)):
            return True, "Active stop covers observed position"
    return False, "No verified active stop for remaining quantity"


class OrderLifecycle:
    def __init__(self, journal, broker, lookup=None, open_orders=None):
        self.journal, self.broker = journal, broker
        self.lookup = lookup or broker.get_order_by_client_id
        self.open_orders = open_orders or broker.get_orders

    def submit_once(self, session, symbol, role, request_factory, reference_price, metadata=None):
        cid = client_id(self.journal.account_id, session, symbol, role)
        request = request_factory(cid)
        payload = request.model_dump(mode="json", exclude_none=True) if hasattr(request, "model_dump") else dict(request)
        self.journal.reserve(session, symbol, role, payload, reference_price, metadata)
        # The intent is durable BEFORE making the only submission attempt.
        try:
            order = self.broker.submit_order(request)
            self.journal.observe(cid, snapshot(order))
        except Exception as exc:
            self.journal.uncertain(cid, "Submission/acknowledgement failed: " + type(exc).__name__)
            raise SubmissionUnknown(f"Resolve {cid} by client_order_id; do not resend") from exc
        return order

    def reconcile(self):
        """Lookup each owned ID directly; 404/outage cannot mean 'not sent'."""
        for row in self.journal.rows():
            if row["settled"]:
                continue
            # Retain history forever, but terminal zero-fill entries need no
            # recurring lookup. Filled entries/exits still need their leg state.
            if row["role"] == "entry" and row["state"] == "observed" and row["status"] in TERMINAL and decimal_value(row["filled_qty"]) == 0:
                continue
            try:
                self.journal.observe(row["client_id"], snapshot(self.lookup(row["client_id"])))
            except Exception as exc:
                self.journal.uncertain(row["client_id"], "Reconciliation failed: " + type(exc).__name__)
        return self.journal.rows()

    def _positions(self):
        rows = self.broker.get_all_positions()
        result = {}
        for row in rows:
            symbol = str(field(row, "symbol"))
            if symbol in result:
                raise OrderBlocked("Duplicate position symbol")
            decimal_value(field(row, "qty"))  # negative/unknown holdings fail closed
            result[symbol] = row
        return result

    def _request_partial_cancel(self, row):
        if self.journal.claim_cancel(row["client_id"]):
            try:
                self.broker.cancel_order_by_id(row["broker_id"])
            except Exception as exc:
                # A second cancel is unnecessary. Poll the parent until its
                # final quantity is known; a cancel response is not a fill report.
                self.journal.uncertain(row["client_id"], "Cancel acknowledgement unknown: " + type(exc).__name__)

    def maintain_opening_protection(self, request_factory):
        """Called independently of notifications on every maintenance poll.

        request_factory(cid, symbol, qty, stop, target) constructs an OCO SELL.
        Returns human-readable blockers. It never market-sells on OCO timeout.
        """
        blockers = []
        positions = self._positions()
        open_orders = [snapshot(o) for o in self.open_orders()]
        for entry in self.journal.rows():
            if entry["settled"] or entry["role"] != "entry" or entry["state"] != "observed":
                continue
            filled = decimal_value(entry["filled_qty"])
            if not filled:
                continue
            if filled != filled.to_integral_value():
                blockers.append("Fractional fill requires manual reconciliation")
                continue
            if entry["status"] not in TERMINAL:
                self._request_partial_cancel(entry)
                blockers.append(f"{entry['symbol']}: partial BUY; waiting for terminal cancellation/fill")
                continue
            if entry["metadata"].get("mode") != "opening":
                # A canceled partial bracket can have hidden/inactive children.
                # Do not attach a competing OCO without broker-specific proof.
                continue
            protection_id = client_id(self.journal.account_id, entry["session"], entry["symbol"], "protection")
            protection = self.journal.get(protection_id)
            if protection:
                # Never retry protection or create an emergency SELL merely
                # because the original request returned an ambiguous response.
                continue
            actual = decimal_value(field(positions.get(entry["symbol"]), "qty", 0))
            if actual != filled:
                blockers.append(f"{entry['symbol']}: cannot protect; position != final filled quantity")
                continue
            active_sells = [r for o in open_orders for r in family(o)
                            if r["symbol"] == entry["symbol"] and r["side"] == "sell" and r["status"] not in TERMINAL]
            if active_sells:
                blockers.append(f"{entry['symbol']}: existing SELL orders must be reconciled first")
                continue
            # Only the immutable planned distance is reused; anchor to the
            # actual final weighted fill price, including late partial fills.
            avg = decimal_value(entry["filled_avg_price"], positive=True)
            dist = decimal_value(entry["metadata"]["stop_distance"], positive=True)
            rr = decimal_value(entry["metadata"]["reward_risk_ratio"], positive=True)
            stop, target = round(avg-dist, 2), round(avg+dist*rr, 2)
            if stop <= 0 or target <= avg:
                blockers.append(f"{entry['symbol']}: invalid protection prices")
                continue
            try:
                self.submit_once(entry["session"], entry["symbol"], "protection",
                    lambda cid: request_factory(cid, entry["symbol"], int(filled), float(stop), float(target)),
                    float(avg), {"entry_client_id": entry["client_id"]})
            except OrderBlocked as exc:
                blockers.append(str(exc))
        return blockers

    def entry_blockers(self, session):
        rows = self.journal.rows()
        positions = self._positions()
        blockers = []
        owned_symbols = {r["symbol"] for r in rows if r["role"] == "entry" and not r["settled"] and decimal_value(r["filled_qty"]) > 0}
        if set(positions)-owned_symbols:
            blockers.append("Unowned positions require manual reconciliation")
        owned_ids = {o["id"] for r in rows if r["snapshot"] for o in family(r["snapshot"])}
        for order in self.open_orders():
            if any(o["id"] not in owned_ids for o in family(snapshot(order))):
                blockers.append("Unowned open orders require manual reconciliation")
        for row in rows:
            if row["settled"]:
                continue
            if row["state"] != "observed":
                blockers.append(f"Unresolved {row['role']}: {row['client_id']}")
                continue
            if row["role"] != "entry":
                continue
            filled = decimal_value(row["filled_qty"])
            if row["status"] not in TERMINAL and row["session"] != session:
                blockers.append("Unresolved BUY from a previous session")
            if filled and row["status"] not in TERMINAL:
                blockers.append(f"{row['symbol']}: partial BUY still active")
            elif filled:
                if row["metadata"].get("mode") == "opening":
                    protective = self.journal.get(client_id(self.journal.account_id, row["session"], row["symbol"], "protection"))
                    snap = protective["snapshot"] if protective and protective["state"] == "observed" else None
                    request = protective["request"] if protective else {}
                else:
                    snap = row["snapshot"]
                    request = row["request"]
                ok, reason = exit_state(row, snap, positions, request.get("stop_loss", {}).get("stop_price"))
                if ok and reason == "Closed and reconciled":
                    self.journal.settle(row["client_id"])
                    if row["metadata"].get("mode") == "opening":
                        self.journal.settle(protective["client_id"])
                if not ok:
                    blockers.append(f"{row['symbol']}: {reason}")
        return blockers

    def reserve_exposure(self, session, exposure_by_symbol):
        """Conservative per-symbol maximum avoids position/pending double count.

        An acknowledged BUY missing from the broker's list still reserves its
        full notional. Reservations remain for the session; release requires a
        future fill/exit accounting implementation, not a guessed flat state.
        """
        result = dict(exposure_by_symbol)
        for row in self.journal.rows():
            if row["role"] != "entry" or row["session"] != session:
                continue
            if row["state"] == "observed" and row["status"] in TERMINAL and decimal_value(row["filled_qty"]) == 0:
                continue
            notional = float(decimal_value(row["request"]["qty"])*decimal_value(row["reference_price"]))
            result[row["symbol"]] = max(result.get(row["symbol"], 0.), notional)
        return result
