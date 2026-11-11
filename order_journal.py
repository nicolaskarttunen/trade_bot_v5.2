"""Durable, account-bound intents. No SDK, network, credentials or order IO.

A committed PREPARED row consumes the right to send exactly once. Recovery is
read-only; neither a timeout nor a 404 grants permission to resend. SQLite
transactions serialize reservations, and event snapshots retain cumulative
fills. This journal must not be deleted/reset to unblock trading.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import sqlite3


TERMINAL = {"filled", "canceled", "expired", "rejected"}
ACTIVE = {"new", "accepted", "pending_new", "partially_filled", "pending_cancel",
          "pending_replace", "accepted_for_bidding", "held", "stopped", "suspended", "calculated"}


class OrderBlocked(RuntimeError):
    pass


class SubmissionUnknown(OrderBlocked):
    pass


def enum_value(value):
    return str(getattr(value, "value", value)).lower()


def decimal_value(value, *, positive=False):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise OrderBlocked("Invalid numeric broker/order value") from None
    if not result.is_finite() or result < 0 or (positive and result <= 0):
        raise OrderBlocked("Invalid numeric broker/order value")
    return result


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def client_id(account_id, session, symbol, role):
    key = canonical([account_id, session, symbol, role])
    return "audit1-" + hashlib.sha256(key.encode()).hexdigest()[:36]


def now():
    return datetime.now(timezone.utc).isoformat()


class OrderJournal:
    def __init__(self, path, account_id):
        self.path = Path(path)
        self.account_id = str(account_id)
        if account_id is None or not self.account_id:
            raise OrderBlocked("Paper account id required")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS intents (
                    client_id TEXT PRIMARY KEY, session TEXT NOT NULL, symbol TEXT NOT NULL,
                    role TEXT NOT NULL, request TEXT NOT NULL, metadata TEXT NOT NULL,
                    reference_price TEXT NOT NULL, state TEXT NOT NULL, broker_id TEXT,
                    settled INTEGER NOT NULL DEFAULT 0, status TEXT, filled_qty TEXT NOT NULL DEFAULT '0', filled_avg_price TEXT,
                    snapshot TEXT, error TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    UNIQUE(session, symbol, role));
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY, client_id TEXT NOT NULL,
                    observed_at TEXT NOT NULL, event TEXT NOT NULL, payload TEXT NOT NULL);
            ''')
            db.execute("BEGIN IMMEDIATE")
            for key, value in [("schema_version", "1"), ("account_id", self.account_id)]:
                old = db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
                if old and old[0] != value:
                    raise OrderBlocked("Journal schema/account mismatch")
                db.execute("INSERT OR IGNORE INTO metadata VALUES (?,?)", (key, value))

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA synchronous=FULL")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _decode(row):
        if row is None:
            return None
        value = dict(row)
        for field in ["request", "metadata", "snapshot"]:
            value[field] = json.loads(value[field]) if value[field] else None
        return value

    def get(self, cid):
        with self.connection() as db:
            return self._decode(db.execute("SELECT * FROM intents WHERE client_id=?", (cid,)).fetchone())

    def rows(self):
        with self.connection() as db:
            return [self._decode(r) for r in db.execute("SELECT * FROM intents ORDER BY created_at, client_id")]

    def reserve(self, session, symbol, role, request, reference_price, metadata=None):
        cid = client_id(self.account_id, session, symbol, role)
        if request.get("client_order_id") != cid or request.get("symbol") != symbol:
            raise OrderBlocked("Intent/request identity mismatch")
        qty = decimal_value(request.get("qty"), positive=True)
        if qty != qty.to_integral_value():
            raise OrderBlocked("This lifecycle supports integer-share orders only")
        if role not in {"entry", "protection"} or request.get("side") != {"entry": "buy", "protection": "sell"}[role]:
            raise OrderBlocked("Invalid order side")
        ref = str(decimal_value(reference_price, positive=True))
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM intents WHERE client_id=?", (cid,)).fetchone():
                raise OrderBlocked("Intent already exists: reconcile; never resend")
            if role == "entry" and db.execute(
                    "SELECT 1 FROM intents WHERE state != 'observed' LIMIT 1").fetchone():
                raise OrderBlocked("An unresolved intent blocks all new BUYs")
            stamp = now()
            db.execute('''INSERT INTO intents
                (client_id,session,symbol,role,request,metadata,reference_price,state,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,'prepared',?,?)''',
                (cid, session, symbol, role, canonical(request), canonical(metadata or {}), ref, stamp, stamp))
            self._event(db, cid, "prepared", request)
        return cid

    @staticmethod
    def _event(db, cid, event, payload):
        db.execute("INSERT INTO events(client_id,observed_at,event,payload) VALUES (?,?,?,?)",
                   (cid, now(), event, canonical(payload)))

    def uncertain(self, cid, reason):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE intents SET state='unknown',error=?,updated_at=? WHERE client_id=?",
                       (str(reason), now(), cid))
            self._event(db, cid, "unknown", {"reason": str(reason)})

    def observe(self, cid, snapshot):
        """Reject mismatched/stale/unsupported snapshots instead of freeing risk."""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = self._decode(db.execute("SELECT * FROM intents WHERE client_id=?", (cid,)).fetchone())
            if old is None:
                raise OrderBlocked("Unowned broker order")
            req = old["request"]
            for field in ["client_order_id", "symbol", "side"]:
                if snapshot.get(field) != req.get(field):
                    raise OrderBlocked("Broker identity mismatch: " + field)
            if not snapshot.get("id") or (old["broker_id"] and snapshot["id"] != old["broker_id"]):
                raise OrderBlocked("Broker order id mismatch")
            for name, key in [("order_class", "order_class"), ("type", "order_type")]:
                if req.get(name) and snapshot.get(key) != req[name]:
                    raise OrderBlocked("Broker order contract mismatch: " + name)
            qty = decimal_value(snapshot.get("qty"), positive=True)
            filled = decimal_value(snapshot.get("filled_qty"))
            if qty != decimal_value(req["qty"]) or filled > qty or filled < decimal_value(old["filled_qty"]):
                raise OrderBlocked("Inconsistent cumulative fill quantity")
            status = snapshot.get("status")
            if status not in ACTIVE | TERMINAL:  # 'replaced' needs explicit successor reconciliation
                raise OrderBlocked("Unsupported broker status: " + str(status))
            if status == "filled" and filled != qty:
                raise OrderBlocked("Filled status without complete quantity")
            if old["status"] in TERMINAL and status != old["status"]:
                raise OrderBlocked("Terminal order status regressed/changed")
            avg = str(decimal_value(snapshot.get("filled_avg_price"), positive=True)) if filled else None
            encoded = canonical(snapshot)
            if encoded != canonical(old["snapshot"]):
                self._event(db, cid, "broker_snapshot", snapshot)
            db.execute('''UPDATE intents SET state='observed',broker_id=?,status=?,filled_qty=?,
                filled_avg_price=?,snapshot=?,error=NULL,updated_at=? WHERE client_id=?''',
                (snapshot["id"], status, str(filled), avg, encoded, now(), cid))
        return self.get(cid)

    def claim_cancel(self, cid):
        """Persist before cancellation RPC. Ack does NOT imply canceled status."""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT broker_id,cancel_requested FROM intents WHERE client_id=?", (cid,)).fetchone()
            if not row or not row[0]:
                raise OrderBlocked("Cannot cancel an unidentified order")
            if row[1]:
                return False
            db.execute("UPDATE intents SET cancel_requested=1,updated_at=? WHERE client_id=?", (now(), cid))
            self._event(db, cid, "cancel_requested", {})
        return True

    def settle(self, cid):
        """Retire only after terminal exits and a flat position were reconciled."""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE intents SET settled=1 WHERE client_id=?", (cid,))
            self._event(db, cid, "closed_reconciled", {})

    def daily_symbols(self, session):
        # Attempts remain consumed even after rejection: conservative daily cap.
        return {r["symbol"] for r in self.rows() if r["role"] == "entry" and r["session"] == session}


class ProcessLock:
    """One paper writer per working directory, Windows and POSIX."""
    def __init__(self, path):
        self.file = open(path, "a+b")
        try:
            # Reading the locked byte itself fails on Windows before locking().
            # Inspect file size without reading another writer's locked region.
            if os.fstat(self.file.fileno()).st_size == 0:
                self.file.write(b"0")
                self.file.flush()
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError):
            self.file.close()
            raise OrderBlocked("Cannot acquire paper writer lock: another process or filesystem error") from None

    def close(self):
        self.file.close()
