"""Fixed development experiment; ordering is the only treatment."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pandas as pd

POLICY = "daily_sha256_v1"
CACHE = "replay_inputs_v1.zip"


def daily_order(symbols, day):
    # Independent of prices, scores, returns and RNG/library implementation.
    return sorted(symbols, key=lambda s: (hashlib.sha256(
        f"{POLICY}|{day}|{s}".encode("utf-8")).digest(), s))


def save_cache(path, symbols, scores, prices, gate, sessions):
    """Only August replay inputs; no model, training rows, secrets or pickle."""
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        meta = dict(version=1, symbols=symbols, sessions=sessions,
                    gate={d:bool(gate[d]) for d in sessions})
        archive.writestr("metadata.json", json.dumps(meta, allow_nan=False))
        for i,symbol in enumerate(symbols):
            for name,frames in [("scores",scores), ("prices",prices)]:
                with archive.open(f"{i}/{name}.csv","w") as raw:
                    with io.TextIOWrapper(raw, encoding="utf-8", newline="") as csv:
                        frames[symbol].to_csv(csv,index=False)


def load_cache(path, symbols, sessions):
    scores,prices = {},{}
    with zipfile.ZipFile(Path(path)) as archive:
        meta = json.loads(archive.read("metadata.json"))
        if (meta.get("version") != 1 or meta["symbols"] != symbols or meta["sessions"] != sessions
                or set(meta["gate"]) != set(sessions) or any(type(v) is not bool for v in meta["gate"].values())):
            raise ValueError("Replay cache metadata mismatch")
        for i,symbol in enumerate(symbols):
            for name,frames in [("scores",scores), ("prices",prices)]:
                with archive.open(f"{i}/{name}.csv") as raw:
                    frame = pd.read_csv(raw,float_precision="round_trip")
                frame["timestamp"] = pd.to_datetime(frame.timestamp,utc=True,errors="raise")
                if name == "scores":
                    frame["signal_available_at"] = pd.to_datetime(frame.signal_available_at,utc=True,errors="raise")
                frames[symbol] = frame
    return scores, prices, meta["gate"]
