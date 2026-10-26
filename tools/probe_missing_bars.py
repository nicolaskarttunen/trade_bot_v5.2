"""Read-only probe of five reported SIP gaps. Never patches source CSVs.

A returned bar is only a candidate: matching surrounding OHLCV does not prove
how the original full dataset was downloaded. Raw API responses and explicit
request settings are retained for review. No trading/account API is used.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from features import normalize_bars

GAPS = [("EQIX", "2026-08-11T18:55:00Z"), ("EQIX", "2026-08-18T18:05:00Z"),
        ("EQIX", "2026-08-26T14:05:00Z"), ("EQIX", "2026-08-28T18:05:00Z"),
        ("BLK", "2026-08-13T17:20:00Z")]
FIELDS = ["open", "high", "low", "close", "volume"]
REQUEST_SETTINGS = {"feed": "sip", "adjustment": "raw", "asof": "2026-08-31",
                    "timeframe": "5Min", "sort": "asc", "limit": 1000}


class DataReadError(RuntimeError):
    def __init__(self, status):
        super().__init__("Historical data request failed")
        self.status = status


class HistoricalReader:
    """Only GET to the fixed market-data endpoint; no redirects/retries/orders."""
    def __init__(self, key, secret):
        self.headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    def __call__(self, symbol, start, end):
        if symbol not in {s for s, _ in GAPS}:
            raise ValueError("Unexpected symbol")
        params = dict(REQUEST_SETTINGS, start=start.isoformat(), end=end.isoformat())
        bars, seen_tokens = [], set()
        for _ in range(10):
            response = requests.get(f"https://data.alpaca.markets/v2/stocks/{symbol}/bars",
                headers=self.headers, params=params, timeout=(5, 20), allow_redirects=False)
            try:
                if response.status_code != 200:
                    raise DataReadError(response.status_code)
                body = response.json()
            finally:
                response.close()
            if body.get("symbol") != symbol or not isinstance(body.get("bars"), list):
                raise ValueError("Invalid historical bar response")
            bars.extend(body["bars"])
            token = body.get("next_page_token")
            if not token:
                return {"symbol": symbol, "bars": bars}
            if not isinstance(token, str) or token in seen_tokens:
                raise ValueError("Repeated/invalid pagination token")
            seen_tokens.add(token)
            params["page_token"] = token
        raise ValueError("Unexpected pagination length for a 25-minute window")


def compare_window(local, returned, target):
    """No interpolation or cross-feed fallback; require all local anchors."""
    local = normalize_bars(local)
    if target in set(local.timestamp):
        return {"status": "already_present_locally", "target_returned": False}, None
    left, right = target-pd.Timedelta(minutes=10), target+pd.Timedelta(minutes=10)
    anchors = local.loc[local.timestamp.between(left, right)].set_index("timestamp")
    if returned.empty:
        return {"status": "absent_from_response", "target_returned": False,
                "context_matches": False, "local_context_rows": len(anchors)}, None
    remote = normalize_bars(returned).set_index("timestamp")
    if not remote.index.to_series().between(left, right).all():
        raise ValueError("Response extends beyond the requested window")
    has_target = target in remote.index
    required_neighbors = {target-pd.Timedelta(minutes=5), target+pd.Timedelta(minutes=5)}
    complete = required_neighbors.issubset(anchors.index) and set(anchors.index).issubset(remote.index)
    matched = 0
    for ts, row in anchors.iterrows():
        if ts in remote.index:
            price_match = np.allclose(row[FIELDS[:4]].to_numpy(dtype=float),
                                      remote.loc[ts, FIELDS[:4]].to_numpy(dtype=float), rtol=0, atol=1e-6)
            if price_match and float(row.volume) == float(remote.loc[ts, "volume"]):
                matched += 1
    matches = bool(complete and matched == len(anchors))
    status = ("candidate_for_review" if matches else "context_mismatch") if has_target else "absent_from_response"
    result = {"status": status, "target_returned": has_target, "context_matches": matches,
              "local_context_rows": len(anchors), "matching_context_rows": matched}
    candidate = remote.loc[[target]].reset_index() if has_target and matches else None
    return result, candidate


def run_probe(data_dir, output_dir, fetcher):
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output_dir == data_dir or data_dir in output_dir.parents:
        raise ValueError("Output must be outside the original data directory")
    if output_dir.exists():
        raise FileExistsError("Use a new output directory")
    originals, hashes = {}, {}
    for symbol in sorted({s for s, _ in GAPS}):
        path = data_dir/f"{symbol}_5min.csv"
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        originals[symbol] = normalize_bars(pd.read_csv(io.BytesIO(raw)))
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {"classification": "data_probe_not_backtest", "status": "started",
              "retrieved_at": datetime.now(timezone.utc).isoformat(),
              "request_settings": REQUEST_SETTINGS, "input_sha256": hashes,
              "source_files_modified_by_tool": False, "profitability_validated": False,
              "original_feed_and_adjustment_proven": False, "gaps": []}
    report_path = output_dir/"report.json"
    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    save()
    for symbol, stamp in GAPS:
        target = pd.Timestamp(stamp)
        entry = {"symbol": symbol, "timestamp": stamp}
        try:
            if target in set(originals[symbol].timestamp):
                entry["status"] = "already_present_locally"
            else:
                start, end = target-pd.Timedelta(minutes=10), target+pd.Timedelta(minutes=10)
                entry.update(start=start.isoformat(), end=end.isoformat())
                response = fetcher(symbol, start, end)
                if response.get("symbol") != symbol or not isinstance(response.get("bars"), list):
                    raise ValueError("Invalid response envelope")
                name = f"{symbol}_{target.strftime('%Y%m%dT%H%M')}"
                raw_path = output_dir/(name+"_response.json")
                raw_path.write_text(json.dumps(response, indent=2, allow_nan=False), encoding="utf-8")
                entry["response_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                remote = pd.DataFrame(response["bars"]).rename(columns={
                    "t":"timestamp", "o":"open", "h":"high", "l":"low", "c":"close",
                    "v":"volume", "n":"trade_count", "vw":"vwap"})
                result, candidate = compare_window(originals[symbol], remote, target)
                entry.update(result)
                if candidate is not None:
                    candidate.to_csv(output_dir/(name+"_candidate_ONLY.csv"), index=False)
        except Exception as exc:
            # Never print exception bodies, request objects, headers or credentials.
            entry.update(status="request_or_validation_error", error_type=type(exc).__name__)
            if isinstance(exc, DataReadError):
                entry["http_status"] = exc.status
        report["gaps"].append(entry)
        save()
        print(symbol, stamp, entry["status"], flush=True)
    unchanged = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in hashes.items())
    report["source_hashes_unchanged_at_end"] = unchanged
    failures = any(r["status"] == "request_or_validation_error" for r in report["gaps"])
    report["status"] = "completed" if unchanged and not failures else "failed"
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True,
                        help="Existing local credentials file; never printed or copied")
    args = parser.parse_args()
    if not args.env_file.is_file():
        parser.error("Credentials file not found; do not paste credentials into chat")
    if args.output_dir.exists():
        parser.error("Output exists; choose a new directory")
    from dotenv import dotenv_values
    credentials = dotenv_values(args.env_file, interpolate=False)
    key, secret = credentials.get("ALPACA_API_KEY"), credentials.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        parser.error("Required Alpaca credentials missing from selected file")
    print("READ-ONLY SIP/RAW DATA PROBE: five windows; originals will not be patched.")
    report = run_probe(args.data_dir, args.output_dir, HistoricalReader(key, secret))
    print("Report:", args.output_dir/"report.json")
    print("Candidates require review. No backtest or profitability result was produced.")
    raise SystemExit(0 if report["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
