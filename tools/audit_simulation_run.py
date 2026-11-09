"""Audit a completed August research run without retraining, network or source edits."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import pandas as pd
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from replay_audit import audit_frames, LIMITATIONS
from v6_5_intraday_replay import SECTOR_BY_SYMBOL

FILES = ["run_report.json", "trades.csv", "daily.csv", "equity.csv", "summary.csv", "spy_comparison.json"]
ARMS = ["intraday_order_no_gate", "intraday_order_spy_ge010"]
WINDOWS = [["2023-01-01", "2026-07-31", "2026-08-01", "2026-08-31"]]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_run(run_dir, output_dir):
    source, out = Path(run_dir).resolve(), Path(output_dir).resolve()
    if out == source or source in out.parents:
        raise ValueError("Audit output must be outside the original run directory")
    if out.exists():
        raise FileExistsError("Use a new audit output directory")
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status="failed_ledger_checks", independent_evidence=False,
                  profitability_validated=False, source_run=str(source), arms={},
                  limitations=LIMITATIONS, historical_prices_reopened=False,
                  decisions_reconstructed=False)
    try:
        manifest_bytes = (source/"manifest.json").read_bytes()
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes)
        if manifest.get("status") != "completed_development_comparison" or manifest.get("windows") != WINDOWS:
            raise ValueError("Requires a completed fixed August development comparison")
        recorded = manifest["outputs"]
        if set(FILES)-set(recorded):
            raise ValueError("Manifest does not hash all required outputs")
        for name, digest in recorded.items():
            if Path(name).name != name or "/" in name or "\\" in name or name in {".", "..", "manifest.json"}:
                raise ValueError("Invalid manifest output path")
            if sha(source/name) != digest:
                raise ValueError(f"Output hash mismatch: {name}")
        # Parse exactly the bytes verified here, not a second unverified read.
        raw = {name:(source/name).read_bytes() for name in FILES}
        if any(hashlib.sha256(value).hexdigest() != recorded[name] for name,value in raw.items()):
            raise ValueError("Output changed before audit")
        report = json.loads(raw["run_report.json"])
        if report.get("status") != "completed_development_comparison" or report.get("source_hashes_unchanged_at_end") is not True:
            raise ValueError("Run was not completed with unchanged historical input hashes")
        cfg = manifest["config"]
        symbols = report["symbols"]
        if symbols != cfg["symbols"] or len(symbols) != len(set(symbols)):
            raise ValueError("Inconsistent symbol universe")
        tables = {name:pd.read_csv(io.BytesIO(raw[name+".csv"])) for name in ["trades", "daily", "equity", "summary"]}
        for name,frame in tables.items():
            if "strategy" not in frame or frame.strategy.isna().any():
                raise ValueError(f"Missing strategy: {name}")
            actual = set(frame.strategy)
            if not actual <= set(ARMS) or (name != "trades" and actual != set(ARMS)):
                raise ValueError(f"Unexpected/missing strategy: {name}")
        sessions = dict.fromkeys(pd.bdate_range("2026-08-03", "2026-08-31").strftime("%Y-%m-%d"), "16:00")
        for arm in ARMS:
            frames = [tables[name].loc[tables[name].strategy == arm] for name in ["trades", "daily", "equity", "summary"]]
            result["arms"][arm] = audit_frames(*frames, cfg, symbols, sessions, SECTOR_BY_SYMBOL)
        result["source_outputs_unchanged_at_end"] = (
            sha(source/"manifest.json") == manifest_hash and
            all(sha(source/name) == digest for name,digest in recorded.items()))
        if not result["source_outputs_unchanged_at_end"]:
            raise ValueError("Original run changed during audit")
        result["source_manifest_sha256"] = manifest_hash
        result["auditor_source_sha256"] = {name:sha(BASE/name) for name in ["replay_audit.py", "tools/audit_simulation_run.py"]}
        result["decision_log_available"] = "decisions.jsonl" in recorded
        result["status"] = ("passed_ledger_checks" if all(a["status"] == "passed_ledger_checks" for a in result["arms"].values()) else "failed_ledger_checks")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    (out/"audit_report.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("Status:", result["status"])
    for name, arm in result["arms"].items():
        print(f"{name}: {arm['trades']} trades; top20={arm['top20_trade_pct']:.1f}%; issues={arm['issues']}")
        for sector in arm["sectors"]:
            print(f"  {sector['name']}: {sector['trades']} trades ({sector['trade_pct']:.1f}%)")
    if "error" in result:
        print(result["error"])
    print("Report:", out/"audit_report.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit_run(args.run_dir, args.output_dir)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc))
    raise SystemExit(0 if result["status"] == "passed_ledger_checks" else 1)


if __name__ == "__main__":
    main()
