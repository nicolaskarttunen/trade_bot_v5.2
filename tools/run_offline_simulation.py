"""One offline demonstration run. Synthetic prices; NOT profitability evidence.

Never imports the paper process, loads .env/pickles or connects to Alpaca.
The separate test suite exercises SDK objects with fake credentials and responses. The actual V6.5 driver is exercised with a temporary two-stock universe;
its production settings/files remain unchanged. Network connections are blocked.
"""
import argparse
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

TRAIN_DAYS = ["2026-08-20", "2026-08-21", "2026-08-24", "2026-08-25",
              "2026-08-26", "2026-08-27", "2026-08-28", "2026-08-31"]
TEST_DAYS = ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04",
             "2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11", "2026-10-19"]


def synthetic_bars(days, scale):
    frames = []
    for i, day in enumerate(days):
        timestamps = pd.date_range(f"{day} 09:30", f"{day} 15:55", freq="5min",
                                   tz="America/New_York").tz_convert("UTC")
        close = (100 + i*.1 + np.sin(np.arange(len(timestamps))*.7))*scale
        frames.append(pd.DataFrame({"timestamp": timestamps, "open": close,
            "high": close+.4, "low": close-.4, "close": close, "volume": 1000.}))
    return pd.concat(frames, ignore_index=True)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def run_demo(root):
    """Create a NEW demo directory and retain all inputs, outputs and labels."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "data_kind": "synthetic",
              "purpose": "functional_demonstration_only", "symbols": ["NVDA", "AMD"],
              "benchmark": "SPY", "sessions": len(TEST_DAYS),
              "is_100_stock_test": False, "profitability_validated": False,
              "broker_access": False, "network_blocked": True,
              "parameter_search": False}
    write_json(root/"run_report.json", report)
    try:
        with ExitStack() as stack:
            for obj, attr in [(socket.socket, "connect"), (socket, "create_connection")]:
                stack.enter_context(patch.object(obj, attr, side_effect=RuntimeError("Network forbidden in simulation")))
            import v6_5_intraday_replay as driver
            old, new, out = root/"synthetic_train", root/"synthetic_test", root/"replay"
            old.mkdir(); new.mkdir()
            for symbol, scale in [("NVDA", 1.), ("AMD", 2.), ("SPY", 3.)]:
                synthetic_bars(TRAIN_DAYS, scale).to_csv(old/f"{symbol}_5min.csv", index=False)
                synthetic_bars(TEST_DAYS, scale).to_csv(new/f"{symbol}_5min.csv", index=False)
            sessions = root/"synthetic_sessions.csv"
            pd.DataFrame({"date": TEST_DAYS, "close": "16:00"}).to_csv(sessions, index=False)
            args = ["v6_5_intraday_replay.py", "--data-dir", str(old), "--extension-dir", str(new),
                    "--sessions", str(sessions), "--output-dir", str(out)]
            stack.enter_context(patch.object(driver, "SYMBOLS", report["symbols"]))
            stack.enter_context(patch.object(driver, "CFG", dict(driver.CFG, symbols=report["symbols"])))
            stack.enter_context(patch.object(driver, "DATA_DIR", old))
            stack.enter_context(patch.object(driver, "EXT_DIR", new))
            stack.enter_context(patch("sys.argv", args))
            with (root/"driver.log").open("w", encoding="utf-8") as log:
                log.write("SYNTHETIC DATA - FUNCTIONAL DEMONSTRATION, NOT MARKET EVIDENCE\n")
                with redirect_stdout(log), redirect_stderr(log):
                    driver.main()
            # Label the driver's generic manifests at their point of use too.
            manifest_path = out/"manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.update(data_kind="synthetic", is_100_stock_test=False,
                            status="completed_synthetic_demonstration")
            comparison_path = out/"spy_comparison.json"
            comparison = json.loads(comparison_path.read_text())
            comparison.update(classification="synthetic_demonstration_only", independent_evidence=False)
            write_json(comparison_path, comparison)
            manifest["outputs"][comparison_path.name] = hashlib.sha256(comparison_path.read_bytes()).hexdigest()
            manifest["source_sha256"]["tools/run_offline_simulation.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            write_json(manifest_path, manifest)
            daily = pd.read_csv(out/"v6_5_intraday_daily.csv")
            summary = pd.read_csv(out/"v6_5_intraday_summary.csv")
            if len(daily) != 2*len(TEST_DAYS) or len(summary) != 2:
                raise ValueError("Incomplete two-arm demonstration")
            report.update(status="completed", daily_rows=len(daily),
                          trades_by_arm={r["strategy"]: int(r["trades"]) for _, r in summary.iterrows()})
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        write_json(root/"run_report.json", report)
        raise
    write_json(root/"run_report.json", report)
    (root/"READ_ME_FIRST.txt").write_text(
        "SYNTHETIC DATA - FUNCTIONAL DEMONSTRATION ONLY\n"
        "Two stocks and a synthetic SPY series, not the 100-stock universe.\n"
        "Any return/win-rate/SPY difference is artificial and must not be used to judge profitability.\n"
        "No live market data, news, brokerage connection, real fills or actual funds.\n"
        "Broker lifecycle fault tests are separate from this OHLC replay.\n"
        "Do not deploy the paper process: legacy EOD acknowledgement handling remains unresolved.\n",
        encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New output folder; existing runs are never overwritten")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if root.exists():
        parser.error("Output already exists; choose a new run directory")
    root.mkdir(parents=True, exist_ok=False)
    print("OFFLINE FUNCTIONAL SIMULATION - SYNTHETIC DATA, NOT A PROFIT TEST", flush=True)
    print("1/2 Running the offline regression/fault suite...", flush=True)
    tests = subprocess.run([sys.executable, "-X", "utf8", str(BASE/"tools/run_audit_tests.py"),
                            "--output-dir", str(root/"tests")], cwd=BASE,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (root/"test_launcher.log").write_bytes(tests.stdout)
    if tests.returncode:
        raise SystemExit(f"Tests failed. See {root/'test_launcher.log'}; replay was not started.")
    result = json.loads((root/"tests/test_results.json").read_text())
    print(f"Tests passed: {result['tests']}/{result['tests']}", flush=True)
    print("2/2 Running the real V6.5 engine on synthetic prices...", flush=True)
    report = run_demo(root/"demo")
    print(f"Completed: {report['sessions']} synthetic sessions, two fixed strategy arms.")
    print("No profitability conclusion; this is not the 100-stock historical test.")
    print(f"Results: {root}")


if __name__ == "__main__":
    main()
