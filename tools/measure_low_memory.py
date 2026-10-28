"""Linux allocation smoke test, SYNTHETIC data, NOT a complete model fit.

Exercises external merge and one full-gradient update each of the main and a
bootstrap logistic model. Full production epochs are intentionally not timed.
Uses an address-space limit in an isolated process; no system settings change.
"""
import argparse
import json
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import numpy as np
import pandas as pd
from disk_training import TrainingStore, fit_logistic, bootstrap_counts
from features import FEATURES
from ml_models import LogisticGD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--symbols", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("This allocation probe uses Linux resource limits, not Windows measurements")
    if args.rows < args.symbols*2 or args.symbols < 1:
        parser.error("Need at least two rows per synthetic symbol")
    import resource
    limit = 2*1024**3
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    store = TrainingStore(args.output_dir/"temporary_training")
    rng = np.random.default_rng(20261028)
    started = time.monotonic()
    print(f"SYNTHETIC allocation smoke: {args.rows:,} rows; address-space limit 2 GiB", flush=True)
    for i in range(args.symbols):
        n = args.rows//args.symbols + (i < args.rows % args.symbols)
        frame = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
        frame["target"] = (frame[FEATURES[0]] > 0).astype(float)
        frame["timestamp"] = pd.date_range("2023-01-01", periods=n, freq="5min", tz="UTC")
        store.append(frame)
        del frame
    store.finish()
    merged = time.monotonic()
    print("Merge complete; testing full-gradient allocation", flush=True)
    fit_logistic(store.path, LogisticGD(epochs=1), store.root)
    counts = bootstrap_counts(rng, store.rows)
    cols = rng.choice(len(FEATURES), 8, replace=False)
    fit_logistic(store.path, LogisticGD(epochs=1), store.root, cols, counts)
    elapsed = time.monotonic()-started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    report = {"classification": "synthetic_allocation_smoke_only", "rows": store.rows,
              "synthetic_symbols": args.symbols, "source": "generated_gaussian_features",
              "address_space_limit_bytes": limit, "peak_rss_bytes": peak,
              "elapsed_seconds": elapsed, "merge_seconds": merged-started,
              "logistic_models_exercised": 2, "epochs_per_model": 1,
              "full_ensemble_tested_at_this_scale": False, "full_replay_tested_at_this_scale": False,
              "windows_memory_measured": False, "profitability_validated": False,
              "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__}
    report["temporary_directory_removed"] = store.cleanup()
    (args.output_dir/"memory_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Peak RSS: {peak/1024**2:.1f} MiB; elapsed: {elapsed:.1f} seconds", flush=True)


if __name__ == "__main__": main()
