"""Pre-specified comparison helpers, separate from execution and strategy code."""
import hashlib
from pathlib import Path
import subprocess
import platform
import numpy as np
import pandas as pd
from features import normalize_bars


def build_spy_gate(bars):
    """Fixed >=+0.10% FIRST completed five-minute bar; never usable for OPG."""
    x = normalize_bars(bars)
    local = x.timestamp.dt.tz_convert("America/New_York")
    first = x.loc[local.dt.strftime("%H:%M") == "09:30"].copy()
    day = first.timestamp.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    # Compare prices directly and allow only machine-roundoff at the fixed boundary.
    threshold_price = first.open * 1.001
    tolerance = np.finfo(float).eps * np.maximum(first.open.abs(), 1) * 8
    allowed = (first.close >= threshold_price - tolerance) & (first.volume > 0)
    return dict(zip(day, allowed))


def paired_daily_comparison(base, gated, block_days=5, repetitions=2000, seed=20261023):
    """Paired moving-block bootstrap of DAILY return differences, including zero days.

    Descriptive uncertainty for dependent observations, not proof of an edge.
    No threshold search, day filtering, annualization or independent-OOS claim.
    """
    for df in (base, gated):
        if df.empty or df["date"].duplicated().any():
            raise ValueError("Daily results must be nonempty with unique dates")
        if not np.isfinite(df[["pnl", "start_equity"]].to_numpy(dtype=float)).all() or (df.start_equity <= 0).any():
            raise ValueError("Invalid daily equity")
    if set(base.date) != set(gated.date):
        raise ValueError("Arms must cover exactly the same sessions, including zero-trade days")
    a = base.sort_values("date").set_index("date")
    b = gated.sort_values("date").set_index("date")
    diff = (b.pnl / b.start_equity - a.pnl / a.start_equity).to_numpy()
    result = {"classification": "development_only_previously_inspected", "sessions": len(diff),
              "mean_daily_difference_bps": float(diff.mean()*10000),
              "positive_difference_days": int((diff > 0).sum()),
              "block_days": block_days, "repetitions": repetitions, "seed": seed,
              "ci95_daily_difference_bps": None, "independent_evidence": False}
    if block_days < 1 or repetitions < 1:
        raise ValueError("Invalid bootstrap settings")
    if len(diff) >= 2*block_days:
        rng = np.random.default_rng(seed)
        means = []
        for _ in range(repetitions):
            starts = rng.integers(0, len(diff)-block_days+1, size=int(np.ceil(len(diff)/block_days)))
            sample = np.concatenate([diff[i:i+block_days] for i in starts])[:len(diff)]
            means.append(sample.mean()*10000)
        result["ci95_daily_difference_bps"] = np.quantile(means, [.025, .975]).tolist()
    return result


def input_manifest(paths, cfg, windows):
    base = Path(__file__).resolve().parent
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=base, text=True).strip())
    except (OSError, subprocess.SubprocessError):
        sha, dirty = "unavailable", True
    code = [base / name for name in ["v6_5_intraday_replay.py", "intraday_replay.py", "features.py",
            "model_pipeline.py", "news_features.py", "research_validation.py", "ml_models.py"]]
    return {"git_sha": sha, "worktree_dirty": dirty, "config": cfg, "windows": windows,
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
            "news_mode": "neutral", "parameter_search": False, "OPG_included": False,
            "strict_price_coverage": True, "profitability_validated": False}
