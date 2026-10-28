"""Explicit offline model training. Never overwrites the deployed model.

Example: python train_models.py --data-dir PATH --train-end 2026-08-31
  --news-mode neutral --output research_runs/model-v2.pkl
No test returns or profitability are inferred from training accuracy.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import pandas as pd
from features import enrich, label
from model_pipeline import fit_bundle, training_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--news-dir", type=Path)
    p.add_argument("--train-start", default="2023-01-01")
    p.add_argument("--train-end", required=True, help="Inclusive New York date")
    p.add_argument("--news-mode", choices=["neutral", "received", "publication_proxy"], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    base = Path(__file__).resolve().parent
    cfg = json.loads((base / "config.json").read_text())
    if args.output.resolve() == (base / "models/ensemble.pkl").resolve() or args.output.exists():
        p.error("Choose a NEW research artifact path; deployed/old model overwrite is prohibited")
    if args.news_mode != "neutral" and args.news_dir is None:
        p.error("--news-dir required for news training")
    files = [args.data_dir / f"{s}_5min.csv" for s in cfg["symbols"] + [cfg["benchmark_symbols"][0]]]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        p.error(f"Missing {len(missing)} required price files: {missing}")
    market = pd.read_csv(files[-1])
    start = pd.Timestamp(args.train_start, tz="America/New_York").tz_convert("UTC")
    end = (pd.Timestamp(args.train_end, tz="America/New_York") + pd.DateOffset(days=1)).tz_convert("UTC")
    frames = []
    hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    for symbol, path in zip(cfg["symbols"], files):
        news = None
        if args.news_mode != "neutral":
            np = args.news_dir / f"{symbol}_news.csv"
            if not np.is_file():
                p.error(f"Missing news coverage for {symbol}; do not silently substitute neutral")
            news = pd.read_csv(np)
            hashes[str(np)] = hashlib.sha256(np.read_bytes()).hexdigest()
        x = label(enrich(pd.read_csv(path), market, news, cfg["news_decay_minutes"],
                         cfg["news_lookback_minutes"], args.news_mode == "received"),
                  cfg["label_horizon_bars"], cfg["label_threshold_pct"])
        x["symbol"] = symbol
        frames.append(training_rows(x, start, end))
    train = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"], kind="stable")
    bundle = fit_bundle(train, cfg, args.news_mode, {"start": str(start), "cutoff": str(end),
                        "input_sha256": hashes, "rows": len(train), "config": cfg})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as f:
        pickle.dump(bundle, f)
    print(f"Trained {len(train)} rows; new research model: {args.output}")


if __name__ == "__main__":
    main()
