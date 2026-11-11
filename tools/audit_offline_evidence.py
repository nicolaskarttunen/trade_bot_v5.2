"""Read-only inventory, real-file feature parity, and baseline counterexample.

Writes only explicitly selected evidence directory. Never loads pickle models,
imports the paper process, accesses Alpaca, or computes market returns.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from features import normalize_bars, enrich, FEATURES
from intraday_replay import replay_intraday
from test_intraday_replay import bars, signals, CFG as TEST_CFG


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((BASE/"config.json").read_text())
    report={"base_commit":"f92c46af5c5ee677505b542fe48035d9a82e3047", "price_files":{}, "news_files":{},
            "missing_config_symbols":[], "profitability_run":False}
    market=pd.read_csv(BASE/"data/SPY_5min.csv")
    market["timestamp"]=pd.to_datetime(market["timestamp"],utc=True)
    for path in sorted((BASE/"data").glob("*_5min.csv")):
        x=pd.read_csv(path)
        item={"rows":len(x),"start":str(x.timestamp.min()),"end":str(x.timestamp.max()),
              "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        try:
            x=normalize_bars(x); item["ohlcv_validation"]="passed"
            if path.stem.split('_')[0] in cfg["symbols"]:
                # Last 2,000 stored rows only: verifies transformations on real
                # file values, not full-history profitability or model training.
                raw=x.tail(2000).reset_index(drop=True)
                m=market[market.timestamp <= raw.timestamp.iloc[-1]]
                full=enrich(raw,m)
                valid=full.dropna(subset=FEATURES)
                if valid.empty:
                    raise ValueError("No feature-valid comparison row")
                cutoff=valid.timestamp.iloc[-1]
                short=enrich(raw[raw.timestamp<=cutoff].tail(250),m[m.timestamp<=cutoff].tail(250))
                expected=valid.iloc[-1][FEATURES].to_numpy(float)
                actual=short.iloc[-1][FEATURES].to_numpy(float)
                np.testing.assert_allclose(expected,actual,atol=1e-12,rtol=1e-12)
                item["feature_parity_max_abs_diff"]=float(np.max(np.abs(expected-actual)))
                item["feature_parity_timestamp"]=str(cutoff)
        except (ValueError,AssertionError) as exc:
            item["validation_error"]=str(exc)
        report["price_files"][path.name]=item
    for path in sorted((BASE/"news").glob("*.csv")):
        x=pd.read_csv(path)
        report["news_files"][path.name]={"rows":len(x),"received_at_present":"received_at" in x,
                                        "columns":list(x.columns),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    report["missing_config_symbols"]=[s for s in cfg["symbols"] if not (BASE/"data"/f"{s}_5min.csv").exists()]
    (args.output_dir/"data_inventory.json").write_text(json.dumps(report,indent=2))
    # Execute ONLY the reviewed pure offline replay module at the audit base.
    source=subprocess.check_output(["git","show",report["base_commit"]+":intraday_replay.py"],cwd=BASE,text=True)
    old=types.ModuleType("baseline_intraday")
    exec(compile(source,"baseline_intraday_replay.py","exec"),old.__dict__)
    price=bars(); score=signals(price); config=dict(TEST_CFG,slippage_bps=3,spread_bps=2)
    results={}
    for name,fn in [("before",old.replay_intraday),("after",replay_intraday)]:
        _,trades,_,_=fn({"A":score},{"A":price},["A"],{},config)
        row=trades.iloc[0]
        results[name]={"qty":int(row.qty),"entry_value":float(row.entry*row.qty)}
    (args.output_dir/"risk_counterexample.json").write_text(json.dumps({
        "fixture":"synthetic; initial equity 100000, reference 100, 5 bps each side, position cap 10%",
        "results":results,"profitability_evidence":False},indent=2))
    print(json.dumps({"price_files":len(report["price_files"]),"missing_stocks":len(report["missing_config_symbols"]),
                      "real_file_feature_checks":sum('feature_parity_max_abs_diff' in v for v in report['price_files'].values()),
                      "errors":{p:v["validation_error"] for p,v in report["price_files"].items() if "validation_error" in v},
                      "synthetic_risk_counterexample":results},indent=2))


if __name__=="__main__":
    main()
