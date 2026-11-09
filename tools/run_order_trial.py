"""One fixed August order experiment, isolated from production and reference results."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path, PureWindowsPath
import sys

if __name__ == "__main__":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from intraday_replay import replay_intraday
from order_trial import CACHE, POLICY, daily_order, load_cache
from replay_audit import audit_frames
from research_validation import paired_daily_comparison
from tools import run_august_simulation as august
from tools.audit_simulation_run import ARMS, audit_run

TABLES = ["trades","daily","equity","summary"]
STATUS = "completed_order_experiment_development_only"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(source, name):
    return pd.read_csv(source/(name+".csv"),float_precision="round_trip")


def same_tables(left, right):
    for name in TABLES:
        try:
            pd.testing.assert_frame_equal(table(left,name),table(right,name),check_dtype=False,
                                          check_exact=False,rtol=0,atol=1e-6)
        except AssertionError as exc:
            raise ValueError(f"Reference reproduction differs: {name}") from exc


def one_arm(scores, prices, symbols, cfg, sessions, gate, gated, rotated, sink):
    cash=cfg["initial_cash"]
    trades, daily, curves=[],[],[]
    original_rank={s:i+1 for i,s in enumerate(symbols)}
    sectors=august.reference.SECTOR_BY_SYMBOL
    # Keep individual days only: all previous positions have been flattened.
    for day,close in sorted(sessions.items()):
        order=daily_order(symbols,day) if rotated else symbols
        start=pd.Timestamp(day,tz=august.NY)
        end=start+pd.DateOffset(days=1)
        def session(frames):
            return {s:x.loc[(x.timestamp>=start)&(x.timestamp<end)] for s,x in frames.items()}
        def record(event):
            row=dict(event, scan_order=event["symbol_order"], symbol_order=original_rank[event["symbol"]])
            sink(row)
        cash,t,d,c=replay_intraday(session(scores),session(prices),order,sectors,dict(cfg,initial_cash=cash),
                                   spy_opening_gate=gate,require_spy=gated,session_closes={day:close},
                                   decision_sink=record if sink is not None else None)
        if not t.empty:
            if rotated:
                t=t.rename(columns={"symbol_order":"scan_order"})
                t["symbol_order"]=t.symbol.map(original_rank)
            trades.append(t)
        daily.append(d); curves.append(c)
    t=pd.concat(trades,ignore_index=True) if trades else pd.DataFrame()
    d,c=pd.concat(daily,ignore_index=True),pd.concat(curves,ignore_index=True)
    s=pd.DataFrame([august.reference.metrics(cfg["initial_cash"],cash,t,d,c)])
    audit=audit_frames(t,d,c,s,cfg,symbols,sessions,sectors)
    if audit["issues"]:
        raise ValueError(f"Trial ledger failed: {audit['issues']}")
    return (t,d,c,s),audit


def experiment(previous_run, data_dir, output_dir):
    prev=Path(previous_run).resolve(); out=Path(output_dir).resolve()
    data=Path(data_dir).resolve() if data_dir is not None else None
    if prev==out or prev in out.parents or (data is not None and (data==out or data in out.parents)):
        raise ValueError("Use a new output directory outside all source directories")
    out.mkdir(parents=True,exist_ok=False)
    report=dict(status="started_not_validated",classification="august_development_only",
                independent_evidence=False,profitability_validated=False,policy=POLICY,
                baseline_order="original_config_order", gated_arms=ARMS,
                parameter_search=False,training_runs=0, previous_run=str(prev))
    report_path=out/"order_report.json"
    august.write_json(report_path,report)
    try:
        old_audit=audit_run(prev,out/"previous_ledger_audit")
        if old_audit["status"]!="passed_ledger_checks":
            raise ValueError("Previous run failed ledger checks; see previous_ledger_audit/audit_report.json")
        old_manifest=read(prev/"manifest.json")
        if august.digest(prev/"manifest.json")!=old_audit["source_manifest_sha256"]:
            raise ValueError("Previous manifest changed")
        cfg=old_manifest["config"]; symbols=cfg["symbols"]
        sessions=august.load_sessions()
        # Lock the protocol before viewing experimental outcomes.
        report.update(previous_manifest_sha256=old_audit["source_manifest_sha256"],
                      order_by_date={d:daily_order(symbols,d) for d in sessions},
                      trial_code_sha256={f:august.digest(BASE/f) for f in
                                         ["intraday_replay.py","tools/run_order_trial.py","order_trial.py","replay_audit.py"]})
        august.write_json(report_path,report)
        src=prev
        if CACHE not in old_manifest["outputs"]:
            if data is None:
                raise ValueError("Old run has no saved signals. Supply --data-dir for one low-memory fit.")
            if cfg!=dict(august.reference.CFG,feed="sip",strict_price_coverage=True):
                raise ValueError("Reference settings differ; no retraining with changed settings")
            symbol_files={s+"_5min.csv" for s in symbols+[august.reference.BENCH]}
            hashes={PureWindowsPath(k).name:v for k,v in read(prev/"run_report.json")["input_sha256"].items()}
            if set(hashes)!=symbol_files or any(august.digest(data/k)!=v for k,v in hashes.items()):
                raise ValueError("History differs from the original run; will not train")
            for name in ["disk_training.py","features.py","ml_models.py","model_pipeline.py","news_features.py"]:
                if old_manifest["source_sha256"].get(name)!=august.digest(BASE/name):
                    raise ValueError(f"Training source differs from original: {name}")
            src=out/"reference"
            print("Previous run saved no signals: fitting once; this can take hours.",flush=True)
            report["training_runs"]=1
            august.write_json(report_path,report)
            fitted=august.run(data,src,execute=True,low_memory=True,retain_replay_inputs=True)
            if {PureWindowsPath(k).name:v for k,v in fitted["input_sha256"].items()} != hashes:
                raise ValueError("Fitted reference used different history")
            same_tables(prev,src)
        manifest=read(src/"manifest.json")
        if manifest["config"]!=cfg or august.digest(src/CACHE)!=manifest["outputs"][CACHE]:
            raise ValueError("Replay cache/config mismatch")
        scores,prices,gate=load_cache(src/CACHE,symbols,sessions)
        report.update(reference_dir=str(src), cache_sha256=manifest["outputs"][CACHE])
        for rotated in (False,True):
            label="changed_order" if rotated else "reference_check"
            work=out/label; work.mkdir()
            groups={name:[] for name in TABLES}; checks={}
            with ExitStack() as stack:
                log=stack.enter_context((work/"decisions.jsonl").open("x",encoding="utf-8")) if rotated else None
                for arm in ARMS:
                    print(f"Replaying {label}: {arm}",flush=True)
                    def sink(event):
                        log.write(json.dumps(dict(event,strategy=arm),allow_nan=False)+"\n")
                    frames,audit=one_arm(scores,prices,symbols,cfg,sessions,gate,arm==ARMS[1],
                                         rotated,sink if rotated else None)
                    checks[arm]=audit
                    for name,frame in zip(TABLES,frames):
                        frame=frame.assign(strategy=arm)
                        if name=="summary":
                            frame=frame[["strategy", *(key for key in frame if key!="strategy")]]
                        groups[name].append(frame)
            for name,frames in groups.items():
                pd.concat(frames,ignore_index=True).to_csv(work/(name+".csv"),index=False)
            august.write_json(work/"ledger_audit.json",checks)
            if not rotated:
                same_tables(prev,work)  # Same frozen inputs must reproduce all original rows first.
                report["original_reproduced"]=True
        original=table(prev,"daily"); changed=table(out/"changed_order","daily")
        report["paired_order_comparison"]={arm:paired_daily_comparison(
            original.loc[original.strategy==arm],changed.loc[changed.strategy==arm]) for arm in ARMS}
        if (august.digest(src/CACHE)!=report["cache_sha256"] or
            august.digest(prev/"manifest.json")!=report["previous_manifest_sha256"] or
            any(august.digest(prev/k)!=v for k,v in old_manifest["outputs"].items())):
            raise ValueError("Reference files changed during experiment")
        report["outputs"]={p.relative_to(out).as_posix():august.digest(p)
                           for folder in ["reference_check","changed_order"] for p in (out/folder).iterdir()}
        report["status"]=STATUS
    except BaseException as exc:
        report.update(status="failed_not_validated",error=str(exc))
        raise
    finally:
        august.write_json(report_path,report)
    print("Status:",report["status"])
    for label in ["reference_check","changed_order"]:
        print(label); print(table(out/label,"summary").to_string(index=False))
    print("Report:",report_path)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run",type=Path,required=True)
    parser.add_argument("--data-dir",type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    try:
        experiment(args.previous_run,args.data_dir,args.output_dir)
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"Stopped: {exc}")


if __name__=="__main__": main()
