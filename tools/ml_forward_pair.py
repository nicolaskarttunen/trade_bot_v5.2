"""Registered end-of-period ML ablation. Reads 19 existing snapshots; owns the last one."""
import argparse
from contextlib import closing, ExitStack
import os
from pathlib import Path
import sys
import tempfile
import zipfile

if __name__ == "__main__":
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from tools import trial_ml_filter as ml
from order_journal import ProcessLock
from features import normalize_bars, FEATURE_HISTORY_BARS
from research_validation import paired_daily_comparison
from replay_audit import audit_frames

fw, aug = ml.forward, ml.aug
NY = fw.NY
LAST = "2026-12-09"
CAL = {**{d:fw.CAL[d] for d in fw.DATES[1:]}, LAST:"16:00"}
ARMS = (ml.ARM_ML, ml.ARM_MARKET)
EXTRA = ("tools/ml_forward_pair.py", "tools/trial_ml_filter.py", "tools/trial_open_priority.py",
         "tools/inspect_cost_margin.py", "order_journal.py")
FILES = ("summary.csv","daily.csv","equity.csv","trades.csv","audit.json")


def close_time(day):
    return pd.Timestamp(f"{day} {CAL[day]}", tz=NY)


def plan(source):
    return dict(sessions=CAL, arms=ARMS, primary="original_ml minus market_only mean paired daily net return",
                model_files={n:aug.digest(source/n) for n in ("model.json","registration.json","ready.json")},
                source_path=str(source), settings=fw.protocol(), extra_sources={n:aug.digest(BASE/n) for n in EXTRA},
                starting_cash_per_arm=100000., spy_opening_gate=False, news="neutral", training_runs=0,
                delay_minutes=30, bootstrap=dict(block_days=5,repetitions=2000,seed=20261023),
                extra_cost_per_side_bps=3, results_after_last_close_only=True,
                secondary=["net_return", "mtm_drawdown", "exposure", "turnover", "cost_stress",
                           "pnl_without_best_day", "trade_count"], rules="no substitutions; no parameter search")


def paths(root,source):
    a,b=map(lambda p:Path(p).resolve(),(root,source))
    if a==b or a in b.parents or b in a.parents: raise ValueError("Use separate sibling directories")
    return a,b


def registration(root,source,create):
    cfg,bundle=fw.load_ready(source)
    if cfg["initial_cash"] != 100000.: raise ValueError("Unexpected initial cash")
    p=plan(source); path=root/"registration.json"
    cutoff=pd.Timestamp(f"{next(iter(CAL))} 09:30",tz=NY)
    if not path.exists():
        if not create: raise ValueError("Register before the first open")
        now=fw.utcnow()
        if now>=cutoff: raise ValueError("Registration closed; do not backdate")
        if set(f.name for f in root.iterdir())!={"writer.lock"}: raise ValueError("Use an empty study directory")
        fw.json_atomic(path,dict(registered_at=now.isoformat(),plan=p,plan_hash=fw.fingerprint(p)))
    reg=fw.read(path)
    at=pd.Timestamp(reg["registered_at"])
    if (pd.isna(at) or at.tzinfo is None or at>=cutoff or at>fw.utcnow()
        or reg["plan_hash"]!=fw.fingerprint(p) or reg["plan_hash"]!=fw.fingerprint(reg["plan"])):
        raise ValueError("Registration, clock, model, settings or source changed")
    return cfg,bundle


def init(root,source):
    root,source=paths(root,source)
    root.mkdir(parents=True,exist_ok=True)
    with closing(ProcessLock(root/"writer.lock")):
        registration(root,source,True)
        print("Registered ML pair:",root,flush=True)


def validate_bars(x,day,s):
    if day!=LAST:
        return fw.check_bars(x,day,s,"SPY")[0]
    start=pd.Timestamp(day,tz=NY); end=close_time(day)
    ts=pd.to_datetime(x.timestamp,utc=True,errors="raise")
    if (ts.empty or ts.isna().any() or not ts.is_monotonic_increasing or ts.duplicated().any()
        or ts.min()<start-pd.DateOffset(days=14) or ts.max()>=end
        or (ts<start).sum()<FEATURE_HISTORY_BARS): raise ValueError(f"Bad final snapshot window: {s}")
    grid=pd.date_range(start+pd.Timedelta(hours=9,minutes=30),end-pd.Timedelta(minutes=5),freq="5min")
    missing=grid.difference(pd.DatetimeIndex(ts))
    if len(missing)==len(grid) or (s=="SPY" and len(missing)): raise ValueError(f"Missing final bars: {s}")
    return normalize_bars(x)


def last_snapshot(root,symbols,env_file,fetch):
    dest=root/"final_snapshot"
    if dest.exists(): return dest
    begin=fw.utcnow()
    if begin<close_time(LAST)+pd.Timedelta(minutes=30): raise ValueError("Final session not yet eligible")
    if fetch is None:
        from dotenv import dotenv_values
        v=dotenv_values(env_file,interpolate=False) if env_file and Path(env_file).is_file() else {}
        headers={"APCA-API-KEY-ID":v.get("ALPACA_API_KEY"),"APCA-API-SECRET-KEY":v.get("ALPACA_SECRET_KEY")}
        if not all(headers.values()): raise ValueError("Missing historical-data credentials")
        def fetch(s,a,b): return ml.locked.fetch_bars(s,headers,start=a,end=b)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        work=Path(tmp)/"new"; work.mkdir()
        with zipfile.ZipFile(work/"bars.zip","x",zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for i,s in enumerate([*symbols,"SPY"]):
                x=fetch(s,pd.Timestamp(LAST,tz=NY)-pd.DateOffset(days=14),close_time(LAST)-pd.Timedelta(seconds=1))
                validate_bars(x,LAST,s)
                z.writestr(f"{i}.csv",x[aug.OHLCV].to_csv(index=False).encode())
        end=fw.utcnow()
        fw.json_atomic(work/"snapshot.json",dict(day=LAST,first_fetch=begin.isoformat(),last_fetch=end.isoformat(),
            fetched_after_session_date=end.tz_convert(NY).date()>close_time(LAST).date(),hash=aug.digest(work/"bars.zip")))
        work.rename(dest)
    return dest


def snapshot_meta(folder,day):
    m=fw.read(folder/"snapshot.json"); first,last=map(pd.Timestamp,(m["first_fetch"],m["last_fetch"]))
    if (pd.isna(first) or pd.isna(last) or first.tzinfo is None or last.tzinfo is None
        or m["day"]!=day or not close_time(day)+pd.Timedelta(minutes=30)<=first<=last<=fw.utcnow()
        or m["fetched_after_session_date"]!=(last.tz_convert(NY).date()>close_time(day).date())
        or aug.digest(folder/"bars.zip")!=m["hash"]):
        raise ValueError(f"Snapshot integrity or timing mismatch: {day}")
    return dict(manifest_sha256=aug.digest(folder/"snapshot.json"),bars_sha256=m["hash"],
                late=m["fetched_after_session_date"])


def compute(root,source,cfg,bundle,fetch,env_file,registered_hash):
    symbols=cfg["symbols"]
    # Preserve the last session's snapshot even if an earlier collector is delayed.
    final=last_snapshot(root,symbols,env_file,fetch)
    folders={d:(final if d==LAST else source/"snapshots"/d) for d in CAL}
    inputs={d:snapshot_meta(f,d) for d,f in folders.items()}
    groups={a:{k:[] for k in ("trades","daily","equity")} for a in ARMS}
    cash=dict.fromkeys(ARMS,cfg["initial_cash"])
    configs={ARMS[0]:cfg, ARMS[1]:dict(cfg,probability_threshold=0.,trade_score_threshold=0.)}
    with ExitStack() as stack:
        fw.offline(stack)
        for day,folder in folders.items():
            print("Replaying",day,flush=True)
            start=pd.Timestamp(day,tz=NY); end=close_time(day)
            with zipfile.ZipFile(folder/"bars.zip") as z:
                if sorted(z.namelist())!=sorted(f"{i}.csv" for i in range(len(symbols)+1)):
                    raise ValueError("Unexpected snapshot members")
                def bars(i,s):
                    with z.open(f"{i}.csv") as f: x=pd.read_csv(f,float_precision="round_trip")
                    return validate_bars(x,day,s)
                market=bars(len(symbols),"SPY"); scores={}; prices={}
                for i,s in enumerate(symbols):
                    x=bars(i,s)
                    scores[s]=ml.locked.score(x,market,cfg,bundle,start,end)
                    prices[s]=x.loc[x.timestamp>=start,aug.OHLCV]
            eligible=ml.market_rows(scores,market,cfg)
            for arm,rows in zip(ARMS,(scores,eligible)):
                frames,_=ml.prior.replay(rows,prices,dict(configs[arm],initial_cash=cash[arm]),
                                         {day:CAL[day]}, {},{},False,False)
                cash[arm]=float(frames["daily"].end_equity.iloc[-1])
                for k in groups[arm]: groups[arm][k].append(frames[k])
        frames={a:{k:pd.concat(v,ignore_index=True) for k,v in g.items()} for a,g in groups.items()}
        summary=[]; audits={}
        for a,g in frames.items():
            t,d,c=[g[k] for k in ("trades","daily","equity")]
            stats=aug.reference.metrics(cfg["initial_cash"],cash[a],t,d,c)
            audits[a]=audit_frames(t,d,c,pd.DataFrame([stats]),configs[a],symbols,CAL,aug.reference.SECTOR_BY_SYMBOL)
            if audits[a]["issues"]: raise ValueError(f"Cumulative ledger failed: {a}")
            summary.append({"strategy":a, **stats, **ml.activity(g,CAL)})
        pair=paired_daily_comparison(frames[ARMS[1]]["daily"],frames[ARMS[0]]["daily"])
        pair["classification"]="preregistered_after_close_ml_pair"
        if (inputs!={d:snapshot_meta(f,d) for d,f in folders.items()} or aug.digest(root/"registration.json")!=registered_hash):
            raise ValueError("Study inputs changed during replay")
        registration(root,source,False)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            work=Path(tmp)/"result"; work.mkdir()
            pd.DataFrame(summary).to_csv(work/"summary.csv",index=False)
            for k in ("trades","daily","equity"):
                pd.concat([frames[a][k].assign(strategy=a) for a in ARMS],ignore_index=True).to_csv(work/(k+".csv"),index=False)
            fw.json_atomic(work/"audit.json",audits)
            fw.json_atomic(work/"report.json",dict(status="complete",sessions=list(CAL),inputs=inputs,
                late_sessions=[d for d,m in inputs.items() if m["late"]],ml_minus_market_only=pair,
                registration_sha256=registered_hash,profitability_validated=False,
                files={n:aug.digest(work/n) for n in FILES}))
            work.rename(root/"result")


def update(root,source,env_file=None,fetch=None):
    root,source=paths(root,source)
    with closing(ProcessLock(root/"writer.lock")):
        cfg,bundle=registration(root,source,False)
        h=aug.digest(root/"registration.json")
        if fw.utcnow()<close_time(LAST)+pd.Timedelta(minutes=30):
            print("Registered; waiting for 20 completed sessions. No result yet."); return
        if not (root/"result").exists():
            compute(root,source,cfg,bundle,fetch,env_file,h)
        result=root/"result"; report=fw.read(result/"report.json")
        if (report["registration_sha256"]!=h or report["status"]!="complete" or report["sessions"]!=list(CAL)
            or set(report["files"])!=set(FILES) or any(aug.digest(result/n)!=v for n,v in report["files"].items())
            or report["inputs"]!={d:snapshot_meta(root/"final_snapshot" if d==LAST else source/"snapshots"/d,d) for d in CAL}):
            raise ValueError("Saved study result or inputs changed")
        print("Status: complete; results:",result,flush=True)
        return report


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("mode",choices=["init","update"])
    p.add_argument("--state",type=Path,required=True); p.add_argument("--source",type=Path,required=True)
    p.add_argument("--env-file",type=Path)
    a=p.parse_args()
    try:
        if a.mode=="init": init(a.state,a.source)
        else: update(a.state,a.source,a.env_file)
    except (OSError,ValueError,KeyError,RuntimeError) as exc: raise SystemExit(f"Stopped: {exc}")


if __name__=="__main__": main()
