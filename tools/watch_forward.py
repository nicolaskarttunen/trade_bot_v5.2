"""Predeclared, after-close forward observation. No brokerage or order API."""
import argparse
from contextlib import ExitStack, closing
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile
from unittest.mock import patch
import zipfile

if __name__=="__main__":
    os.environ["OPENBLAS_NUM_THREADS"]="1"
    os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from tools import run_locked_period as locked
from order_journal import ProcessLock
from research_model_state import encode, decode
from research_validation import paired_daily_comparison
from features import normalize_bars, FEATURE_HISTORY_BARS
from replay_audit import audit_frames

aug=locked.aug
NY=locked.NY
DATES=("2026-11-10 2026-11-11 2026-11-12 2026-11-13 2026-11-16 2026-11-17 2026-11-18 "
       "2026-11-19 2026-11-20 2026-11-23 2026-11-24 2026-11-25 2026-11-27 2026-11-30 "
       "2026-12-01 2026-12-02 2026-12-03 2026-12-04 2026-12-07 2026-12-08").split()
CAL={d:"13:00" if d=="2026-11-27" else "16:00" for d in DATES}
SOURCES=("tools/watch_forward.py research_model_state.py tools/run_locked_period.py tools/run_order_trial.py "
         "tools/run_august_simulation.py v6_5_intraday_replay.py intraday_replay.py research_validation.py "
         "features.py model_pipeline.py ml_models.py disk_training.py news_features.py replay_audit.py order_trial.py").split()


def utcnow(): return pd.Timestamp.now(tz="UTC")


def canonical(x): return json.dumps(x,sort_keys=True,allow_nan=False,separators=(",",":")).encode()


def fingerprint(x): return hashlib.sha256(canonical(x)).hexdigest()


def read(path): return json.loads(path.read_bytes())


def json_atomic(path,x):
    with tempfile.NamedTemporaryFile("wb",dir=path.parent,delete=False) as f:
        tmp=Path(f.name); f.write(canonical(x)); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def env_versions():
    return dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)


def protocol():
    return dict(sessions=CAL,arms=locked.ARMS,policy=locked.POLICY,
                config=dict(aug.reference.CFG,feed="sip",strict_price_coverage=True),
                sources={n:aug.digest(BASE/n) for n in SOURCES},versions=env_versions(),
                history=dict(feed="sip",adjustment="raw",asof="2026-08-31",timeframe="5Min"),
                delay_minutes=30,warmup_calendar_days=14,news="neutral",parameter_search=False)


def offline(stack):
    for obj,key in [(socket.socket,"connect"),(socket.socket,"connect_ex"),(socket,"create_connection")]:
        stack.enter_context(patch.object(obj,key,side_effect=RuntimeError("Network disabled during computation")))


def initialize(root,history,reference):
    root,history,reference=map(lambda p:Path(p).resolve(),(root,history,reference))
    if any(root==p or p in root.parents for p in (history,reference)): raise ValueError("Use a separate research directory")
    root.mkdir(parents=True,exist_ok=True)
    with closing(ProcessLock(root/"writer.lock")):
        ready=root/"ready.json"
        if ready.is_file():
            load_ready(root); print("Model already frozen. No retraining."); return
        plan=protocol(); start=root/"registration.json"
        if start.exists():
            reg=read(start)
            if reg["protocol_hash"]!=fingerprint(plan): raise ValueError("Registration differs; cannot resume")
        else:
            now=utcnow()
            if now>=pd.Timestamp(DATES[0]+" 09:30",tz=NY):
                raise ValueError("Registration is closed: do not relabel observed days as prospective")
            reg=dict(registered_at=now.isoformat(),protocol=plan,protocol_hash=fingerprint(plan))
            json_atomic(start,reg)
        if pd.Timestamp(reg["registered_at"])>=pd.Timestamp(DATES[0]+" 09:30",tz=NY):
            raise ValueError("Late registration")
        cfg=plan["config"]; symbols=list(cfg["symbols"]); bench=aug.reference.BENCH
        with tempfile.TemporaryDirectory(dir=root,prefix="fit-") as tmp, ExitStack() as stack:
            offline(stack); tmp=Path(tmp)
            log=dict(source_sha256=plan["sources"])
            paths,expected,m,ref,ref_hash=locked.training_reference(history,reference,tmp,symbols,bench,cfg,log)
            bundle=locked.rebuild(reference,tmp,symbols,bench,cfg,paths,expected,ref,log,
                                  lambda:json_atomic(tmp/"progress.json",log))
            if (aug.digest(reference/"manifest.json")!=ref_hash or
                aug.digest(reference/locked.CACHE)!=m["outputs"][locked.CACHE] or
                aug.digest(reference/"run_report.json")!=m["outputs"]["run_report.json"] or
                any(aug.digest(p)!=expected[p.name] for p in paths.values())): raise ValueError("Reference changed")
            model=encode(bundle); decode(model,cfg); json_atomic(root/"model.json",model)
            json_atomic(ready,dict(registration_sha256=aug.digest(start),model_sha256=aug.digest(root/"model.json"),
                                   training_matrix=log["training_cache_sha256"],anchors=log["august_anchor_predictions_checked"],
                                   finished_at=utcnow().isoformat()))
        print("Model frozen and August anchors verified:",root,flush=True)


def load_ready(root):
    reg=read(root/"registration.json"); ready=read(root/"ready.json")
    if (reg["protocol_hash"]!=fingerprint(protocol()) or reg["protocol_hash"]!=fingerprint(reg["protocol"])
        or aug.digest(root/"registration.json")!=ready["registration_sha256"]
        or aug.digest(root/"model.json")!=ready["model_sha256"]):
        raise ValueError("Frozen code, settings, environment or model changed")
    return reg["protocol"]["config"],decode(read(root/"model.json"),reg["protocol"]["config"])


def bounds(day):
    midnight=pd.Timestamp(day,tz=NY)
    return midnight-pd.DateOffset(days=14),midnight,pd.Timestamp(f"{day} {CAL[day]}",tz=NY)


def check_bars(frame,day,s,bench):
    warm,start,end=bounds(day); ts=pd.to_datetime(frame.timestamp,utc=True,errors="raise")
    if ts.empty or not ts.is_monotonic_increasing or ts.isna().any() or ts.min()<warm or ts.max()>=end:
        raise ValueError(f"{s}: bad observation window")
    x=normalize_bars(frame)
    if (ts<start).sum()<FEATURE_HISTORY_BARS: raise ValueError(f"{s}: insufficient warm-up")
    grid=pd.date_range(f"{day} 09:30",end.tz_localize(None)-pd.Timedelta(minutes=5),freq="5min",tz=NY)
    missed=grid.difference(pd.DatetimeIndex(x.timestamp))
    if len(missed)==len(grid) or (s==bench and len(missed)): raise ValueError(f"{s}: missing session or benchmark bars")
    return x,[t.isoformat() for t in missed]


def snapshot(root,day,symbols,bench,fetch):
    dest=root/"snapshots"/day
    if not dest.exists():
        warm,_,end=bounds(day)
        with tempfile.TemporaryDirectory(dir=root) as stage:
            work=Path(stage)/"new"; work.mkdir()
            first=utcnow()
            with zipfile.ZipFile(work/"bars.zip","x",zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                for i,s in enumerate([*symbols,bench]):
                    x=fetch(s,warm,end-pd.Timedelta(seconds=1))
                    check_bars(x,day,s,bench)
                    z.writestr(f"{i}.csv",x[aug.OHLCV].to_csv(index=False).encode())
            finish=utcnow()
            json_atomic(work/"snapshot.json",dict(day=day,first_fetch=first.isoformat(),last_fetch=finish.isoformat(),
                        fetched_after_session_date=finish.tz_convert(NY).date()>end.date(),hash=aug.digest(work/"bars.zip")))
            work.rename(dest)
    meta=read(dest/"snapshot.json")
    if meta["day"]!=day or meta["hash"]!=aug.digest(dest/"bars.zip"): raise ValueError("Snapshot hash mismatch")
    return dest,meta


def do_day(root,day,previous,cash,cfg,bundle,fetch):
    symbols=cfg["symbols"]; bench=aug.reference.BENCH
    snap,meta=snapshot(root,day,symbols,bench,fetch)
    _,start,end=bounds(day)
    with tempfile.TemporaryDirectory(dir=root) as stage, ExitStack() as stack:
        offline(stack); work=Path(stage)/"new"; work.mkdir(); gaps={}
        with zipfile.ZipFile(snap/"bars.zip") as z:
            def bars(i,s):
                with z.open(f"{i}.csv") as f: x=pd.read_csv(f,float_precision="round_trip")
                x,gaps[s]=check_bars(x,day,s,bench); return x
            market=bars(len(symbols),bench); prices={}; scores={}
            for i,s in enumerate(symbols):
                x=bars(i,s)
                scores[s]=locked.score(x,market,cfg,bundle,start,end)
                prices[s]=x.loc[x.timestamp>=start,aug.OHLCV]
        gate=locked.build_spy_gate(market); audits={}; tables={n:[] for n in locked.TABLES}
        for arm,rotated,gated in locked.ARMS:
            frames,audits[arm]=locked.one_arm(scores,prices,symbols,dict(cfg,initial_cash=cash[arm]),
                                              {day:CAL[day]},gate,gated,rotated,None)
            for n,f in zip(locked.TABLES,frames): tables[n].append(f.assign(strategy=arm))
        for n,fs in tables.items(): pd.concat(fs,ignore_index=True).to_csv(work/(n+".csv"),index=False)
        json_atomic(work/"audit.json",audits)
        json_atomic(work/"manifest.json",dict(day=day,previous=previous,spy_gate=bool(gate[day]),
                    missing_bars=gaps,snapshot_manifest=aug.digest(snap/"snapshot.json"),
                    files={p.name:aug.digest(p) for p in work.iterdir()}))
        work.rename(root/"days"/day)


def refresh(root,env_file,*,fetch=None):
    root=Path(root).resolve()
    with closing(ProcessLock(root/"writer.lock")):
        cfg,bundle=load_ready(root)
        for folder in ["days","snapshots"]: (root/folder).mkdir(exist_ok=True)
        if fetch is None:
            def fetch(s,start,end):
                from dotenv import dotenv_values
                if not Path(env_file).is_file(): raise ValueError("Market data credentials file missing")
                creds=dotenv_values(env_file,interpolate=False)
                headers={"APCA-API-KEY-ID":creds.get("ALPACA_API_KEY"),"APCA-API-SECRET-KEY":creds.get("ALPACA_SECRET_KEY")}
                if not all(headers.values()): raise ValueError("Market data credentials missing")
                return locked.fetch_bars(s,headers,start=start,end=end)
        now=utcnow(); dates=[d for d in DATES if now>=bounds(d)[2]+pd.Timedelta(minutes=30)]
        already=sorted(p.name for p in (root/"days").iterdir())
        if already!=dates[:len(already)]: raise ValueError("Non-contiguous results or clock moved backwards")
        arms=[a for a,_,_ in locked.ARMS]; cash=dict.fromkeys(arms,cfg["initial_cash"])
        previous=aug.digest(root/"ready.json")
        daily=[]; curves=[]; trades=[]; late=[]; gates=[]
        for day in dates:
            folder=root/"days"/day
            if not folder.exists():
                print("Collecting and replaying",day,flush=True)
                do_day(root,day,previous,cash,cfg,bundle,fetch)
            m=read(folder/"manifest.json"); snap=root/"snapshots"/day
            if (m["day"]!=day or m["previous"]!=previous or aug.digest(snap/"snapshot.json")!=m["snapshot_manifest"]
                or set(m["files"])!={*(n+".csv" for n in locked.TABLES),"audit.json"}
                or any(aug.digest(folder/n)!=h for n,h in m["files"].items())): raise ValueError("Daily ledger chain mismatch")
            sm=read(snap/"snapshot.json")
            if aug.digest(snap/"bars.zip")!=sm["hash"]: raise ValueError("Saved market data changed")
            if sm["fetched_after_session_date"]: late.append(day)
            if m["spy_gate"]: gates.append(day)
            def table(name): return pd.read_csv(folder/(name+".csv"),float_precision="round_trip")
            d=table("daily")
            for arm in arms:
                row=d.loc[d.strategy==arm].iloc[0]
                if row.start_equity!=cash[arm]: raise ValueError("Cash carry differs")
                cash[arm]=float(row.end_equity)
            daily.append(d); curves.append(table("equity")); trades.append(table("trades"))
            previous=aug.digest(folder/"manifest.json")
        status=dict(status="complete" if dates==DATES else ("collecting" if dates else "waiting_first_close"),
                    completed_sessions=dates,planned_sessions=DATES,late_sessions=late,gate_allowed_days=gates,
                    last_hash=previous,profitability_validated=False)
        if dates:
            d,c,t=[pd.concat(fs,ignore_index=True) for fs in [daily,curves,trades]]
            summary=[]; comparisons={}
            for arm in arms:
                ta,da,ca=[x.loc[x.strategy==arm] for x in [t,d,c]]
                stats=aug.reference.metrics(cfg["initial_cash"],cash[arm],ta,da,ca)
                audit=audit_frames(ta,da,ca,pd.DataFrame([stats]),cfg,cfg["symbols"],{k:CAL[k] for k in dates},aug.reference.SECTOR_BY_SYMBOL)
                if audit["issues"]: raise ValueError(f"Combined ledger failed: {audit['issues']}")
                summary.append(dict(strategy=arm,**stats,top20_trade_pct=audit["top20_trade_pct"]))
            frame=pd.DataFrame(summary)
            with tempfile.NamedTemporaryFile("w",suffix=".csv",dir=root,delete=False,newline="") as f:
                frame.to_csv(f,index=False); tmp=Path(f.name)
            os.replace(tmp,root/"summary.csv")
            if dates==DATES:
                for a,b in [(0,1),(2,3),(0,2),(1,3)]:
                    pair=paired_daily_comparison(d.loc[d.strategy==arms[a]],d.loc[d.strategy==arms[b]])
                    pair["classification"]="preregistered_after_close_bar_simulation"
                    comparisons[arms[b]+"_minus_"+arms[a]]=pair
                status["paired_daily_comparisons"]=comparisons
        json_atomic(root/"status.json",status)
        print(f"Status: {status['status']}, sessions {len(dates)}/{len(DATES)}",flush=True)
        return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest="mode",required=True)
    init=subs.add_parser("init"); update=subs.add_parser("update")
    for p in [init,update]: p.add_argument("--state",required=True,type=Path)
    for arg in ["train-data","reference"]: init.add_argument("--"+arg,required=True,type=Path)
    update.add_argument("--env-file",required=True,type=Path)
    args=parser.parse_args()
    try:
        if args.mode=="init": initialize(args.state,args.train_data,args.reference)
        else:
            try:
                result=refresh(args.state,args.env_file)
            except Exception as exc:
                if args.state.is_dir():
                    json_atomic(args.state/"last_attempt.json",dict(status="update_failed",at=utcnow().isoformat(),
                                                                   error=str(exc)))
                raise
            json_atomic(args.state/"last_attempt.json",dict(status=result["status"],at=utcnow().isoformat()))
    except (OSError,ValueError,KeyError,RuntimeError) as exc: raise SystemExit(f"Stopped: {exc}")


if __name__=="__main__": main()
