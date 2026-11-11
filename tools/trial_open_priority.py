"""Exploratory, cache-only trial. Never updates the registered four-arm study."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from tools import run_locked_period as locked
from tools import watch_forward as forward
from tools.run_order_trial import TABLES
from intraday_replay import replay_intraday
from replay_audit import audit_frames
from order_trial import CACHE, load_cache
from research_validation import paired_daily_comparison

aug = locked.aug
POLICY = "0935_completed_opening_ml_desc_v1"
AUG_ARMS = ["intraday_order_no_gate", "intraday_order_spy_ge010"]
LOCKED_ARMS = ["original_no_gate", "original_spy_ge010"]


def opening_order(symbols, scores, day):
    opening = pd.Timestamp(f"{day} 09:30", tz=locked.NY)
    decision = opening + pd.Timedelta(minutes=5)
    priority = {}
    for symbol in symbols:
        frame = scores[symbol]
        row = frame.loc[frame.timestamp == opening]
        if len(row) > 1:
            raise ValueError(f"Duplicate opening signal: {day} {symbol}")
        if row.empty:
            continue
        observed = row.iloc[0]
        if observed.signal_available_at != decision:
            raise ValueError(f"Opening availability differs: {day} {symbol}")
        p = float(observed.ml_prob)
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f"Invalid opening prediction: {day} {symbol}")
        priority[symbol] = p
    # Python's stable sort retains configuration order for exact ties and missing scores.
    order = sorted(symbols, key=lambda s: -priority.get(s, -1.))
    return order, priority


def replay(scores, prices, cfg, sessions, gate, orders, gated, ranked):
    symbols = cfg["symbols"]
    rank = {s:i+1 for i,s in enumerate(symbols)}
    sectors = aug.reference.SECTOR_BY_SYMBOL
    cash = cfg["initial_cash"]
    trades, days, curves = [], [], []
    for day, close in sessions.items():
        start = pd.Timestamp(day, tz=locked.NY)
        end = start + pd.DateOffset(days=1)
        def session(frames):
            return {s:x.loc[(x.timestamp>=start)&(x.timestamp<end)] for s,x in frames.items()}
        order = orders[day] if ranked else symbols
        cash,t,d,c = replay_intraday(session(scores),session(prices),order,sectors,
                                     dict(cfg,initial_cash=cash),spy_opening_gate=gate,
                                     require_spy=gated,session_closes={day:close})
        if not t.empty:
            if ranked:
                t=t.rename(columns={"symbol_order":"scan_order"})
                t["symbol_order"]=t.symbol.map(rank)
            trades.append(t)
        days.append(d); curves.append(c)
    t = pd.concat(trades,ignore_index=True) if trades else pd.DataFrame()
    d,c = pd.concat(days,ignore_index=True),pd.concat(curves,ignore_index=True)
    s = pd.DataFrame([aug.reference.metrics(cfg["initial_cash"],cash,t,d,c)])
    audit = audit_frames(t,d,c,s,cfg,symbols,sessions,sectors)
    if audit["issues"]:
        raise ValueError(f"Ledger mismatch: {audit['issues']}")
    return dict(trades=t,daily=d,equity=c,summary=s),audit


def verify_control(actual, reference, arm):
    for name in TABLES:
        saved=pd.read_csv(reference/f"{name}.csv",float_precision="round_trip")
        expected=saved.loc[saved.strategy==arm].drop(columns="strategy").reset_index(drop=True)
        present=actual[name].reset_index(drop=True).copy()
        if name=="trades" and present.empty and expected.empty:
            continue
        # The October file also contains the alternate-order arm's optional column.
        if "scan_order" in expected and expected.scan_order.isna().all(): expected=expected.drop(columns="scan_order")
        for key in ["entry_ts","exit_ts"]:
            if key in expected:
                expected[key]=pd.to_datetime(expected[key],utc=True).astype("datetime64[ns, UTC]")
                present[key]=pd.to_datetime(present[key],utc=True).astype("datetime64[ns, UTC]")
        try:
            pd.testing.assert_frame_equal(expected.sort_index(axis=1),present.sort_index(axis=1),
                                          check_dtype=False,check_exact=False,rtol=0,atol=1e-6)
        except AssertionError as exc:
            raise ValueError(f"Original control differs: {arm} {name}") from exc


def study(reference, cfg, sessions, scores, prices, gate, control_names):
    orders, probabilities = {},{}
    for day in sessions:
        orders[day],probabilities[day]=opening_order(cfg["symbols"],scores,day)
    groups={n:[] for n in TABLES}; audits={}; pairs={}; daily={}
    for ranked in [False,True]:
        for i,gated in enumerate([False,True]):
            arm=("opening_ml" if ranked else "original")+("_spy_ge010" if gated else "_no_gate")
            print("Replaying",arm,flush=True)
            frames,audits[arm]=replay(scores,prices,cfg,sessions,gate,orders,gated,ranked)
            if not ranked: verify_control(frames,reference,control_names[i])
            daily[arm]=frames["daily"]
            for n in TABLES: groups[n].append(frames[n].assign(strategy=arm))
    for suffix in ["no_gate","spy_ge010"]:
        p=paired_daily_comparison(daily["original_"+suffix],daily["opening_ml_"+suffix])
        p["classification"]="exploratory_previously_inspected"
        pairs[suffix]=p
    return ({n:pd.concat(fs,ignore_index=True) for n,fs in groups.items()},
            dict(ledger=audits,paired_daily=pairs,orders=orders,opening_ml_prob=probabilities))


def run(august, october, frozen, output):
    august,october,frozen,output=[Path(p).resolve() for p in (august,october,frozen,output)]
    if any(p==output or p in output.parents for p in [august,october,frozen]):
        raise ValueError("Use a separate new research directory")
    output.mkdir(parents=True,exist_ok=False)
    info=dict(status="started",policy=POLICY,classification="historical_exploration_only",
              independent_evidence=False,profitability_validated=False,training_runs=0,
              primary="opening_ml minus original mean paired daily return, no SPY",
              secondary="same contrast with SPY",periods=["august","october_november"],
              fitted_thresholds=False,source_sha256=aug.digest(Path(__file__)))
    report=output/"report.json"; aug.write_json(report,info)
    watched={}
    def checked(path, expected=None):
        digest=aug.digest(path)
        if expected is not None and digest!=expected: raise ValueError(f"Input hash mismatch: {path.name}")
        watched[path]=digest
    try:
        with ExitStack() as stack:
            forward.offline(stack)
            for p in [august/"manifest.json",october/"report.json",frozen/"ready.json",
                      frozen/"model.json",frozen/"registration.json"]: checked(p)
            cfg,bundle=forward.load_ready(frozen)
            manifest=forward.read(august/"manifest.json"); old=forward.read(october/"report.json")
            ready=forward.read(frozen/"ready.json")
            if (manifest["status"]!="completed_development_comparison" or manifest["config"]!=cfg
                or manifest["windows"]!=[list(aug.WINDOW)] or old["status"]!="completed_fixed_period_comparison"
                or old["initial_config"]!=cfg or old["sessions"]!=locked.SESSIONS or old["inputs_unchanged"] is not True
                or old["training_cache_sha256"]!=ready["training_matrix"]
                or old["reference_manifest_sha256"]!=watched[august/"manifest.json"]):
                raise ValueError("Reference periods, model or configuration differ")
            for n in TABLES:
                file=n+".csv"
                checked(august/file, manifest["outputs"][file])
                checked(october/file, old["outputs"][file])
            checked(august/CACHE,manifest["outputs"][CACHE])
            sessions=aug.load_sessions(); symbols=cfg["symbols"]
            scores,prices,gate=load_cache(august/CACHE,symbols,sessions)
            first=study(august,cfg,sessions,scores,prices,gate,AUG_ARMS)
            del scores,prices,gate
            bench=aug.reference.BENCH
            def extension(s):
                path=october/"sip_extension"/f"{s}_5min.csv"
                x,h,_=locked.read_extension(path,s,bench)
                if h!=old["extension_sha256"][s]: raise ValueError(f"New-period source differs: {s}")
                watched[path]=h
                return x
            market=extension(bench); gate=locked.build_spy_gate(market); scores={}; prices={}
            for s in symbols:
                x=extension(s); scores[s]=locked.score(x,market,cfg,bundle)
                prices[s]=x.loc[x.timestamp>=locked.START,aug.OHLCV]
            second=study(october,cfg,locked.SESSIONS,scores,prices,gate,LOCKED_ARMS)
            if any(aug.digest(p)!=h for p,h in watched.items()): raise ValueError("Source changed during trial")
            for label,(tables,detail) in [("august",first),("october_november",second)]:
                dest=output/label; dest.mkdir()
                for n,df in tables.items(): df.to_csv(dest/f"{n}.csv",index=False)
                aug.write_json(dest/"audit.json",detail)
            info.update(status="completed_exploratory_trial",all_controls_reproduced=True,
                        verified_inputs={str(p):h for p,h in watched.items()},
                        outputs={p.relative_to(output).as_posix():aug.digest(p) for p in output.glob("*/*")})
            aug.write_json(report,info)
            for label,(tables,_) in [("august",first),("october_november",second)]:
                print(label); print(tables["summary"].to_string(index=False))
    except BaseException as exc:
        info.update(status="failed_not_validated",error=str(exc)); aug.write_json(report,info); raise
    print("Status:",info["status"]); print("Report:",report)
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--august",type=Path,default=BASE/"research_runs/jarjestys-20261109-130806/reference")
    p.add_argument("--october",type=Path,default=BASE/"research_runs/lukittu-20261109-151005")
    p.add_argument("--frozen",type=Path,default=BASE/"research_runs/forward-20261110")
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    try: run(a.august,a.october,a.frozen,a.output_dir)
    except (ValueError,OSError,KeyError,RuntimeError) as exc: raise SystemExit(f"Stopped: {exc}")


if __name__=="__main__": main()
