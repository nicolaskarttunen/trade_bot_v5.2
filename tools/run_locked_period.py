"""Locked 2026-10-20..11-06 historical comparison. Research only."""
import argparse
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import socket
import sys
from unittest.mock import patch
import zipfile

if __name__ == "__main__":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import requests

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from disk_training import TrainingStore, fit_disk_bundle, RECORD
from features import enrich, normalize_bars, FEATURES, FEATURE_HISTORY_BARS
from model_pipeline import predict_bundle, trade_score
from order_trial import CACHE, POLICY, daily_order
from research_validation import build_spy_gate, paired_daily_comparison
from tools import run_august_simulation as aug
from tools.run_order_trial import one_arm, TABLES

NY = aug.NY
WARM = pd.Timestamp("2026-10-01", tz=NY)
START = pd.Timestamp("2026-10-20", tz=NY)
END = pd.Timestamp("2026-11-07", tz=NY)  # exclusive
DAYS = ("2026-10-20 2026-10-21 2026-10-22 2026-10-23 "
        "2026-10-26 2026-10-27 2026-10-28 2026-10-29 2026-10-30 "
        "2026-11-02 2026-11-03 2026-11-04 2026-11-05 2026-11-06").split()
SESSIONS = dict.fromkeys(DAYS, "16:00")
ARMS = [("original_no_gate", False, False), ("original_spy_ge010", False, True),
        ("daily_order_no_gate", True, False), ("daily_order_spy_ge010", True, True)]
PARAMS = dict(feed="sip", adjustment="raw", timeframe="5Min", asof="2026-08-31",
              sort="asc", limit=10000, start=WARM.tz_convert("UTC").isoformat(),
              end=(END-pd.Timedelta(seconds=1)).tz_convert("UTC").isoformat())


def fetch_bars(symbol, headers):
    """GET historical bars only. Never pass credentials through redirects or logs."""
    params = dict(PARAMS)
    bars, seen = [], set()
    for _ in range(32):
        with requests.get(f"https://data.alpaca.markets/v2/stocks/{symbol}/bars",
                          headers=headers, params=params, timeout=(5,45),
                          allow_redirects=False) as response:
            if response.status_code != 200:
                raise ValueError(f"{symbol}: historical SIP GET returned HTTP {response.status_code}")
            body = response.json()
        if body.get("symbol") != symbol or not isinstance(body.get("bars"),list):
            raise ValueError(f"{symbol}: invalid historical data envelope")
        bars.extend(body["bars"])
        token=body.get("next_page_token")
        if token is None:
            if not bars:
                raise ValueError(f"{symbol}: empty historical SIP response")
            return pd.DataFrame(bars).rename(columns=dict(t="timestamp",o="open",h="high",l="low",c="close",v="volume"))[aug.OHLCV]
        if not isinstance(token,str) or not token or token in seen:
            raise ValueError(f"{symbol}: invalid pagination")
        seen.add(token)
        params["page_token"]=token
    raise ValueError(f"{symbol}: pagination limit exceeded")


def read_extension(path, symbol, benchmark):
    raw=path.read_bytes()
    frame=pd.read_csv(io.BytesIO(raw),usecols=aug.OHLCV,float_precision="round_trip")
    ts=pd.to_datetime(frame.timestamp,utc=True,errors="raise")
    if ts.isna().any() or not ts.is_monotonic_increasing or ts.duplicated().any():
        raise ValueError(f"{symbol}: missing, unordered or duplicate timestamp")
    frame=normalize_bars(frame)
    if frame.empty or ts.min()<WARM or ts.max()>=END:
        raise ValueError(f"{symbol}: extension must contain only the locked fetch window")
    if (ts<START).sum()<FEATURE_HISTORY_BARS:
        raise ValueError(f"{symbol}: insufficient warm-up")
    expected=pd.DatetimeIndex([t for d in DAYS for t in pd.date_range(
        f"{d} 09:30",f"{d} 15:55",freq="5min",tz=NY)]).tz_convert("UTC")
    local=ts.dt.tz_convert(NY)
    regular=ts.loc[(ts>=START)&local.dt.strftime("%H:%M").between("09:30","15:55")]
    if set(regular.dt.tz_convert(NY).dt.strftime("%Y-%m-%d"))!=set(DAYS):
        raise ValueError(f"{symbol}: missing or unexpected evaluation session")
    missing=expected.difference(pd.DatetimeIndex(regular))
    if symbol==benchmark and len(missing):
        raise ValueError(f"{symbol}: missing {len(missing)} scheduled benchmark bars")
    return frame,hashlib.sha256(raw).hexdigest(),[t.isoformat() for t in missing]


def score(frame, market, cfg, bundle):
    # Inference never calls label() or reads targets.
    x=enrich(frame,market,news=None,news_decay_minutes=cfg["news_decay_minutes"],
             news_lookback_minutes=cfg["news_lookback_minutes"])
    x=x.loc[(x.timestamp>=START)&(x.timestamp<END)].copy()
    x=x.loc[np.isfinite(x[[*FEATURES,"atr"]].to_numpy(dtype=float)).all(axis=1)]
    if x.empty:
        raise ValueError("No usable evaluation features")
    p=predict_bundle(bundle,x,cfg)
    return x[["timestamp","signal_available_at","atr","news_score"]].assign(
        ml_prob=p,trade_score=trade_score(p,x.market_ret_3,x.news_score,cfg))


def run(train_data, reference, out, extension=None, env_file=None):
    train_data,reference,out=map(lambda p:Path(p).resolve(),[train_data,reference,out])
    extension=Path(extension).resolve() if extension is not None else None
    if (env_file is None)==(extension is None):
        raise ValueError("Select exactly one of --extension-dir or --env-file")
    if any(out==p or p in out.parents for p in [train_data,reference]+([extension] if extension else [])):
        raise ValueError("Output must be separate from input directories")
    out.mkdir(parents=True,exist_ok=False)
    symbols=list(aug.reference.SYMBOLS); bench=aug.reference.BENCH
    cfg=dict(aug.reference.CFG,feed="sip",strict_price_coverage=True)
    report=dict(status="locked_before_evaluation",period=[DAYS[0],DAYS[-1]],sessions=SESSIONS,
                training_window=list(aug.WINDOW[:2]), training_label_cutoff=aug.CUTOFF.isoformat(),
                arms=ARMS,policy=POLICY,order_by_day={d:daily_order(symbols,d) for d in DAYS},
                news_mode="neutral",parameter_search=False,broker_access=False,initial_config=cfg,
                previously_unseen_confirmed=False,independent_evidence=False,profitability_validated=False,
                extension_source="alpaca_historical_get" if env_file else "user_supplied_not_verified",
                requested_feed_params=PARAMS if env_file else None, source_sha256={
                    n:aug.digest(BASE/n) for n in ["tools/run_locked_period.py","order_trial.py",
                    "tools/run_order_trial.py","intraday_replay.py","replay_audit.py","disk_training.py",
                    "features.py","model_pipeline.py","ml_models.py","news_features.py",
                    "tools/run_august_simulation.py","research_validation.py","v6_5_intraday_replay.py"]})
    report_file=out/"report.json"
    def save(): aug.write_json(report_file,report)
    save()
    try:
        paths={s:train_data/f"{s}_5min.csv" for s in [*symbols,bench]}
        missing=[p.name for p in paths.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)}/{len(paths)} training inputs; first: {missing[:3]}")
        manifest_path=reference/"manifest.json"
        mbytes=manifest_path.read_bytes(); manifest=json.loads(mbytes)
        ref_hash=hashlib.sha256(mbytes).hexdigest()
        ref_report_path=reference/"run_report.json"
        rbytes=ref_report_path.read_bytes(); ref=json.loads(rbytes)
        if (manifest.get("config")!=cfg or manifest.get("windows")!=[list(aug.WINDOW)] or
            manifest.get("status")!="completed_development_comparison" or ref.get("low_memory") is not True
            or ref.get("status")!="completed_development_comparison" or ref.get("source_hashes_unchanged_at_end") is not True
            or manifest["outputs"].get("run_report.json")!=hashlib.sha256(rbytes).hexdigest()
            or manifest["outputs"].get(CACHE)!=aug.digest(reference/CACHE)):
            raise ValueError("Need the verified reference subfolder of the completed August order trial")
        for name in ["features.py","model_pipeline.py","ml_models.py","news_features.py","disk_training.py"]:
            if manifest["source_sha256"].get(name)!=report["source_sha256"][name]:
                raise ValueError(f"Reference training source differs: {name}")
        expected={PureWindowsPath(k).name:v for k,v in ref["input_sha256"].items()}
        if set(expected)!={p.name for p in paths.values()} or any(aug.digest(p)!=expected[p.name] for p in paths.values()):
            raise ValueError("Training history differs from the verified August run")
        report.update(reference_manifest_sha256=ref_hash,training_sources=expected)
        required=int(ref["raw_training_rows_upper_bound"])*RECORD.itemsize*3+1024**3
        if shutil.disk_usage(out).free<required:
            raise ValueError(f"Need at least {required/1024**3:.1f} GiB free")
        if env_file:
            from dotenv import dotenv_values
            if not Path(env_file).is_file(): raise ValueError("Historical-data credentials file not found")
            values=dotenv_values(env_file,interpolate=False)
            if not all(values.get(k) for k in ["ALPACA_API_KEY","ALPACA_SECRET_KEY"]):
                raise ValueError("Historical-data credentials missing in selected file")
            headers={"APCA-API-KEY-ID":values["ALPACA_API_KEY"],"APCA-API-SECRET-KEY":values["ALPACA_SECRET_KEY"]}
            extension=out/"sip_extension"; extension.mkdir()
            for i,s in enumerate(paths,1):
                print(f"SIP history {i}/{len(paths)} {s}",flush=True)
                fetch_bars(s,headers).to_csv(extension/f"{s}_5min.csv",index=False)
            del headers,values
        with ExitStack() as stack:
            for obj,name in [(socket.socket,"connect"),(socket.socket,"connect_ex"),(socket,"create_connection")]:
                stack.enter_context(patch.object(obj,name,side_effect=RuntimeError("Network disabled during replay")))
            ext_paths={s:extension/f"{s}_5min.csv" for s in paths}
            ext_hash={}; gaps={}
            for s,path in ext_paths.items():
                _,ext_hash[s],gaps[s]=read_extension(path,s,bench)
            report.update(status="validated_inputs",extension_sha256=ext_hash,missing_regular_bars=gaps)
            save()
            market,_=aug.read_source(paths[bench],expected[paths[bench].name])
            store=TrainingStore(out/"training_cache"); samples={}
            with zipfile.ZipFile(reference/CACHE) as old_cache:
                for i,s in enumerate(symbols):
                    hist,_=aug.read_source(paths[s],expected[paths[s].name])
                    train,test=aug.prepare(hist,market,cfg)
                    store.append(train)
                    with old_cache.open(f"{i}/scores.csv") as f:
                        saved=pd.read_csv(f,usecols=["timestamp","ml_prob"],float_precision="round_trip")
                    anchors=saved.iloc[sorted(set([0,len(saved)//2,len(saved)-1]))].copy()
                    anchors["timestamp"]=pd.to_datetime(anchors.timestamp,utc=True)
                    sample=anchors.merge(test[["timestamp",*FEATURES]],on="timestamp",validate="one_to_one")
                    if len(sample)!=len(anchors): raise ValueError(f"{s}: missing August model anchors")
                    samples[s]=sample
                    print(f"Prepared training history {i+1}/{len(symbols)} {s}",flush=True)
            del train,test,hist,market
            store.finish()
            if store.rows!=ref["training_rows"] or aug.digest(store.path)!=ref["training_cache_sha256"]:
                raise ValueError("Training matrix differs from August reference")
            report.update(status="training",training_rows=store.rows,training_cache_sha256=ref["training_cache_sha256"])
            save()
            bundle=fit_disk_bundle(store,cfg,"neutral")
            for s,x in samples.items():
                if not np.allclose(predict_bundle(bundle,x,cfg),x.ml_prob,rtol=0,atol=1e-10):
                    raise ValueError(f"{s}: reconstructed model differs at August anchors")
            report["august_anchor_predictions_checked"]=sum(map(len,samples.values()))
            report["training_cache_removed"]=store.cleanup()
            report["status"]="replaying"; save()
            def verified(s):
                x,h,_=read_extension(ext_paths[s],s,bench)
                if h!=ext_hash[s]: raise ValueError(f"{s}: extension changed")
                return x
            market=verified(bench); gate=build_spy_gate(market)
            scores,prices={},{}
            for s in symbols:
                x=verified(s); scores[s]=score(x,market,cfg,bundle)
                prices[s]=x.loc[x.timestamp>=START,aug.OHLCV]
            groups={n:[] for n in TABLES}; audits={}; by_arm={}
            for arm,rotated,gated in ARMS:
                print(f"Replaying {arm}",flush=True)
                frames,audits[arm]=one_arm(scores,prices,symbols,cfg,SESSIONS,gate,gated,rotated,None)
                by_arm[arm]=frames[1]
                for n,f in zip(TABLES,frames): groups[n].append(f.assign(strategy=arm))
            comparisons={}
            for a,b in [(ARMS[0][0],ARMS[1][0]),(ARMS[2][0],ARMS[3][0]),
                        (ARMS[0][0],ARMS[2][0]),(ARMS[1][0],ARMS[3][0])]:
                pair=paired_daily_comparison(by_arm[a],by_arm[b])
                pair["classification"]="fixed_historical_period_novelty_unverified"
                comparisons[f"{b}_minus_{a}"]=pair
            if (any(aug.digest(p)!=expected[p.name] for p in paths.values()) or
                any(aug.digest(p)!=ext_hash[s] for s,p in ext_paths.items()) or
                aug.digest(manifest_path)!=ref_hash or aug.digest(reference/CACHE)!=manifest["outputs"][CACHE]
                or aug.digest(ref_report_path)!=manifest["outputs"]["run_report.json"]):
                raise ValueError("Inputs changed during replay")
            for n,frames in groups.items(): pd.concat(frames,ignore_index=True).to_csv(out/(n+".csv"),index=False)
            aug.write_json(out/"ledger_audit.json",audits)
            report.update(status="completed_fixed_period_comparison",paired_daily_comparisons=comparisons,
                          spy_allowed_sessions=sum(bool(gate[d]) for d in DAYS),inputs_unchanged=True,
                          outputs={p:aug.digest(out/p) for p in [*(n+".csv" for n in TABLES),"ledger_audit.json"]})
            save()
    except BaseException as exc:
        report.update(status="failed_not_validated",error=str(exc)); save(); raise
    print("Status:",report["status"])
    print(pd.read_csv(out/"summary.csv").to_string(index=False))
    print("Report:",report_file)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ["train-data","reference","output-dir"]: p.add_argument("--"+flag,type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument("--env-file",type=Path); source.add_argument("--extension-dir",type=Path)
    a=p.parse_args()
    try: run(a.train_data,a.reference,a.output_dir,a.extension_dir,a.env_file)
    except (OSError,ValueError,KeyError,RuntimeError) as exc: raise SystemExit(f"Stopped: {exc}")


if __name__=="__main__": main()
