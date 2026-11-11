import json,pickle
from pathlib import Path
import pandas as pd,numpy as np
from features import FEATURES,enrich
from model_pipeline import validate_bundle
from backtest_execution import next_entry_and_path, simulate_long_exit

BASE=Path(__file__).resolve().parent; cfg=json.loads((BASE/"config.json").read_text())
with open(BASE/"models/ensemble.pkl","rb") as f: bundle=pickle.load(f)
validate_bundle(bundle,cfg)
models=bundle["models"]; data=BASE/"data"; newsdir=BASE/"news"
bench=enrich(pd.read_csv(data/f"{cfg['benchmark_symbols'][0]}_5min.csv"))
frames=[]
price_paths={}
for s in cfg["symbols"]:
    p=data/f"{s}_5min.csv"
    if not p.exists(): continue
    nf=newsdir/f"{s}_news.csv"; news=pd.read_csv(nf) if nf.exists() else None
    raw=pd.read_csv(p)
    raw["timestamp"]=pd.to_datetime(raw["timestamp"],utc=True)
    raw=raw.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    # Full unfiltered OHLC data is used for fills, never the signal frame.
    price_paths[s]=raw
    x=enrich(raw,bench,news,cfg["news_decay_minutes"],cfg["news_lookback_minutes"]).dropna(subset=FEATURES+["atr"])
    x=x.iloc[int(len(x)*cfg["train_fraction"]):].copy()
    probs=[m.predict_proba(x[FEATURES].to_numpy())[:,1] for m in models]
    x["ml_prob"]=np.mean(probs,axis=0); x["symbol"]=s
    x["regime_score"]=0.5
    x.loc[x.market_vs_ema21>0.002,"regime_score"]+=0.25
    x.loc[x.market_vs_ema21<-0.002,"regime_score"]-=0.25
    x.loc[x.market_atr_pct>0.015,"regime_score"]-=0.10
    frames.append(x)
df=pd.concat(frames).sort_values("timestamp").reset_index(drop=True)

def run(prob_threshold,score_threshold):
    cash=cfg["initial_cash"]; trades=[]; daily={}
    slip=(cfg["slippage_bps"]+cfg["spread_bps"])/10000
    # One position at a time globally: conservative and avoids accidental overlapping cash.
    i=0
    while i < len(df)-1:
        r=df.iloc[i]; day=str(pd.Timestamp(r.timestamp).date())
        if daily.get(day,0)>=cfg["max_trades_per_day"]:
            i+=1; continue
        news_score=float(r.news_score)
        ml_score=max(0,min(100,(r.ml_prob-.5)*200))
        market_score=max(0,min(100,r.regime_score*100))
        news_component=50+50*max(-1,min(1,news_score))
        total=cfg["ml_weight"]*ml_score+cfg["news_weight"]*news_component+cfg["market_weight"]*market_score
        if r.ml_prob<prob_threshold or total<score_threshold:
            i+=1; continue
        entry_bar,g=next_entry_and_path(price_paths[str(r.symbol)],r.timestamp,max_bars=12)
        if entry_bar is None:
            i+=1; continue
        entry=float(entry_bar.open)*(1+slip)
        dist=max(float(r.atr)*cfg["stop_atr_multiplier"],entry*.003)
        stop=entry-dist; target=entry+dist*cfg["reward_risk_ratio"]
        qty=int(min((cash*cfg["risk_per_trade_pct"])/dist,(cash*cfg["max_position_pct"])/entry))
        if qty<1: i+=1; continue
        # The entry/exit path is strictly the original symbol, including
        # price bars that might lack usable ML/news features.
        exitp,reason,exit_ts=simulate_long_exit(g,stop,target,slip)
        if exitp is None:
            i+=1; continue
        pnl=(exitp-entry)*qty-cfg["commission_per_trade"]; cash+=pnl
        trades.append({"timestamp":entry_bar.timestamp,"symbol":r.symbol,"pnl":pnl,"score":total,"ml_prob":r.ml_prob,"news_score":news_score,"reason":reason})
        daily[day]=daily.get(day,0)+1
        # Advance past exit to prevent overlapping trades.
        later=df.index[df.timestamp>exit_ts]
        i=int(later[0]) if len(later) else len(df)
    out=pd.DataFrame(trades)
    return cash,out

print("\n=== V5.2 DEVELOPMENT BACKTEST / NOT INDEPENDENT VALIDATION ===")
print("NOTE: best thresholds are selected on this same data; results are NOT OOS evidence.")
best=None
for pt in [0.52,0.55,0.58,0.61]:
    for st in [50,55,60,65,70]:
        cash,o=run(pt,st)
        ret=(cash/cfg["initial_cash"]-1)*100
        row=(ret,pt,st,cash,len(o),o)
        if best is None or ret>best[0]: best=row
        print(f"prob>={pt:.2f} score>={st:2d} | return {ret:7.2f}% | trades {len(o):4d}")
ret,pt,st,cash,_,out=best
print("\n--- DEVELOPMENT SET: HIGHEST RETURN (NOT AN OOS RESULT) ---")
print(f"Initial: ${cfg['initial_cash']:,.2f}")
print(f"Final:   ${cash:,.2f}")
print(f"Return:  {ret:.2f}%")
print(f"Trades:  {len(out)}")
print(f"Chosen thresholds: probability>={pt:.2f}, score>={st}")
if len(out):
    print(f"Win rate: {(out.pnl>0).mean()*100:.2f}%")
    gp=out.loc[out.pnl>0,"pnl"].sum(); gl=-out.loc[out.pnl<0,"pnl"].sum()
    print(f"Profit factor: {gp/gl if gl else float('inf'):.2f}")
    eq=pd.concat([pd.Series([cfg["initial_cash"]]),cfg["initial_cash"]+out.pnl.cumsum()],ignore_index=True)
    dd=(eq/eq.cummax()-1).min()*100
    print(f"Max drawdown: {dd:.2f}%")
    print(f"Avg news score: {out.news_score.mean():.3f}")
    out.to_csv(BASE/"backtest_trades.csv",index=False)
else:
    print("Ei kauppoja. Tarkista news/threshold-diagnostiikka.")

