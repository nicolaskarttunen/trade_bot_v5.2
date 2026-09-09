import json,pickle
from pathlib import Path
import pandas as pd,numpy as np
from features import FEATURES,enrich,label
from ml_models import LogisticGD,RandomLinearEnsemble,CentroidClassifier

BASE=Path(__file__).resolve().parent
cfg=json.loads((BASE/"config.json").read_text())
data=BASE/"data"
newsdir=BASE/"news"

SYMBOLS=cfg["symbols"]
BENCH=cfg["benchmark_symbols"][0]

WINDOWS=[
    ("2023-01-01","2023-12-31","2024-01-01","2024-06-30"),
    ("2023-01-01","2024-06-30","2024-07-01","2025-01-31"),
    ("2023-01-01","2025-01-31","2025-02-01","2025-08-31"),
    ("2023-01-01","2025-08-31","2025-09-01","2026-02-28"),
    ("2023-01-01","2026-02-28","2026-03-01","2026-08-31"),
]

def load_market():
    m=pd.read_csv(data/f"{BENCH}_5min.csv")
    m["timestamp"]=pd.to_datetime(m["timestamp"],utc=True)
    pc=m["close"].shift(1)
    tr=pd.concat([
        m["high"]-m["low"],
        (m["high"]-pc).abs(),
        (m["low"]-pc).abs()
    ],axis=1).max(axis=1)
    m["atr_pct"]=tr.rolling(14).mean()/m["close"]
    return m

def load_symbol(s,market):
    p=data/f"{s}_5min.csv"
    if not p.exists():
        return None
    nf=newsdir/f"{s}_news.csv"
    news=pd.read_csv(nf) if nf.exists() else None
    x=enrich(
        pd.read_csv(p),
        market,
        news,
        cfg["news_decay_minutes"],
        cfg["news_lookback_minutes"]
    )
    x=label(
        x,
        horizon=3,
        threshold=0.002
    )
    x=x.dropna(subset=FEATURES+["atr","target"]).copy()
    x["symbol"]=s
    return x

def train_ensemble(train):
    X=train[FEATURES].to_numpy()
    y=train["target"].to_numpy().astype(int)

    models=[
        LogisticGD(lr=0.03,epochs=500,l2=0.01),
        RandomLinearEnsemble(n_models=25,seed=42),
        CentroidClassifier()
    ]

    for m in models:
        m.fit(X,y)

    return models

def predict(models,x):
    probs=[
        m.predict_proba(x[FEATURES].to_numpy())[:,1]
        for m in models
    ]
    return np.mean(probs,axis=0)

def run_test(df,models,prob_threshold=0.52,score_threshold=50):
    cash=cfg["initial_cash"]
    trades=[]
    daily={}
    slip=(cfg["slippage_bps"]+cfg["spread_bps"])/10000

    df=df.sort_values("timestamp").reset_index(drop=True)

    i=0
    while i<len(df)-1:
        r=df.iloc[i]
        day=str(pd.Timestamp(r.timestamp).date())

        if daily.get(day,0)>=cfg["max_trades_per_day"]:
            i+=1
            continue

        news_score=float(r.news_score)
        ml_score=max(0,min(100,(r.ml_prob-.5)*200))
        market_score=max(0,min(100,r.regime_score*100))

        news_component=50+50*max(-1,min(1,news_score))

        total=(
            cfg["ml_weight"]*ml_score+
            cfg["news_weight"]*news_component+
            cfg["market_weight"]*market_score
        )

        if r.ml_prob<prob_threshold or total<score_threshold:
            i+=1
            continue

        entry_bar=df.iloc[i+1]
        entry=float(entry_bar.open)*(1+slip)

        dist=max(
            float(r.atr)*cfg["stop_atr_multiplier"],
            entry*.003
        )

        stop=entry-dist
        target=entry+dist*cfg["reward_risk_ratio"]

        qty=int(min(
            (cash*cfg["risk_per_trade_pct"])/dist,
            (cash*cfg["max_position_pct"])/entry
        ))

        if qty<1:
            i+=1
            continue

        g=df[
            (df.symbol==r.symbol) &
            (df.timestamp>=entry_bar.timestamp)
        ].head(12)

        if len(g)==0:
            i+=1
            continue

        exitp=None
        reason="time_exit"
        exit_ts=g.iloc[-1].timestamp

        for _,b in g.iterrows():
            if b.low<=stop:
                exitp=stop*(1-slip)
                reason="stop"
                exit_ts=b.timestamp
                break
            if b.high>=target:
                exitp=target*(1-slip)
                reason="target"
                exit_ts=b.timestamp
                break

        if exitp is None:
            exitp=float(g.iloc[-1].close)*(1-slip)

        pnl=(exitp-entry)*qty-cfg["commission_per_trade"]
        cash+=pnl

        trades.append({
            "timestamp":entry_bar.timestamp,
            "symbol":r.symbol,
            "pnl":pnl,
            "score":total,
            "ml_prob":r.ml_prob,
            "news_score":news_score,
            "reason":reason
        })

        daily[day]=daily.get(day,0)+1

        later=df.index[df.timestamp>exit_ts]
        i=int(later[0]) if len(later) else len(df)

    return cash,pd.DataFrame(trades)

print("\n=== V5.2 TRUE WALK-FORWARD ML + NEWS ===")
print("Mallia koulutetaan aina vain testijaksoa edeltävällä datalla.")
print("Testidata ei osallistu mallin koulutukseen.\n")

market=load_market()
all_data={}

for s in SYMBOLS:
    print(f"Ladataan {s}...")
    x=load_symbol(s,market)
    if x is not None:
        all_data[s]=x
        print(f"  {len(x):,} riviä")

results=[]

for wi,(tr_start,tr_end,te_start,te_end) in enumerate(WINDOWS,1):

    print(f"\n{'='*60}")
    print(f"WINDOW {wi}")
    print(f"Train: {tr_start} -> {tr_end}")
    print(f"Test:  {te_start} -> {te_end}")
    print(f"{'='*60}")

    train_frames=[]
    test_frames=[]

    for s,x in all_data.items():

        train=x[
            (x.timestamp>=pd.Timestamp(tr_start,tz="UTC")) &
            (x.timestamp<=pd.Timestamp(tr_end+" 23:59:59",tz="UTC"))
        ].copy()

        test=x[
            (x.timestamp>=pd.Timestamp(te_start,tz="UTC")) &
            (x.timestamp<=pd.Timestamp(te_end+" 23:59:59",tz="UTC"))
        ].copy()

        if len(train):
            train_frames.append(train)

        if len(test):
            test_frames.append(test)

    train=pd.concat(train_frames,ignore_index=True)
    test=pd.concat(test_frames,ignore_index=True)

    print(f"Train rows: {len(train):,}")
    print(f"Test rows:  {len(test):,}")

    print("Koulutetaan ML-ensemble...")
    models=train_ensemble(train)

    test["ml_prob"]=predict(models,test)

    test["regime_score"]=0.5
    test.loc[test.market_vs_ema21>0.002,"regime_score"]+=0.25
    test.loc[test.market_vs_ema21<-0.002,"regime_score"]-=0.25
    test.loc[test.market_atr_pct>0.015,"regime_score"]-=0.10

    best=None
    pt=0.52
    st=50

    for _pt in [0.52]:
        for _st in [50]:

            cash,out=run_test(
                test,
                models,
                _pt,
                _st
            )

            ret=(cash/cfg["initial_cash"]-1)*100

            if best is None or ret>best[0]:
                best=(ret,pt,st,cash,out)

    ret,pt,st,cash,out=best

    if len(out):
        win=(out.pnl>0).mean()*100
        gp=out.loc[out.pnl>0,"pnl"].sum()
        gl=-out.loc[out.pnl<0,"pnl"].sum()
        pf=gp/gl if gl else float("inf")
        eq=cfg["initial_cash"]+out.pnl.cumsum()
        dd=(eq/eq.cummax()-1).min()*100
    else:
        win=0
        pf=0
        dd=0

    print(f"\nWINDOW {wi} TULOS")
    print(f"Return:       {ret:.2f}%")
    print(f"Trades:       {len(out)}")
    print(f"Win rate:     {win:.2f}%")
    print(f"Profit factor:{pf:.2f}")
    print(f"Max drawdown: {dd:.2f}%")
    print(f"Thresholds:   prob>={pt:.2f}, score>={st}")

    if len(out):
        print("Syyt:")
        print(out.reason.value_counts().to_string())

    results.append({
        "window":wi,
        "train_start":tr_start,
        "train_end":tr_end,
        "test_start":te_start,
        "test_end":te_end,
        "return_pct":ret,
        "trades":len(out),
        "win_rate":win,
        "profit_factor":pf,
        "max_drawdown_pct":dd
    })

result_df=pd.DataFrame(results)

print("\n\n=== WALK-FORWARD YHTEENVETO ===")
print(result_df.to_string(index=False))

total_return=((1+result_df.return_pct/100).prod()-1)*100
print(f"\nYhdistetty testijaksojen tuotto: {total_return:.2f}%")
print(f"Testijaksoja voitollisena: {(result_df.return_pct>0).sum()}/{len(result_df)}")
print(f"Kauppoja yhteensä: {result_df.trades.sum()}")

result_df.to_csv(BASE/"walk_forward_results.csv",index=False)
print("\nTallennettu: walk_forward_results.csv")

