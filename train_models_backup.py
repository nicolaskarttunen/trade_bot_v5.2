import json,pickle
from pathlib import Path
import pandas as pd
from ml_models import LogisticGD,RandomLinearEnsemble,CentroidClassifier
from features import FEATURES,enrich,label

BASE=Path(__file__).resolve().parent; cfg=json.loads((BASE/"config.json").read_text())
data=BASE/"data"; newsdir=BASE/"news"; models=BASE/"models"; models.mkdir(exist_ok=True)
bench=enrich(pd.read_csv(data/f"{cfg['benchmark_symbols'][0]}_5min.csv"))
train=[]; test=[]
for s in cfg["symbols"]:
    p=data/f"{s}_5min.csv"
    if not p.exists(): continue
    nfile=newsdir/f"{s}_news.csv"
    news=pd.read_csv(nfile) if nfile.exists() else None
    x=label(enrich(pd.read_csv(p),bench,news,cfg["news_decay_minutes"],cfg["news_lookback_minutes"]),
            cfg["label_horizon_bars"],cfg["label_threshold_pct"])
    x=x.dropna(subset=FEATURES+["target"])
    cut=int(len(x)*cfg["train_fraction"]); train.append(x.iloc[:cut]); test.append(x.iloc[cut:])
if not train: raise SystemExit("Data puuttuu. Aja download_data.py ensin.")
tr=pd.concat(train); te=pd.concat(test); X,y=tr[FEATURES].to_numpy(),tr.target.astype(int).to_numpy()
models_list=[LogisticGD(),RandomLinearEnsemble(),CentroidClassifier()]
for m in models_list: m.fit(X,y)
with open(models/"ensemble.pkl","wb") as f: pickle.dump({"models":models_list,"features":FEATURES},f)
print(f"Train: {len(tr):,} | Test: {len(te):,}")
print(f"News feature: {'KYLLÄ' if 'news_score' in FEATURES else 'EI'}")
print("V5.2 ML-ensemble tallennettu:",models/"ensemble.pkl")
