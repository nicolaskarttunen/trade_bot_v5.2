from pathlib import Path
import pandas as pd,numpy as np
BASE=Path(__file__).resolve().parent
p=BASE/"backtest_trades.csv"
if not p.exists(): raise SystemExit("Aja backtest.py ensin.")
pnl=pd.read_csv(p)["pnl"].to_numpy()
if len(pnl)<2: raise SystemExit("Liian vähän kauppoja Monte Carloon.")
rng=np.random.default_rng(42); finals=[]; maxdds=[]
for _ in range(5000):
    s=rng.choice(pnl,size=len(pnl),replace=True)
    equity=100000+np.cumsum(s); peak=np.maximum.accumulate(equity)
    dd=(equity-peak)/peak
    finals.append(equity[-1]); maxdds.append(dd.min())
print("=== MONTE CARLO 5000 ===")
print(f"Median final: ${np.median(finals):,.0f}")
print(f"5% final:     ${np.percentile(finals,5):,.0f}")
print(f"95% final:    ${np.percentile(finals,95):,.0f}")
print(f"Median max DD: {np.median(maxdds)*100:.2f}%")
print(f"Worst 5% DD:   {np.percentile(maxdds,5)*100:.2f}%")
