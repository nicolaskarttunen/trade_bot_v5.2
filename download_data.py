import os,json,time
from pathlib import Path
from datetime import datetime,timedelta
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame,TimeFrameUnit
from alpaca.data.enums import DataFeed

BASE=Path(__file__).resolve().parent
load_dotenv(BASE/".env")
key=os.getenv("ALPACA_API_KEY"); secret=os.getenv("ALPACA_SECRET_KEY")
if not key or not secret:
    raise SystemExit("Alpaca-avaimet puuttuvat .env-tiedostosta.")

cfg=json.loads((BASE/"config.json").read_text())
client=StockHistoricalDataClient(key,secret)
out=BASE/"data"
out.mkdir(exist_ok=True)
symbols=list(dict.fromkeys(cfg["symbols"]+cfg["benchmark_symbols"]))

start_all=datetime.fromisoformat(cfg.get("history_start","2025-01-01"))
end_all=datetime.fromisoformat(cfg.get("history_end","2026-09-01"))

for s in symbols:
    print(f"Ladataan {s}...")
    try:
        parts=[]
        chunk_start=start_all

        while chunk_start < end_all:
            chunk_end=min(chunk_start+timedelta(days=45),end_all)

            req=StockBarsRequest(
                symbol_or_symbols=s,
                timeframe=TimeFrame(cfg["timeframe_minutes"],TimeFrameUnit.Minute),
                start=chunk_start.isoformat(),
                end=chunk_end.isoformat(),
                feed=DataFeed.IEX,
                limit=10000
            )

            df=client.get_stock_bars(req).df

            if df is not None and len(df)>0:
                if isinstance(df.index,pd.MultiIndex):
                    df=df.xs(s,level=0)
                df=df.reset_index()
                parts.append(df)
                print(f"  {chunk_start.date()} -> {chunk_end.date()}: {len(df):,} bars")

            chunk_start=chunk_end
            time.sleep(.3)

        if not parts:
            print("Ei dataa")
            continue

        df=pd.concat(parts,ignore_index=True)
        df=df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df.to_csv(out/f"{s}_5min.csv",index=False)

        print(f"VALMIS {s}: {len(df):,} bars | {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")

    except Exception as e:
        print("ERROR",e)

    time.sleep(.5)
