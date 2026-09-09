import json, os, re, time
from pathlib import Path
from datetime import datetime, timezone
import requests
import pandas as pd
from dotenv import load_dotenv

BASE=Path(__file__).resolve().parent
load_dotenv(BASE/".env")
KEY=os.getenv("ALPACA_API_KEY"); SECRET=os.getenv("ALPACA_SECRET_KEY")
if not KEY or not SECRET: raise SystemExit("Alpaca-avaimet puuttuvat .env-tiedostosta.")

POS=set("""beat beats beating strong stronger strongest bullish bull upside upsurge surge surged rises rise rising gain gains gained growth
positive optimism optimistic upgrade upgraded outperform buy bought winner win wins record breakout momentum robust improve improved
expands expansion demand demand-growth revenue revenues profit profits earnings eps guidance raised raises rally rallied partnership launch launched
approval approved contract contracts deal deals acquisition acquires""".split())
NEG=set("""miss misses missed weak weaker weakest bearish bear downside downtrend fall falls fell falling loss losses lost decline declines declined
negative pessimism pessimistic downgrade downgraded underperform sell sold risk risks warning warns warned cut cuts lowered lower guidance weakens
lawsuit probe investigation investigation? fraud recall recalls delay delayed layoffs layoffs inflation recession""".replace("?","").split())

def sentiment(text):
    words=re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", (text or "").lower())
    if not words: return 0.0
    p=sum(w in POS for w in words); n=sum(w in NEG for w in words)
    raw=(p-n)/max((p+n),1)
    return float(max(-1,min(1,raw)))

def fetch_symbol(symbol,start,end):
    url="https://data.alpaca.markets/v1beta1/news"
    headers={"APCA-API-KEY-ID":KEY,"APCA-API-SECRET-KEY":SECRET}
    params={"symbols":symbol,"start":start,"end":end,"limit":50,"include_content":"false"}
    rows=[]; token=None
    while True:
        if token: params["page_token"]=token
        resp=requests.get(url,headers=headers,params=params,timeout=30)
        resp.raise_for_status()
        data=resp.json()
        for item in data.get("news",[]):
            ts=item.get("created_at") or item.get("updated_at")
            text=" ".join(str(item.get(k) or "") for k in ("headline","summary"))
            rows.append({"timestamp":ts,"symbol":symbol,"sentiment":sentiment(text),
                         "headline":item.get("headline",""),"source":item.get("source","")})
        token=data.get("next_page_token")
        if not token or len(rows)>=50000: break
    return pd.DataFrame(rows)

def main():
    cfg=json.loads((BASE/"config.json").read_text())
    out=BASE/"news"; out.mkdir(exist_ok=True)
    for s in cfg["symbols"]:
        print(f"Ladataan uutiset {s}...")
        try:
            df=fetch_symbol(s,cfg["history_start"],cfg["history_end"])
            if len(df):
                df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True)
                df=df.sort_values("timestamp").drop_duplicates(subset=["timestamp","headline"])
                df.to_csv(out/f"{s}_news.csv",index=False)
                print(f"{len(df):,} uutista")
            else: print("Ei uutisia")
        except Exception as e: print("ERROR",e)
        time.sleep(.3)
if __name__=="__main__": main()
