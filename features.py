import numpy as np
import pandas as pd

FEATURES = [
    "ret_1","ret_3","ret_12","range_pct","close_vs_ema9","close_vs_ema21",
    "ema9_vs_ema21","rsi14","atr_pct","volume_ratio","hour_sin","hour_cos",
    "market_ret_3","market_ret_12","market_vs_ema21","market_atr_pct",
    "news_score"
]

def enrich(df, market=None, news=None, news_decay_minutes=120, news_lookback_minutes=360):
    x=df.copy()
    x["timestamp"]=pd.to_datetime(x["timestamp"],utc=True)
    x=x.sort_values("timestamp").reset_index(drop=True)
    c=x.close; h=x.high; l=x.low; v=x.volume
    x["ret_1"]=c.pct_change(); x["ret_3"]=c.pct_change(3); x["ret_12"]=c.pct_change(12)
    x["ema9"]=c.ewm(span=9,adjust=False).mean(); x["ema21"]=c.ewm(span=21,adjust=False).mean()
    x["close_vs_ema9"]=c/x.ema9-1; x["close_vs_ema21"]=c/x.ema21-1; x["ema9_vs_ema21"]=x.ema9/x.ema21-1
    d=c.diff(); gain=d.clip(lower=0).rolling(14).mean(); loss=(-d.clip(upper=0)).rolling(14).mean()
    x["rsi14"]=100-100/(1+gain/loss.replace(0,np.nan))
    pc=c.shift(1); tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    x["atr"]=tr.rolling(14).mean(); x["atr_pct"]=x.atr/c; x["range_pct"]=(h-l)/c
    x["volume_ratio"]=v/v.rolling(20).mean()
    mins=x.timestamp.dt.hour*60+x.timestamp.dt.minute
    x["hour_sin"]=np.sin(2*np.pi*mins/390); x["hour_cos"]=np.cos(2*np.pi*mins/390)
    if market is not None:
        m=market[["timestamp","close","atr_pct"]].copy().rename(columns={"close":"mclose","atr_pct":"market_atr_pct"})
        x=pd.merge_asof(x.sort_values("timestamp"),m.sort_values("timestamp"),left_on="timestamp",right_on="timestamp",direction="backward")
        x["market_ret_3"]=x.mclose.pct_change(3); x["market_ret_12"]=x.mclose.pct_change(12)
        ma=x.mclose.ewm(span=21,adjust=False).mean(); x["market_vs_ema21"]=x.mclose/ma-1
    else:
        for c2 in ["market_ret_3","market_ret_12","market_vs_ema21","market_atr_pct"]: x[c2]=0.0
    x["news_score"]=0.0
    if news is not None and len(news):
        n=news.copy(); n["timestamp"]=pd.to_datetime(n["timestamp"],utc=True)
        n=n.sort_values("timestamp")[["timestamp","sentiment"]].rename(columns={"sentiment":"news_sentiment"})
        # Only news published before/equal to the bar can affect it. Apply exponential decay.
        n=n.rename(columns={"timestamp":"news_timestamp"})
        x=pd.merge_asof(x.sort_values("timestamp"),n,left_on="timestamp",right_on="news_timestamp",
                        direction="backward",tolerance=pd.Timedelta(minutes=news_lookback_minutes))
        age=(x["timestamp"]-x["news_timestamp"]).dt.total_seconds()/60.0
        x["news_score"]=x["news_sentiment"].fillna(0.0)*np.exp(-np.maximum(age,0)/max(news_decay_minutes,1))
        x=x.drop(columns=["news_timestamp","sentiment"],errors="ignore")
    return x

def label(x,horizon=3,threshold=.002):
    f=x.close.shift(-horizon)/x.close-1
    y=x.copy(); y["target"]=(f>=threshold).astype(int); y.loc[f.isna(),"target"]=np.nan
    return y






