import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent
CSV = BASE / "trades.csv"

st.set_page_config(
    page_title="V5.2 Trading Dashboard",
    layout="wide"
)

st.title("V5.2 AI Day Trader")
st.caption("Paper Trading Dashboard")

if CSV.exists():
    df = pd.read_csv(CSV)

    if len(df):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        buys = df[df["side"].str.upper() == "BUY"]
        sells = df[df["side"].str.upper() == "SELL"]

        pnl = pd.to_numeric(df["pnl"], errors="coerce").fillna(0).sum()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Trades", len(df))
        c2.metric("Buys", len(buys))
        c3.metric("Sells", len(sells))
        c4.metric("P/L", f"${pnl:,.2f}")

        st.subheader("Trades")
        st.dataframe(
            df.sort_values("timestamp", ascending=False),
            use_container_width=True
        )

        st.subheader("Symbols")
        st.bar_chart(df["symbol"].value_counts())

    else:
        st.info("Ei vielä treidejä. Dashboard odottaa V5.2:n ensimmäisiä paper-kauppoja.")
else:
    st.warning("trades.csv puuttuu.")

st.caption(
    "Päivitetty: " +
    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)
