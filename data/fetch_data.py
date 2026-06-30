# downloads historic ohlcv data for nse
# open high low close volume

import yfinance as yf
import pandas as pd
from pathlib import Path

RAW_DIR=Path(__file__).parent / "raw"
RAW_DIR.mkdir(exist_ok=True)

WATCHLIST = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
]

def fetch_one(symbol: str, interval : str="5m", period: str="60d")-> pd.DataFrame:
    print(f"Fetching {symbol}...")

    df=yf.download(
        symbol,
        interval=interval,
        period=period,
        progress=False,
        auto_adjust=True
    )

    if df.empty:
        print(f"  ✗ {symbol}: no data returned. Skipping.")
        return pd.DataFrame()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns=df.columns.get_level_values(0)

    df=df.reset_index()

    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"datetime": "timestamp", "date": "timestamp"}, inplace=True)

    # Add a column so we know which stock this row belongs to.
    df["symbol"] = symbol

    print(f"  ✓ {symbol}: {len(df)} rows fetched")
    return df

def fetch_all(symbols: list = None, interval: str = "5m", period: str = "60d") -> pd.DataFrame:
    symbols=symbols or WATCHLIST
    all_frames=[]

    for sym in symbols:
        df=fetch_one(sym,interval=interval, period=period)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        raise RuntimeError("No data fetched for any symbol. Check your internet connection.")

    combined = pd.concat(all_frames, ignore_index=True)
    return combined

def save(df: pd.DataFrame, filename: str = "intraday_raw.csv"):
    path = RAW_DIR / filename
    df.to_csv(path, index=False)
    print(f"\nSaved {len(df)} rows → {path}")
    return path

if __name__=="__main__":
    print("Fetching intraday data for watchlist...\n")
    data=fetch_all(interval="5m", period="60d")

    print(f"\nTotal rows : {len(data)}")
    print(f"Symbols    : {data['symbol'].unique()}")
    print(f"Date range : {data['timestamp'].min()} → {data['timestamp'].max()}")
    print(f"\nSample:\n{data.head()}")

    save(data)