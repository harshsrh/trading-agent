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

# Market index — fetched separately and merged as context features
MARKET_INDEX = "^NSEI"

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

def fetch_with_market_context(interval: str = "1d", period: str = "5y") -> pd.DataFrame:
    """
    Fetches stock data and merges Nifty 50 index features as market context.
    This gives the model awareness of the broader market environment.
    """
    # Fetch stocks
    stocks_df = fetch_all(WATCHLIST, interval=interval, period=period)

    # Fetch Nifty 50
    print(f"\n  Fetching market index {MARKET_INDEX}...")
    nifty = yf.download(MARKET_INDEX, interval=interval, period=period,
                        progress=False, auto_adjust=True)

    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)

    nifty = nifty.reset_index()
    nifty.columns = [c.lower() for c in nifty.columns]
    nifty.rename(columns={"datetime": "timestamp", "date": "timestamp"}, inplace=True)

    # Compute Nifty-specific features
    nifty = nifty.sort_values("timestamp").reset_index(drop=True)
    nifty["nifty_returns_1"]  = nifty["close"].pct_change(1)
    nifty["nifty_returns_5"]  = nifty["close"].pct_change(5)
    nifty["nifty_ema_9"]      = nifty["close"].ewm(span=9).mean()
    nifty["nifty_ema_21"]     = nifty["close"].ewm(span=21).mean()
    nifty["nifty_trend"]      = nifty["nifty_ema_9"] - nifty["nifty_ema_21"]
    nifty["nifty_volatility"] = nifty["close"].pct_change().rolling(10).std()

    # Keep only the context columns + timestamp for merging
    nifty_context = nifty[[
        "timestamp", "nifty_returns_1", "nifty_returns_5",
        "nifty_trend", "nifty_volatility"
    ]].copy()

    # Normalize timestamp to date only for merging (daily data)
    stocks_df["date"]    = pd.to_datetime(stocks_df["timestamp"]).dt.date
    nifty_context["date"] = pd.to_datetime(nifty_context["timestamp"]).dt.date
    nifty_context        = nifty_context.drop(columns=["timestamp"])

    # Merge on date
    merged = stocks_df.merge(nifty_context, on="date", how="left")
    merged = merged.drop(columns=["date"])

    print(f"  ✓ Market context merged. NaN context rows: "
          f"{merged['nifty_trend'].isna().sum()}")

    return merged

if __name__ == "__main__":
    print("Fetching daily data with market context (5 years)...\n")
    data = fetch_with_market_context(interval="1d", period="5y")

    print(f"\nTotal rows : {len(data)}")
    print(f"Symbols    : {data['symbol'].unique()}")
    print(f"Date range : {data['timestamp'].min()} → {data['timestamp'].max()}")
    print(f"\nSample:\n{data.head()}")

    save(data, filename="daily_raw.csv")