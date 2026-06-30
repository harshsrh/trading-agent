import pandas as pd
import numpy as np
import ta
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"

FEATURE_COLUMNS = [
    "rsi_14",
    "stoch_k",
    "macd",
    "macd_signal",
    "macd_diff",
    "ema_diff",
    "bb_width",
    "atr_14",
    "volume_ratio",
    "returns_1",
    "returns_5",
    "high_low_pct",
]

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all technical indicators for a single symbol's data.
    Input df must be sorted by timestamp ascending.
    """
    df = df.sort_values("timestamp").reset_index(drop=True).copy()

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # ── Momentum ───────────────────────────────────────────────────────────
    # RSI: 0-100 scale. >70 = overbought, <30 = oversold.
    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # Stochastic: similar to RSI but uses high/low range instead of closes.
    df["stoch_k"] = ta.momentum.StochasticOscillator(
        high, low, close, window=14
    ).stoch()

    # ── Trend ──────────────────────────────────────────────────────────────
    # MACD: difference between 12-period and 26-period EMAs.
    # macd_diff (histogram) is the most actionable — positive = bullish momentum.
    macd_obj       = ta.trend.MACD(close)
    df["macd"]        = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_diff"]   = macd_obj.macd_diff()

    # EMA crossover: 9-period vs 21-period.
    # Positive ema_diff = short-term average above long-term = uptrend.
    df["ema_9"]    = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    df["ema_21"]   = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    df["ema_diff"] = df["ema_9"] - df["ema_21"]

    # ── Volatility ─────────────────────────────────────────────────────────
    # Bollinger Bands: measures how wide price range is vs recent average.
    # bb_width > normal = high volatility = bigger moves expected.
    bb             = ta.volatility.BollingerBands(close, window=20)
    df["bb_high"]  = bb.bollinger_hband()
    df["bb_low"]   = bb.bollinger_lband()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / close  # normalized

    # ATR: average size of each candle. Used later for stop-loss placement.
    df["atr_14"] = ta.volatility.AverageTrueRange(
        high, low, close, window=14
    ).average_true_range()

    # ── Volume ─────────────────────────────────────────────────────────────
    # Volume ratio: current bar's volume vs 20-bar average.
    # > 1.5 = unusually high volume = stronger signal reliability.
    df["volume_sma_20"] = volume.rolling(20).mean()
    df["volume_ratio"]  = volume / df["volume_sma_20"]

    # ── Price action ───────────────────────────────────────────────────────
    # How much did price move in the last 1 and 5 bars?
    df["returns_1"] = close.pct_change(1)
    df["returns_5"] = close.pct_change(5)

    # How wide was this candle relative to price? Proxy for indecision/momentum.
    df["high_low_pct"] = (high - low) / close

    return df

def add_target(df: pd.DataFrame, horizon: int = 3, threshold: float = 0.003) -> pd.DataFrame:
    df = df.copy()

    # pct change from current close to close `horizon` bars later
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    df["future_return"] = future_return

    # default everything to 0 (no trade)
    df["target"] = 0
    df.loc[future_return >  threshold, "target"] =  1
    df.loc[future_return < -threshold, "target"] = -1

    return df

def build_features(df: pd.DataFrame, horizon: int = 3, threshold: float = 0.003) -> pd.DataFrame:
    df = add_indicators(df)
    df = add_target(df, horizon=horizon, threshold=threshold)
    df = df.dropna().reset_index(drop=True)
    return df

def process_all(input_file: str = "intraday_raw.csv") -> pd.DataFrame:
    """
    Loads raw data, processes each symbol separately, combines results.
    We process per symbol so rolling windows don't bleed across stocks.
    """
    path = RAW_DIR / input_file
    raw  = pd.read_csv(path, parse_dates=["timestamp"])

    print(f"Loaded {len(raw)} rows from {path}")
    print(f"Symbols: {raw['symbol'].unique()}\n")

    all_featured = []

    for symbol in raw["symbol"].unique():
        symbol_df = raw[raw["symbol"] == symbol].copy()
        featured  = build_features(symbol_df)
        all_featured.append(featured)
        print(f"  ✓ {symbol}: {len(featured)} rows after feature engineering")
    
    combined = pd.concat(all_featured, ignore_index=True)
    return combined


if __name__ == "__main__":
    featured = process_all()

    print(f"\nTotal rows     : {len(featured)}")
    print(f"Feature columns: {FEATURE_COLUMNS}")
    print(f"\nTarget distribution (all symbols combined):")
    print(featured["target"].value_counts())
    print(f"\nSample row:")
    print(featured[["timestamp", "symbol", "close"] + FEATURE_COLUMNS + ["target"]].head())

    # Save for use in model training
    out_path = RAW_DIR / "intraday_featured.csv"
    featured.to_csv(out_path, index=False)
    print(f"\nSaved featured data → {out_path}")

