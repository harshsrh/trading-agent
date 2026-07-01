"""
predict.py

Loads the trained model and generates trading signals with confidence scores.
Called by the agent during live trading to decide what to do each bar.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "saved"

FEATURE_COLUMNS = [
    "rsi_14", "stoch_k", "macd", "macd_signal", "macd_diff",
    "ema_diff", "bb_width", "atr_14", "volume_ratio",
    "returns_1", "returns_5", "high_low_pct",
    "nifty_returns_1", "nifty_returns_5",
    "nifty_trend", "nifty_volatility",
]

# Maps encoded label back to human-readable signal
SIGNAL_LABELS = {-1: "SELL", 0: "HOLD", 1: "BUY"}


def load_model():
    """Loads trained model and label encoder from disk."""
    with open(MODEL_DIR / "trading_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le


def predict_signal(model, le, features: dict) -> dict:
    """
    Generates a trading signal for a single bar of data.

    features: dict with keys matching FEATURE_COLUMNS
              e.g. {"rsi_14": 45.2, "macd": 0.003, ...}

    Returns a dict with:
        signal     : "BUY", "SELL", or "HOLD"
        signal_int : 1, -1, or 0
        confidence : probability of the predicted class (0.0 - 1.0)
        probabilities: full probability distribution across all classes
    """
    # Build feature row in the correct column order
    X = np.array([[features[col] for col in FEATURE_COLUMNS]])

    # Get class probabilities (e.g. [0.12, 0.21, 0.67] for sell/hold/buy)
    proba = model.predict_proba(X)[0]

    # Pick the highest probability class
    encoded_pred = np.argmax(proba)
    confidence   = proba[encoded_pred]

    # Decode back to -1/0/1
    signal_int = le.inverse_transform([encoded_pred])[0]
    signal     = SIGNAL_LABELS[signal_int]

    # Full probability breakdown
    classes = le.inverse_transform([0, 1, 2])
    prob_dict = {SIGNAL_LABELS[c]: round(float(p), 4) for c, p in zip(classes, proba)}

    return {
        "signal"       : signal,
        "signal_int"   : int(signal_int),
        "confidence"   : round(float(confidence), 4),
        "probabilities": prob_dict,
    }


def predict_dataframe(model, le, df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates signals for an entire dataframe of featured data.
    Useful for backtesting and batch prediction.
    """
    X = df[FEATURE_COLUMNS].values
    proba        = model.predict_proba(X)
    encoded_pred = np.argmax(proba, axis=1)
    confidence   = proba[np.arange(len(proba)), encoded_pred]
    signal_int   = le.inverse_transform(encoded_pred)

    result = df.copy()
    result["signal"]     = [SIGNAL_LABELS[s] for s in signal_int]
    result["signal_int"] = signal_int
    result["confidence"] = confidence

    return result


if __name__ == "__main__":
    print("Loading model...")
    model, le = load_model()
    print("Model loaded successfully.\n")

    # Test with a sample feature row (realistic values)
    sample_features = {
        "rsi_14"     : 58.3,
        "stoch_k"    : 72.1,
        "macd"       : 0.85,
        "macd_signal": 0.62,
        "macd_diff"  : 0.23,
        "ema_diff"   : 12.4,
        "bb_width"   : 0.032,
        "atr_14"     : 45.2,
        "volume_ratio": 1.6,
        "returns_1"  : 0.008,
        "returns_5"  : 0.021,
        "high_low_pct": 0.015,
        "nifty_returns_1" : 0.002,   # <-- ADDED
        "nifty_returns_5" : 0.011,   # <-- ADDED
        "nifty_trend"     : 1,       # <-- ADDED (Assuming 1=up, -1=down etc.)
        "nifty_volatility": 0.012,   # <-- ADDED
    }

    result = predict_signal(model, le, sample_features)

    print("Sample prediction:")
    print(f"  Signal     : {result['signal']}")
    print(f"  Confidence : {result['confidence']:.2%}")
    print(f"  Probabilities: {result['probabilities']}")