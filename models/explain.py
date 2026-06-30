"""
explain.py

Uses SHAP values to explain WHY the model made a specific prediction.
This is the "thinking process" of the agent — every signal comes with
a plain-English reason so you always know what drove the decision.
"""

import shap
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from predict import load_model, FEATURE_COLUMNS, SIGNAL_LABELS

MODEL_DIR = Path(__file__).parent / "saved"


def build_explainer(model, background_data: np.ndarray):
    """
    Creates a SHAP TreeExplainer.
    Uses tree_path_dependent to avoid XGBoost categorical split errors.
    """
    explainer = shap.TreeExplainer(
        model, 
        feature_perturbation="tree_path_dependent"
    )
    return explainer

def explain_prediction(explainer, le, features: dict, signal_int: int) -> dict:
    """
    Computes SHAP values for a single prediction and formats them
    into a human-readable explanation.

    Returns:
        shap_values   : raw SHAP values per feature
        top_reasons   : top 3 features driving this prediction
        explanation   : plain-English summary string
    """
    X = np.array([[features[col] for col in FEATURE_COLUMNS]])

    # SHAP values shape: (1, n_features, n_classes)
    shap_vals = explainer.shap_values(X)

    # Find which class index corresponds to the predicted signal
    class_idx = list(le.classes_).index(signal_int)

    # Get SHAP values for the predicted class
    feature_shap = shap_vals[0, :, class_idx]

    # Pair each feature with its SHAP value and sort by absolute impact
    feature_impact = sorted(
        zip(FEATURE_COLUMNS, feature_shap),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # Build top 3 reasons
    top_reasons = []
    for feat, shap_val in feature_impact[:3]:
        direction = "↑ bullish" if shap_val > 0 else "↓ bearish"
        actual_val = features[feat]
        top_reasons.append({
            "feature"  : feat,
            "shap_value": round(float(shap_val), 4),
            "direction": direction,
            "value"    : round(float(actual_val), 4),
        })

    # Build plain-English explanation
    signal_label = SIGNAL_LABELS[signal_int]
    reason_lines = []
    for r in top_reasons:
        reason_lines.append(
            f"{r['feature']} = {r['value']} ({r['direction']}, impact: {r['shap_value']:+.3f})"
        )

    explanation = (
        f"Signal: {signal_label}\n"
        f"Top reasons:\n  " + "\n  ".join(reason_lines)
    )

    return {
        "top_reasons": top_reasons,
        "explanation": explanation,
        "all_shap"   : dict(zip(FEATURE_COLUMNS, [round(float(v), 4) for v in feature_shap])),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from models.predict import load_model, predict_signal

    print("Loading model...")
    model, le = load_model()

    # Load a small sample of training data as SHAP background
    data_path = Path(__file__).parent.parent / "data" / "raw" / "daily_featured.csv"
    df_background = pd.read_csv(data_path)[FEATURE_COLUMNS].sample(100, random_state=42)

    print("Building SHAP explainer (this takes ~10 seconds)...")
    explainer = build_explainer(model, df_background.values)

    # Same sample features as predict.py
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
    }

    # Get signal first
    result = predict_signal(model, le, sample_features)
    print(f"\nPrediction: {result['signal']} ({result['confidence']:.2%} confidence)")

    # Then explain it
    explanation = explain_prediction(
        explainer, le, sample_features, result["signal_int"]
    )

    print(f"\n{explanation['explanation']}")
    print(f"\nFull SHAP values:")
    for feat, val in explanation["all_shap"].items():
        bar = "█" * int(abs(val) * 100)
        sign = "+" if val > 0 else "-"
        print(f"  {feat:<20} {sign}{abs(val):.4f}  {bar}")