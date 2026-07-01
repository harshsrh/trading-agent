import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

DATA_DIR  = Path(__file__).parent.parent / "data" / "raw"
MODEL_DIR = Path(__file__).parent / "saved"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = [
    "rsi_14", "stoch_k", "macd", "macd_signal", "macd_diff",
    "ema_diff", "bb_width", "atr_14", "volume_ratio",
    "returns_1", "returns_5", "high_low_pct",
    # Market context — these were missing before
    "nifty_returns_1", "nifty_returns_5",
    "nifty_trend", "nifty_volatility",
]

def load_data() -> pd.DataFrame:
    path=DATA_DIR / "daily_featured.csv"
    df   = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} rows from {path}")
    return df

def prepare_xy(df: pd.DataFrame):
    """
    Splits featured dataframe into X (features) and y (target).

    We also encode target:
        -1 → 0  (sell)
         0 → 1  (hold)
         1 → 2  (buy)

    XGBoost needs integer class labels starting from 0.
    We'll decode predictions back to -1/0/1 when generating signals.
    """
    X = df[FEATURE_COLUMNS].values
    y_raw = df["target"].values

    # encode -1/0/1 → 0/1/2
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    return X, y, le

def compute_class_weights(y: np.ndarray) -> dict:
    """
    Computes class weights to handle the 83% / 8% / 8% imbalance.

    We weight the rare classes (buy/sell) higher so the model
    pays more attention to them during training.
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    # weight = total / (num_classes * count_of_class)
    weights = {cls: total / (len(classes) * cnt) for cls, cnt in zip(classes, counts)}
    print(f"\nClass weights: {weights}")
    return weights

def walk_forward_evaluation(X, y, le, n_splits: int = 5):
    """
    Evaluates model quality using walk-forward time-series cross validation.

    n_splits=5 means we test on 5 different future periods.
    We print results for each fold so you can see if the model
    is consistent or degrades over time.
    """
    tscv    = TimeSeriesSplit(n_splits=n_splits)
    weights = compute_class_weights(y)

    fold_scores = []

    print(f"\nWalk-forward validation ({n_splits} folds):")
    print("-" * 50)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # sample_weight maps each training row to its class weight
        sample_weights = np.array([weights[label] for label in y_train])

        model = xgb.XGBClassifier(
            n_estimators=200,       # number of trees
            max_depth=5,            # depth of each tree (controls complexity)
            learning_rate=0.05,     # smaller = slower learning = less overfit
            subsample=0.8,          # use 80% of rows per tree (regularization)
            colsample_bytree=0.8,   # use 80% of features per tree
            #use_label_encoder=False,
            eval_metric="mlogloss", # multi-class log loss
            random_state=42,
            n_jobs=-1               # use all CPU cores
        )

        model.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred = model.predict(X_test)

        # decode back to -1/0/1 for readable reporting
        y_test_decoded = le.inverse_transform(y_test)
        y_pred_decoded = le.inverse_transform(y_pred)

        # accuracy only on buy/sell predictions (ignore hold)
        # this is more meaningful than overall accuracy for trading
        signal_mask = y_test_decoded != 0
        if signal_mask.sum() > 0:
            signal_acc = (y_test_decoded[signal_mask] == y_pred_decoded[signal_mask]).mean()
        else:
            signal_acc = 0.0

        fold_scores.append(signal_acc)
        print(f"  Fold {fold}: signal accuracy = {signal_acc:.2%}  "
              f"(test size: {len(y_test)} rows)")

    print("-" * 50)
    print(f"  Mean signal accuracy: {np.mean(fold_scores):.2%}")
    print(f"  Std  signal accuracy: {np.std(fold_scores):.2%}")

    return fold_scores

def train_final_model(X, y, le):
    """
    Trains the final model on ALL available data.

    After walk-forward validation gives us confidence in the approach,
    we train on everything so the model has maximum historical context.
    This is the model we'll actually use for live predictions.
    """
    weights        = compute_class_weights(y)
    sample_weights = np.array([weights[label] for label in y])

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        #use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y, sample_weight=sample_weights)
    print("\nFinal model trained on full dataset.")
    return model

def save_model(model, le):
    """Save model and label encoder together so we can decode predictions later."""
    model_path = MODEL_DIR / "trading_model.pkl"
    le_path    = MODEL_DIR / "label_encoder.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(le_path, "wb") as f:
        pickle.dump(le, f)

    print(f"Model saved    → {model_path}")
    print(f"Encoder saved  → {le_path}")

def print_feature_importance(model, le):
    """Shows which features the model relies on most — useful for understanding its behavior."""
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLUMNS
    ).sort_values(ascending=False)

    print("\nFeature importance (higher = model relies on this more):")
    print("-" * 45)
    for feat, score in importance.items():
        bar = "█" * int(score * 200)
        print(f"  {feat:<20} {score:.4f}  {bar}")

if __name__ == "__main__":
    # 1. Load data
    df = load_data()

    # 2. Chronological train/test split — 80% train, 20% test
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Then split 80% train / 20% test
    split_idx  = int(len(df) * 0.80)
    df_train   = df.iloc[:split_idx].copy()
    df_test    = df.iloc[split_idx:].copy()

    print(f"\nTrain size : {len(df_train)} rows "
          f"({df_train['timestamp'].min()} → {df_train['timestamp'].max()})")
    print(f"Test size  : {len(df_test)} rows "
          f"({df_test['timestamp'].min()} → {df_test['timestamp'].max()})")

    # Save test set separately — backtest will use ONLY this
    test_path = DATA_DIR / "daily_test.csv"
    df_test.to_csv(test_path, index=False)
    print(f"Test set saved → {test_path}")

    # 3. Prepare features — train set only
    X_train, y_train, le = prepare_xy(df_train)
    print(f"\nX_train shape: {X_train.shape}")
    print(f"y distribution: {dict(zip(le.classes_, np.bincount(y_train)))}")

    # 4. Walk-forward validation on training set only
    fold_scores = walk_forward_evaluation(X_train, y_train, le, n_splits=5)

    # 5. Train final model on full training set
    model = train_final_model(X_train, y_train, le)

    # 6. Evaluate on test set (out-of-sample — the honest number)
    X_test, y_test, _ = prepare_xy(df_test)
    y_pred_encoded     = model.predict(X_test)

    y_test_decoded = le.inverse_transform(y_test)
    y_pred_decoded = le.inverse_transform(y_pred_encoded)

    signal_mask = y_test_decoded != 0
    if signal_mask.sum() > 0:
        oos_accuracy = (
            y_test_decoded[signal_mask] == y_pred_decoded[signal_mask]
        ).mean()
    else:
        oos_accuracy = 0.0

    print(f"\nOut-of-sample signal accuracy : {oos_accuracy:.2%}")
    print("(This is the honest number — model never saw this data)")

    # 7. Feature importance
    print_feature_importance(model, le)

    # 8. Save model
    save_model(model, le)

    print("\nDone. Model is ready for prediction.")