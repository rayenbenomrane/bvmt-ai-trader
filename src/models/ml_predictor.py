import os
import sys
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config.database import SessionLocal
from src.database.models import DailyPrice, CandlestickPattern

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, balanced_accuracy_score
from sklearn.dummy import DummyClassifier

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('date').copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['daily_return'] = df['close'].pct_change()
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma50'] = df['close'].rolling(window=50).mean()
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()

    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    rolling_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['sma20'] + (rolling_std * 2)
    df['bb_lower'] = df['sma20'] - (rolling_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma20']

    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr14'] = true_range.rolling(window=14).mean()

    return df

def prepare_ml_dataset(db, symbol: str, target_type: str = "binary", threshold_pct: float = 0.005) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    prices_query = db.query(DailyPrice).filter(DailyPrice.symbol == symbol).order_by(DailyPrice.date.asc()).statement
    df_prices = pd.read_sql(prices_query, db.bind)
    
    if df_prices.empty or len(df_prices) < 60:
        return pd.DataFrame(), pd.Series(), []

    df = compute_technical_indicators(df_prices)

    patterns_query = db.query(CandlestickPattern).filter(CandlestickPattern.symbol == symbol).statement
    df_patterns = pd.read_sql(patterns_query, db.bind)

    if not df_patterns.empty:
        df_patterns['date'] = pd.to_datetime(df_patterns['date'])
        df['date'] = pd.to_datetime(df['date'])
        pivot_patterns = df_patterns.pivot_table(index='date', columns='pattern_name', values='direction', aggfunc='first')
        pivot_patterns = pivot_patterns.notnull().astype(int)
        pivot_patterns.columns = [f"pat_{c.lower().replace(' ', '_')}" for c in pivot_patterns.columns]
        df = pd.merge(df, pivot_patterns, on='date', how='left')

    pat_cols = [c for c in df.columns if c.startswith('pat_')]
    for c in pat_cols:
        df[c] = df[c].fillna(0)

    # Next return calculation
    df['next_close'] = df['close'].shift(-1)
    df['next_return'] = (df['next_close'] - df['close']) / df['close']

    if target_type == "binary":
        # 1 = UP, 0 = DOWN or EQUAL
        df['target'] = (df['next_return'] > 0).astype(int)
    elif target_type == "three_class":
        # 2 = UP (> +0.5%), 0 = DOWN (< -0.5%), 1 = NEUTRAL
        conditions = [
            df['next_return'] > threshold_pct,
            df['next_return'] < -threshold_pct
        ]
        choices = [2, 0] # 2: UP, 0: DOWN, 1: NEUTRAL
        df['target'] = np.select(conditions, choices, default=1)
    else:
        raise ValueError("Invalid target_type. Use 'binary' or 'three_class'.")

    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'daily_return', 'sma20', 'sma50', 'ema12', 'ema26',
        'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_lower', 'bb_width', 'atr14'
    ] + pat_cols

    df_clean = df.dropna(subset=feature_cols + ['target']).copy()
    X = df_clean[feature_cols]
    y = df_clean['target']
    
    return X, y, feature_cols

def get_class_probability(model, X, target_class=1):
    """Safely extracts probability for a given target class using model.classes_ to avoid index bugs."""
    probs = model.predict_proba(X)
    classes = list(model.classes_)
    if target_class in classes:
        idx = classes.index(target_class)
        return probs[:, idx]
    return probs[:, -1] # fallback

def analyze_features_and_probabilities(symbol: str) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        X, y, feature_cols = prepare_ml_dataset(db, symbol, target_type="binary")
        if len(X) < 40:
            return {"symbol": symbol, "status": "Insufficient data"}

        print(f"\n==========================================")
        print(f" DEEP INVESTIGATION & FEATURE ANALYSIS: {symbol}")
        print(f"==========================================")

        # 1. Feature Correlation with Target
        df_full = X.copy()
        df_full['target'] = y
        corrs = df_full.corr()['target'].drop('target').sort_values(ascending=False)
        print("\n--- Feature Correlations with Target ---")
        print(corrs.to_string())

        # 2. Chronological Split (Train: 60%, Val: 20%, Test: 20%)
        n = len(X)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=42)
        rf.fit(X_train, y_train)

        # Feature Importance
        importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print("\n--- Top 5 Feature Importances ---")
        print(importances.head(5).to_string())

        # 3. Probability Analysis & Threshold Sweep on Validation Data
        val_probs = get_class_probability(rf, X_val, target_class=1)
        print("\n--- Threshold Sweep on Validation Data ---")
        thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
        sweep_results = []
        for th in thresholds:
            preds = (val_probs >= th).astype(int)
            acc = accuracy_score(y_val, preds)
            prec = precision_score(y_val, preds, zero_division=0)
            rec = recall_score(y_val, preds, zero_division=0)
            f1 = f1_score(y_val, preds, zero_division=0)
            b_acc = balanced_accuracy_score(y_val, preds)
            sweep_results.append({"threshold": th, "accuracy": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "balanced_acc": round(b_acc, 4)})
            print(f" Threshold {th:.2f} -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, Bal-Acc: {b_acc:.4f}")

        # Select best threshold based on F1 on validation set
        best_th = max(sweep_results, key=lambda x: x["f1"])["threshold"]
        print(f"\nSelected Best Threshold (Validation F1): {best_th}")

        # 4. Test Set Evaluation with Best Threshold
        test_probs = get_class_probability(rf, X_test, target_class=1)
        test_preds = (test_probs >= best_th).astype(int)

        test_acc = accuracy_score(y_test, test_preds)
        test_prec = precision_score(y_test, test_preds, zero_division=0)
        test_rec = recall_score(y_test, test_preds, zero_division=0)
        test_f1 = f1_score(y_test, test_preds, zero_division=0)
        test_auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else 0.5
        test_cm = confusion_matrix(y_test, test_preds).tolist()
        pred_dist = pd.Series(test_preds).value_counts().to_dict()

        print(f"\n--- Test Set Results (Threshold = {best_th}) ---")
        print(f" Accuracy  : {test_acc:.4f}")
        print(f" Precision : {test_prec:.4f}")
        print(f" Recall    : {test_rec:.4f}")
        print(f" F1-Score  : {test_f1:.4f}")
        print(f" ROC-AUC   : {test_auc:.4f}")
        print(f" Pred Dist : {pred_dist}")
        print(f" Conf Matrix: {test_cm}")

        # 5. Compare Binary vs Three-Class Target Distribution
        _, y_3class, _ = prepare_ml_dataset(db, symbol, target_type="three_class")
        class_3_dist = y_3class.value_counts().to_dict()
        print(f"\n--- Three-Class Target Distribution (0: DOWN, 1: NEUTRAL, 2: UP) ---")
        print(class_3_dist)

        report = {
            "symbol": symbol,
            "correlations": corrs.head(5).to_dict(),
            "feature_importances": importances.head(5).to_dict(),
            "threshold_sweep": sweep_results,
            "best_threshold": best_th,
            "test_evaluation": {
                "accuracy": round(float(test_acc), 4),
                "precision": round(float(test_prec), 4),
                "recall": round(float(test_rec), 4),
                "f1_score": round(float(test_f1), 4),
                "roc_auc": round(float(test_auc), 4),
                "confusion_matrix": test_cm,
                "prediction_distribution": pred_dist
            },
            "three_class_distribution": class_3_dist
        }

        os.makedirs("data/processed/reports", exist_ok=True)
        with open(f"data/processed/reports/{symbol}_deep_investigation.json", "w") as f:
            json.dump(report, f, indent=4)

        return report

    finally:
        db.close()

def main():
    for sym in ["BT", "SFBT"]:
        analyze_features_and_probabilities(sym)

if __name__ == "__main__":
    main()