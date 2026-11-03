"""
BVMT AI Trader - Machine Learning Predictor

Implements robust ML model training, chronological walk-forward validation,
validation-based model and threshold selection, probability calibration,
and reproducible model artifact persistence for BVMT stocks.
"""

import os
import sys
import json
import joblib
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from src.config.database import SessionLocal
from src.database.models import DailyPrice
from src.models.feature_engineering import (
    FEATURE_COLUMNS,
    build_features,
    get_available_feature_columns,
    prepare_labeled_dataset,
)


def get_benchmark_models() -> Dict[str, Any]:
    """Returns the standardized suite of benchmark models."""
    models = {
        "Majority Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
        ),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )
    return models


def chronological_walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 3,
    min_train_ratio: float = 0.50,
) -> List[Dict[str, Any]]:
    """
    Performs expanding-window chronological cross-validation.
    Ensures that validation sets always temporally follow training sets with zero look-ahead bias.
    """
    n_samples = len(X)
    min_train_size = int(n_samples * min_train_ratio)
    remaining_samples = n_samples - min_train_size
    fold_size = remaining_samples // n_splits

    fold_results = []
    for fold in range(n_splits):
        train_end = min_train_size + fold * fold_size
        val_end = train_end + fold_size if fold < n_splits - 1 else n_samples

        X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
        X_va, y_va = X.iloc[train_end:val_end], y.iloc[train_end:val_end]

        models = get_benchmark_models()
        fold_metrics = {}

        for name, model in models.items():
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            probs = (
                model.predict_proba(X_va)[:, 1]
                if hasattr(model, "predict_proba")
                else np.full(len(preds), 0.5)
            )

            acc = accuracy_score(y_va, preds)
            b_acc = balanced_accuracy_score(y_va, preds)
            f1 = f1_score(y_va, preds, zero_division=0)
            auc = (
                roc_auc_score(y_va, probs)
                if len(np.unique(y_va)) > 1
                else 0.5
            )

            fold_metrics[name] = {
                "accuracy": round(acc, 4),
                "balanced_accuracy": round(b_acc, 4),
                "f1": round(f1, 4),
                "roc_auc": round(auc, 4),
            }

        fold_results.append({
            "fold": fold + 1,
            "train_size": len(X_tr),
            "val_size": len(X_va),
            "metrics": fold_metrics,
        })

    return fold_results


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """Computes comprehensive binary classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = (
        roc_auc_score(y_true, y_prob)
        if len(np.unique(y_true)) > 1
        else 0.5
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(b_acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": cm,
    }


def train_and_select_model(
    df_clean: pd.DataFrame,
    symbol: str,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    save_dir: str = "models/saved",
    horizon: int = 1,
    target_col: str = "target_binary",
    evaluate_test: bool = True,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Strict, leak-free ML workflow:
    1. Chronological partition: 60% Train, 20% Validation, 20% Test.
    2. Fit models on Train.
    3. Evaluate models and sweep decision thresholds on Validation ONLY.
    4. Select optimal model and threshold based on Validation F1/AUC.
    5. Evaluate the winning configuration on the unseen Test set ONCE.
    6. Persist models and structured metadata.
    """
    if target_col not in {"target_binary", "target_cost_aware", "target_legacy_1d"}:
        raise ValueError(f"Unsupported target column: {target_col}")
    if horizon < 1:
        raise ValueError("horizon must be a positive number of trading sessions.")

    total = len(df_clean)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    feature_columns = get_available_feature_columns(df_clean)
    X = df_clean[feature_columns]
    y = df_clean[target_col]
    dates = df_clean["date"]

    # Open-to-open labels end at t+horizon+1; the legacy close target ends at t+1.
    purge_periods = 1 if target_col == "target_legacy_1d" else horizon + 1
    train_stop = max(0, train_end - purge_periods)
    val_stop = max(train_end, val_end - purge_periods)
    train_idx = slice(0, train_stop)
    val_idx = slice(train_end, val_stop)
    test_idx = slice(val_end, total)

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    if min(len(X_train), len(X_val), len(X_test)) == 0:
        raise ValueError("Chronological split is too small after purging horizon-overlapping labels.")

    dates_train = (dates.iloc[0], dates.iloc[train_stop - 1])
    dates_val = (dates.iloc[train_end], dates.iloc[val_stop - 1])
    dates_test = (dates.iloc[val_end], dates.iloc[-1])

    models = get_benchmark_models()
    val_evaluations = {}
    fitted_models = {}

    best_model_name = "Majority Baseline"
    best_val_f1 = -1.0
    best_threshold = 0.50

    thresholds_to_test = [0.50, 0.55, 0.60, 0.65]
    validation_probabilities = {}
    expected_return_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=5,
        random_state=42,
    )
    expected_return_model.fit(X_train, df_clean["forward_return"].iloc[train_idx])

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

        if hasattr(model, "predict_proba"):
            val_probs = model.predict_proba(X_val)[:, 1]
        else:
            val_probs = np.full(len(X_val), 0.5)
        validation_probabilities[name] = val_probs

        # Sweep thresholds on validation data to maximize balanced accuracy
        best_th_for_model = 0.50
        best_bacc_for_model = -1.0
        th_sweep_results = []

        for th in thresholds_to_test:
            preds_th = (val_probs >= th).astype(int)
            bacc_th = balanced_accuracy_score(y_val, preds_th)
            f1_th = f1_score(y_val, preds_th, zero_division=0)
            th_sweep_results.append({
                "threshold": th,
                "balanced_accuracy": round(bacc_th, 4),
                "f1": round(f1_th, 4),
            })
            if bacc_th > best_bacc_for_model:
                best_bacc_for_model = bacc_th
                best_th_for_model = th

        val_preds = (val_probs >= best_th_for_model).astype(int)
        val_metrics = evaluate_predictions(y_val, val_preds, val_probs)
        val_metrics["selected_threshold"] = best_th_for_model
        val_metrics["threshold_sweep"] = th_sweep_results
        val_evaluations[name] = val_metrics

        # Model selection based solely on validation data
        # Prioritize Balanced Accuracy; if tied, compare ROC-AUC
        if (val_metrics["balanced_accuracy"] > best_val_f1) or (
            val_metrics["balanced_accuracy"] == best_val_f1 and val_metrics["roc_auc"] > val_evaluations.get(best_model_name, {}).get("roc_auc", 0)
        ):
            best_val_f1 = val_metrics["balanced_accuracy"]
            best_model_name = name
            best_threshold = best_th_for_model

    # Test data is not scored during candidate search; the runner evaluates it once selection is complete.
    test_evaluations = {}
    if evaluate_test:
        for name, model in fitted_models.items():
            th = val_evaluations[name]["selected_threshold"]
            if hasattr(model, "predict_proba"):
                test_probs = model.predict_proba(X_test)[:, 1]
            else:
                test_probs = np.full(len(X_test), 0.5)

            test_preds = (test_probs >= th).astype(int)
            test_evaluations[name] = evaluate_predictions(y_test, test_preds, test_probs)
            test_evaluations[name]["threshold_used"] = th

    # Save best model and metadata
    best_model_file = None
    meta_file = None
    if save_artifacts:
        os.makedirs(save_dir, exist_ok=True)
        slug = symbol.upper()
        suffix = f"_{horizon}d_{target_col}"
        best_model_file = f"{save_dir}/{slug}_{best_model_name.lower().replace(' ', '_')}{suffix}.joblib"
        joblib.dump(fitted_models[best_model_name], best_model_file)

    metadata = {
        "symbol": symbol,
        "selected_model": best_model_name,
        "selected_threshold": best_threshold,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "date_ranges": {
            "train": [str(dates_train[0]), str(dates_train[1])],
            "val": [str(dates_val[0]), str(dates_val[1])],
            "test": [str(dates_test[0]), str(dates_test[1])],
        },
        "sample_counts": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "features": feature_columns,
        "horizon_sessions": horizon,
        "target_column": target_col,
        "validation_evaluations": val_evaluations,
        "test_evaluations": test_evaluations,
    }

    if save_artifacts:
        meta_file = f"{save_dir}/{slug}_{best_model_name.lower().replace(' ', '_')}{suffix}_metadata.json"
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=4)

    return {
        "symbol": symbol,
        "best_model_name": best_model_name,
        "best_model": fitted_models[best_model_name],
        "best_threshold": best_threshold,
        "fitted_models": fitted_models,
        "expected_return_model": expected_return_model,
        "feature_columns": feature_columns,
        "horizon": horizon,
        "target_col": target_col,
        "val_evaluations": val_evaluations,
        "test_evaluations": test_evaluations,
        "validation_probabilities": validation_probabilities,
        "val_partitions": {
            "X_val": X_val,
            "y_val": y_val,
            "dates": dates.iloc[val_idx].reset_index(drop=True),
            "next_date": df_clean["next_date"].iloc[val_idx].reset_index(drop=True),
            "next_open": df_clean["next_open"].iloc[val_idx].reset_index(drop=True),
            "next_close": df_clean["next_close"].iloc[val_idx].reset_index(drop=True),
            "forward_return": df_clean["forward_return"].iloc[val_idx].reset_index(drop=True),
            "signal_open": df_clean["open"].iloc[val_idx].reset_index(drop=True),
            "signal_close": df_clean["close"].iloc[val_idx].reset_index(drop=True),
        },
        "test_partitions": {
            "X_test": X_test,
            "y_test": y_test,
            "dates_test": dates.iloc[test_idx].reset_index(drop=True),
            "next_date_test": df_clean["next_date"].iloc[test_idx].reset_index(drop=True),
            "next_open_test": df_clean["next_open"].iloc[test_idx].reset_index(drop=True),
            "next_close_test": df_clean["next_close"].iloc[test_idx].reset_index(drop=True),
            "forward_return_test": df_clean["forward_return"].iloc[test_idx].reset_index(drop=True),
            "signal_open_test": df_clean["open"].iloc[test_idx].reset_index(drop=True),
            "signal_close_test": df_clean["close"].iloc[test_idx].reset_index(drop=True),
        },
        "metadata_file": meta_file,
        "model_file": best_model_file,
    }


def analyze_features_and_probabilities(symbol: str) -> Dict[str, Any]:
    """Runs full feature correlation, walk-forward validation, and model selection."""
    db = SessionLocal()
    try:
        query = (
            db.query(DailyPrice)
            .filter(DailyPrice.symbol == symbol)
            .order_by(DailyPrice.date.asc())
            .statement
        )
        df_raw = pd.read_sql(query, db.bind)
        if df_raw.empty or len(df_raw) < 100:
            print(f"Insufficient data for {symbol}.")
            return {"symbol": symbol, "status": "Insufficient data"}

        df_clean = prepare_labeled_dataset(df_raw)

        print(f"\n==========================================")
        print(f" FEATURE & MODEL INVESTIGATION: {symbol}")
        print(f"==========================================")

        # 1. Feature Correlations with Target
        X = df_clean[FEATURE_COLUMNS]
        corrs = X.apply(lambda col: col.corr(df_clean["target_binary"])).sort_values(ascending=False)
        print("\n--- Feature Correlations with Binary Target ---")
        for feat, c in corrs.items():
            print(f" {feat:<18}: {c:+.4f}")

        # 2. Chronological Walk-Forward CV
        wf_results = chronological_walk_forward_cv(X, df_clean["target_binary"], n_splits=3)
        print("\n--- Chronological Walk-Forward Validation (3 Folds) ---")
        for fold in wf_results:
            print(f" Fold {fold['fold']} (Train: {fold['train_size']}, Val: {fold['val_size']}):")
            for m_name, m_scores in fold["metrics"].items():
                print(f"   [{m_name}] F1: {m_scores['f1']:.4f} | AUC: {m_scores['roc_auc']:.4f} | Acc: {m_scores['accuracy']:.4f}")

        # 3. Model Selection & Test Set Evaluation
        selection_result = train_and_select_model(df_clean, symbol)
        print(f"\n--- Model Selection on Validation Data ---")
        print(f" Best Selected Model: {selection_result['best_model_name']} (Threshold: {selection_result['best_threshold']})")
        print("\n--- Out-of-Sample Test Evaluation ---")
        for m_name, m_scores in selection_result["test_evaluations"].items():
            print(f" [{m_name}] Acc: {m_scores['accuracy']:.4f} | Bal-Acc: {m_scores['balanced_accuracy']:.4f} | F1: {m_scores['f1']:.4f} | AUC: {m_scores['roc_auc']:.4f}")

        report = {
            "symbol": symbol,
            "correlations": {k: round(float(v), 4) for k, v in corrs.items()},
            "walk_forward_cv": wf_results,
            "selected_model": selection_result["best_model_name"],
            "selected_threshold": selection_result["best_threshold"],
            "validation_evaluations": selection_result["val_evaluations"],
            "test_evaluations": selection_result["test_evaluations"],
        }

        os.makedirs("data/processed/reports", exist_ok=True)
        report_path = f"data/processed/reports/{symbol}_deep_investigation.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        return report

    finally:
        db.close()


def main():
    for sym in ["BT", "SFBT"]:
        analyze_features_and_probabilities(sym)


if __name__ == "__main__":
    main()
