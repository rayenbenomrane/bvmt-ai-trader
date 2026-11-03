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
from sklearn.ensemble import RandomForestClassifier
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
    purged_chronological_splits,
)


# ---- Edge-detection constants (used by both ML and backtest layers) --------
# A model whose mean walk-forward validation AUC is below this threshold
# is statistically indistinguishable from a random classifier.
MIN_EDGE_AUC = 0.55

# A model whose fold-to-fold AUC standard deviation exceeds this threshold
# is not *stably* predictive, regardless of how high the mean is.
# One lucky fold must not masquerade as edge. Empirically, stable edges
# have fold std <= 0.03-0.04.
MAX_FOLD_STD = 0.04


def _safe_roc_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Return ROC-AUC, or 0.5 when only one class is present (undefined AUC)."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return 0.5


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
    Ensures that validation sets always temporally follow training sets
    with zero look-ahead bias.
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
            auc = _safe_roc_auc(y_va, probs)

            fold_metrics[name] = {
                "accuracy": round(float(acc), 4),
                "balanced_accuracy": round(float(b_acc), 4),
                "f1": round(float(f1), 4),
                "roc_auc": round(float(auc), 4),
            }

        fold_results.append({
            "fold": fold + 1,
            "train_size": len(X_tr),
            "val_size": len(X_va),
            "metrics": fold_metrics,
        })

    return fold_results


def _mean_walk_forward_auc(
    wf_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Average ROC-AUC per model across all walk-forward folds.

    This is the honest, stable metric for edge detection: it depends only
    on model ranking (threshold-independent) and it exposes the fold-to-fold
    variance so that one lucky fold cannot masquerade as real edge.

    Returns per-model: mean, std (population), n_folds, min_fold, max_fold,
    and `is_stable` (True only when mean >= MIN_EDGE_AUC AND std <= MAX_FOLD_STD).
    """
    if not wf_results:
        return {}
    model_names = list(wf_results[0]["metrics"].keys())
    summary: Dict[str, Dict[str, float]] = {}
    for name in model_names:
        aucs = np.array(
            [fold["metrics"][name]["roc_auc"] for fold in wf_results],
            dtype=float,
        )
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))
        summary[name] = {
            "mean": round(mean_auc, 4),
            "std": round(std_auc, 4),
            "n_folds": int(len(aucs)),
            "min_fold": round(float(np.min(aucs)), 4),
            "max_fold": round(float(np.max(aucs)), 4),
            "is_stable": bool(mean_auc >= MIN_EDGE_AUC and std_auc <= MAX_FOLD_STD),
        }
    return summary


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
    auc = _safe_roc_auc(y_true, y_prob)
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
    purge_horizon: Optional[int] = None,
    split_boundaries: Optional[Dict[str, int]] = None,
    min_edge_auc: float = MIN_EDGE_AUC,
) -> Dict[str, Any]:
    """
    Strict, leak-free ML workflow:
    1. Chronological partition: 60% Train, 20% Validation, 20% Test.
    2. Fit models on Train only.
    3. Sweep decision thresholds on Validation only (for later trading use).
    4. Select the model by VALIDATION ROC-AUC (threshold-independent, stable).
       Tiebreak by balanced accuracy, then by name.
    5. If best validation ROC-AUC < `min_edge_auc`, mark `status = "NO_EDGE"`.
       The best-AUC model is still returned (so legacy code paths keep working),
       but the caller should check `status`/`has_edge` before actually trading.
    6. Evaluate on the unseen Test set only if `status == "EDGE_DETECTED"`.
    7. Persist artifacts only when edge is present.
    """
    if target_col not in {"target_binary", "target_legacy_1d"}:
        raise ValueError(f"Unsupported target column: {target_col}")
    if horizon < 1:
        raise ValueError("horizon must be a positive number of trading sessions.")

    total = len(df_clean)
    split = purged_chronological_splits(
        total,
        horizon,
        train_ratio,
        val_ratio,
        legacy_target=target_col == "target_legacy_1d",
        purge_horizon=purge_horizon,
        split_boundaries=split_boundaries,
    )
    train_end = split["train_end"]
    val_end = split["test_start"]
    feature_columns = get_available_feature_columns(df_clean)
    X = df_clean[feature_columns]
    y = df_clean[target_col]
    dates = df_clean["date"]

    train_stop = split["train_stop"]
    val_stop = split["validation_stop"]
    train_idx = split["train"]
    val_idx = split["validation"]
    test_idx = split["test"]

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    required_split_lengths = [len(X_train), len(X_val)]
    if evaluate_test:
        required_split_lengths.append(len(X_test))
    if min(required_split_lengths) == 0:
        raise ValueError(
            "Chronological split is too small after purging horizon-overlapping labels."
        )

    dates_train = (dates.iloc[0], dates.iloc[train_stop - 1])
    dates_val = (dates.iloc[train_end], dates.iloc[val_stop - 1])
    dates_test = (
        (dates.iloc[test_idx.start], dates.iloc[test_idx.stop - 1])
        if len(X_test) > 0
        else ("N/A", "N/A")
    )

    models = get_benchmark_models()
    val_evaluations: Dict[str, Dict[str, Any]] = {}
    fitted_models: Dict[str, Any] = {}

    best_model_name = "Majority Baseline"
    best_val_auc = -1.0
    best_val_bacc = -1.0
    best_threshold = 0.50

    thresholds_to_test = [0.50, 0.55, 0.60, 0.65]
    validation_probabilities: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

        if hasattr(model, "predict_proba"):
            val_probs = model.predict_proba(X_val)[:, 1]
        else:
            val_probs = np.full(len(X_val), 0.5)
        validation_probabilities[name] = val_probs

        # Threshold sweep is used only to configure the trading threshold,
        # NOT to compare models. Comparison uses ROC-AUC below.
        best_th_for_model = 0.50
        best_bacc_for_model = -1.0
        th_sweep_results = []
        for th in thresholds_to_test:
            preds_th = (val_probs >= th).astype(int)
            bacc_th = balanced_accuracy_score(y_val, preds_th)
            f1_th = f1_score(y_val, preds_th, zero_division=0)
            th_sweep_results.append({
                "threshold": th,
                "balanced_accuracy": round(float(bacc_th), 4),
                "f1": round(float(f1_th), 4),
            })
            if bacc_th > best_bacc_for_model:
                best_bacc_for_model = bacc_th
                best_th_for_model = th

        val_preds = (val_probs >= best_th_for_model).astype(int)
        val_metrics = evaluate_predictions(y_val, val_preds, val_probs)
        val_metrics["selected_threshold"] = float(best_th_for_model)
        val_metrics["threshold_sweep"] = th_sweep_results
        val_evaluations[name] = val_metrics

        # MODEL SELECTION: primary = validation ROC-AUC (threshold-independent,
        # far more stable than balanced accuracy or F1). Tiebreak by balanced
        # accuracy, then by model name for determinism.
        val_auc = float(val_metrics["roc_auc"])
        val_bacc = float(val_metrics["balanced_accuracy"])
        if (val_auc > best_val_auc) or (
            val_auc == best_val_auc and val_bacc > best_val_bacc
        ):
            best_val_auc = val_auc
            best_val_bacc = val_bacc
            best_model_name = name
            best_threshold = float(best_th_for_model)

    has_edge = best_val_auc >= min_edge_auc
    status = "EDGE_DETECTED" if has_edge else "NO_EDGE"

    # Test data is never scored when edge is absent: scoring it and then
    # deciding not to trade would leak future information into the decision.
    test_evaluations: Dict[str, Dict[str, Any]] = {}
    if evaluate_test and has_edge:
        for name, model in fitted_models.items():
            th = val_evaluations[name]["selected_threshold"]
            if hasattr(model, "predict_proba"):
                test_probs = model.predict_proba(X_test)[:, 1]
            else:
                test_probs = np.full(len(X_test), 0.5)
            test_preds = (test_probs >= th).astype(int)
            test_evaluations[name] = evaluate_predictions(y_test, test_preds, test_probs)
            test_evaluations[name]["threshold_used"] = float(th)

    # Persist artifacts only when edge detected.
    best_model_file = None
    meta_file = None
    slug = symbol.upper()
    suffix = f"_{horizon}d_{target_col}"
    if save_artifacts and has_edge:
        os.makedirs(save_dir, exist_ok=True)
        best_model_file = (
            f"{save_dir}/{slug}_{best_model_name.lower().replace(' ', '_')}{suffix}.joblib"
        )
        joblib.dump(fitted_models[best_model_name], best_model_file)

    metadata = {
        "symbol": symbol,
        "status": status,
        "has_edge": has_edge,
        "min_edge_auc": float(min_edge_auc),
        "best_validation_auc": float(best_val_auc),
        "selected_model": best_model_name if has_edge else None,
        "best_auc_model_name": best_model_name,
        "selected_threshold": best_threshold if has_edge else None,
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

    if save_artifacts and has_edge:
        meta_file = (
            f"{save_dir}/{slug}_{best_model_name.lower().replace(' ', '_')}{suffix}_metadata.json"
        )
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=4)

    return {
        "symbol": symbol,
        "status": status,
        "has_edge": bool(has_edge),
        "min_edge_auc": float(min_edge_auc),
        "best_validation_auc": float(best_val_auc),
        # Backwards-compatible fields: always populated so existing callers
        # (e.g. benchmark_runner) do not crash. Callers MUST check `status`
        # or `has_edge` before executing trades.
        "best_model_name": best_model_name,
        "best_model": fitted_models[best_model_name],
        "best_threshold": best_threshold,
        "fitted_models": fitted_models,
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
        corrs = X.apply(
            lambda col: col.corr(df_clean["target_binary"])
        ).sort_values(ascending=False)
        print("\n--- Feature Correlations with Binary Target ---")
        for feat, c in corrs.items():
            print(f" {feat:<18}: {c:+.4f}")

        # 2. Chronological Walk-Forward CV (mean AUC across folds and fold
        #    standard deviation are our honest, stable edge measurements).
        wf_results = chronological_walk_forward_cv(
            X, df_clean["target_binary"], n_splits=3
        )
        print("\n--- Chronological Walk-Forward Validation (3 Folds) ---")
        for fold in wf_results:
            print(
                f" Fold {fold['fold']} "
                f"(Train: {fold['train_size']}, Val: {fold['val_size']}):"
            )
            for m_name, m_scores in fold["metrics"].items():
                print(
                    f"   [{m_name}] F1: {m_scores['f1']:.4f} | "
                    f"AUC: {m_scores['roc_auc']:.4f} | "
                    f"Acc: {m_scores['accuracy']:.4f}"
                )

        # 2b. Mean + std AUC across folds — the honest edge measurement.
        auc_summary = _mean_walk_forward_auc(wf_results)
        stable_models = {
            name: stats
            for name, stats in auc_summary.items()
            if stats["is_stable"]
        }

        if stable_models:
            best_name = max(stable_models, key=lambda n: stable_models[n]["mean"])
            best_mean = stable_models[best_name]["mean"]
            best_std = stable_models[best_name]["std"]
            has_edge = True
        else:
            best_name = (
                max(auc_summary, key=lambda n: auc_summary[n]["mean"])
                if auc_summary
                else None
            )
            best_mean = auc_summary[best_name]["mean"] if best_name else 0.0
            best_std = auc_summary[best_name]["std"] if best_name else 0.0
            has_edge = False

        print("\n--- Mean Walk-Forward AUC per Model ---")
        for m_name, m_stats in sorted(
            auc_summary.items(), key=lambda x: -x[1]["mean"]
        ):
            marker = "✅" if m_stats["is_stable"] else "  "
            print(
                f" {marker} {m_name:<22}: "
                f"mean={m_stats['mean']:.4f}  "
                f"std={m_stats['std']:.4f}  "
                f"range=[{m_stats['min_fold']:.4f}, {m_stats['max_fold']:.4f}]"
            )
        print(
            f"\n Edge criteria: mean AUC >= {MIN_EDGE_AUC:.2f} "
            f"AND fold std <= {MAX_FOLD_STD:.2f}"
        )
        if best_name:
            print(
                f" Best model: {best_name} "
                f"(mean={best_mean:.4f}, std={best_std:.4f})"
            )
        if not has_edge:
            print(
                f" ⚠️  NO_EDGE: {symbol} has no *stable* predictive edge on validation."
            )
            print(f"     → System MUST output NO_TRADE for {symbol}.")
        else:
            print(f" ✅ EDGE_DETECTED on {symbol} via {best_name}.")

        # 3. Single-split model selection & test evaluation
        #    (train_and_select_model also uses AUC-based selection now.)
        selection_result = train_and_select_model(df_clean, symbol)
        print("\n--- Model Selection on Single Validation Split ---")
        if selection_result.get("has_edge"):
            print(
                f" Best Selected Model: {selection_result['best_model_name']} "
                f"(Val AUC: {selection_result['best_validation_auc']:.4f}, "
                f"Threshold: {selection_result['best_threshold']})"
            )
        else:
            print(
                f" NO_EDGE at threshold {MIN_EDGE_AUC:.2f}. "
                f"Best val AUC: {selection_result['best_validation_auc']:.4f}."
            )

        if selection_result.get("test_evaluations"):
            print("\n--- Out-of-Sample Test Evaluation ---")
            for m_name, m_scores in selection_result["test_evaluations"].items():
                print(
                    f" [{m_name}] Acc: {m_scores['accuracy']:.4f} | "
                    f"Bal-Acc: {m_scores['balanced_accuracy']:.4f} | "
                    f"F1: {m_scores['f1']:.4f} | "
                    f"AUC: {m_scores['roc_auc']:.4f}"
                )
        else:
            print(
                "\n(No out-of-sample test evaluation: "
                "no edge detected on validation.)"
            )

        report = {
            "symbol": symbol,
            "status": "EDGE_DETECTED" if has_edge else "NO_EDGE",
            "has_edge": bool(has_edge),
            "min_edge_auc": float(MIN_EDGE_AUC),
            "max_fold_std": float(MAX_FOLD_STD),
            "correlations": {k: round(float(v), 4) for k, v in corrs.items()},
            "walk_forward_cv": wf_results,
            "mean_walk_forward_auc_per_model": auc_summary,
            "best_mean_walk_forward_auc_model": best_name,
            "best_mean_walk_forward_auc": best_mean,
            "best_mean_walk_forward_std": best_std,
            "single_split_selected_model": selection_result.get("best_model_name"),
            "single_split_selected_threshold": selection_result.get("best_threshold"),
            "single_split_best_validation_auc": selection_result.get(
                "best_validation_auc"
            ),
            "single_split_status": selection_result.get("status"),
            "validation_evaluations": selection_result.get("val_evaluations"),
            "test_evaluations": selection_result.get("test_evaluations"),
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