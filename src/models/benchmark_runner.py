import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config.database import SessionLocal
from src.database.models import DailyPrice
from src.models.feature_engineering import build_features

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.dummy import DummyClassifier

try:
    import xgboost as xgb
    XGB_AVAIL = True
except ImportError:
    XGB_AVAIL = False

def run_benchmark_for_symbol(symbol: str) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        query = db.query(DailyPrice).filter(DailyPrice.symbol == symbol).order_by(DailyPrice.date.asc()).statement
        df_raw = pd.read_sql(query, db.bind)
        if df_raw.empty or len(df_raw) < 60:
            return {"symbol": symbol, "status": "Insufficient data"}

        df = build_features(df_raw)
        df['next_return'] = (df['close'].shift(-1) - df['close']) / df['close']
        
        # Targets
        df['target_binary'] = (df['next_return'] > 0).astype(int)
        
        # Three-class target (0: DOWN < -0.5%, 1: NEUTRAL, 2: UP > +0.5%)
        conditions = [df['next_return'] > 0.005, df['next_return'] < -0.005]
        df['target_3class'] = np.select(conditions, [2, 0], default=1)

        feature_cols = [
            'return_1d', 'return_2d', 'return_5d', 'dist_sma20', 'dist_sma50',
            'ema_ratio', 'macd', 'macd_hist', 'macd_hist_diff',
            'volatility_5d', 'volatility_20d', 'volume_rel_20', 'atr14_norm'
        ]

        df_clean = df.dropna(subset=feature_cols + ['target_binary', 'target_3class', 'next_return']).copy()
        
        X = df_clean[feature_cols]
        y_bin = df_clean['target_binary']
        y_3cl = df_clean['target_3class']
        dates = df_clean['date']
        closes = df_clean['close']
        next_returns = df_clean['next_return']

        # Chronological Split: 60% Train, 20% Val, 20% Test
        n = len(X)
        t_end, v_end = int(n * 0.6), int(n * 0.8)

        X_train, X_val, X_test = X.iloc[:t_end], X.iloc[t_end:v_end], X.iloc[v_end:]
        yb_train, yb_val, yb_test = y_bin.iloc[:t_end], y_bin.iloc[t_end:v_end], y_bin.iloc[v_end:]
        test_closes = closes.iloc[v_end:]
        test_returns = next_returns.iloc[v_end:]

        print(f"\n==========================================")
        print(f" BENCHMARK & SIMULATION: {symbol}")
        print(f"==========================================")

        # Models Evaluation (Binary)
        models = {
            "Majority Baseline": DummyClassifier(strategy="most_frequent"),
            "Logistic Regression": LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=5, class_weight='balanced', random_state=42)
        }
        if XGB_AVAIL:
            models["XGBoost"] = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)

        model_results = {}
        best_model_name = "Majority Baseline"
        best_model = models[best_model_name]
        best_f1 = -1

        for name, m in models.items():
            m.fit(X_train, yb_train)
            preds = m.predict(X_test)
            probs = m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else [0.5]*len(preds)

            acc = accuracy_score(yb_test, preds)
            b_acc = balanced_accuracy_score(yb_test, preds)
            prec = precision_score(yb_test, preds, zero_division=0)
            rec = recall_score(yb_test, preds, zero_division=0)
            f1 = f1_score(yb_test, preds, zero_division=0)
            auc = roc_auc_score(yb_test, probs) if len(np.unique(yb_test)) > 1 else 0.5

            model_results[name] = {
                "accuracy": round(acc, 4), "balanced_accuracy": round(b_acc, 4),
                "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "roc_auc": round(auc, 4)
            }
            print(f" [{name}] Acc: {acc:.4f} | Bal-Acc: {b_acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model = m
                best_model_name = name

        # --- Simple Trading Simulation (10,000 TND initial capital) ---
        # Rules: Long-only, buy on UP signal (1), hold cash otherwise. 
        # Tunis Stock Exchange transaction costs: 0.15% exchange commission + ~0.50% brokerage + 19% VAT on commissions (~0.77% round trip or per trade)
        initial_capital = 10000.0
        cash = initial_capital
        shares = 0.0
        trade_count = 0
        total_costs = 0.0

        test_preds = best_model.predict(X_test)

        # Simulation Loop over out-of-sample test period
        portfolio_values = []
        for i in range(len(test_preds)):
            current_close = test_closes.iloc[i]
            signal = test_preds[i]
            nxt_ret = test_returns.iloc[i]

            # Execution logic with transaction fee (approx 0.40% per trade execution)
            fee_rate = 0.0040

            if signal == 1 and cash > 0:
                # Buy stock
                invest_amount = cash
                fee = invest_amount * fee_rate
                total_costs += fee
                shares = (invest_amount - fee) / current_close
                cash = 0.0
                trade_count += 1
            elif signal == 0 and shares > 0:
                # Sell stock to cash
                sell_value = shares * current_close
                fee = sell_value * fee_rate
                total_costs += fee
                cash = sell_value - fee
                shares = 0.0
                trade_count += 1

            port_val = cash + (shares * current_close)
            portfolio_values.append(port_val)

        final_portfolio_val = portfolio_values[-1] if portfolio_values else initial_capital
        total_return_pct = ((final_portfolio_val - initial_capital) / initial_capital) * 100

        # Maximum Drawdown calculation
        port_series = pd.Series(portfolio_values)
        rolling_max = port_series.cummax()
        drawdown = (port_series - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100

        # Buy-and-Hold Baseline comparison
        bh_shares = (initial_capital * (1 - 0.0040)) / test_closes.iloc[0]
        bh_final_val = bh_shares * test_closes.iloc[-1]
        bh_return_pct = ((bh_final_val - initial_capital) / initial_capital) * 100

        print(f"\n--- Trading Simulation Results ({symbol}) ---")
        print(f" Model Used        : {best_model_name}")
        print(f" Final Portfolio   : {final_portfolio_val:.3f} TND (Return: {total_return_pct:.2f}%)")
        print(f" Buy & Hold Return : {bh_return_pct:.2f}%")
        print(f" Max Drawdown      : {max_dd:.2f}%")
        print(f" Total Trades      : {trade_count}")
        print(f" Transaction Costs : {total_costs:.3f} TND")

        sim_results = {
            "model_used": best_model_name,
            "final_portfolio_value": round(final_portfolio_val, 3),
            "total_return_pct": round(total_return_pct, 2),
            "buy_and_hold_return_pct": round(bh_return_pct, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "total_trades": trade_count,
            "total_transaction_costs": round(total_costs, 3)
        }

        os.makedirs("data/processed/reports", exist_ok=True)
        report = {
            "symbol": symbol,
            "model_benchmarks": model_results,
            "simulation": sim_results
        }
        with open(f"data/processed/reports/{symbol}_benchmark_simulation.json", "w") as f:
            json.dump(report, f, indent=4)

        return report

    finally:
        db.close()

def main():
    for sym in ["BT", "SFBT"]:
        run_benchmark_for_symbol(sym)

if __name__ == "__main__":
    main()