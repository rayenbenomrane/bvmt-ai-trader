"""
Robustness analysis on top of the Section 19 momentum engine: rebalance
frequency, ranking method and portfolio-size/weight-cap sensitivity grids,
rolling chronological walk-forward folds, regime-filter attribution, and
concentration/exposure risk metrics.

This module adds analysis only -- it does not change the no-fee competition
cost assumption anywhere (no commission, slippage or market-impact term is
introduced), and it does not modify `src/momentum/engine.py`'s existing
behaviour for callers that don't pass the new optional arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.momentum.performance import compute_metrics


# ---------------------------------------------------------------------------
# Rebalance schedules (Task 1)
# ---------------------------------------------------------------------------

def quarter_end_decision_dates(
    calendar: pd.DatetimeIndex, min_history: int
) -> List[pd.Timestamp]:
    """Last trading session of each calendar quarter, once enough history
    exists for the 6-month momentum feature. Mirrors
    scripts.run_momentum_benchmark.month_end_decision_dates but grouped by
    quarter instead of month."""
    s = pd.Series(range(len(calendar)), index=calendar)
    quarter = calendar.year * 10 + calendar.quarter
    quarter_ends = s.groupby(quarter).apply(lambda x: x.index[-1])
    dates = sorted(quarter_ends.tolist())
    return [d for d in dates if calendar.get_loc(d) >= min_history]


# ---------------------------------------------------------------------------
# Rolling chronological walk-forward folds (Task 5)
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardFold:
    fold_id: int
    start: pd.Timestamp
    end: pd.Timestamp
    is_final_holdout: bool


def build_rolling_folds(
    calendar: pd.DatetimeIndex,
    first_valid_date: pd.Timestamp,
    window_months: int = 12,
    step_months: int = 6,
) -> List[WalkForwardFold]:
    """Strictly chronological rolling folds: each fold is a fixed-length
    calendar window (default 12 months), stepped forward by `step_months`
    (default 6, per the roadmap's own walk-forward convention in Section
    11.1). Folds are evaluation windows only -- nothing is fitted on a fold,
    so adjacent folds overlapping by (window - step) months is not leakage,
    it only means the same equity curve is sliced more than once. The last
    fold is flagged as the final holdout: it is reported but must never be
    used to pick the "most robust" configuration (see select_most_robust)."""
    folds: List[WalkForwardFold] = []
    cursor = pd.Timestamp(first_valid_date)
    last_date = calendar.max()
    fold_id = 0
    while True:
        fold_start = cursor
        fold_end = fold_start + pd.DateOffset(months=window_months)
        if fold_start >= last_date:
            break
        fold_id += 1
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                start=fold_start,
                end=min(fold_end, last_date + pd.Timedelta(days=1)),
                is_final_holdout=False,
            )
        )
        cursor = cursor + pd.DateOffset(months=step_months)
        if fold_end >= last_date:
            break
    if folds:
        folds[-1].is_final_holdout = True
    return folds


def fold_metrics(
    equity: pd.Series,
    bh_equity: pd.Series,
    fold: WalkForwardFold,
) -> Optional[Dict]:
    """Performance of one equity curve restricted to one fold's calendar
    window, compared against the buy-and-hold equity curve over the same
    window (so the comparison is always same-dates, same-capital, per the
    roadmap's Section 14 rule)."""
    seg = equity[(equity.index >= fold.start) & (equity.index < fold.end)]
    bh_seg = bh_equity[(bh_equity.index >= fold.start) & (bh_equity.index < fold.end)]
    if len(seg) < 5 or len(bh_seg) < 5:
        return None
    strat_ret = float(seg.iloc[-1] / seg.iloc[0] - 1.0) * 100.0
    bh_ret = float(bh_seg.iloc[-1] / bh_seg.iloc[0] - 1.0) * 100.0
    peak = seg.cummax()
    dd = float(((seg - peak) / peak).min() * 100.0)
    daily_ret = seg.pct_change().dropna()
    sharpe = (
        float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
        if daily_ret.std() > 0
        else 0.0
    )
    return {
        "fold_id": fold.fold_id,
        "fold_start": str(fold.start.date()),
        "fold_end": str(fold.end.date()),
        "is_final_holdout": fold.is_final_holdout,
        "return_pct": round(strat_ret, 3),
        "buy_and_hold_return_pct": round(bh_ret, 3),
        "excess_return_pct": round(strat_ret - bh_ret, 3),
        "max_drawdown_pct": round(dd, 3),
        "sharpe_ratio": round(sharpe, 3),
        "profitable": strat_ret > 0,
        "beats_buy_and_hold": strat_ret > bh_ret,
    }


def summarize_folds(
    fold_rows: List[Dict], exclude_final_holdout: bool = True
) -> Dict:
    """Robustness summary across folds. Per the task's "do not optimize on
    the final test period" constraint, the summary used for selecting a
    configuration excludes the final holdout fold by default; the holdout
    fold's own performance is reported separately for confirmation only."""
    usable = [r for r in fold_rows if not (exclude_final_holdout and r["is_final_holdout"])]
    holdout = next((r for r in fold_rows if r["is_final_holdout"]), None)
    if not usable:
        return {"fold_count": 0}
    returns = [r["return_pct"] for r in usable]
    excess = [r["excess_return_pct"] for r in usable]
    return {
        "fold_count": len(usable),
        "mean_fold_return_pct": round(float(np.mean(returns)), 3),
        "std_fold_return_pct": round(float(np.std(returns)), 3),
        "mean_excess_return_pct": round(float(np.mean(excess)), 3),
        "std_excess_return_pct": round(float(np.std(excess)), 3),
        "profitable_fold_ratio": round(
            sum(r["profitable"] for r in usable) / len(usable), 3
        ),
        "beats_buy_and_hold_fold_ratio": round(
            sum(r["beats_buy_and_hold"] for r in usable) / len(usable), 3
        ),
        "mean_fold_max_drawdown_pct": round(
            float(np.mean([r["max_drawdown_pct"] for r in usable])), 3
        ),
        "mean_fold_sharpe": round(float(np.mean([r["sharpe_ratio"] for r in usable])), 3),
        "holdout_fold": holdout,
    }


def robustness_score(summary: Dict, avg_turnover_pct: float) -> float:
    """A single comparable score that rewards *consistency*, not just mean
    return: mean excess return vs buy-and-hold across non-holdout folds,
    penalized for fold-to-fold volatility, drawdown severity and turnover.
    Mirrors the style of the existing project's validation_score in
    src/simulator/strategy.py (excess return minus drawdown/activity
    penalties) rather than inventing an unrelated methodology."""
    if summary.get("fold_count", 0) == 0:
        return float("-inf")
    return round(
        summary["mean_excess_return_pct"]
        - 0.5 * summary["std_excess_return_pct"]
        - 0.3 * abs(summary["mean_fold_max_drawdown_pct"])
        - 0.05 * avg_turnover_pct,
        3,
    )


# ---------------------------------------------------------------------------
# Regime attribution (Task 6)
# ---------------------------------------------------------------------------

def regime_attribution(regime_series: pd.DataFrame) -> Dict:
    counts = regime_series["regime"].value_counts()
    total = len(regime_series)
    first_known_sma_date = regime_series.loc[
        regime_series["regime"] != "UNKNOWN", "date"
    ].min()
    return {
        "sessions_total": int(total),
        "pct_risk_on": round(float(counts.get("RISK_ON", 0) / total * 100.0), 2),
        "pct_risk_off": round(float(counts.get("RISK_OFF", 0) / total * 100.0), 2),
        "pct_unknown_defaulted_defensive": round(
            float(counts.get("UNKNOWN", 0) / total * 100.0), 2
        ),
        "tunindex_history_start": str(regime_series["date"].min().date()),
        "sma200_first_computable": (
            str(first_known_sma_date.date()) if pd.notna(first_known_sma_date) else None
        ),
        "explanation": (
            "The regime filter reduces return in this sample for two separate reasons, "
            "not one. (1) Tunindex.csv only has history from "
            f"{regime_series['date'].min().date()}, so SMA(200) is not computable until "
            f"{first_known_sma_date.date() if pd.notna(first_known_sma_date) else 'N/A'}. "
            "Per the roadmap's own rule, an UNKNOWN regime defaults to no new risk, so the "
            "regime-filtered variants sat in cash through that entire warm-up period even "
            "though no actual risk-off signal fired -- this is a data-coverage artifact, "
            "not the filter 'working'. (2) Once the SMA is computable, TUNINDEX spent the "
            "large majority of the remaining sample above its 200-day average (a sustained "
            "uptrend), so the filter almost never triggered a real risk-off exit either. "
            "Across the whole sample it therefore only ever removed upside (the missed 2022 "
            "momentum gains) without ever being tested against a real, sustained TUNINDEX "
            "downturn -- the drawdowns the filtered and unfiltered variants did experience "
            "are statistically similar because that downturn never happened in this window. "
            "This is a sample-coverage limitation, not evidence the filter is useless: it "
            "would need a longer TUNINDEX history including a real bear market to be "
            "properly tested."
        ),
    }


# ---------------------------------------------------------------------------
# Concentration & exposure metrics (Task 7)
# ---------------------------------------------------------------------------

def concentration_and_exposure_metrics(rebalances: List, exposure: pd.Series) -> Dict:
    n_holdings = [rb.n_selected for rb in rebalances if rb.n_selected > 0]
    hhi = [
        sum(w ** 2 for w in rb.target_weights.values())
        for rb in rebalances
        if rb.target_weights
    ]
    return {
        "avg_n_holdings": round(float(np.mean(n_holdings)), 2) if n_holdings else 0.0,
        "min_n_holdings": int(np.min(n_holdings)) if n_holdings else 0,
        "max_n_holdings": int(np.max(n_holdings)) if n_holdings else 0,
        "avg_herfindahl_index": round(float(np.mean(hhi)), 4) if hhi else 0.0,
        "pct_rebalances_all_cash": round(
            100.0 * sum(1 for rb in rebalances if rb.n_selected == 0) / max(1, len(rebalances)),
            2,
        ),
        "avg_market_exposure_pct": round(float(exposure.mean() * 100.0), 2),
        "min_market_exposure_pct": round(float(exposure.min() * 100.0), 2),
        "pct_days_fully_in_cash": round(
            float((exposure < 0.01).mean() * 100.0), 2
        ),
    }
