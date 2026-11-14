"""Performance metrics (roadmap Section 13) for an equity curve."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min() * 100.0)


def compute_metrics(equity: pd.Series, label: str = "") -> Dict[str, float]:
    equity = equity.dropna()
    daily_ret = equity.pct_change().dropna()
    total_return_pct = (equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0
    n_days = (equity.index[-1] - equity.index[0]).days
    years = max(n_days / 365.25, 1e-9)
    cagr_pct = ((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0) * 100.0
    vol_pct = float(daily_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)
    mean_daily = float(daily_ret.mean())
    sharpe = (
        (mean_daily / daily_ret.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
        if daily_ret.std() > 0
        else 0.0
    )
    downside = daily_ret[daily_ret < 0]
    sortino = (
        (mean_daily / downside.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
        if len(downside) > 1 and downside.std() > 0
        else 0.0
    )
    mdd = max_drawdown(equity)
    calmar = (cagr_pct / abs(mdd)) if mdd < 0 else 0.0
    worst_month = (
        equity.resample("ME").last().pct_change().dropna().min() * 100.0
        if len(equity) > 22
        else 0.0
    )
    return {
        "label": label,
        "total_return_pct": round(float(total_return_pct), 3),
        "cagr_pct": round(float(cagr_pct), 3),
        "annualized_volatility_pct": round(vol_pct, 3),
        "sharpe_ratio": round(float(sharpe), 3),
        "sortino_ratio": round(float(sortino), 3),
        "max_drawdown_pct": round(mdd, 3),
        "calmar_ratio": round(float(calmar), 3),
        "worst_month_pct": round(float(worst_month), 3),
    }


def yearly_returns(equity: pd.Series) -> pd.DataFrame:
    yearly = equity.resample("YE").last()
    yearly_start = equity.resample("YE").first()
    rets = (yearly / yearly_start - 1.0) * 100.0
    return pd.DataFrame({"year": rets.index.year, "return_pct": rets.values.round(3)})
