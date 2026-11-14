"""
First implementation task (roadmap Section 19): a standalone cross-sectional
3M/6M momentum benchmark, long-only, equal-weight, monthly rebalance,
next-session-open execution, top-20% vs top-30%, with and without the
TUNINDEX SMA(200) regime filter -- compared against cash and an
equal-weight buy-and-hold of the same eligible universe.

Run with:
    python -m scripts.run_momentum_benchmark
(run from the repository root)

This is a no-fee "competition" cost mode, matching the existing project's
competition convention (README: "no broker commission or slippage").
Realistic costs (Section 11.3) are Phase 4 work, not part of this first
task.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.momentum.data_loader import (  # noqa: E402
    build_close_panel,
    build_open_panel,
    build_trading_calendar,
    build_volume_panel,
    load_price_panel,
    load_tunindex,
)
from src.momentum.engine import (  # noqa: E402
    TRADING_SESSIONS_6M,
    PortfolioBacktestConfig,
    compute_momentum_scores,
    compute_point_in_time_universe,
    compute_regime_series,
    run_portfolio_backtest,
)
from src.momentum.performance import compute_metrics, yearly_returns  # noqa: E402

REPORTS_DIR = ROOT / "reports"


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def month_end_decision_dates(calendar: pd.DatetimeIndex, min_history: int) -> List[pd.Timestamp]:
    """Last trading session of each calendar month, once enough history
    exists for the 6-month momentum feature."""
    s = pd.Series(range(len(calendar)), index=calendar)
    month_ends = s.groupby([calendar.year, calendar.month]).apply(lambda x: x.index[-1])
    dates = sorted(month_ends.tolist())
    return [d for d in dates if calendar.get_loc(d) >= min_history]


def equal_weight_buy_and_hold(
    open_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    eligible_symbols: List[str],
    start_date: pd.Timestamp,
    initial_capital: float,
) -> pd.Series:
    """Benchmark 3 (roadmap Section 14): equal-weight buy-and-hold of the
    universe eligible at the first decision date, bought once at the first
    execution open, held to the end with no rebalancing."""
    calendar = close_panel.index
    valuation_close = close_panel.ffill()
    opens = open_panel.loc[start_date]
    per_name = initial_capital / len(eligible_symbols)
    shares = {}
    cash = initial_capital
    for sym in eligible_symbols:
        px = opens.get(sym)
        if pd.isna(px) or px is None or px <= 0:
            continue
        qty = int(per_name // px)
        if qty > 0:
            shares[sym] = qty
            cash -= qty * px
    equity = []
    for d in calendar:
        if d < start_date:
            equity.append(initial_capital)
            continue
        val = cash
        px_row = valuation_close.loc[d]
        for sym, qty in shares.items():
            p = px_row.get(sym)
            if pd.notna(p):
                val += qty * p
        equity.append(val)
    return pd.Series(equity, index=calendar, name="equal_weight_buy_and_hold")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading raw BVMT price panel ...")
    panel = load_price_panel()
    quality_log = panel.attrs.get("quality_log", [])

    close_panel = build_close_panel(panel)
    open_panel = build_open_panel(panel)
    volume_panel = build_volume_panel(panel)
    calendar = build_trading_calendar(panel)
    close_panel = close_panel.reindex(calendar)
    open_panel = open_panel.reindex(calendar)
    volume_panel = volume_panel.reindex(calendar)

    tunindex = load_tunindex()
    regime_series = compute_regime_series(tunindex, sma_window=200)

    print(f"Universe: {close_panel.shape[1]} symbols, "
          f"{len(calendar)} sessions "
          f"({calendar.min().date()} -> {calendar.max().date()})")

    decision_dates = month_end_decision_dates(calendar, min_history=TRADING_SESSIONS_6M)
    print(f"Decision dates (monthly, after 6M history warm-up): {len(decision_dates)} "
          f"({decision_dates[0].date()} -> {decision_dates[-1].date()})")

    print("Computing point-in-time universe & liquidity filters ...")
    universe = compute_point_in_time_universe(close_panel, volume_panel, decision_dates)

    print("Computing 3M/6M momentum & composite ranks ...")
    momentum_by_date = compute_momentum_scores(close_panel, universe)

    initial_capital = 10_000.0
    results: Dict[str, Dict] = {}
    variants = [
        ("momentum_top20", 0.20, False),
        ("momentum_top20_regime", 0.20, True),
        ("momentum_top30", 0.30, False),
        ("momentum_top30_regime", 0.30, True),
    ]
    for name, top_fraction, use_regime in variants:
        print(f"Backtesting variant: {name} ...")
        cfg = PortfolioBacktestConfig(
            initial_capital=initial_capital,
            top_fraction=top_fraction,
            use_regime_filter=use_regime,
        )
        results[name] = run_portfolio_backtest(
            close_panel,
            open_panel,
            universe,
            momentum_by_date,
            decision_dates,
            regime_series if use_regime else None,
            cfg,
        )

    # Use the first decision date that actually has an eligible universe as
    # the buy-and-hold benchmark's inception point (some of the earliest
    # monthly decision dates have zero eligible names while history/liquidity
    # windows are still warming up).
    first_decision = next(
        d for d in decision_dates
        if any(info["eligible"] for info in universe[d].values())
    )
    first_exec_rb = next(
        rb for rb in results["momentum_top20"]["rebalances"]
        if rb.decision_date == first_decision
    )
    first_exec = first_exec_rb.execution_date
    eligible_at_start = [
        s for s, info in universe[first_decision].items() if info["eligible"]
    ]
    bh_equity = equal_weight_buy_and_hold(
        open_panel, close_panel, eligible_at_start, first_exec, initial_capital
    )
    cash_equity = pd.Series(initial_capital, index=calendar, name="cash")

    # ---------------- reports ----------------
    print("Writing reports ...")

    # 1. data quality
    with (REPORTS_DIR / "data_quality_report.json").open("w") as f:
        json.dump(
            {
                "symbols_loaded": int(close_panel.shape[1]),
                "sessions": int(len(calendar)),
                "date_range": [str(calendar.min().date()), str(calendar.max().date())],
                "tunindex_rows": int(len(tunindex)),
                "tunindex_date_range": [
                    str(tunindex["date"].min().date()),
                    str(tunindex["date"].max().date()),
                ],
                "per_file_quality_log": quality_log,
            },
            f,
            indent=2,
            default=str,
        )

    # 2. universe snapshot (long format, every decision date x symbol)
    uni_rows = []
    for d, syms in universe.items():
        for sym, info in syms.items():
            uni_rows.append({"decision_date": d, "symbol": sym, **info})
    pd.DataFrame(uni_rows).to_csv(REPORTS_DIR / "universe_snapshot.csv", index=False)

    # 3. ranked candidates (momentum scores, all decision dates)
    rank_rows = []
    for d, frame in momentum_by_date.items():
        if frame.empty:
            continue
        f2 = frame.reset_index()
        f2.insert(0, "decision_date", d)
        rank_rows.append(f2)
    if rank_rows:
        pd.concat(rank_rows, ignore_index=True).to_csv(
            REPORTS_DIR / "ranked_candidates.csv", index=False
        )

    # 4. target portfolios / rebalance log, one file across all variants
    tp_rows = []
    for name, res in results.items():
        for rb in res["rebalances"]:
            tp_rows.append(
                {
                    "strategy": name,
                    "decision_date": rb.decision_date,
                    "execution_date": rb.execution_date,
                    "regime": rb.regime,
                    "n_eligible": rb.n_eligible,
                    "n_selected": rb.n_selected,
                    "selected": ",".join(rb.selected),
                    "turnover_pct": round(rb.turnover * 100.0, 3),
                }
            )
    pd.DataFrame(tp_rows).to_csv(REPORTS_DIR / "target_portfolio.csv", index=False)

    # 5. daily portfolio equity, all variants + benchmarks
    equity_frame = pd.DataFrame(
        {name: res["equity_curve"] for name, res in results.items()}
    )
    equity_frame["cash"] = cash_equity
    equity_frame["equal_weight_buy_and_hold"] = bh_equity
    equity_frame.index.name = "date"
    equity_frame.to_csv(REPORTS_DIR / "daily_portfolio_equity.csv")

    # 6. strategy comparison (metrics summary)
    comparison_rows = []
    for name, res in results.items():
        comparison_rows.append(compute_metrics(res["equity_curve"], label=name))
    comparison_rows.append(compute_metrics(bh_equity, label="equal_weight_buy_and_hold"))
    comparison_rows.append(
        {
            "label": "cash",
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "annualized_volatility_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "calmar_ratio": 0.0,
            "worst_month_pct": 0.0,
        }
    )
    comparison_df = pd.DataFrame(comparison_rows)
    # turnover summary per variant
    turnover_avg = {
        name: round(
            100.0 * sum(rb.turnover for rb in res["rebalances"]) / max(1, len(res["rebalances"])),
            3,
        )
        for name, res in results.items()
    }
    comparison_df["avg_turnover_per_rebalance_pct"] = comparison_df["label"].map(turnover_avg)
    comparison_df.to_csv(REPORTS_DIR / "strategy_comparison.csv", index=False)

    # 7. yearly returns per variant + benchmark
    yr_rows = []
    for name, res in results.items():
        yr = yearly_returns(res["equity_curve"])
        yr.insert(0, "strategy", name)
        yr_rows.append(yr)
    yr_bh = yearly_returns(bh_equity)
    yr_bh.insert(0, "strategy", "equal_weight_buy_and_hold")
    yr_rows.append(yr_bh)
    pd.concat(yr_rows, ignore_index=True).to_csv(REPORTS_DIR / "yearly_returns.csv", index=False)

    # 8. experiment config
    with (REPORTS_DIR / "experiment_config.json").open("w") as f:
        json.dump(
            {
                "git_commit_hash": _git_commit_hash(),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "initial_capital": initial_capital,
                "execution": "next_session_open",
                "cost_model": "competition mode: no commission, no slippage",
                "rebalance_schedule": "monthly (last trading session of each calendar month)",
                "momentum_formation": "3M (63 sessions) and 6M (126 sessions) PRICE return "
                "(not total return -- dividend/corporate-action data unavailable in the "
                "raw CSV source for this first task; this is the price-return sensitivity "
                "test the roadmap calls for when dividend data is incomplete)",
                "liquidity_filters": {
                    "max_zero_volume_ratio_60d": 0.30,
                    "min_avg_traded_value_60d_tnd": 100_000.0,
                    "min_history_sessions": TRADING_SESSIONS_6M,
                },
                "variants": [v[0] for v in variants],
                "decision_dates_count": len(decision_dates),
                "decision_date_range": [str(decision_dates[0].date()), str(decision_dates[-1].date())],
            },
            f,
            indent=2,
            default=str,
        )

    # ---------------- console summary ----------------
    print("\n" + "=" * 72)
    print(" STRATEGY COMPARISON (no-fee competition mode)")
    print("=" * 72)
    print(
        comparison_df[
            ["label", "total_return_pct", "cagr_pct", "max_drawdown_pct", "sharpe_ratio",
             "avg_turnover_per_rebalance_pct"]
        ].to_string(index=False)
    )
    print(f"\nReports written to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
