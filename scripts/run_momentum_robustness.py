"""
Robustness analysis for the BVMT cross-sectional momentum strategy.

Extends scripts/run_momentum_benchmark.py (left unmodified, along with its
reports/ outputs) with:
  1. Monthly vs quarterly rebalancing
  2. Top 20% vs top 30% portfolios
  3. 3M-only vs 6M-only vs combined momentum ranking
  4. Yearly performance breakdowns, 2021-2026
  5. Rolling chronological walk-forward folds (12-month window, 6-month step)
  6. TUNINDEX SMA(200) regime-filter attribution
  7. Risk metrics: Sortino, worst month, annualized volatility, exposure,
     concentration (Herfindahl index, holding count)
  8. Sensitivity to portfolio size (top_fraction) and max position weight

No commission, slippage or market-impact term is introduced anywhere in this
script -- every variant below is evaluated under the same no-fee competition
assumption as scripts/run_momentum_benchmark.py. The "most robust" strategy
is selected using only non-holdout walk-forward folds (see
src/momentum/robustness.py:select never peeks at the final fold); the final
fold's performance is reported afterwards for confirmation only.

Run with:
    python3 scripts/run_momentum_robustness.py
(from the repository root; depends on scripts/run_momentum_benchmark.py
having been runnable, i.e. the same raw CSV panel under
src/data_collection/raw/.)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

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
from src.momentum.robustness import (  # noqa: E402
    build_rolling_folds,
    concentration_and_exposure_metrics,
    fold_metrics,
    quarter_end_decision_dates,
    regime_attribution,
    robustness_score,
    summarize_folds,
)
from scripts.run_momentum_benchmark import (  # noqa: E402
    _git_commit_hash,
    equal_weight_buy_and_hold,
    month_end_decision_dates,
)

OUT_DIR = ROOT / "reports" / "robustness"

RANKING_WEIGHTS = {
    "3m_only": (1.0, 0.0),
    "6m_only": (0.0, 1.0),
    "combined_3m_6m": (0.5, 0.5),
}


def build_variant_name(rebalance: str, top_fraction: float, ranking: str, regime: bool) -> str:
    return f"{rebalance}_top{int(top_fraction * 100)}_{ranking}{'_regime' if regime else ''}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading raw BVMT price panel ...")
    panel = load_price_panel()
    close_panel = build_close_panel(panel)
    open_panel = build_open_panel(panel)
    volume_panel = build_volume_panel(panel)
    calendar = build_trading_calendar(panel)
    close_panel = close_panel.reindex(calendar)
    open_panel = open_panel.reindex(calendar)
    volume_panel = volume_panel.reindex(calendar)

    tunindex = load_tunindex()
    regime_series = compute_regime_series(tunindex, sma_window=200)

    monthly_dates = month_end_decision_dates(calendar, min_history=TRADING_SESSIONS_6M)
    quarterly_dates = quarter_end_decision_dates(calendar, min_history=TRADING_SESSIONS_6M)
    print(f"Monthly decision dates:   {len(monthly_dates)}")
    print(f"Quarterly decision dates: {len(quarterly_dates)}")

    # Universe is a function of (calendar position, liquidity history), not
    # of rebalance frequency, so it's computed once per decision-date set
    # and reused across every ranking-method / top-fraction / regime
    # combination evaluated on that schedule (avoids exhaustive recompute,
    # per the "don't perform exhaustive unrelated testing" constraint).
    universe_by_schedule = {
        "monthly": compute_point_in_time_universe(close_panel, volume_panel, monthly_dates),
        "quarterly": compute_point_in_time_universe(close_panel, volume_panel, quarterly_dates),
    }
    decision_dates_by_schedule = {"monthly": monthly_dates, "quarterly": quarterly_dates}

    momentum_cache: Dict[tuple, Dict] = {}

    def get_momentum(schedule: str, ranking: str) -> Dict:
        key = (schedule, ranking)
        if key not in momentum_cache:
            w3, w6 = RANKING_WEIGHTS[ranking]
            momentum_cache[key] = compute_momentum_scores(
                close_panel,
                universe_by_schedule[schedule],
                weight_3m=w3,
                weight_6m=w6,
            )
        return momentum_cache[key]

    # ------------------------------------------------------------------
    # Core grid: Tasks 1-3 -- rebalance frequency x top_fraction x ranking
    # method, regime filter OFF (regime impact is isolated separately in
    # Task 6 so its effect isn't tangled up with the other three axes).
    # ------------------------------------------------------------------
    print("Running core grid (rebalance x top_fraction x ranking) ...")
    core_variants: Dict[str, Dict] = {}
    core_grid_rows = []
    for schedule, top_fraction, ranking in product(
        ("monthly", "quarterly"), (0.20, 0.30), RANKING_WEIGHTS.keys()
    ):
        name = build_variant_name(schedule, top_fraction, ranking, regime=False)
        momentum_by_date = get_momentum(schedule, ranking)
        cfg = PortfolioBacktestConfig(top_fraction=top_fraction, use_regime_filter=False)
        res = run_portfolio_backtest(
            close_panel,
            open_panel,
            universe_by_schedule[schedule],
            momentum_by_date,
            decision_dates_by_schedule[schedule],
            None,
            cfg,
        )
        core_variants[name] = res

    # Regime on/off, for the headline monthly/combined/top20/top30 pair
    # (Task 6) -- same engine, same schedule, only the regime flag differs.
    print("Running regime on/off comparison (monthly, combined ranking) ...")
    regime_variants: Dict[str, Dict] = {}
    for top_fraction, use_regime in product((0.20, 0.30), (False, True)):
        name = build_variant_name("monthly", top_fraction, "combined_3m_6m", use_regime)
        if name in core_variants:
            regime_variants[name] = core_variants[name]
            continue
        momentum_by_date = get_momentum("monthly", "combined_3m_6m")
        cfg = PortfolioBacktestConfig(top_fraction=top_fraction, use_regime_filter=use_regime)
        regime_variants[name] = run_portfolio_backtest(
            close_panel,
            open_panel,
            universe_by_schedule["monthly"],
            momentum_by_date,
            monthly_dates,
            regime_series if use_regime else None,
            cfg,
        )

    all_variants: Dict[str, Dict] = {**core_variants, **regime_variants}

    # ------------------------------------------------------------------
    # Benchmark: equal-weight buy-and-hold, anchored to the monthly
    # schedule's first eligible universe (same benchmark convention as
    # scripts/run_momentum_benchmark.py, reused rather than redefined).
    # ------------------------------------------------------------------
    first_decision = next(
        d for d in monthly_dates
        if any(info["eligible"] for info in universe_by_schedule["monthly"][d].values())
    )
    first_exec = next(
        rb.execution_date
        for rb in next(iter(core_variants.values()))["rebalances"]
        if rb.decision_date == first_decision
    )
    eligible_at_start = [
        s for s, info in universe_by_schedule["monthly"][first_decision].items() if info["eligible"]
    ]
    bh_equity = equal_weight_buy_and_hold(
        open_panel, close_panel, eligible_at_start, first_exec, 10_000.0
    )

    # ------------------------------------------------------------------
    # Task 8: sensitivity to portfolio size (top_fraction) and max
    # position weight. Held fixed: monthly rebalance, combined ranking,
    # no regime filter -- isolates each axis rather than a full
    # cross-product sweep.
    # ------------------------------------------------------------------
    print("Running portfolio-size / weight-cap sensitivity sweeps ...")
    combined_monthly_momentum = get_momentum("monthly", "combined_3m_6m")
    size_sensitivity_rows = []
    for top_fraction in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        cfg = PortfolioBacktestConfig(top_fraction=top_fraction, use_regime_filter=False)
        res = run_portfolio_backtest(
            close_panel, open_panel, universe_by_schedule["monthly"],
            combined_monthly_momentum, monthly_dates, None, cfg,
        )
        metrics = compute_metrics(res["equity_curve"], label=f"top_fraction={top_fraction}")
        conc = concentration_and_exposure_metrics(res["rebalances"], res["exposure"])
        size_sensitivity_rows.append({"top_fraction": top_fraction, **metrics, **conc})

    weight_cap_sensitivity_rows = []
    for weight_cap in (0.10, 0.15, 0.20, 0.25, 0.30, 0.50):
        cfg = PortfolioBacktestConfig(
            top_fraction=0.20, use_regime_filter=False, single_stock_weight_cap=weight_cap
        )
        res = run_portfolio_backtest(
            close_panel, open_panel, universe_by_schedule["monthly"],
            combined_monthly_momentum, monthly_dates, None, cfg,
        )
        metrics = compute_metrics(res["equity_curve"], label=f"weight_cap={weight_cap}")
        conc = concentration_and_exposure_metrics(res["rebalances"], res["exposure"])
        weight_cap_sensitivity_rows.append({"max_position_weight": weight_cap, **metrics, **conc})

    # ------------------------------------------------------------------
    # Task 5: rolling chronological walk-forward folds, every variant.
    # ------------------------------------------------------------------
    print("Building rolling walk-forward folds ...")
    folds = build_rolling_folds(calendar, first_valid_date=first_exec, window_months=12, step_months=6)
    print(f"  {len(folds)} folds, {folds[0].start.date()} -> {folds[-1].end.date()} "
          f"(fold {folds[-1].fold_id} is the final holdout, excluded from selection)")

    fold_rows_by_variant: Dict[str, List[Dict]] = {}
    summary_by_variant: Dict[str, Dict] = {}
    for name, res in all_variants.items():
        rows = [
            r for r in (fold_metrics(res["equity_curve"], bh_equity, f) for f in folds)
            if r is not None
        ]
        fold_rows_by_variant[name] = rows
        summary_by_variant[name] = summarize_folds(rows, exclude_final_holdout=True)

    # ------------------------------------------------------------------
    # Task 7: risk / concentration / exposure metrics for every variant.
    # ------------------------------------------------------------------
    print("Computing risk and concentration metrics ...")
    risk_rows = []
    for name, res in all_variants.items():
        metrics = compute_metrics(res["equity_curve"], label=name)
        conc = concentration_and_exposure_metrics(res["rebalances"], res["exposure"])
        avg_turnover_pct = round(
            100.0 * sum(rb.turnover for rb in res["rebalances"]) / max(1, len(res["rebalances"])),
            3,
        )
        summary = summary_by_variant[name]
        risk_rows.append(
            {
                **metrics,
                **conc,
                "avg_turnover_per_rebalance_pct": avg_turnover_pct,
                "walk_forward_fold_count": summary.get("fold_count", 0),
                "walk_forward_profitable_fold_ratio": summary.get("profitable_fold_ratio"),
                "walk_forward_beats_bh_fold_ratio": summary.get("beats_buy_and_hold_fold_ratio"),
                "walk_forward_mean_excess_return_pct": summary.get("mean_excess_return_pct"),
                "walk_forward_std_excess_return_pct": summary.get("std_excess_return_pct"),
                "robustness_score": robustness_score(summary, avg_turnover_pct),
                "holdout_fold_return_pct": (
                    summary["holdout_fold"]["return_pct"] if summary.get("holdout_fold") else None
                ),
                "holdout_fold_beats_bh": (
                    summary["holdout_fold"]["beats_buy_and_hold"] if summary.get("holdout_fold") else None
                ),
            }
        )
    bh_metrics = compute_metrics(bh_equity, label="equal_weight_buy_and_hold")
    risk_rows.append({**bh_metrics, "robustness_score": None})

    risk_df = pd.DataFrame(risk_rows).sort_values("robustness_score", ascending=False, na_position="last")

    # ------------------------------------------------------------------
    # Task 4: yearly breakdown, every variant + benchmark.
    # ------------------------------------------------------------------
    print("Computing yearly breakdowns ...")
    yearly_rows = []
    for name, res in all_variants.items():
        yr = yearly_returns(res["equity_curve"])
        yr.insert(0, "strategy", name)
        yearly_rows.append(yr)
    yr_bh = yearly_returns(bh_equity)
    yr_bh.insert(0, "strategy", "equal_weight_buy_and_hold")
    yearly_rows.append(yr_bh)
    yearly_df = pd.concat(yearly_rows, ignore_index=True)

    # ------------------------------------------------------------------
    # Select the most robust configuration (not the highest raw return):
    # highest robustness_score among non-regime, core-grid variants,
    # computed entirely from non-holdout folds. Regime variants are
    # reported but excluded from "most robust" selection since Task 6's
    # finding is that the regime filter's effect here is a data-coverage
    # artifact, not a genuine signal (see regime_analysis.json).
    # ------------------------------------------------------------------
    core_only = risk_df[risk_df["label"].isin(core_variants.keys())]
    most_robust = core_only.iloc[0]["label"] if not core_only.empty else None

    # ------------------------------------------------------------------
    # Task 6: regime attribution.
    # ------------------------------------------------------------------
    print("Writing regime attribution ...")
    regime_report = regime_attribution(regime_series)
    regime_report["return_comparison"] = {
        name: {
            "total_return_pct": compute_metrics(res["equity_curve"])["total_return_pct"],
            "max_drawdown_pct": compute_metrics(res["equity_curve"])["max_drawdown_pct"],
        }
        for name, res in regime_variants.items()
    }

    # ------------------------------------------------------------------
    # Write reports
    # ------------------------------------------------------------------
    print("Writing reports to", OUT_DIR)
    risk_df.to_csv(OUT_DIR / "risk_and_robustness_scorecard.csv", index=False)
    yearly_df.to_csv(OUT_DIR / "yearly_breakdown_2021_2026.csv", index=False)
    pd.DataFrame(size_sensitivity_rows).to_csv(OUT_DIR / "sensitivity_portfolio_size.csv", index=False)
    pd.DataFrame(weight_cap_sensitivity_rows).to_csv(OUT_DIR / "sensitivity_max_position_weight.csv", index=False)

    fold_rows_flat = []
    for name, rows in fold_rows_by_variant.items():
        for r in rows:
            fold_rows_flat.append({"strategy": name, **r})
    pd.DataFrame(fold_rows_flat).to_csv(OUT_DIR / "walk_forward_rolling_folds.csv", index=False)

    with (OUT_DIR / "regime_analysis.json").open("w") as f:
        json.dump(regime_report, f, indent=2, default=str)

    with (OUT_DIR / "experiment_config.json").open("w") as f:
        json.dump(
            {
                "git_commit_hash": _git_commit_hash(),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "cost_model": "competition mode preserved everywhere: no commission, "
                "no slippage, no market-impact term in any variant in this report",
                "execution": "next_session_open, integer shares, same convention as "
                "scripts/run_momentum_benchmark.py",
                "core_grid_axes": {
                    "rebalance_frequency": ["monthly", "quarterly"],
                    "top_fraction": [0.20, 0.30],
                    "ranking_method": list(RANKING_WEIGHTS.keys()),
                },
                "regime_variants_evaluated_separately": list(regime_variants.keys()),
                "walk_forward": {
                    "method": "rolling chronological folds, 12-month window, 6-month step",
                    "fold_count": len(folds),
                    "final_holdout_fold_id": folds[-1].fold_id,
                    "selection_rule": "robustness_score computed from non-holdout folds only; "
                    "the final holdout fold's performance is reported for confirmation and "
                    "was never used to choose a configuration",
                },
                "most_robust_core_variant": most_robust,
                "selection_methodology": "robustness_score = mean excess return vs "
                "buy-and-hold across non-holdout folds - 0.5*std(excess return across "
                "folds) - 0.3*|mean fold max drawdown| - 0.05*avg turnover per rebalance",
                "known_limitations": [
                    "Momentum uses price returns, not total returns (no dividend data in "
                    "the raw CSV source) -- unchanged from the first-task benchmark.",
                    "TUNINDEX history starts 2023-01-02 in this repo's CSV, so the SMA(200) "
                    "regime filter is UNKNOWN/defensive by default before ~Oct 2023; see "
                    "regime_analysis.json for the full attribution.",
                    "The eligible universe is whatever symbols exist in "
                    "src/data_collection/raw/ as of this snapshot; if that source already "
                    "excludes delisted/suspended BVMT symbols, survivorship bias upstream "
                    "of this code is possible and is not corrected here.",
                    "Folds overlap by (window - step) = 6 months by construction; this is "
                    "fine for evaluation-only metrics (nothing is fitted per fold) but means "
                    "fold rows are not independent observations.",
                ],
            },
            f,
            indent=2,
            default=str,
        )

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(" ROBUSTNESS SCORECARD (core grid, no regime filter, sorted by robustness_score)")
    print("=" * 78)
    cols = [
        "label", "total_return_pct", "cagr_pct", "max_drawdown_pct", "sharpe_ratio",
        "walk_forward_profitable_fold_ratio", "walk_forward_beats_bh_fold_ratio",
        "robustness_score", "holdout_fold_return_pct",
    ]
    print(core_only[cols].to_string(index=False))
    print(f"\nMost robust core-grid configuration: {most_robust}")

    print("\n" + "=" * 78)
    print(" REGIME FILTER: ON vs OFF (monthly, combined ranking)")
    print("=" * 78)
    print(
        risk_df[risk_df["label"].isin(regime_variants.keys())][
            ["label", "total_return_pct", "max_drawdown_pct", "avg_market_exposure_pct"]
        ].to_string(index=False)
    )
    print(f"\nRegime attribution: {regime_report['pct_unknown_defaulted_defensive']}% of TUNINDEX "
          f"sessions are UNKNOWN (defaulted defensive), {regime_report['pct_risk_on']}% RISK_ON, "
          f"{regime_report['pct_risk_off']}% RISK_OFF.")

    print(f"\nReports written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
