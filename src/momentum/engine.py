"""
Core momentum-portfolio engine: universe/liquidity filters (roadmap Section 5),
momentum features and ranking (Section 6), the TUNINDEX regime filter
(Section 7), and a multi-asset portfolio backtest with next-session-open
execution (Sections 11-12). All functions are point-in-time: at decision date
`d`, nothing past `d` is read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_SESSIONS_3M = 63
TRADING_SESSIONS_6M = 126
LIQUIDITY_LOOKBACK = 60
MAX_ZERO_VOLUME_RATIO = 0.30  # roadmap 5.1 initial threshold
MIN_AVG_TRADED_VALUE_TND = 100_000.0  # roadmap 5.1 initial threshold


# ---------------------------------------------------------------------------
# Universe & liquidity (Section 5)
# ---------------------------------------------------------------------------

def compute_point_in_time_universe(
    close_panel: pd.DataFrame,
    volume_panel: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    min_history_sessions: int = TRADING_SESSIONS_6M,
    liquidity_lookback: int = LIQUIDITY_LOOKBACK,
    max_zero_volume_ratio: float = MAX_ZERO_VOLUME_RATIO,
    min_avg_traded_value: float = MIN_AVG_TRADED_VALUE_TND,
) -> Dict[pd.Timestamp, Dict[str, Dict[str, float]]]:
    """For every decision date, return every symbol's eligibility and the
    liquidity statistics behind the decision, using only data up to and
    including that date.

    Returns: {decision_date: {symbol: {eligible, reason, zero_volume_ratio,
                                         avg_traded_value_60d, history_sessions}}}
    """
    calendar = close_panel.index
    traded_value_panel = close_panel.fillna(0.0) * volume_panel.reindex(
        columns=close_panel.columns
    ).fillna(0.0)
    traded_flag = close_panel.notna()  # a session on which the symbol actually traded

    result: Dict[pd.Timestamp, Dict[str, Dict[str, float]]] = {}
    for d in decision_dates:
        pos = calendar.searchsorted(d, side="right")  # exclusive of future
        if pos == 0:
            result[d] = {}
            continue
        history_slice = slice(max(0, pos - liquidity_lookback), pos)
        liq_window_traded = traded_flag.iloc[history_slice]
        liq_window_value = traded_value_panel.iloc[history_slice]
        history_sessions_count = traded_flag.iloc[:pos].sum(axis=0)

        zero_vol_ratio = 1.0 - liq_window_traded.mean(axis=0)
        avg_traded_value = liq_window_value.mean(axis=0)
        last_traded_pos = traded_flag.iloc[:pos].apply(
            lambda col: col[col].index.max() if col.any() else pd.NaT
        )
        sessions_since_last_trade = pd.Series(
            {
                sym: (pos - 1 - calendar.get_loc(ts)) if pd.notna(ts) else np.inf
                for sym, ts in last_traded_pos.items()
            }
        )

        per_symbol: Dict[str, Dict[str, float]] = {}
        for sym in close_panel.columns:
            hist = float(history_sessions_count.get(sym, 0))
            zvr = float(zero_vol_ratio.get(sym, 1.0))
            atv = float(avg_traded_value.get(sym, 0.0))
            stale = float(sessions_since_last_trade.get(sym, np.inf))

            reasons = []
            if hist < min_history_sessions:
                reasons.append("insufficient_history")
            if zvr > max_zero_volume_ratio:
                reasons.append("zero_volume_ratio_too_high")
            if atv < min_avg_traded_value:
                reasons.append("avg_traded_value_too_low")
            if stale > 5:
                reasons.append("not_currently_tradable")

            per_symbol[sym] = {
                "eligible": len(reasons) == 0,
                "reason": ";".join(reasons) if reasons else "OK",
                "zero_volume_ratio_60d": round(zvr, 4),
                "avg_traded_value_60d_tnd": round(atv, 2),
                "history_sessions": hist,
                "sessions_since_last_trade": stale,
            }
        result[d] = per_symbol
    return result


# ---------------------------------------------------------------------------
# Momentum features & ranking (Section 6)
# ---------------------------------------------------------------------------

def compute_momentum_scores(
    close_panel: pd.DataFrame,
    universe: Dict[pd.Timestamp, Dict[str, Dict[str, float]]],
    lookback_3m: int = TRADING_SESSIONS_3M,
    lookback_6m: int = TRADING_SESSIONS_6M,
    max_price_staleness: int = 5,
    weight_3m: float = 0.5,
    weight_6m: float = 0.5,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """For every decision date, compute price-return momentum (3M, 6M),
    percentile ranks and the composite score, restricted to that date's
    eligible universe. Returns NaN-free DataFrames indexed by symbol.

    NOTE (roadmap 6.1): dividend/corporate-action data is not available in
    the raw CSV source for this first task, so this uses PRICE returns, not
    total returns. This is explicitly the "clearly labeled price-return
    sensitivity test" the roadmap calls for when dividend data is
    incomplete -- it is not presented as a total-return momentum signal.
    """
    calendar = close_panel.index
    out: Dict[pd.Timestamp, pd.DataFrame] = {}
    for d, elig_map in universe.items():
        pos = calendar.get_loc(d)
        eligible_syms = [s for s, info in elig_map.items() if info["eligible"]]
        if not eligible_syms:
            out[d] = pd.DataFrame(
                columns=["return_3m", "return_6m", "pct_3m", "pct_6m", "momentum_score"]
            )
            continue

        def last_traded_close_near(sym: str, target_pos: int) -> Optional[float]:
            if target_pos < 0:
                return None
            lo = max(0, target_pos - max_price_staleness)
            window = close_panel[sym].iloc[lo : target_pos + 1]
            window = window.dropna()
            if window.empty:
                return None
            return float(window.iloc[-1])

        rows = []
        for sym in eligible_syms:
            px_now = last_traded_close_near(sym, pos)
            px_3m = last_traded_close_near(sym, pos - lookback_3m)
            px_6m = last_traded_close_near(sym, pos - lookback_6m)
            if px_now is None or px_3m is None or px_6m is None or px_3m <= 0 or px_6m <= 0:
                continue
            rows.append(
                {
                    "symbol": sym,
                    "return_3m": px_now / px_3m - 1.0,
                    "return_6m": px_now / px_6m - 1.0,
                }
            )
        if not rows:
            out[d] = pd.DataFrame(
                columns=["return_3m", "return_6m", "pct_3m", "pct_6m", "momentum_score"]
            )
            continue
        frame = pd.DataFrame(rows).set_index("symbol")
        frame["pct_3m"] = frame["return_3m"].rank(pct=True)
        frame["pct_6m"] = frame["return_6m"].rank(pct=True)
        # weight_3m/weight_6m let the caller isolate a pure 3M or pure 6M
        # ranking (weight=1.0/0.0) from the combined 50/50 score, while still
        # requiring both lookbacks to be computable so the eligible universe
        # is identical across ranking methods -- only the ranking formula
        # changes, not the data available to it.
        frame["momentum_score"] = weight_3m * frame["pct_3m"] + weight_6m * frame["pct_6m"]
        out[d] = frame.sort_values("momentum_score", ascending=False)
    return out


# ---------------------------------------------------------------------------
# TUNINDEX regime filter (Section 7)
# ---------------------------------------------------------------------------

def compute_regime_series(
    tunindex: pd.DataFrame, sma_window: int = 200
) -> pd.DataFrame:
    """Point-in-time RISK_ON / RISK_OFF / UNKNOWN regime per TUNINDEX session.

    RISK_ON  : close > SMA(window), both using only data up to that session.
    RISK_OFF : close <= SMA(window).
    UNKNOWN  : insufficient history for the SMA; defaults to no new risk.
    """
    s = tunindex.sort_values("date").reset_index(drop=True).copy()
    s["sma"] = s["close"].rolling(sma_window, min_periods=sma_window).mean()
    s["regime"] = np.where(
        s["sma"].isna(),
        "UNKNOWN",
        np.where(s["close"] > s["sma"], "RISK_ON", "RISK_OFF"),
    )
    return s[["date", "close", "sma", "regime"]]


def regime_as_of(regime_series: pd.DataFrame, d: pd.Timestamp) -> str:
    """The regime using only TUNINDEX sessions at or before d. UNKNOWN
    (defensive) if TUNINDEX has no observation at/before d."""
    eligible = regime_series[regime_series["date"] <= d]
    if eligible.empty:
        return "UNKNOWN"
    return str(eligible.iloc[-1]["regime"])


# ---------------------------------------------------------------------------
# Portfolio construction & multi-asset backtest (Sections 6.3-6.4, 11, 12)
# ---------------------------------------------------------------------------

@dataclass
class PortfolioBacktestConfig:
    initial_capital: float = 10_000.0
    top_fraction: float = 0.20  # top 20% vs top 30% (Section 6.3)
    use_regime_filter: bool = False
    single_stock_weight_cap: float = 0.20  # Section 12
    rebalance_rank_change_quintiles: int = 2  # "more than two-quintile deterioration"


@dataclass
class RebalanceRecord:
    decision_date: pd.Timestamp
    execution_date: Optional[pd.Timestamp]
    regime: str
    n_eligible: int
    n_selected: int
    selected: List[str]
    target_weights: Dict[str, float]
    turnover: float


def _select_target_portfolio(
    momentum: pd.DataFrame,
    top_fraction: float,
    weight_cap: float,
    current_holdings_syms: List[str],
    rank_change_quintiles: int,
) -> List[str]:
    """Top-`top_fraction` of the eligible, ranked universe. Existing holdings
    are kept while still in the eligible universe, unless their quintile has
    deteriorated by more than `rank_change_quintiles` quintiles below the cut
    line (roadmap 6.4 turnover control) -- this avoids unnecessary turnover
    from marginal rank reshuffling between adjacent names."""
    if momentum.empty:
        return []
    n = len(momentum)
    cutoff = max(1, int(np.ceil(n * top_fraction)))
    ranked = momentum.reset_index()  # already sorted desc by momentum_score
    ranked["rank"] = np.arange(1, n + 1)
    ranked["quintile"] = np.minimum(5, ((ranked["rank"] - 1) * 5 // n) + 1)
    top_set = set(ranked.iloc[:cutoff]["symbol"])

    kept = []
    for sym in current_holdings_syms:
        row = ranked[ranked["symbol"] == sym]
        if row.empty:
            continue  # no longer eligible at all -> drop
        quintile = int(row.iloc[0]["quintile"])
        cutoff_quintile = int(ranked.iloc[cutoff - 1]["quintile"])
        if quintile <= cutoff_quintile + rank_change_quintiles:
            kept.append(sym)

    selected = list(dict.fromkeys(kept + list(top_set)))
    # Enforce single-stock weight cap by capping the number of names so that
    # equal weight never exceeds the cap (simpler and more transparent than
    # a post-hoc re-weighting for this first-task benchmark).
    min_names_for_cap = int(np.ceil(1.0 / weight_cap))
    if len(selected) < min_names_for_cap and len(ranked) >= min_names_for_cap:
        extra = [s for s in ranked["symbol"] if s not in selected]
        selected = selected + extra[: max(0, min_names_for_cap - len(selected))]
    return selected


def run_portfolio_backtest(
    close_panel: pd.DataFrame,
    open_panel: pd.DataFrame,
    universe: Dict[pd.Timestamp, Dict[str, Dict[str, float]]],
    momentum_by_date: Dict[pd.Timestamp, pd.DataFrame],
    decision_dates: List[pd.Timestamp],
    regime_series: Optional[pd.DataFrame],
    config: PortfolioBacktestConfig,
) -> Dict:
    """Simulate the long-only equal-weight momentum portfolio.

    Timing (roadmap 11.2): a decision taken at the close of `decision_date`
    executes no earlier than the next calendar-trading session's OPEN. Daily
    mark-to-market uses the last traded close (forward-filled for valuation
    only -- never for computing a return), exactly per Section 4.1's rule
    that a no-trade session is never treated as a genuine zero return.
    """
    calendar = close_panel.index
    valuation_close = close_panel.ffill()  # valuation only; momentum uses raw returns above

    cash = config.initial_capital
    shares: Dict[str, int] = {}
    rebalance_log: List[RebalanceRecord] = []
    equity_curve = []
    daily_exposure = []

    def portfolio_value(at_date: pd.Timestamp) -> float:
        val = cash
        if shares:
            px = valuation_close.loc[at_date]
            for sym, qty in shares.items():
                p = px.get(sym, np.nan)
                if pd.notna(p):
                    val += qty * p
        return val

    # Build the execution-date mapping: first calendar session strictly after
    # each decision date.
    exec_date_for: Dict[pd.Timestamp, Optional[pd.Timestamp]] = {}
    for d in decision_dates:
        pos = calendar.searchsorted(d, side="right")
        exec_date_for[d] = calendar[pos] if pos < len(calendar) else None

    decision_set = set(decision_dates)
    rebalance_pointer = 0
    sorted_decisions = sorted(decision_dates)

    for i, today in enumerate(calendar):
        # Did a prior decision's execution land today? Process all that do,
        # in decision order (there is at most one per rebalance schedule).
        while (
            rebalance_pointer < len(sorted_decisions)
            and exec_date_for[sorted_decisions[rebalance_pointer]] == today
        ):
            d = sorted_decisions[rebalance_pointer]
            rebalance_pointer += 1

            elig_map = universe.get(d, {})
            n_eligible = sum(1 for v in elig_map.values() if v["eligible"])
            momentum = momentum_by_date.get(d, pd.DataFrame())

            regime = regime_as_of(regime_series, d) if regime_series is not None else "RISK_ON"
            risk_on = (regime == "RISK_ON") if config.use_regime_filter else True

            current_syms = [s for s, q in shares.items() if q > 0]
            if risk_on and not momentum.empty:
                selected = _select_target_portfolio(
                    momentum,
                    config.top_fraction,
                    config.single_stock_weight_cap,
                    current_syms,
                    config.rebalance_rank_change_quintiles,
                )
            else:
                selected = []  # defensive / no eligible candidates -> hold cash

            pre_rebalance_value = portfolio_value(today)
            target_weight = (1.0 / len(selected)) if selected else 0.0

            # SELL everything not in the new target (at today's open).
            opens_today = open_panel.loc[today]
            turnover_value = 0.0
            for sym in list(shares.keys()):
                if sym not in selected and shares[sym] > 0:
                    px = opens_today.get(sym, np.nan)
                    if pd.notna(px):
                        proceeds = shares[sym] * float(px)
                        cash += proceeds
                        turnover_value += proceeds
                        shares[sym] = 0

            # BUY/adjust toward equal target weight for selected names.
            if selected:
                target_value_per_name = pre_rebalance_value * target_weight
                for sym in selected:
                    px = opens_today.get(sym, np.nan)
                    if pd.isna(px) or px <= 0:
                        continue  # could not fill -- documented limitation
                    held_value = shares.get(sym, 0) * float(px)
                    delta_value = target_value_per_name - held_value
                    if delta_value > 0:
                        buy_shares = int(delta_value // float(px))
                        buy_shares = min(buy_shares, int(cash // float(px)))
                        if buy_shares > 0:
                            cost = buy_shares * float(px)
                            cash -= cost
                            shares[sym] = shares.get(sym, 0) + buy_shares
                            turnover_value += cost
                    elif delta_value < 0:
                        sell_shares = min(
                            shares.get(sym, 0), int((-delta_value) // float(px))
                        )
                        if sell_shares > 0:
                            proceeds = sell_shares * float(px)
                            cash += proceeds
                            shares[sym] -= sell_shares
                            turnover_value += proceeds

            shares = {s: q for s, q in shares.items() if q > 0}
            post_value = portfolio_value(today)
            rebalance_log.append(
                RebalanceRecord(
                    decision_date=d,
                    execution_date=today,
                    regime=regime,
                    n_eligible=n_eligible,
                    n_selected=len(selected),
                    selected=selected,
                    target_weights={s: target_weight for s in selected},
                    turnover=turnover_value / post_value if post_value > 0 else 0.0,
                )
            )

        val = portfolio_value(today)
        equity_curve.append(val)
        invested = val - cash
        daily_exposure.append(invested / val if val > 0 else 0.0)

    equity = pd.Series(equity_curve, index=calendar, name="equity")
    exposure = pd.Series(daily_exposure, index=calendar, name="exposure")
    return {
        "equity_curve": equity,
        "exposure": exposure,
        "rebalances": rebalance_log,
        "final_cash": cash,
        "final_shares": shares,
    }
