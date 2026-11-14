"""
Cross-sectional momentum portfolio engine for BVMT AI Trader.

Implements the "First implementation task" from
BVMT_AI_Trader_Profitability_Roadmap.md, Section 19:
a standalone, long-only, cross-sectional 3M/6M momentum benchmark,
with and without the TUNINDEX SMA(200) regime filter, executed at
next-session open on an equal-weight basis with cash as a valid
position. This module does not touch or duplicate the existing
single-stock `src.models.benchmark_runner` / `src.models.ml_predictor`
pipeline; it is an independent, portfolio-level system per the
roadmap's Section 3 architecture.
"""
