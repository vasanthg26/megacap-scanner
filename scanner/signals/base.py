"""Signal protocol definition, action classification, and compositing utilities."""

import math
from enum import StrEnum
from typing import Protocol, Sequence
import duckdb


class Action(StrEnum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


def classify_action(score: float, universe_scores: Sequence[float]) -> Action:
    """
    Convert a raw score to a discrete action via cross-sectional quintile rank.

    Percentile is computed as the fraction of universe scores strictly below
    `score` (i.e. scipy-style 'strict' rank), so ties always round down.
    NaN scores in universe_scores are ignored; a NaN `score` returns HOLD.

    Q5 (pct >= 0.80) → STRONG_BUY
    Q4 (0.60–0.80)   → BUY
    Q2-Q3 (0.20–0.60) → HOLD
    Q1 (< 0.20)      → SELL
    """
    if math.isnan(score):
        return Action.HOLD
    valid = [s for s in universe_scores if not math.isnan(s)]
    if not valid:
        return Action.HOLD
    pct = sum(s < score for s in valid) / len(valid)
    if pct >= 0.80:
        return Action.STRONG_BUY
    elif pct >= 0.60:
        return Action.BUY
    elif pct >= 0.20:
        return Action.HOLD
    else:
        return Action.SELL


def score_rank(score: float, universe_scores: Sequence[float]) -> tuple[int, int]:
    """
    Return (ordinal_rank, total_n) where rank 1 = highest score.
    NaN scores in universe_scores are excluded from the count.
    """
    valid = [s for s in universe_scores if not math.isnan(s)]
    if not valid:
        return (1, 1)
    rank = sum(s > score for s in valid) + 1
    return (rank, len(valid))


TRADEABLE_REGIMES: frozenset[str] = frozenset({"CORRECTION", "MILD_PULLBACK"})

# Parents with demonstrated positive backtest IC (CORRECTION + MILD_PULLBACK regimes only).
# All other mega-caps are unvalidated — scan shows WAIT labels for their dependents.
VALIDATED_PARENTS: frozenset[str] = frozenset({"MSFT", "META"})


def classify_regime(parent_ret_20d: float) -> str:
    """Classify a parent's 20-day return into a regime label for signal gating.

    CORRECTION and MILD_PULLBACK are the tradeable regimes (positive backtest IC).
    UP and DRAWDOWN suppress action labels — signal has no demonstrated edge there.
    """
    if math.isnan(parent_ret_20d):
        return "UNKNOWN"
    if parent_ret_20d > 0:
        return "UP"
    if parent_ret_20d >= -0.05:
        return "MILD_PULLBACK"
    if parent_ret_20d >= -0.15:
        return "CORRECTION"
    return "DRAWDOWN"


class Signal(Protocol):
    name: str

    def compute(self, ticker: str, date: str, conn: duckdb.DuckDBPyConnection) -> float:
        """
        Return a scalar score for ticker as of date.
        MUST only use data available strictly before or on date (no lookahead).
        Returns float('nan') when insufficient data.
        """
        ...


def weighted_sum(
    signals: list[tuple[Signal, float]],
    ticker: str,
    date: str,
    conn: duckdb.DuckDBPyConnection,
) -> float:
    """Compute a weighted composite score from multiple signals."""
    import math

    total_weight = 0.0
    total_score = 0.0
    for signal, weight in signals:
        score = signal.compute(ticker, date, conn)
        if not math.isnan(score):
            total_score += score * weight
            total_weight += weight

    return total_score / total_weight if total_weight > 0 else float("nan")
