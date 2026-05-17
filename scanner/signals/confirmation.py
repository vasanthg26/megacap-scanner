"""Cross-signal confirmation score.

Two separate paths:
  - compute_confirmation_live: 0-5, includes Est Rev (for scan display)
  - compute_confirmation_backtest: 0-4, excludes Est Rev (for walk-forward backtest
    until 20+ daily Est Rev snapshots are available)
"""

import math


_RVOL_THRESHOLD = 1.5


def compute_confirmation_megacap(
    rs_20: float,
    rvol: float,
    rev_score: float | None,
    insider_buying: bool,
    trend_raw: int,
) -> int:
    """Live score 0-5 for a mega-cap ticker.

    +1  RS-20 positive vs QQQ
    +1  RVOL >= 1.5x
    +1  Est Rev score > 0 (NULL = 0 contribution)
    +1  Insider buying (any open-market buy in prior 30d)
    +1  Trend structure 2/2 (price above both 50d and 200d MA)
    """
    score = 0
    if not math.isnan(rs_20) and rs_20 > 0:
        score += 1
    if not math.isnan(rvol) and rvol >= _RVOL_THRESHOLD:
        score += 1
    if rev_score is not None and not math.isnan(rev_score) and rev_score > 0:
        score += 1
    if insider_buying:
        score += 1
    if trend_raw >= 2:
        score += 1
    return score


def compute_confirmation_dependent(
    rs_score: float,
    rvol: float,
    rev_score: float | None,
    insider_buying: bool,
    above_50ma: bool,
) -> int:
    """Live score 0-5 for a dependent ticker.

    +1  RS score positive
    +1  RVOL >= 1.5x
    +1  Est Rev score > 0 (NULL = 0 contribution)
    +1  Insider buying (any open-market buy in prior 30d)
    +1  Price above 50d MA
    """
    score = 0
    if not math.isnan(rs_score) and rs_score > 0:
        score += 1
    if not math.isnan(rvol) and rvol >= _RVOL_THRESHOLD:
        score += 1
    if rev_score is not None and not math.isnan(rev_score) and rev_score > 0:
        score += 1
    if insider_buying:
        score += 1
    if above_50ma:
        score += 1
    return score


def compute_confirmation_backtest(
    rs_positive: bool,
    rvol: float,
    insider_buying: bool,
    above_50ma: bool,
) -> int:
    """Backtest score 0-4 (Est Rev excluded until 20+ daily snapshots available).

    +1  RS positive (20d return > 0 for mega-caps; score > 0 for dependents)
    +1  RVOL >= 1.5x
    +1  Insider buying (filed_date <= T, transaction_date >= T - 30d)
    +1  Price above 50d MA
    """
    score = 0
    if rs_positive:
        score += 1
    if not math.isnan(rvol) and rvol >= _RVOL_THRESHOLD:
        score += 1
    if insider_buying:
        score += 1
    if above_50ma:
        score += 1
    return score
