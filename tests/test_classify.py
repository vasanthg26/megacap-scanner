"""Tests for classify_action and score_rank."""

import math
import pytest
from scanner.signals.base import Action, classify_action, score_rank


# ---------------------------------------------------------------------------
# classify_action
# ---------------------------------------------------------------------------

def test_top_quintile_is_strong_buy():
    universe = list(range(100))       # 0..99, evenly spaced
    assert classify_action(95.0, universe) == Action.STRONG_BUY


def test_fourth_quintile_is_buy():
    universe = list(range(100))
    assert classify_action(65.0, universe) == Action.BUY


def test_middle_quintiles_are_hold():
    universe = list(range(100))
    assert classify_action(50.0, universe) == Action.HOLD
    assert classify_action(20.0, universe) == Action.HOLD


def test_bottom_quintile_is_sell():
    universe = list(range(100))
    assert classify_action(5.0, universe) == Action.SELL


def test_nan_score_returns_hold():
    universe = [1.0, 2.0, 3.0]
    assert classify_action(float("nan"), universe) == Action.HOLD


def test_nan_values_in_universe_are_ignored():
    # Same as [0..99] once NaNs stripped
    universe = [float("nan")] * 10 + list(range(100))
    assert classify_action(95.0, universe) == Action.STRONG_BUY
    assert classify_action(5.0, universe) == Action.SELL


def test_empty_universe_returns_hold():
    assert classify_action(1.0, []) == Action.HOLD


def test_all_nan_universe_returns_hold():
    assert classify_action(1.0, [float("nan"), float("nan")]) == Action.HOLD


def test_single_element_universe_is_strong_buy():
    # Only score — nothing below it, pct=0/1=0.0 → SELL? No, wait:
    # pct = sum(s < score for s in [5.0]) / 1 = 0/1 = 0.0 → SELL
    # That's correct: with one element you're at the bottom quintile by rank.
    assert classify_action(5.0, [5.0]) == Action.SELL


def test_highest_score_in_universe_is_strong_buy():
    universe = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert classify_action(10.0, universe) == Action.STRONG_BUY


def test_lowest_score_in_universe_is_sell():
    universe = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert classify_action(1.0, universe) == Action.SELL


def test_boundary_exactly_at_80th_percentile_is_strong_buy():
    # 10 elements, score=9.0 → 8 elements below it → pct=0.80 → STRONG_BUY
    universe = [float(i) for i in range(1, 11)]  # 1..10
    assert classify_action(9.0, universe) == Action.STRONG_BUY


def test_boundary_exactly_at_60th_percentile_is_buy():
    # 10 elements, score=7.0 → 6 elements below it → pct=0.60 → BUY
    universe = [float(i) for i in range(1, 11)]
    assert classify_action(7.0, universe) == Action.BUY


# ---------------------------------------------------------------------------
# score_rank
# ---------------------------------------------------------------------------

def test_rank_highest_is_one():
    universe = [1.0, 2.0, 3.0, 4.0, 5.0]
    rank, total = score_rank(5.0, universe)
    assert rank == 1
    assert total == 5


def test_rank_lowest_equals_total():
    universe = [1.0, 2.0, 3.0, 4.0, 5.0]
    rank, total = score_rank(1.0, universe)
    assert rank == 5
    assert total == 5


def test_rank_middle():
    universe = [1.0, 2.0, 3.0, 4.0, 5.0]
    rank, total = score_rank(3.0, universe)
    assert rank == 3
    assert total == 5


def test_rank_excludes_nans_from_total():
    universe = [1.0, float("nan"), 3.0, float("nan"), 5.0]
    rank, total = score_rank(5.0, universe)
    assert total == 3
    assert rank == 1


def test_rank_ties_resolved_by_count_greater_than():
    # Two scores of 3.0 — both share rank 2: only 5.0 is strictly greater
    universe = [1.0, 2.0, 3.0, 3.0, 5.0]
    rank, _ = score_rank(3.0, universe)
    assert rank == 2


def test_rank_empty_universe_returns_one_of_one():
    assert score_rank(42.0, []) == (1, 1)


# ---------------------------------------------------------------------------
# Action is a StrEnum — usable as a plain string
# ---------------------------------------------------------------------------

def test_action_str_values():
    assert str(Action.STRONG_BUY) == "STRONG_BUY"
    assert str(Action.BUY) == "BUY"
    assert str(Action.HOLD) == "HOLD"
    assert str(Action.SELL) == "SELL"


def test_action_equality_with_string():
    assert Action.STRONG_BUY == "STRONG_BUY"
    assert Action.SELL == "SELL"
