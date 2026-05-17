"""Tests for graph loader."""

import pytest
from scanner.graph.loader import (
    get_dependents,
    get_parents,
    get_edge_weight,
    get_all_tickers,
    MEGA_CAPS,
    _load_edges,
)


@pytest.fixture(autouse=True)
def clear_edge_cache():
    _load_edges.cache_clear()
    yield
    _load_edges.cache_clear()


def test_mega_caps_present_in_universe():
    tickers = get_all_tickers()
    for mc in MEGA_CAPS:
        assert mc in tickers


def test_get_dependents_nvda():
    edges = get_dependents("NVDA")
    children = [e.child for e in edges]
    assert "VRT" in children
    assert "ANET" in children
    assert "SMCI" in children
    for e in edges:
        assert e.parent == "NVDA"
        assert 0 < e.weight <= 1.0


def test_get_dependents_unknown_parent_returns_empty():
    edges = get_dependents("ZZZZZ")
    assert edges == []


def test_get_parents_vrt():
    edges = get_parents("VRT")
    parents = [e.parent for e in edges]
    assert "NVDA" in parents
    assert "MSFT" in parents


def test_get_edge_weight_known():
    w = get_edge_weight("NVDA", "ANET")
    assert w is not None
    assert 0 < w <= 1.0


def test_get_edge_weight_unknown_returns_none():
    assert get_edge_weight("NVDA", "ZZZZZ") is None


def test_all_edges_have_valid_weights():
    from scanner.graph.loader import _load_edges
    for e in _load_edges():
        assert 0 < e.weight <= 1.0, f"Bad weight on edge {e.parent}->{e.child}"


def test_all_edges_have_nonempty_type():
    from scanner.graph.loader import _load_edges
    for e in _load_edges():
        assert e.edge_type, f"Missing type on edge {e.parent}->{e.child}"
