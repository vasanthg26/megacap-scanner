"""Dependency graph loader from config/dependencies.yaml."""

from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass

import yaml

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "dependencies.yaml"

MEGA_CAPS = ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "TSM"]


@dataclass(frozen=True)
class Edge:
    parent: str
    child: str
    edge_type: str
    weight: float
    notes: str


@lru_cache(maxsize=1)
def _load_edges() -> list[Edge]:
    with open(_CONFIG_PATH) as f:
        data = yaml.safe_load(f)
    edges = []
    for e in data["edges"]:
        edges.append(Edge(
            parent=e["parent"],
            child=e["child"],
            edge_type=e["type"],
            weight=float(e["weight"]),
            notes=e.get("notes", ""),
        ))
    return edges


def get_all_tickers() -> list[str]:
    edges = _load_edges()
    tickers = set(MEGA_CAPS)
    for e in edges:
        tickers.add(e.parent)
        tickers.add(e.child)
    tickers.update(["QQQ", "XLK"])  # benchmarks required for megacap RS calculations
    return sorted(tickers)


def get_dependents(parent: str) -> list[Edge]:
    """Return all child edges for a given parent mega-cap."""
    return [e for e in _load_edges() if e.parent == parent]


def get_parents(child: str) -> list[Edge]:
    """Return all parent edges for a given child ticker."""
    return [e for e in _load_edges() if e.child == child]


def get_edge_weight(parent: str, child: str) -> float | None:
    """Return the edge weight between parent and child, or None if no edge exists."""
    for e in _load_edges():
        if e.parent == parent and e.child == child:
            return e.weight
    return None


class Graph:
    """Thin wrapper around the loaded edges providing an object-oriented interface."""

    def __init__(self, edges: list[Edge]) -> None:
        self._edges = edges
        self.mega_caps: frozenset[str] = frozenset(MEGA_CAPS)

    def get_dependents(self, parent: str) -> list[Edge]:
        return [e for e in self._edges if e.parent == parent]

    def get_parents(self, child: str) -> list[Edge]:
        return [e for e in self._edges if e.child == child]

    def get_edge_weight(self, parent: str, child: str) -> float | None:
        for e in self._edges:
            if e.parent == parent and e.child == child:
                return e.weight
        return None

    def all_tickers(self) -> list[str]:
        tickers = set(MEGA_CAPS)
        for e in self._edges:
            tickers.add(e.parent)
            tickers.add(e.child)
        return sorted(tickers)


def load_graph() -> Graph:
    return Graph(_load_edges())
