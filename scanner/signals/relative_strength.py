"""Relative strength of a child vs its parent, scaled by edge weight."""

import math
import duckdb

from scanner.graph.loader import get_edge_weight


class RelativeStrengthVsParent:
    """
    Rolling return differential (child - parent) over `lookback` trading days,
    scaled by the edge weight from the dependency graph.

    No lookahead: all prices are fetched as of `date` or earlier.
    Returns nan when fewer than `lookback` rows are available for either ticker.
    """

    name = "rs"

    def __init__(self, parent: str, lookback: int = 20):
        self._parent = parent
        self._lookback = lookback

    def _rolling_return(
        self, ticker: str, date: str, conn: duckdb.DuckDBPyConnection
    ) -> float:
        rows = conn.execute("""
            SELECT adj_close
            FROM prices
            WHERE ticker = ? AND date <= ?
            ORDER BY date DESC
            LIMIT ?
        """, [ticker, date, self._lookback + 1]).fetchall()

        if len(rows) < self._lookback + 1:
            return float("nan")

        latest = rows[0][0]
        oldest = rows[-1][0]
        if oldest == 0:
            return float("nan")
        return (latest - oldest) / oldest

    def compute(self, ticker: str, date: str, conn: duckdb.DuckDBPyConnection) -> float:
        child_ret = self._rolling_return(ticker, date, conn)
        parent_ret = self._rolling_return(self._parent, date, conn)

        if math.isnan(child_ret) or math.isnan(parent_ret):
            return float("nan")

        raw_rs = child_ret - parent_ret
        weight = get_edge_weight(self._parent, ticker)
        if weight is None:
            weight = 1.0

        return raw_rs * weight
