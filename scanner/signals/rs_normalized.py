"""Volatility-normalized relative strength signal."""

import math

import numpy as np
import duckdb

from scanner.graph.loader import get_edge_weight


class RSNormalized:
    """
    Rolling return differential (child - parent), scaled by edge weight,
    then divided by child's 20-day annualised realized vol.

    Prevents high-vol names from dominating the cross-section.
    Returns nan when insufficient data or vol is zero.
    """

    name = "rs_normalized"

    def __init__(self, parent: str, lookback: int = 20):
        self._parent = parent
        self._lookback = lookback

    def _rolling_return(self, ticker: str, date: str, conn: duckdb.DuckDBPyConnection) -> float:
        rows = conn.execute("""
            SELECT adj_close FROM prices
            WHERE ticker = ? AND date <= ?
            ORDER BY date DESC LIMIT ?
        """, [ticker, date, self._lookback + 1]).fetchall()

        if len(rows) < self._lookback + 1:
            return float("nan")
        latest, oldest = rows[0][0], rows[-1][0]
        if oldest == 0:
            return float("nan")
        return (latest - oldest) / oldest

    def _realized_vol(self, ticker: str, date: str, conn: duckdb.DuckDBPyConnection) -> float:
        rows = conn.execute("""
            SELECT adj_close FROM prices
            WHERE ticker = ? AND date <= ?
            ORDER BY date DESC LIMIT ?
        """, [ticker, date, self._lookback + 1]).fetchall()

        if len(rows) < self._lookback + 1:
            return float("nan")
        closes = [r[0] for r in reversed(rows)]
        daily = [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes))
            if closes[i - 1] != 0
        ]
        if len(daily) < 2:
            return float("nan")
        return float(np.std(daily, ddof=1) * np.sqrt(252))

    def compute(self, ticker: str, date: str, conn: duckdb.DuckDBPyConnection) -> float:
        child_ret = self._rolling_return(ticker, date, conn)
        parent_ret = self._rolling_return(self._parent, date, conn)

        if math.isnan(child_ret) or math.isnan(parent_ret):
            return float("nan")

        raw_rs = child_ret - parent_ret
        weight = get_edge_weight(self._parent, ticker)
        if weight is None:
            weight = 1.0

        vol = self._realized_vol(ticker, date, conn)
        if math.isnan(vol) or vol == 0:
            return float("nan")

        return (raw_rs * weight) / vol
