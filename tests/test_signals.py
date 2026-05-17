"""Tests for the relative strength signal."""

import math
import pytest
import duckdb
from datetime import date, timedelta


def _make_conn_with_prices(rows: list[tuple]) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE prices (
            ticker VARCHAR NOT NULL,
            date DATE NOT NULL,
            open DOUBLE, high DOUBLE, low DOUBLE,
            close DOUBLE NOT NULL,
            volume BIGINT,
            adj_close DOUBLE NOT NULL,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.executemany("INSERT INTO prices VALUES (?,?,?,?,?,?,?,?)", rows)
    return conn


def _make_price_rows(ticker: str, start: date, prices: list[float]) -> list[tuple]:
    rows = []
    d = start
    for p in prices:
        rows.append((ticker, str(d), p, p, p, p, 1000, p))
        d += timedelta(days=1)
    return rows


def test_rs_positive_when_child_outperforms():
    from scanner.signals.relative_strength import RelativeStrengthVsParent
    from unittest.mock import patch

    start = date(2024, 1, 2)
    # Parent flat; child trends up 90→111 across the window so oldest≠latest
    parent_prices = [100.0] * 22
    child_prices = [float(90 + i) for i in range(22)]

    rows = _make_price_rows("NVDA", start, parent_prices)
    rows += _make_price_rows("VRT", start, child_prices)
    conn = _make_conn_with_prices(rows)

    with patch("scanner.signals.relative_strength.get_edge_weight", return_value=1.0):
        signal = RelativeStrengthVsParent(parent="NVDA", lookback=20)
        score = signal.compute("VRT", str(start + timedelta(days=21)), conn)

    assert score > 0, f"Expected positive RS score, got {score}"


def test_rs_negative_when_child_underperforms():
    from scanner.signals.relative_strength import RelativeStrengthVsParent
    from unittest.mock import patch

    start = date(2024, 1, 2)
    # Parent trends up 90→111; child flat — child underperforms
    parent_prices = [float(90 + i) for i in range(22)]
    child_prices = [100.0] * 22

    rows = _make_price_rows("NVDA", start, parent_prices)
    rows += _make_price_rows("VRT", start, child_prices)
    conn = _make_conn_with_prices(rows)

    with patch("scanner.signals.relative_strength.get_edge_weight", return_value=1.0):
        signal = RelativeStrengthVsParent(parent="NVDA", lookback=20)
        score = signal.compute("VRT", str(start + timedelta(days=21)), conn)

    assert score < 0, f"Expected negative RS score, got {score}"


def test_rs_returns_nan_on_insufficient_data():
    from scanner.signals.relative_strength import RelativeStrengthVsParent
    from unittest.mock import patch

    start = date(2024, 1, 2)
    rows = _make_price_rows("NVDA", start, [100.0] * 5)
    rows += _make_price_rows("VRT", start, [100.0] * 5)
    conn = _make_conn_with_prices(rows)

    with patch("scanner.signals.relative_strength.get_edge_weight", return_value=1.0):
        signal = RelativeStrengthVsParent(parent="NVDA", lookback=20)
        score = signal.compute("VRT", str(start + timedelta(days=4)), conn)

    assert math.isnan(score)


def test_rs_scaled_by_edge_weight():
    from scanner.signals.relative_strength import RelativeStrengthVsParent
    from unittest.mock import patch

    start = date(2024, 1, 2)
    parent_prices = [100.0] * 22
    child_prices = [100.0] + [110.0] * 21

    rows = _make_price_rows("NVDA", start, parent_prices)
    rows += _make_price_rows("VRT", start, child_prices)
    conn1 = _make_conn_with_prices(rows)
    conn2 = _make_conn_with_prices(rows)

    as_of = str(start + timedelta(days=21))
    signal = RelativeStrengthVsParent(parent="NVDA", lookback=20)

    with patch("scanner.signals.relative_strength.get_edge_weight", return_value=1.0):
        score_full = signal.compute("VRT", as_of, conn1)
    with patch("scanner.signals.relative_strength.get_edge_weight", return_value=0.5):
        score_half = signal.compute("VRT", as_of, conn2)

    assert abs(score_half - score_full * 0.5) < 1e-9
