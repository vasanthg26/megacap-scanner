"""Tests for the ingest pipeline (mocked — no network calls)."""

import pytest
import pandas as pd
import duckdb
from datetime import date, timedelta
from unittest.mock import patch, MagicMock


def _make_mock_df(ticker: str, n_days: int = 30) -> pd.DataFrame:
    start = date(2024, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    return pd.DataFrame({
        "ticker": ticker,
        "date": [str(d) for d in dates],
        "open": [100.0] * n_days,
        "high": [105.0] * n_days,
        "low": [95.0] * n_days,
        "close": [102.0] * n_days,
        "volume": [1_000_000] * n_days,
        "adj_close": [102.0] * n_days,
    })


def _make_ingest_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE prices (
            ticker VARCHAR NOT NULL, date DATE NOT NULL,
            open DOUBLE, high DOUBLE, low DOUBLE,
            close DOUBLE NOT NULL, volume BIGINT, adj_close DOUBLE NOT NULL,
            PRIMARY KEY (ticker, date)
        );
        CREATE TABLE ingest_log (
            ticker VARCHAR NOT NULL, fetched_at TIMESTAMP NOT NULL,
            rows_written INTEGER NOT NULL, status VARCHAR NOT NULL,
            error_msg VARCHAR, PRIMARY KEY (ticker, fetched_at)
        );
    """)
    return conn


def test_ingest_ticker_writes_rows():
    from scanner.ingest.prices import ingest_ticker

    df = _make_mock_df("NVDA", 30)
    conn = _make_ingest_conn()

    with patch("scanner.ingest.prices._fetch_ohlcv", return_value=df):
        rows = ingest_ticker("NVDA", conn, "2024-01-02", "2024-02-01")

    assert rows == 30
    count = conn.execute("SELECT COUNT(*) FROM prices WHERE ticker='NVDA'").fetchone()[0]
    assert count == 30


def test_ingest_ticker_idempotent():
    from scanner.ingest.prices import ingest_ticker

    df = _make_mock_df("NVDA", 30)
    conn = _make_ingest_conn()

    with patch("scanner.ingest.prices._fetch_ohlcv", return_value=df):
        ingest_ticker("NVDA", conn, "2024-01-02", "2024-02-01")
        rows = ingest_ticker("NVDA", conn, "2024-01-02", "2024-02-01")

    assert rows == 30
    count = conn.execute("SELECT COUNT(*) FROM prices WHERE ticker='NVDA'").fetchone()[0]
    assert count == 30


def test_ingest_ticker_returns_zero_on_empty_fetch():
    from scanner.ingest.prices import ingest_ticker

    conn = _make_ingest_conn()
    with patch("scanner.ingest.prices._fetch_ohlcv", return_value=pd.DataFrame()):
        rows = ingest_ticker("BADTICKER", conn, "2024-01-02", "2024-02-01")

    assert rows == 0


def test_fetch_ohlcv_returns_empty_on_yfinance_error():
    from scanner.ingest.prices import _fetch_ohlcv

    mock_ticker = MagicMock()
    mock_ticker.history.side_effect = Exception("network error")

    with patch("scanner.ingest.prices.yf.Ticker", return_value=mock_ticker):
        df = _fetch_ohlcv("NVDA", "2024-01-02", "2024-02-01")

    assert df.empty
