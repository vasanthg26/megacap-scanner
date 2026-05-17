"""Tests for DuckDB schema and connection."""

import pytest
import duckdb
from unittest.mock import patch
from pathlib import Path
import tempfile


def _make_test_conn() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with the production DDL applied."""
    conn = duckdb.connect(":memory:")
    ddl = """
    CREATE TABLE IF NOT EXISTS prices (
        ticker   VARCHAR NOT NULL,
        date     DATE    NOT NULL,
        open     DOUBLE,
        high     DOUBLE,
        low      DOUBLE,
        close    DOUBLE NOT NULL,
        adj_close DOUBLE NOT NULL,
        volume   BIGINT,
        PRIMARY KEY (ticker, date)
    );
    CREATE TABLE IF NOT EXISTS ingest_log (
        ticker      VARCHAR NOT NULL,
        fetched_at  TIMESTAMP NOT NULL,
        rows_written INTEGER  NOT NULL,
        status      VARCHAR  NOT NULL,
        error_msg   VARCHAR,
        PRIMARY KEY (ticker, fetched_at)
    );
    """
    conn.execute(ddl)
    return conn


def test_prices_table_schema():
    conn = _make_test_conn()
    conn.execute("""
        INSERT INTO prices VALUES ('NVDA', '2024-01-02', 100, 105, 98, 103, 103, 1000000)
    """)
    rows = conn.execute("SELECT ticker, close FROM prices").fetchall()
    assert rows == [("NVDA", 103.0)]


def test_prices_primary_key_prevents_duplicate():
    conn = _make_test_conn()
    conn.execute("""
        INSERT INTO prices VALUES ('NVDA', '2024-01-02', 100, 105, 98, 103, 103, 1000000)
    """)
    with pytest.raises(Exception):
        conn.execute("""
            INSERT INTO prices VALUES ('NVDA', '2024-01-02', 110, 115, 108, 112, 112, 2000000)
        """)


def test_ingest_log_records_status():
    conn = _make_test_conn()
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    conn.execute("""
        INSERT INTO ingest_log VALUES (?, ?, ?, ?, ?)
    """, ["NVDA", now, 756, "ok", None])
    row = conn.execute("SELECT ticker, status FROM ingest_log").fetchone()
    assert row == ("NVDA", "ok")
