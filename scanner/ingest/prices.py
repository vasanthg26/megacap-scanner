"""OHLCV ingest — Massive API primary, yfinance fallback when no key set."""

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml

try:
    import yfinance as yf
except ImportError:
    yf = None  # type: ignore

from scanner.db import get_connection
from scanner.graph.loader import get_all_tickers

logger = logging.getLogger(__name__)


def _redact_key(exc) -> str:
    """Replace apiKey value in exception/URL strings before logging."""
    s = str(exc)
    if "apiKey=" not in s:
        return s
    parts = s.split("apiKey=")
    result = parts[0] + "apiKey=***"
    for part in parts[1:]:
        for delim in ("&", " ", '"', "'", ")"):
            idx = part.find(delim)
            if idx != -1:
                result += part[idx:]
                break
    return result

_LOOKBACK_DAYS = 730  # 2 years
_INCREMENTAL_DAYS = 30
_RATE_SLEEP = 6.0  # Massive starter plan: ≤10 req/min
_MASSIVE_BASE = "https://api.massive.com"
_SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"

# Theme basket tickers + benchmark ETFs + SPY (not in get_all_tickers()).
_THEME_TICKERS: list[str] = [
    "BB", "NOK", "AEHR", "HUT", "FPS", "BE", "YSS", "NVTS",  # theme tickers
    "DOCN", "NET", "AKAM", "DDOG",                             # Developer Platforms & Edge Compute
    "IGV", "IGN", "SMH", "WGMI", "GRID", "XLI", "ITA",       # benchmark ETFs
    "WCLD",                                                     # WisdomTree Cloud Computing ETF
    "SPY",                                                      # needed for theme_active computation
]


def _get_massive_key() -> str | None:
    key = os.environ.get("MASSIVE_API_KEY", "").strip()
    if key:
        return key
    if _SETTINGS_PATH.exists():
        with _SETTINGS_PATH.open() as f:
            settings = yaml.safe_load(f) or {}
        key = (settings.get("MASSIVE_API_KEY") or "").strip()
        if key and key != "YOUR_MASSIVE_API_KEY_HERE":
            return key
    return None


def _fetch_ohlcv_massive(ticker: str, from_date: str, to_date: str, api_key: str) -> pd.DataFrame:
    """Fetch OHLCV from Massive API (Polygon-compatible bars endpoint).

    adjusted=true means the close price is split-adjusted; we use it as adj_close.
    """
    url = f"{_MASSIVE_BASE}/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return pd.DataFrame()
        rows = []
        for r in results:
            d = datetime.fromtimestamp(r["t"] / 1000).date()
            close = r.get("c")
            rows.append({
                "ticker": ticker,
                "date": d,
                "open": r.get("o"),
                "high": r.get("h"),
                "low": r.get("l"),
                "close": close,
                "volume": int(r["v"]) if r.get("v") is not None else None,
                "adj_close": close,  # adjusted=true: close IS the adjusted close
            })
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.error("Massive fetch failed for %s: %s", ticker, _redact_key(exc))
        return pd.DataFrame()


def _fetch_ohlcv_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV from yfinance. Returns empty DataFrame on failure."""
    if yf is None:
        logger.error("yfinance not installed — cannot use fallback")
        return pd.DataFrame()
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end, auto_adjust=False)
        if df.empty:
            return pd.DataFrame()
        df = df[["Open", "High", "Low", "Close", "Volume", "Adj Close"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = "date"
        df.columns = ["open", "high", "low", "close", "volume", "adj_close"]
        df["ticker"] = ticker
        return df.reset_index()
    except Exception as exc:
        logger.error("yfinance fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def _fetch_ohlcv(ticker: str, start: str, end: str, api_key: str | None = None) -> pd.DataFrame:
    """Fetch OHLCV — Massive API if key set, yfinance fallback."""
    if api_key:
        return _fetch_ohlcv_massive(ticker, start, end, api_key)
    return _fetch_ohlcv_yfinance(ticker, start, end)


def _has_existing_prices(ticker: str, conn) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM prices WHERE ticker = ?", [ticker]
    ).fetchone()
    return (row[0] if row else 0) > 0


def ingest_ticker(ticker: str, conn, start: str, end: str, api_key: str | None = None) -> int:
    """Upsert OHLCV rows for one ticker. Returns number of rows written."""
    df = _fetch_ohlcv(ticker, start, end, api_key)
    if df.empty:
        return 0

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[["ticker", "date", "open", "high", "low", "close", "volume", "adj_close"]]
    df = df.dropna(subset=["close", "adj_close"])

    conn.register("_staging", df)
    conn.execute("""
        INSERT OR REPLACE INTO prices
        SELECT ticker, date, open, high, low, close, volume, adj_close
        FROM _staging
    """)
    conn.unregister("_staging")
    return len(df)


def ingest_all(tickers: list[str] | None = None, force_full: bool = False) -> dict[str, str]:
    """
    Pull OHLCV for all universe tickers (or a subset) into DuckDB.
    Returns a status dict: ticker -> 'ok' | 'empty' | 'error'.
    Idempotent: re-running overwrites existing rows.

    When MASSIVE_API_KEY is set:
      - force_full=True or no existing data: fetch full 2-year history
      - force_full=False and data exists: fetch last 30 days (incremental)
    Use --force-full on first Massive run to avoid a data seam with any
    prior yfinance history (which uses a different adjustment basis).

    When no key is set, falls back to yfinance (3-year lookback, no rate limit).
    """
    today = datetime.today()
    full_start = (today - timedelta(days=_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    incremental_start = (today - timedelta(days=_INCREMENTAL_DAYS)).strftime("%Y-%m-%d")
    yf_start = (today - timedelta(days=3 * 365 + 5)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    if tickers:
        universe = tickers
    else:
        base = list(dict.fromkeys(get_all_tickers() + _THEME_TICKERS))
        try:
            _conn = get_connection()
            try:
                discovery_rows = _conn.execute(
                    """
                    SELECT DISTINCT ticker FROM discovery_candidates
                    WHERE status IN ('ACCUMULATING', 'READY_FOR_BACKTEST')
                    """
                ).fetchall()
                discovery_tickers = [r[0] for r in discovery_rows]
            finally:
                _conn.close()
        except Exception:
            discovery_tickers = []
        universe = list(dict.fromkeys(base + discovery_tickers))
    results: dict[str, str] = {}

    api_key = _get_massive_key()
    using_massive = api_key is not None
    if using_massive:
        logger.info("Using Massive API for price ingest (force_full=%s)", force_full)
    else:
        logger.info("MASSIVE_API_KEY not set — using yfinance fallback")

    with get_connection() as conn:
        for ticker in universe:
            fetched_at = datetime.now(timezone.utc)
            try:
                if using_massive:
                    if force_full or not _has_existing_prices(ticker, conn):
                        start = full_start
                    else:
                        start = incremental_start
                else:
                    start = yf_start

                rows = ingest_ticker(ticker, conn, start, end, api_key)
                status = "ok" if rows > 0 else "empty"
                error_msg = None
                logger.info("%-6s  %s  rows=%d", ticker, status, rows)
            except Exception as exc:
                rows = 0
                status = "error"
                error_msg = str(exc)
                logger.error("%-6s  error: %s", ticker, _redact_key(exc))

            conn.execute("""
                INSERT OR REPLACE INTO ingest_log (ticker, fetched_at, rows_written, status, error_msg)
                VALUES (?, ?, ?, ?, ?)
            """, [ticker, fetched_at, rows, status, error_msg])

            results[ticker] = status

            if using_massive:
                time.sleep(_RATE_SLEEP)

    return results
