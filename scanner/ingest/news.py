"""News sentiment ingest from Massive API (Polygon-compatible /v2/reference/news)."""

import logging
import os
import time
from pathlib import Path

import requests
import yaml

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

_MASSIVE_BASE = "https://api.massive.com"
_RATE_SLEEP = 1.0
_SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"


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


def ingest_news(tickers: list[str] | None = None) -> dict[str, str]:
    """Fetch recent news sentiment for universe tickers into DuckDB.

    Returns status dict: ticker -> 'ok' | 'empty' | 'error' | 'no_key'.
    """
    api_key = _get_massive_key()
    if not api_key:
        logger.warning("MASSIVE_API_KEY not set — skipping news ingest")
        return {}

    universe = tickers or list(dict.fromkeys(get_all_tickers()))
    results: dict[str, str] = {}

    with get_connection() as conn:
        for ticker in universe:
            try:
                url = f"{_MASSIVE_BASE}/v2/reference/news"
                resp = requests.get(
                    url,
                    params={"ticker": ticker, "limit": 5, "apiKey": api_key},
                    timeout=30,
                )
                time.sleep(_RATE_SLEEP)
                resp.raise_for_status()
                data = resp.json()
                items = data.get("results", [])
                if not items:
                    results[ticker] = "empty"
                    continue

                rows_written = 0
                for item in items:
                    published_utc = item.get("published_utc", "")
                    published_date = published_utc[:10] if published_utc else None
                    if not published_date:
                        continue
                    title = (item.get("title") or "").strip()
                    if not title:
                        continue
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO news_sentiment
                            (ticker, published_date, title, sentiment, article_url)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        [
                            ticker,
                            published_date,
                            title,
                            item.get("sentiment"),
                            item.get("article_url"),
                        ],
                    )
                    rows_written += 1

                results[ticker] = "ok"
                logger.info("%-6s  news: %d articles", ticker, rows_written)

            except Exception as exc:
                logger.error("%-6s  news ingest error: %s", ticker, _redact_key(exc))
                results[ticker] = "error"

    return results


def get_recent_sentiment(ticker: str, conn, days: int = 7) -> list[dict]:
    """Return recent news sentiment items for a ticker, newest first."""
    try:
        rows = conn.execute(
            """
            SELECT title, sentiment, published_date, article_url
            FROM news_sentiment
            WHERE ticker = ?
              AND published_date >= CURRENT_DATE - INTERVAL ? DAY
            ORDER BY published_date DESC
            LIMIT 5
            """,
            [ticker, days],
        ).fetchall()
        return [
            {"title": r[0], "sentiment": r[1], "published_date": str(r[2]), "article_url": r[3]}
            for r in rows
        ]
    except Exception:
        return []
