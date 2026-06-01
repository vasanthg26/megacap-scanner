"""Short interest ingest from Massive API."""

import logging
import os
import time
from pathlib import Path

import requests
import yaml

from scanner.db import get_connection
from scanner.graph.loader import get_all_tickers

logger = logging.getLogger(__name__)

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


def ingest_short_interest(tickers: list[str] | None = None) -> dict[str, str]:
    """Fetch short interest data for all universe tickers into DuckDB.

    Returns status dict: ticker -> 'ok' | 'empty' | 'error' | 'no_key'.
    """
    api_key = _get_massive_key()
    if not api_key:
        logger.warning("MASSIVE_API_KEY not set — skipping short interest ingest")
        return {}

    universe = tickers or list(dict.fromkeys(get_all_tickers()))
    results: dict[str, str] = {}

    with get_connection() as conn:
        for ticker in universe:
            try:
                url = f"{_MASSIVE_BASE}/stocks/v1/short-interest"
                resp = requests.get(
                    url,
                    params={"ticker": ticker, "apiKey": api_key},
                    timeout=30,
                )
                time.sleep(_RATE_SLEEP)
                resp.raise_for_status()
                data = resp.json()
                items = data.get("results", [])
                if not items:
                    results[ticker] = "empty"
                    logger.info("%-6s  short interest: empty", ticker)
                    continue

                rows_written = 0
                for item in items:
                    settlement_date = item.get("settlement_date")
                    if not settlement_date:
                        continue
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO short_interest
                            (ticker, settlement_date, short_interest, days_to_cover, short_percent_of_float)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        [
                            ticker,
                            settlement_date,
                            item.get("short_interest"),
                            item.get("days_to_cover"),
                            item.get("short_percent_of_float"),
                        ],
                    )
                    rows_written += 1

                results[ticker] = "ok"
                logger.info("%-6s  short interest: %d rows", ticker, rows_written)

            except Exception as exc:
                logger.error("%-6s  short interest error: %s", ticker, exc)
                results[ticker] = "error"

    return results
