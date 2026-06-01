"""8-K and S-3 filing ingest via Massive API.

Replaces the previous SEC EDGAR HTML scraping. The Massive API returns
plain text directly in the response — no MarkItDown conversion needed.

SEC_USER_AGENT is no longer required for filings (still needed for insiders).
"""

import logging
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import requests
import yaml

from scanner.db import get_connection
from scanner.graph.loader import get_all_tickers

logger = logging.getLogger(__name__)

_RATE_SLEEP = 1.0  # conservative; adjust if plan allows higher throughput
_MASSIVE_BASE = "https://api.massive.com"
_SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"

# Items considered material enough to ingest
MATERIAL_ITEMS = {
    "1.01": "Material Agreement",
    "1.02": "Agreement Terminated",
    "2.01": "Asset Acquisition/Disposal",
    "2.02": "Results of Operations",
    "2.05": "Employee Departure",
    "5.02": "Executive Change",
    "7.01": "Regulation FD",
    "8.01": "Other Material Event",
}

# Auto-impact classification constants
_HIGH_ITEMS = {"2.02", "1.01", "2.01"}
_MEDIUM_ITEMS = {"7.01", "8.01", "1.02", "2.05"}

_BOOST_KEYWORDS = [
    "billion", "acquisition", "merger", "resign", "terminate",
    "CEO", "CFO", "supply agreement", "partnership",
]
_DOWNGRADE_KEYWORDS = [
    "amendment", "correction", "technical", "administrative", "exhibit",
]

_DOLLAR_PATTERN = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|B|M)\b", re.IGNORECASE)

_IMPACT_RANK = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
_IMPACT_FROM_RANK = {2: "HIGH", 1: "MEDIUM", 0: "LOW"}


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


def _get_anthropic_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    if _SETTINGS_PATH.exists():
        with _SETTINGS_PATH.open() as f:
            settings = yaml.safe_load(f) or {}
        key = (settings.get("anthropic_api_key") or "").strip()
        if key and key != "YOUR_ANTHROPIC_API_KEY_HERE":
            return key
    logger.warning("No Anthropic API key found — 8-K summaries will be disabled")
    return None


_haiku_client = None
_sonnet_client = None


def _get_haiku_client():
    global _haiku_client
    if _haiku_client is None:
        import anthropic
        key = _get_anthropic_key()
        if not key:
            return None
        _haiku_client = anthropic.Anthropic(api_key=key, timeout=10.0)
    return _haiku_client


def _get_sonnet_client():
    global _sonnet_client
    if _sonnet_client is None:
        import anthropic
        key = _get_anthropic_key()
        if not key:
            return None
        _sonnet_client = anthropic.Anthropic(api_key=key, timeout=10.0)
    return _sonnet_client


def generate_filing_analysis(
    item_labels: str, title: str | None, filing_text: str | None, ticker: str = ""
) -> tuple[str | None, str | None, str | None]:
    """Two-model cascade: Haiku extracts facts, Sonnet returns JSON {summary, sentiment, impact}.

    Returns (summary, sentiment, impact_explanation). All may be None on failure.
    Sentiment is one of: POSITIVE, NEGATIVE, NEUTRAL.
    """
    if not filing_text:
        return None, None, None

    haiku = _get_haiku_client()
    sonnet = _get_sonnet_client()
    if not haiku or not sonnet:
        return None, None, None

    text_cap = filing_text[:5000]
    try:
        haiku_prompt = (
            f"Extract the key facts from this SEC 8-K filing as 3-4 bullet points. "
            f"Items: {item_labels}. Title: {title or 'N/A'}.\n\nFiling text:\n{text_cap}"
        )
        h_msg = haiku.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": haiku_prompt}],
        )
        facts = h_msg.content[0].text.strip() if h_msg.content else ""
        if not facts:
            logger.warning("generate_filing_analysis [%s]: Haiku returned empty facts", ticker)
    except Exception as exc:
        logger.error("generate_filing_analysis [%s]: Haiku failed: %s: %s", ticker, type(exc).__name__, exc)
        return None, None, None

    try:
        company = ticker.upper() if ticker else "the company"
        sonnet_prompt = (
            f"Analyze this SEC 8-K filing event. Respond in JSON only, no preamble, no markdown.\n\n"
            f"Company: {company}\n"
            f"Filing type: {item_labels}\n"
            f"Key facts extracted:\n{facts}\n\n"
            f'{{\n'
            f'  "summary": "1-2 sentences: what happened, include {company} and specific details",\n'
            f'  "sentiment": "POSITIVE or NEGATIVE or NEUTRAL",\n'
            f'  "impact": "1-2 sentences specific to {company} and this event — not generic. '
            f'Explain the actual business impact for investors."\n'
            f'}}'
        )
        s_msg = sonnet.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": sonnet_prompt}],
        )
        raw = s_msg.content[0].text.strip() if s_msg.content else ""
        logger.info("generate_filing_analysis [%s]: Sonnet raw (%d chars): %s", ticker, len(raw), raw[:300])
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            logger.warning("generate_filing_analysis [%s]: no JSON in Sonnet response", ticker)
            return None, None, None
        import json
        data = json.loads(m.group())
        summary = (data.get("summary") or "").strip() or None
        sentiment = (data.get("sentiment") or "").strip().upper() or None
        if sentiment not in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
            logger.warning("generate_filing_analysis [%s]: unexpected sentiment %r", ticker, sentiment)
            sentiment = None
        impact_explanation = (data.get("impact") or "").strip() or None
        return summary, sentiment, impact_explanation
    except Exception as exc:
        logger.error("generate_filing_analysis [%s]: Sonnet failed: %s: %s", ticker, type(exc).__name__, exc)
        return None, None, None


def cleanup_filings(conn) -> dict:
    """Delete old unsaved filings per retention policy. Never deletes saved_to_journal=TRUE rows."""
    low_row = conn.execute(
        """
        SELECT COUNT(*) FROM filings_8k
        WHERE saved_to_journal = FALSE AND impact = 'LOW'
          AND filed_date < CURRENT_DATE - INTERVAL 30 DAY
        """
    ).fetchone()
    med_row = conn.execute(
        """
        SELECT COUNT(*) FROM filings_8k
        WHERE saved_to_journal = FALSE AND impact = 'MEDIUM'
          AND filed_date < CURRENT_DATE - INTERVAL 90 DAY
        """
    ).fetchone()
    high_row = conn.execute(
        """
        SELECT COUNT(*) FROM filings_8k
        WHERE saved_to_journal = FALSE AND impact = 'HIGH'
          AND filed_date < CURRENT_DATE - INTERVAL 365 DAY
        """
    ).fetchone()
    protected_row = conn.execute(
        "SELECT COUNT(*) FROM filings_8k WHERE saved_to_journal = TRUE"
    ).fetchone()

    low_n = low_row[0] if low_row else 0
    med_n = med_row[0] if med_row else 0
    high_n = high_row[0] if high_row else 0
    protected_n = protected_row[0] if protected_row else 0

    conn.execute(
        """
        DELETE FROM filings_8k
        WHERE saved_to_journal = FALSE AND impact = 'LOW'
          AND filed_date < CURRENT_DATE - INTERVAL 30 DAY
        """
    )
    conn.execute(
        """
        DELETE FROM filings_8k
        WHERE saved_to_journal = FALSE AND impact = 'MEDIUM'
          AND filed_date < CURRENT_DATE - INTERVAL 90 DAY
        """
    )
    conn.execute(
        """
        DELETE FROM filings_8k
        WHERE saved_to_journal = FALSE AND impact = 'HIGH'
          AND filed_date < CURRENT_DATE - INTERVAL 365 DAY
        """
    )

    logger.info(
        "Filings cleanup: LOW removed %d / MEDIUM removed %d / HIGH removed %d / Protected %d",
        low_n, med_n, high_n, protected_n,
    )
    return {"low_removed": low_n, "medium_removed": med_n, "high_removed": high_n, "protected": protected_n}


def _classify_impact(item_numbers: str, title: str, description: str, form_type: str) -> str:
    """Auto-classify impact as HIGH, MEDIUM, or LOW."""
    text = f"{title or ''} {description or ''}".lower()
    items = [i.strip() for i in (item_numbers or "").split(",") if i.strip()]

    if form_type.upper().endswith("/A"):
        rank = _IMPACT_RANK["LOW"]
    elif any(i in _HIGH_ITEMS for i in items):
        rank = _IMPACT_RANK["HIGH"]
        if "5.02" in items and any(kw in text for kw in ("ceo", "cfo", "chief executive", "chief financial")):
            rank = _IMPACT_RANK["HIGH"]
    elif any(i in _MEDIUM_ITEMS for i in items):
        rank = _IMPACT_RANK["MEDIUM"]
    else:
        rank = _IMPACT_RANK["LOW"]

    for m in _DOLLAR_PATTERN.finditer(text):
        amount_str = m.group(1).replace(",", "")
        unit = m.group(2).lower()
        amount = float(amount_str)
        if unit in ("billion", "b"):
            amount_in_m = amount * 1000
        else:
            amount_in_m = amount
        if amount_in_m >= 1000:
            rank = max(rank, _IMPACT_RANK["HIGH"])
        elif amount_in_m >= 100:
            rank = max(rank, _IMPACT_RANK["MEDIUM"])

    boost = any(kw.lower() in text for kw in _BOOST_KEYWORDS)
    downgrade = any(kw.lower() in text for kw in _DOWNGRADE_KEYWORDS)
    if boost and not downgrade:
        rank = min(rank + 1, _IMPACT_RANK["HIGH"])
    elif downgrade and not boost:
        rank = max(rank - 1, _IMPACT_RANK["LOW"])

    return _IMPACT_FROM_RANK[rank]


def _extract_material_items(items_raw) -> list[str]:
    """Given a raw 'items' value (str or list), return material item codes."""
    if not items_raw:
        return []
    if isinstance(items_raw, list):
        candidates = [str(i).strip() for i in items_raw]
    else:
        candidates = [i.strip() for i in str(items_raw).split(",")]
    return [c for c in candidates if c in MATERIAL_ITEMS]


def _fetch_filings_massive(ticker: str, cutoff: date, api_key: str) -> list[dict]:
    """Fetch 8-K and S-3 filings for a single ticker via Massive API since `cutoff`."""
    results = []

    # 8-K filings — text included directly in response
    try:
        url = f"{_MASSIVE_BASE}/stocks/filings/8-K/v1/text"
        resp = requests.get(
            url,
            params={"ticker": ticker, "limit": 10, "apiKey": api_key},
            timeout=30,
        )
        time.sleep(_RATE_SLEEP)
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("results", []):
            filed_str = (item.get("filed_at") or "")[:10]
            try:
                filed = date.fromisoformat(filed_str)
            except (ValueError, TypeError):
                continue
            if filed < cutoff:
                continue

            items_raw = item.get("items", [])
            material = _extract_material_items(items_raw)
            if not material:
                continue

            item_numbers = ",".join(material)
            filing_text = item.get("text") or ""
            impact = _classify_impact(item_numbers, None, filing_text[:500], "8-K")

            results.append({
                "ticker": ticker,
                "filed_date": filed,
                "accession_number": item.get("accession_number", ""),
                "form_type": "8-K",
                "item_numbers": item_numbers,
                "title": None,
                "description": None,
                "filing_url": None,
                "impact": impact,
                "impact_source": "auto",
                "filing_text": filing_text or None,
            })
    except Exception as exc:
        logger.warning("%s: Massive 8-K fetch failed: %s", ticker, exc)

    # S-3 shelf registrations — fetched separately from the filings index
    try:
        url = f"{_MASSIVE_BASE}/stocks/filings/v1/index"
        resp = requests.get(
            url,
            params={"ticker": ticker, "form_type": "S-3", "apiKey": api_key},
            timeout=30,
        )
        time.sleep(_RATE_SLEEP)
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("results", []):
            filed_str = (item.get("filed_at") or "")[:10]
            try:
                filed = date.fromisoformat(filed_str)
            except (ValueError, TypeError):
                continue
            if filed < cutoff:
                continue

            results.append({
                "ticker": ticker,
                "filed_date": filed,
                "accession_number": item.get("accession_number", ""),
                "form_type": item.get("form_type", "S-3"),
                "item_numbers": "S-3: Shelf Registration",
                "title": None,
                "description": None,
                "filing_url": None,
                "impact": "HIGH",
                "impact_source": "auto",
                "summary": (
                    f"{ticker} filed Form S-3 shelf registration. "
                    "This signals potential upcoming capital raise which may be dilutive to existing shareholders."
                ),
                "sentiment": "NEGATIVE",
                "impact_explanation": None,
                "filing_text": None,
            })
    except Exception as exc:
        logger.warning("%s: Massive S-3 fetch failed: %s", ticker, exc)

    return results


def ingest_filings(
    tickers: list[str] | None = None,
    days_back: int = 7,
    conn=None,
) -> dict[str, str]:
    """
    Fetch 8-K filings for universe tickers (or subset) into DuckDB.
    Returns status dict: ticker -> 'ok' | 'skipped' | 'error'.
    Idempotent: accession_number UNIQUE constraint prevents duplicates.
    """
    api_key = _get_massive_key()
    if not api_key:
        logger.warning("MASSIVE_API_KEY not set — skipping filings ingest")
        return {t: "skipped" for t in (tickers or get_all_tickers())}

    universe = tickers or get_all_tickers()
    cutoff = date.today() - timedelta(days=days_back)
    ingested_at = datetime.now(timezone.utc)
    results: dict[str, str] = {}

    close_conn = conn is None
    if conn is None:
        conn = get_connection()

    anthropic_key = _get_anthropic_key()
    summaries_enabled = anthropic_key is not None

    try:
        for ticker in universe:
            try:
                filings = _fetch_filings_massive(ticker, cutoff, api_key)
                written = 0
                for f in filings:
                    existing = conn.execute(
                        "SELECT 1 FROM filings_8k WHERE accession_number = ?",
                        [f["accession_number"]],
                    ).fetchone()
                    if existing:
                        continue

                    row = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM filings_8k").fetchone()
                    next_id = row[0]

                    summary = f.get("summary")
                    sentiment = f.get("sentiment")
                    impact_explanation = f.get("impact_explanation")

                    if summaries_enabled and f["form_type"] not in ("S-3", "S-3/A"):
                        filing_text = f.get("filing_text")
                        if filing_text:
                            item_labels = ", ".join(
                                MATERIAL_ITEMS.get(i.strip(), i.strip())
                                for i in (f["item_numbers"] or "").split(",")
                                if i.strip()
                            )
                            summary, sentiment, impact_explanation = generate_filing_analysis(
                                item_labels, f["title"], filing_text, ticker
                            )
                            time.sleep(0.5)

                    try:
                        conn.execute(
                            """
                            INSERT INTO filings_8k
                                (id, ticker, filed_date, accession_number, form_type,
                                 item_numbers, title, description, filing_url,
                                 saved_to_journal, ingested_at, impact, impact_source,
                                 summary, sentiment, impact_explanation)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            [
                                next_id,
                                f["ticker"],
                                f["filed_date"],
                                f["accession_number"],
                                f["form_type"],
                                f["item_numbers"],
                                f["title"],
                                f["description"],
                                f["filing_url"],
                                False,
                                ingested_at,
                                f["impact"],
                                f["impact_source"],
                                summary,
                                sentiment,
                                impact_explanation,
                            ],
                        )
                        written += 1
                    except Exception as exc:
                        if "UNIQUE constraint" in str(exc) or "unique" in str(exc).lower():
                            pass
                        else:
                            logger.error("%s: insert error for %s: %s", ticker, f["accession_number"], exc)

                logger.info("%-6s  %d filings written (since %s)", ticker, written, cutoff)
                results[ticker] = "ok"
            except Exception as exc:
                logger.error("%-6s  filings ingest error: %s", ticker, exc)
                results[ticker] = "error"

        cleanup_filings(conn)
    finally:
        if close_conn:
            conn.close()

    return results
