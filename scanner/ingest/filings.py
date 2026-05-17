"""8-K filing ingest via SEC EDGAR REST API.

Fetches 8-K filings for all universe tickers, filters to material items only,
runs auto-impact classification, and writes to the filings_8k DuckDB table.
Idempotent: existing accession_numbers are skipped (UNIQUE constraint).
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

_RATE_SLEEP = 0.11  # SEC policy: <=10 req/s

_EDGAR_BASE = "https://data.sec.gov"
_CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
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

_PLACEHOLDER_SUBSTRINGS = (
    "your name", "your.email@example.com", "your_email@example.com",
    "example@example.com", "name@example.com", "changeme", "placeholder",
)

_IMPACT_RANK = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
_IMPACT_FROM_RANK = {2: "HIGH", 1: "MEDIUM", 0: "LOW"}


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


def _fetch_filing_text(url: str, session: requests.Session) -> str | None:
    """Fetch filing HTML and return stripped plain text (first 3000 chars)."""
    try:
        resp = session.get(url, timeout=15)
        time.sleep(_RATE_SLEEP)
        resp.raise_for_status()
        content = resp.text
        if "<html" in content.lower() or "<body" in content.lower():
            content = re.sub(r"<[^>]+>", " ", content)
            content = re.sub(r"\s+", " ", content).strip()
        return content[:3000] if content else None
    except Exception as exc:
        logger.warning("Failed to fetch filing text from %s: %s", url, exc)
        return None


def _generate_summary(item_labels: str, title: str | None, filing_text: str | None) -> str | None:
    key = _get_anthropic_key()
    if not key:
        return None
    if not filing_text:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key, timeout=10.0)
        prompt = (
            f"Summarize this SEC 8-K filing in one sentence (max 20 words). "
            f"Items: {item_labels}. Title: {title or 'N/A'}.\n\nFiling text:\n{filing_text}"
        )
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip() if msg.content else None
    except Exception as exc:
        logger.warning("Summary generation failed: %s", exc)
        return None


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


def _get_user_agent() -> str:
    import os
    value = os.environ.get("SEC_USER_AGENT", "")
    if not value and _SETTINGS_PATH.exists():
        with _SETTINGS_PATH.open() as f:
            settings = yaml.safe_load(f) or {}
        value = settings.get("SEC_USER_AGENT", "")
    if not value or any(p in value.lower() for p in _PLACEHOLDER_SUBSTRINGS):
        return "MegaCapScanner research@example.com"
    return value


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": _get_user_agent(), "Accept-Encoding": "gzip, deflate"})
    return s


def _load_cik_map(session: requests.Session) -> dict[str, str]:
    """Return {ticker: zero-padded 10-digit CIK} for all SEC-registered companies."""
    resp = session.get(_CIK_MAP_URL, timeout=30)
    time.sleep(_RATE_SLEEP)
    resp.raise_for_status()
    raw = resp.json()
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in raw.values()}


def _classify_impact(item_numbers: str, title: str, description: str, form_type: str) -> str:
    """Auto-classify impact as HIGH, MEDIUM, or LOW."""
    text = f"{title or ''} {description or ''}".lower()
    items = [i.strip() for i in (item_numbers or "").split(",") if i.strip()]

    # Amendment gets LOW baseline
    if form_type.upper().endswith("/A"):
        rank = _IMPACT_RANK["LOW"]
    elif any(i in _HIGH_ITEMS for i in items):
        rank = _IMPACT_RANK["HIGH"]
        # 5.02 CEO/CFO → HIGH
        if "5.02" in items and any(kw in text for kw in ("ceo", "cfo", "chief executive", "chief financial")):
            rank = _IMPACT_RANK["HIGH"]
    elif any(i in _MEDIUM_ITEMS for i in items):
        rank = _IMPACT_RANK["MEDIUM"]
    else:
        rank = _IMPACT_RANK["LOW"]

    # Dollar detection — override rank floor
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

    # Keyword boost / downgrade
    boost = any(kw.lower() in text for kw in _BOOST_KEYWORDS)
    downgrade = any(kw.lower() in text for kw in _DOWNGRADE_KEYWORDS)
    if boost and not downgrade:
        rank = min(rank + 1, _IMPACT_RANK["HIGH"])
    elif downgrade and not boost:
        rank = max(rank - 1, _IMPACT_RANK["LOW"])

    return _IMPACT_FROM_RANK[rank]


def _fetch_filings_for_ticker(
    ticker: str,
    cik: str,
    session: requests.Session,
    cutoff: date,
) -> list[dict]:
    """Fetch 8-K filings for a single ticker since `cutoff` date."""
    url = f"{_EDGAR_BASE}/submissions/CIK{cik}.json"
    try:
        resp = session.get(url, timeout=30)
        time.sleep(_RATE_SLEEP)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            logger.warning("%s: CIK %s not found on EDGAR", ticker, cik)
            return []
        raise

    data = resp.json()
    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return []

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    items_list = recent.get("items", [])
    descriptions = recent.get("primaryDocDescription", [])

    results = []
    for i, (form, filed_str, accession, primary_doc) in enumerate(
        zip(forms, dates, accessions, primary_docs)
    ):
        if form not in ("8-K", "8-K/A"):
            continue
        try:
            filed = date.fromisoformat(filed_str)
        except (ValueError, TypeError):
            continue
        if filed < cutoff:
            continue

        items_raw = items_list[i] if i < len(items_list) else ""
        material = _extract_material_items(items_raw)
        if not material:
            continue

        item_numbers = ",".join(material)
        title = descriptions[i] if i < len(descriptions) else None

        acc_clean = accession.replace("-", "")
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}"
            f"/{acc_clean}/{primary_doc}"
        )

        impact = _classify_impact(item_numbers, title, None, form)

        results.append({
            "ticker": ticker,
            "filed_date": filed,
            "accession_number": accession,
            "form_type": form,
            "item_numbers": item_numbers,
            "title": title,
            "description": None,
            "filing_url": filing_url,
            "impact": impact,
            "impact_source": "auto",
        })

    return results


def _extract_material_items(items_raw) -> list[str]:
    """Given a raw 'items' value (str or list), return material item codes."""
    if not items_raw:
        return []
    if isinstance(items_raw, list):
        candidates = [str(i).strip() for i in items_raw]
    else:
        candidates = [i.strip() for i in str(items_raw).split(",")]

    return [c for c in candidates if c in MATERIAL_ITEMS]


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
    universe = tickers or get_all_tickers()
    cutoff = date.today() - timedelta(days=days_back)
    ingested_at = datetime.now(timezone.utc)
    results: dict[str, str] = {}

    close_conn = conn is None
    if conn is None:
        conn = get_connection()

    anthropic_key = _get_anthropic_key()
    summaries_enabled = anthropic_key is not None

    session = _make_session()
    try:
        logger.info("Loading CIK map from SEC EDGAR…")
        cik_map = _load_cik_map(session)

        for ticker in universe:
            cik = cik_map.get(ticker.upper())
            if not cik:
                logger.warning("%s: no CIK found, skipping", ticker)
                results[ticker] = "skipped"
                continue

            try:
                filings = _fetch_filings_for_ticker(ticker, cik, session, cutoff)
                written = 0
                for f in filings:
                    existing = conn.execute(
                        "SELECT 1 FROM filings_8k WHERE accession_number = ?",
                        [f["accession_number"]],
                    ).fetchone()
                    if existing:
                        continue  # already ingested — skip API call and insert

                    row = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM filings_8k").fetchone()
                    next_id = row[0]

                    summary = None
                    if summaries_enabled:
                        filing_text = _fetch_filing_text(f["filing_url"], session)
                        logger.debug(
                            "Filing text length for %s: %d",
                            ticker,
                            len(filing_text) if filing_text else 0,
                        )
                        if filing_text:
                            item_labels = ", ".join(
                                MATERIAL_ITEMS.get(i.strip(), i.strip())
                                for i in (f["item_numbers"] or "").split(",")
                                if i.strip()
                            )
                            summary = _generate_summary(item_labels, f["title"], filing_text)
                            time.sleep(0.5)

                    try:
                        conn.execute(
                            """
                            INSERT INTO filings_8k
                                (id, ticker, filed_date, accession_number, form_type,
                                 item_numbers, title, description, filing_url,
                                 saved_to_journal, ingested_at, impact, impact_source, summary)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                            ],
                        )
                        written += 1
                    except Exception as exc:
                        if "UNIQUE constraint" in str(exc) or "unique" in str(exc).lower():
                            pass  # already ingested
                        else:
                            logger.error("%s: insert error for %s: %s", ticker, f["accession_number"], exc)
                logger.info("%-6s  %d filings written (since %s)", ticker, written, cutoff)
                results[ticker] = "ok"
            except Exception as exc:
                logger.error("%-6s  filings ingest error: %s", ticker, exc)
                results[ticker] = "error"

        cleanup_filings(conn)
    finally:
        session.close()
        if close_conn:
            conn.close()

    return results
