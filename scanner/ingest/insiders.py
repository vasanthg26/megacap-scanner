"""Form 4 insider transaction ingest via SEC EDGAR REST API.

CRITICAL DESIGN NOTE — Offering Participation False Positives
=============================================================
Secondary aggregators (e.g. OpenInsider, some data vendors) often classify
offering-participation purchases as "insider buys", creating strong false-positive
signals. A CEO buying in a secondary offering looks identical to an open-market buy
in raw Form 4 data but has zero predictive value (price is set by bankers, not market).

Filter applied:
  1. Transaction code must be 'P' (open-market purchase).
  2. Check footnote text (case-insensitive) for _OFFERING_KEYWORDS.
     Match → store with is_open_market=False (excluded from buy signals).
  3. Check footnote text for _PROSPECTUS_PATTERNS (S-1, S-3, 424B).
     Match (no offering keyword) → route to insider_filter_review for human review.

Implementation note: spec called for sec-edgar-downloader but that package is not
installed. Using SEC EDGAR REST API directly (data.sec.gov / www.sec.gov/Archives).
"""

import logging
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import requests
import yaml

from scanner.db import get_connection

logger = logging.getLogger(__name__)

_RATE_SLEEP = 0.11  # 9 req/s — SEC policy requires <=10 req/s

_SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"
_PLACEHOLDER_SUBSTRINGS = (
    "your name",
    "your.email@example.com",
    "your_email@example.com",
    "example@example.com",
    "name@example.com",
    "changeme",
    "placeholder",
)

_EDGAR_BASE = "https://data.sec.gov"
_EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
_CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

_OFFERING_KEYWORDS = frozenset([
    "underwritten offering",
    "public offering",
    "registered direct",
    "follow-on offering",
    "secondary offering",
    "private placement",
    "pipe",
    # ESPP purchases are plan-driven at predetermined prices, not discretionary
    "employee stock purchase plan",
    "espp",
])

_PROSPECTUS_PATTERNS = ("s-1", "s-3", "424b")


def _is_xsl_path(document: str) -> bool:
    """Return True if document is an XSL-rendered path rather than raw XML."""
    lo = document.lower()
    return lo.startswith("xsl") or "/xsl" in lo


def _get_user_agent() -> str:
    import os
    value = os.environ.get("SEC_USER_AGENT", "")
    if not value and _SETTINGS_PATH.exists():
        with _SETTINGS_PATH.open() as f:
            settings = yaml.safe_load(f) or {}
        value = settings.get("SEC_USER_AGENT", "")
    if not value or not value.strip():
        raise ValueError(
            "SEC_USER_AGENT is not set. "
            "Set it as an environment variable or in config/settings.yaml. "
            "Example: 'AppName/Version (contact: real-email@domain.com)'"
        )
    lower = value.lower()
    if any(ph in lower for ph in _PLACEHOLDER_SUBSTRINGS):
        raise ValueError(
            f"SEC_USER_AGENT still contains placeholder text: {value!r}. "
            "Update it with your real application name and contact email per SEC EDGAR requirements. "
            "Example: 'AppName/Version (contact: real-email@domain.com)'"
        )
    return value


def _make_session() -> requests.Session:
    user_agent = _get_user_agent()
    sess = requests.Session()
    sess.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    return sess


def _load_cik_map(session: requests.Session) -> dict[str, int]:
    """Fetch ticker→CIK map from EDGAR company_tickers.json."""
    resp = session.get(_CIK_MAP_URL, timeout=30)
    resp.raise_for_status()
    time.sleep(_RATE_SLEEP)
    return {v["ticker"].upper(): int(v["cik_str"]) for v in resp.json().values()}


def _extract_form4_from_batch(batch: dict, start_date: date) -> list[dict]:
    forms = batch.get("form", [])
    dates = batch.get("filingDate", [])
    accessions = batch.get("accessionNumber", [])
    documents = batch.get("primaryDocument", [])
    results = []
    for form, filing_date, accession, document in zip(forms, dates, accessions, documents):
        if form not in ("4", "4/A"):
            continue
        if filing_date < str(start_date):
            continue
        if not document:
            continue
        results.append({
            "accession_number": accession,
            "filing_date": filing_date,
            "primary_document": document,
        })
    return results


def _list_form4_filings(cik: int, start_date: date, session: requests.Session) -> list[dict]:
    """Return Form 4 / 4/A filings for cik filed on or after start_date."""
    url = f"{_EDGAR_BASE}/submissions/CIK{cik:010d}.json"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    time.sleep(_RATE_SLEEP)
    data = resp.json()

    filings_block = data.get("filings", {})
    recent = filings_block.get("recent", {})
    results = _extract_form4_from_batch(recent, start_date)

    # Paginate older batches only if recent doesn't fully cover the lookback window.
    # recent is sorted newest-first; if oldest date is still >= start_date, there may be more.
    recent_dates = recent.get("filingDate", [])
    if recent_dates and recent_dates[-1] >= str(start_date):
        for batch_meta in filings_block.get("files", []):
            batch_url = f"{_EDGAR_BASE}/submissions/{batch_meta['name']}"
            batch_resp = session.get(batch_url, timeout=30)
            batch_resp.raise_for_status()
            time.sleep(_RATE_SLEEP)
            batch_data = batch_resp.json()
            batch_results = _extract_form4_from_batch(batch_data, start_date)
            results.extend(batch_results)
            batch_dates = batch_data.get("filingDate", [])
            if batch_dates and batch_dates[-1] < str(start_date):
                break

    return results


def _resolve_raw_xml(cik: int, accession: str, document: str, session: requests.Session) -> str:
    """Return the raw XML document name for a Form 4 filing.

    EDGAR submissions.json sometimes lists XSL-rendered paths such as
    'xslF345X06/form4.xml'. Fetching those URLs returns HTML, not XML.
    When an XSL path is detected, fetch the filing's index.json to locate
    the real .xml file. Falls back to stripping the leading xsl directory
    if the index fetch fails.
    """
    if not _is_xsl_path(document):
        return document

    accession_path = accession.replace("-", "")
    index_url = f"{_EDGAR_ARCHIVES}/{cik}/{accession_path}/index.json"
    try:
        resp = session.get(index_url, timeout=30)
        resp.raise_for_status()
        time.sleep(_RATE_SLEEP)
        items = resp.json().get("directory", {}).get("item", [])
        for item in items:
            name = item.get("name", "")
            if name.lower().endswith(".xml") and "/" not in name:
                return name
    except Exception:
        logger.warning("Could not fetch filing index for %s/%s; falling back to path strip", cik, accession)

    return document.rsplit("/", 1)[-1] if "/" in document else document


def _fetch_xml(cik: int, accession: str, document: str, session: requests.Session) -> tuple[str, str]:
    """Fetch Form 4 XML from EDGAR archives. Returns (xml_text, filing_url)."""
    raw_doc = _resolve_raw_xml(cik, accession, document, session)
    accession_path = accession.replace("-", "")
    url = f"{_EDGAR_ARCHIVES}/{cik}/{accession_path}/{raw_doc}"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    time.sleep(_RATE_SLEEP)
    return resp.text, url


def _parse_float(el) -> Optional[float]:
    if el is None or not el.text:
        return None
    try:
        return float(el.text.strip())
    except ValueError:
        return None


def _parse_bool(el) -> bool:
    if el is None or not el.text:
        return False
    return el.text.strip() in ("1", "true", "True")


def check_offering_keywords(footnote_text: str) -> bool:
    """Return True if footnote contains an offering-participation keyword."""
    lower = footnote_text.lower()
    return any(kw in lower for kw in _OFFERING_KEYWORDS)


def check_prospectus_reference(footnote_text: str) -> bool:
    """Return True if footnote references a prospectus filing (S-1, S-3, 424B)."""
    lower = footnote_text.lower()
    return any(pat in lower for pat in _PROSPECTUS_PATTERNS)


def parse_form4_xml(
    xml_text: str,
    accession_number: str,
    ticker: str,
    filed_date: str,
    filing_url: str,
) -> tuple[list[dict], list[dict]]:
    """Parse Form 4 XML into (transactions, review_items).

    P + offering keyword  → transactions with is_open_market=False.
    P + prospectus ref    → review_items only (excluded from transactions).
    P + clean             → transactions with is_open_market=True.
    S                     → transactions with is_open_market=None.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.error("XML parse error for %s %s: %s", ticker, filing_url, exc)
        return [], []

    footnotes: dict[str, str] = {}
    for fn in root.findall(".//footnotes/footnote"):
        fn_id = fn.get("id", "")
        if fn_id:
            footnotes[fn_id] = (fn.text or "").strip()

    owner_el = root.find(".//reportingOwner")
    insider_name = insider_title = None
    is_officer = is_director = is_ten_pct = False
    if owner_el is not None:
        name_el = owner_el.find(".//rptOwnerName")
        if name_el is not None and name_el.text:
            insider_name = name_el.text.strip()
        rel_el = owner_el.find(".//reportingOwnerRelationship")
        if rel_el is not None:
            title_el = rel_el.find("officerTitle")
            if title_el is not None and title_el.text:
                insider_title = title_el.text.strip()
            is_officer = _parse_bool(rel_el.find("isOfficer"))
            is_director = _parse_bool(rel_el.find("isDirector"))
            is_ten_pct = _parse_bool(rel_el.find("isTenPercentOwner"))

    transactions: list[dict] = []
    review_items: list[dict] = []

    for seq, txn_el in enumerate(root.findall(".//nonDerivativeTransaction")):
        code_el = txn_el.find(".//transactionCode")
        code = (code_el.text or "").strip() if code_el is not None else ""
        if code not in ("P", "S"):
            continue

        ref_ids = [el.get("id", "") for el in txn_el.findall(".//footnoteId")]
        fn_text = " ".join(footnotes.get(fid, "") for fid in ref_ids if fid).strip()

        date_el = txn_el.find(".//transactionDate/value")
        shares_el = txn_el.find(".//transactionShares/value")
        price_el = txn_el.find(".//transactionPricePerShare/value")
        owned_el = txn_el.find(".//sharesOwnedFollowingTransaction/value")

        txn_date_str = (date_el.text or "").strip() if date_el is not None else ""
        try:
            txn_date: Optional[date] = date.fromisoformat(txn_date_str) if txn_date_str else None
        except ValueError:
            txn_date = None

        shares = _parse_float(shares_el)
        price = _parse_float(price_el)
        shares_after = _parse_float(owned_el)
        total_value = shares * price if (shares is not None and price is not None) else None

        if code == "P":
            if check_offering_keywords(fn_text):
                is_open_market: Optional[bool] = False
            elif check_prospectus_reference(fn_text):
                review_items.append({
                    "accession_number": accession_number,
                    "transaction_seq": seq,
                    "ticker": ticker,
                    "filed_date": filed_date,
                    "reason": "prospectus_reference",
                    "footnote_text": fn_text or None,
                    "filing_url": filing_url,
                })
                continue
            else:
                is_open_market = True
        else:
            is_open_market = None

        transactions.append({
            "accession_number": accession_number,
            "transaction_seq": seq,
            "ticker": ticker,
            "filed_date": filed_date,
            "transaction_date": txn_date,
            "insider_name": insider_name,
            "insider_title": insider_title,
            "is_officer": is_officer,
            "is_director": is_director,
            "is_ten_percent_owner": is_ten_pct,
            "transaction_code": code,
            "shares": shares,
            "price_per_share": price,
            "total_value": total_value,
            "shares_owned_after": shares_after,
            "is_open_market": is_open_market,
            "footnote_text": fn_text or None,
            "filing_url": filing_url,
        })

    return transactions, review_items


def ingest_form4(
    ticker: str,
    lookback_days: int = 365,
    conn=None,
    cik_map: Optional[dict[str, int]] = None,
    session: Optional[requests.Session] = None,
) -> int:
    """Ingest Form 4 filings for ticker. Returns rows written to insider_transactions.

    Pass conn/cik_map/session for batch callers to reuse across tickers.
    """
    _own_conn = conn is None
    _own_session = session is None
    if _own_conn:
        conn = get_connection()
    if _own_session:
        session = _make_session()

    try:
        if cik_map is None:
            cik_map = _load_cik_map(session)

        cik = cik_map.get(ticker.upper())
        if cik is None:
            logger.warning("%s: no CIK found — skipping (may be foreign private issuer)", ticker)
            return 0

        start_date = date.today() - timedelta(days=lookback_days)
        filings = _list_form4_filings(cik, start_date, session)
        logger.info("%s: %d Form 4 filings since %s", ticker, len(filings), start_date)

        rows_written = 0
        for filing in filings:
            accession = filing["accession_number"]
            filed_date = filing["filing_date"]
            document = filing["primary_document"]

            try:
                xml_text, filing_url = _fetch_xml(cik, accession, document, session)
            except requests.HTTPError as exc:
                logger.error("%s: fetch failed for %s: %s", ticker, accession, exc)
                continue

            transactions, review_items = parse_form4_xml(
                xml_text, accession, ticker, filed_date, filing_url
            )

            for txn in transactions:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO insider_transactions (
                        accession_number, transaction_seq, ticker, filed_date,
                        transaction_date, insider_name, insider_title,
                        is_officer, is_director, is_ten_percent_owner,
                        transaction_code, shares, price_per_share, total_value,
                        shares_owned_after, is_open_market, footnote_text, filing_url
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        txn["accession_number"], txn["transaction_seq"], txn["ticker"],
                        txn["filed_date"], txn["transaction_date"], txn["insider_name"],
                        txn["insider_title"], txn["is_officer"], txn["is_director"],
                        txn["is_ten_percent_owner"], txn["transaction_code"],
                        txn["shares"], txn["price_per_share"], txn["total_value"],
                        txn["shares_owned_after"], txn["is_open_market"],
                        txn["footnote_text"], txn["filing_url"],
                    ],
                )
                rows_written += 1

            for rev in review_items:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO insider_filter_review (
                        accession_number, transaction_seq, ticker, filed_date,
                        reason, footnote_text, filing_url
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        rev["accession_number"], rev["transaction_seq"], rev["ticker"],
                        rev["filed_date"], rev["reason"], rev["footnote_text"], rev["filing_url"],
                    ],
                )

        return rows_written

    finally:
        if _own_conn:
            conn.close()
        if _own_session:
            session.close()
