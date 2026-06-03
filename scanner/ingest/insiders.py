"""Form 4 insider transaction ingest — Massive API primary, EDGAR XML for P-transaction footnotes.

Architecture
============
Massive /stocks/filings/vX/form-4?issuer_cik={cik} lists all Form 4s for a ticker.
Filings that contain at least one P (purchase) transaction → fetch raw EDGAR XML to
extract footnotes and run the offering-participation filter (unchanged logic).
Filings with only S (sale) or non-P transactions → insert directly from Massive data,
no XML fetch needed, cutting EDGAR call volume significantly.

CRITICAL DESIGN NOTE — Offering Participation False Positives
=============================================================
Secondary aggregators (e.g. OpenInsider, some data vendors) often classify
offering-participation purchases as "insider buys", creating strong false-positive
signals. A CEO buying in a secondary offering looks identical to an open-market buy
in raw Form 4 data but has zero predictive value (price is set by bankers, not market).

Filter applied (P transactions only, via EDGAR XML):
  1. Transaction code must be 'P' (open-market purchase).
  2. Check footnote text (case-insensitive) for _OFFERING_KEYWORDS.
     Match → store with is_open_market=False (excluded from buy signals).
  3. Check footnote text for _PROSPECTUS_PATTERNS (S-1, S-3, 424B).
     Match (no offering keyword) → route to insider_filter_review for human review.

CIK caching: ticker_cik_map table, populated from EDGAR company_tickers.json.
SEC_USER_AGENT: required only for EDGAR XML fetches; Massive calls use MASSIVE_API_KEY.
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

_EDGAR_RATE_SLEEP = 0.15   # EDGAR: ≤10 req/s; using 0.15 for margin
_MASSIVE_RATE_SLEEP = 6.0  # Massive starter: ≤10 req/min

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
_MASSIVE_BASE = "https://api.massive.com"

_OFFERING_KEYWORDS = frozenset([
    "underwritten offering",
    "public offering",
    "registered direct",
    "follow-on offering",
    "secondary offering",
    "private placement",
    "pipe",
    "employee stock purchase plan",
    "espp",
])

_PROSPECTUS_PATTERNS = ("s-1", "s-3", "424b")


# ---------- config helpers -------------------------------------------------------

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
            "Update it with your real application name and contact email."
        )
    return value


def _get_massive_key() -> Optional[str]:
    import os
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


def _make_edgar_session() -> requests.Session:
    user_agent = _get_user_agent()
    sess = requests.Session()
    sess.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    return sess


# ---------- CIK lookup + caching -------------------------------------------------

def _load_cik_map(session: requests.Session) -> dict[str, str]:
    """Fetch ticker→CIK map from EDGAR company_tickers.json. Returns zero-padded CIK strings."""
    resp = session.get(_CIK_MAP_URL, timeout=30)
    resp.raise_for_status()
    time.sleep(_EDGAR_RATE_SLEEP)
    return {v["ticker"].upper(): f"{int(v['cik_str']):010d}" for v in resp.json().values()}


def _get_cik_for_ticker(
    ticker: str,
    conn,
    edgar_session: requests.Session,
    cik_map_cache: Optional[dict[str, str]] = None,
) -> Optional[str]:
    """Return zero-padded CIK string for ticker, using DB cache then EDGAR fallback."""
    row = conn.execute(
        "SELECT cik FROM ticker_cik_map WHERE ticker = ?", [ticker.upper()]
    ).fetchone()
    if row:
        return row[0]

    if cik_map_cache is None:
        cik_map_cache = _load_cik_map(edgar_session)

    cik = cik_map_cache.get(ticker.upper())
    if cik is None:
        return None

    conn.execute(
        """
        INSERT OR REPLACE INTO ticker_cik_map (ticker, cik, updated_date)
        VALUES (?, ?, CURRENT_DATE)
        """,
        [ticker.upper(), cik],
    )
    return cik


# ---------- Massive Form 4 listing -----------------------------------------------

def _list_form4_massive(
    issuer_cik: str,
    start_date: date,
    api_key: str,
) -> list[dict]:
    """Return all non-derivative P/S Form 4 transactions for issuer_cik since start_date.

    One dict per transaction row, grouped so caller can identify which accessions
    contain P transactions. Keys: accession_number, filing_date, transaction_code,
    transaction_date, owner_name, officer_title, is_officer, is_director,
    is_ten_percent_owner, transaction_shares, transaction_price_per_share,
    shares_owned_after, filing_url.
    """
    results: list[dict] = []
    url: Optional[str] = (
        f"{_MASSIVE_BASE}/stocks/filings/vX/form-4"
        f"?issuer_cik={issuer_cik}"
        f"&filing_date.gte={start_date.isoformat()}"
        f"&limit=100"
        f"&apiKey={api_key}"
    )

    while url:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Massive Form 4 fetch failed for CIK %s: %s", issuer_cik, _redact_key(exc))
            break

        data = resp.json()
        if data.get("status") != "OK":
            logger.error("Massive Form 4 error for CIK %s: %s", issuer_cik, data.get("error"))
            break

        for r in data.get("results", []):
            code = r.get("transaction_code") or ""
            record_type = r.get("record_type") or ""
            security_type = r.get("security_type") or ""

            if record_type != "transaction":
                continue
            if security_type != "non_derivative":
                continue
            if code not in ("P", "S"):
                continue

            results.append({
                "accession_number": r.get("accession_number", ""),
                "filing_date": r.get("filing_date", ""),
                "transaction_code": code,
                "transaction_date": r.get("transaction_date"),
                "owner_name": r.get("owner_name"),
                "officer_title": r.get("officer_title"),
                "is_officer": bool(r.get("is_officer", False)),
                "is_director": bool(r.get("is_director", False)),
                "is_ten_percent_owner": bool(r.get("is_ten_percent_owner", False)),
                "transaction_shares": r.get("transaction_shares"),
                "transaction_price_per_share": r.get("transaction_price_per_share"),
                "shares_owned_after": r.get("shares_owned_following_transaction"),
                "filing_url": r.get("filing_url", ""),
            })

        next_url = data.get("next_url")
        url = f"{next_url}&apiKey={api_key}" if next_url else None
        time.sleep(_MASSIVE_RATE_SLEEP)

    return results


# ---------- EDGAR XML fetch + parse (P-transaction accessions) -------------------

def _is_xsl_path(document: str) -> bool:
    lo = document.lower()
    return lo.startswith("xsl") or "/xsl" in lo


def _resolve_raw_xml(cik: str, accession: str, document: str, session: requests.Session) -> str:
    if not _is_xsl_path(document):
        return document

    accession_path = accession.replace("-", "")
    index_url = f"{_EDGAR_ARCHIVES}/{int(cik)}/{accession_path}/index.json"
    try:
        resp = session.get(index_url, timeout=30)
        resp.raise_for_status()
        time.sleep(_EDGAR_RATE_SLEEP)
        items = resp.json().get("directory", {}).get("item", [])
        for item in items:
            name = item.get("name", "")
            if name.lower().endswith(".xml") and "/" not in name:
                return name
    except Exception:
        logger.warning("Could not fetch filing index for %s/%s", cik, accession)

    return document.rsplit("/", 1)[-1] if "/" in document else document


def _fetch_xml(cik: str, accession: str, document: str, session: requests.Session) -> tuple[str, str]:
    """Fetch Form 4 XML from EDGAR archives. Returns (xml_text, filing_url)."""
    # Need to find the primary document name from the filing index
    accession_path = accession.replace("-", "")
    index_url = f"{_EDGAR_ARCHIVES}/{int(cik)}/{accession_path}/index.json"
    try:
        resp = session.get(index_url, timeout=30)
        resp.raise_for_status()
        time.sleep(_EDGAR_RATE_SLEEP)
        items = resp.json().get("directory", {}).get("item", [])
        raw_doc = None
        for item in items:
            name = item.get("name", "")
            if name.lower().endswith(".xml") and "/" not in name and not _is_xsl_path(name):
                raw_doc = name
                break
        if raw_doc is None:
            # Fall back to .txt filing
            raw_doc = f"{accession}.txt"
    except Exception:
        raw_doc = f"{accession}.txt"

    url = f"{_EDGAR_ARCHIVES}/{int(cik)}/{accession_path}/{raw_doc}"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    time.sleep(_EDGAR_RATE_SLEEP)
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


# ---------- DB write helpers -----------------------------------------------------

def _upsert_transaction(conn, txn: dict) -> None:
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


def _upsert_review(conn, rev: dict) -> None:
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


# ---------- main ingest ----------------------------------------------------------

def ingest_form4(
    ticker: str,
    lookback_days: int = 365,
    conn=None,
    cik_map_cache: Optional[dict[str, str]] = None,
    edgar_session: Optional[requests.Session] = None,
) -> int:
    """Ingest Form 4 filings for ticker. Returns rows written to insider_transactions.

    Pass conn/cik_map_cache/edgar_session for batch callers to reuse across tickers.
    """
    _own_conn = conn is None
    _own_session = edgar_session is None
    if _own_conn:
        conn = get_connection()
    if _own_session:
        edgar_session = _make_edgar_session()

    try:
        cik = _get_cik_for_ticker(ticker, conn, edgar_session, cik_map_cache)
        if cik is None:
            logger.warning("%s: no CIK found — skipping (may be foreign private issuer)", ticker)
            return 0

        massive_key = _get_massive_key()
        start_date = date.today() - timedelta(days=lookback_days)

        if massive_key:
            return _ingest_hybrid(ticker, cik, start_date, conn, edgar_session, massive_key)
        else:
            logger.warning(
                "%s: MASSIVE_API_KEY not set — falling back to full EDGAR ingest", ticker
            )
            return _ingest_edgar_only(ticker, cik, start_date, conn, edgar_session)

    finally:
        if _own_conn:
            conn.close()
        if _own_session:
            edgar_session.close()


def _ingest_hybrid(
    ticker: str,
    cik: str,
    start_date: date,
    conn,
    edgar_session: requests.Session,
    massive_key: str,
) -> int:
    """Hybrid path: Massive lists filings, EDGAR XML only for accessions with P transactions."""
    massive_rows = _list_form4_massive(cik, start_date, massive_key)
    logger.info(
        "%s: Massive returned %d non-derivative P/S transaction rows since %s",
        ticker, len(massive_rows), start_date,
    )

    # Group rows by accession_number
    from collections import defaultdict
    by_accession: dict[str, list[dict]] = defaultdict(list)
    for row in massive_rows:
        by_accession[row["accession_number"]].append(row)

    rows_written = 0

    for accession, rows in by_accession.items():
        filed_date = rows[0]["filing_date"]
        has_purchase = any(r["transaction_code"] == "P" for r in rows)

        if has_purchase:
            # Fetch XML from EDGAR to get footnotes and run offering-participation filter
            try:
                xml_text, filing_url = _fetch_xml(cik, accession, "", edgar_session)
            except requests.HTTPError as exc:
                logger.error("%s: EDGAR XML fetch failed for %s: %s", ticker, accession, exc)
                continue

            transactions, review_items = parse_form4_xml(
                xml_text, accession, ticker, filed_date, filing_url
            )
            for txn in transactions:
                _upsert_transaction(conn, txn)
                rows_written += 1
            for rev in review_items:
                _upsert_review(conn, rev)

        else:
            # S-only accession — insert directly from Massive data, no XML needed
            for seq, row in enumerate(rows):
                txn_date_str = row.get("transaction_date") or ""
                try:
                    txn_date: Optional[date] = (
                        date.fromisoformat(txn_date_str) if txn_date_str else None
                    )
                except ValueError:
                    txn_date = None

                shares = row.get("transaction_shares")
                price = row.get("transaction_price_per_share")
                total_value = (
                    shares * price
                    if (shares is not None and price is not None)
                    else None
                )

                txn = {
                    "accession_number": accession,
                    "transaction_seq": seq,
                    "ticker": ticker,
                    "filed_date": filed_date,
                    "transaction_date": txn_date,
                    "insider_name": row.get("owner_name"),
                    "insider_title": row.get("officer_title"),
                    "is_officer": row.get("is_officer", False),
                    "is_director": row.get("is_director", False),
                    "is_ten_percent_owner": row.get("is_ten_percent_owner", False),
                    "transaction_code": row["transaction_code"],
                    "shares": shares,
                    "price_per_share": price,
                    "total_value": total_value,
                    "shares_owned_after": row.get("shares_owned_after"),
                    "is_open_market": None,
                    "footnote_text": None,
                    "filing_url": row.get("filing_url", ""),
                }
                _upsert_transaction(conn, txn)
                rows_written += 1

    return rows_written


def _ingest_edgar_only(
    ticker: str,
    cik: str,
    start_date: date,
    conn,
    edgar_session: requests.Session,
) -> int:
    """EDGAR-only fallback: list filings from submissions.json, fetch every XML."""
    filings = _list_form4_edgar(cik, start_date, edgar_session)
    logger.info("%s: %d Form 4 filings since %s (EDGAR fallback)", ticker, len(filings), start_date)

    rows_written = 0
    for filing in filings:
        accession = filing["accession_number"]
        filed_date = filing["filing_date"]
        document = filing["primary_document"]

        try:
            xml_text, filing_url = _fetch_xml_by_doc(
                cik, accession, document, edgar_session
            )
        except requests.HTTPError as exc:
            logger.error("%s: fetch failed for %s: %s", ticker, accession, _redact_key(exc))
            continue

        transactions, review_items = parse_form4_xml(
            xml_text, accession, ticker, filed_date, filing_url
        )
        for txn in transactions:
            _upsert_transaction(conn, txn)
            rows_written += 1
        for rev in review_items:
            _upsert_review(conn, rev)

    return rows_written


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


def _list_form4_edgar(cik: str, start_date: date, session: requests.Session) -> list[dict]:
    """List Form 4 filings from EDGAR submissions.json (EDGAR-only fallback)."""
    url = f"{_EDGAR_BASE}/submissions/CIK{cik}.json"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    time.sleep(_EDGAR_RATE_SLEEP)
    data = resp.json()

    filings_block = data.get("filings", {})
    recent = filings_block.get("recent", {})
    results = _extract_form4_from_batch(recent, start_date)

    recent_dates = recent.get("filingDate", [])
    if recent_dates and recent_dates[-1] >= str(start_date):
        for batch_meta in filings_block.get("files", []):
            batch_url = f"{_EDGAR_BASE}/submissions/{batch_meta['name']}"
            batch_resp = session.get(batch_url, timeout=30)
            batch_resp.raise_for_status()
            time.sleep(_EDGAR_RATE_SLEEP)
            batch_data = batch_resp.json()
            batch_results = _extract_form4_from_batch(batch_data, start_date)
            results.extend(batch_results)
            batch_dates = batch_data.get("filingDate", [])
            if batch_dates and batch_dates[-1] < str(start_date):
                break

    return results


def _fetch_xml_by_doc(
    cik: str, accession: str, document: str, session: requests.Session
) -> tuple[str, str]:
    """Fetch Form 4 XML from EDGAR by known document name (EDGAR fallback path)."""
    if _is_xsl_path(document):
        accession_path = accession.replace("-", "")
        index_url = f"{_EDGAR_ARCHIVES}/{int(cik)}/{accession_path}/index.json"
        try:
            resp = session.get(index_url, timeout=30)
            resp.raise_for_status()
            time.sleep(_EDGAR_RATE_SLEEP)
            items = resp.json().get("directory", {}).get("item", [])
            for item in items:
                name = item.get("name", "")
                if name.lower().endswith(".xml") and "/" not in name:
                    document = name
                    break
        except Exception:
            logger.warning("Could not fetch index for %s/%s", cik, accession)
        if _is_xsl_path(document):
            document = document.rsplit("/", 1)[-1] if "/" in document else document

    accession_path = accession.replace("-", "")
    url = f"{_EDGAR_ARCHIVES}/{int(cik)}/{accession_path}/{document}"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    time.sleep(_EDGAR_RATE_SLEEP)
    return resp.text, url
