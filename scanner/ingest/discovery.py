"""Pre-screener discovery pipeline.

Stages
======
1. discover()        — weekly: fetch ETF universe, read each candidate's own 10-K to
                       find >10% revenue concentration in a known parent, apply quality
                       gates, insert passing tickers as ACCUMULATING, backfill RS history.
2. accumulate_rs()   — daily: compute rolling RS score for each ACCUMULATING candidate,
                       advance to READY_FOR_BACKTEST at 60 days
3. run_discovery_backtests() — triggered: IC backtest on rs_accumulation, auto-promote
                               to dependencies.yaml on PASS, mark FAILED otherwise

Discovery source
================
Candidate's own 10-K (business + risk_factors via Massive). SEC-mandated disclosure of
>10% customer concentration means mega-cap dependents are reliably surfaced. Two paths:
  - Explicit: customer name matches a PARENT_ALIASES entry
  - Placeholder: "Customer A" / "one customer" → regex context clues → haiku fallback

Placeholder resolutions are cached permanently in the placeholder_resolutions table —
haiku is called at most once per (ticker, accession, placeholder) tuple.

ETF universe
============
6 ETFs: SMH, SOXX, IGV, XLI, DRIV, GRID
Massive ETF holdings endpoint returns 404 — using hardcoded fallback constituent lists.
~150-300 candidates after dedup and filtering.

Promotion criteria (same evidentiary bar as main signal):
  ic_h1 > 0.05 AND ic_h2 > 0.05 (both halves positive)
"""

import logging
import math
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import requests
import yaml

from scanner.db import get_connection
from scanner.graph.loader import get_all_tickers, MEGA_CAPS, _load_edges

logger = logging.getLogger(__name__)

_MASSIVE_BASE = "https://api.massive.com"
_MASSIVE_RATE_SLEEP = 6.0   # 10 req/min starter plan

_RS_LOOKBACK = 20           # trading days for rolling return
_RS_MIN_DAYS = 60           # days of accumulation before backtest
_IC_THRESHOLD = 0.05
_FAILED_REELIGIBLE_DAYS = 90

# Quality gate thresholds
_MIN_PRICE = 5.0            # USD
_MIN_ADV = 10_000_000.0     # USD (avg daily close × volume)
_MIN_LIST_DAYS = 180        # calendar days since IPO/listing

_SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"
_DEPS_YAML_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "dependencies.yaml"

# ---------- ETF universe ---------------------------------------------------------

KNOWN_PARENTS = [
    "NVDA", "MSFT", "META", "AAPL", "AMZN",
    "GOOGL", "GOOG", "TSLA", "AVGO", "ORCL",
    "TSM",
]

DISCOVERY_ETFS = ["SMH", "SOXX", "IGV", "XLI", "DRIV", "GRID"]

# Hardcoded fallback — Massive ETF holdings endpoint returns 404
ETF_FALLBACK_UNIVERSE: dict[str, list[str]] = {
    "SMH": [
        "AMAT", "LRCX", "KLAC", "MCHP",
        "MPWR", "SWKS", "QCOM", "TXN",
        "MU", "WOLF", "COHR", "AMKR",
        "ONTO", "AZTA", "KLIC", "GFS",
        "ENTG", "CCMP", "ACLS", "AXTI",
    ],
    "SOXX": [
        "NVDA", "AVGO", "TSM", "QCOM",
        "TXN", "MU", "AMAT", "LRCX",
        "KLAC", "MCHP", "MPWR",
        "STM", "WOLF", "COHR", "AMKR",
        "MRVL", "SWKS", "QRVO", "CRUS",
    ],
    "IGV": [
        "CRM", "NOW", "SNOW", "DDOG",
        "NET", "OKTA", "WDAY", "PLTR",
        "ADBE", "CDNS", "SNPS", "ANSS",
        "PTC", "MANH", "PAYC", "PCTY",
        "GWRE", "NCNO", "BAND", "ALTR",
    ],
    "XLI": [
        "GEV", "ETN", "PWR", "FIX",
        "HUBB", "AMETEK", "ROK", "EMR",
        "ITT", "TDY", "FWRD", "CARR",
        "OTIS", "JCI", "GNRC", "AOS",
    ],
    "DRIV": [
        "STM", "APTV", "MGA",
        "RIVN", "LCID", "ALB", "SQM",
        "ALTM", "LTHM", "CHPT", "BLNK",
        "EVGO", "NXPI", "INF", "WOLF",
    ],
    "GRID": [
        "VRT", "ETN", "PWR", "FIX",
        "GEV", "CEG", "VST", "NRG",
        "AES", "CWEN", "ARRY", "NOVA",
        "RUN", "SEDG", "ENPH", "FSLR",
    ],
}

# ---------- extraction constants -------------------------------------------------

CONCENTRATION_PATTERNS = [
    # "NVIDIA accounted for 67% of revenue"
    r'(.{3,60}?)\s+accounted\s+for\s+(?:approximately\s+)?(\d+(?:\.\d+)?)\s*%\s+of\s+(?:our\s+)?(?:total\s+)?(?:net\s+)?revenue',
    # "67% of revenue was from NVIDIA"
    r'(\d+(?:\.\d+)?)\s*%\s+of\s+(?:our\s+)?(?:total\s+)?(?:net\s+)?revenue\s+(?:was\s+)?(?:from|derived\s+from)\s+(.{3,60}?)[,\.]',
    # "NVIDIA represented 67% of net revenue"
    r'(.{3,60}?)\s+represented\s+(?:approximately\s+)?(\d+(?:\.\d+)?)\s*%\s+of\s+(?:our\s+)?(?:total\s+)?(?:net\s+)?revenue',
    # "NVIDIA was responsible for 67%"
    r'(.{3,60}?)\s+was\s+responsible\s+for\s+(?:approximately\s+)?(\d+(?:\.\d+)?)\s*%\s+of\s+(?:our\s+)?(?:total\s+)?revenue',
    # "revenue concentration of 67% from NVIDIA"
    r'revenue\s+concentration\s+of\s+(\d+(?:\.\d+)?)\s*%\s+(?:from|with)\s+(.{3,60}?)[,\.]',
]

PARENT_ALIASES: dict[str, list[str]] = {
    "NVDA": ["NVIDIA", "NVIDIA Corporation", "NVIDIA Corp"],
    "MSFT": ["Microsoft", "Microsoft Corporation", "Azure"],
    "META": ["Meta", "Meta Platforms", "Facebook", "Instagram"],
    "AAPL": ["Apple", "Apple Inc", "App Store"],
    "AMZN": ["Amazon", "Amazon.com", "AWS", "Amazon Web Services"],
    "GOOGL": ["Google", "Alphabet", "Google LLC", "YouTube"],
    "TSLA": ["Tesla", "Tesla Inc", "Tesla Motors"],
    "AVGO": ["Broadcom", "Broadcom Inc", "Broadcom Corporation"],
    "ORCL": ["Oracle", "Oracle Corporation"],
    "TSM": ["TSMC", "Taiwan Semiconductor", "Taiwan Semiconductor Manufacturing"],
}

PLACEHOLDER_PATTERNS = [
    r'^Customer\s+[A-Z]$',
    r'^Customer\s+\d+$',
    r'^(?:a|one)\s+(?:single\s+)?customer$',
    r'^(?:our\s+)?(?:largest|primary|key|significant)\s+customer$',
    r'^(?:one|a)\s+(?:large|major)\s+customer$',
]

CONTEXT_CLUES: dict[str, list[str]] = {
    "NVDA": [
        "GPU", "graphics processing", "data center GPU", "H100", "A100",
        "Blackwell", "Hopper", "NVLink", "CUDA", "accelerated computing",
    ],
    "TSLA": [
        "electric vehicle", "EV", "silicon carbide", "SiC",
        "autopilot", "battery electric", "drivetrain", "powertrain",
    ],
    "AAPL": [
        "iPhone", "App Store", "iOS", "Mac", "AirPods",
        "Apple Watch", "iPad",
    ],
    "AMZN": [
        "AWS", "Amazon Web Services", "Prime", "Alexa",
        "Kindle", "fulfillment center",
    ],
    "MSFT": [
        "Azure", "Office 365", "Teams", "Xbox",
        "Windows", "Dynamics", "GitHub",
    ],
    "META": [
        "Facebook", "Instagram", "WhatsApp", "Oculus",
        "Reality Labs", "metaverse",
    ],
    "GOOGL": [
        "Google Search", "YouTube", "AdWords", "AdSense",
        "Google Cloud", "Android", "Chrome",
    ],
    "AVGO": [
        "custom AI accelerator", "XPU", "networking ASIC",
        "Tomahawk", "Jericho",
    ],
    "ORCL": [
        "Oracle Cloud", "OCI", "Oracle Database", "Oracle ERP",
    ],
    "TSM": [
        "semiconductor fabrication", "wafer fabrication",
        "foundry services", "advanced node",
    ],
}


# ---------- config helpers -------------------------------------------------------

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


# ---------- ETF universe fetch ---------------------------------------------------

def _get_etf_universe(massive_key: str, conn) -> list[str]:
    """Return deduplicated, filtered candidate ticker list from ETF constituents.

    Tries the Massive ETF holdings endpoint first; falls back to hardcoded lists
    when the endpoint is unavailable (confirmed 404 on all Massive endpoints).
    """
    etf_counts: dict[str, int] = {}
    combined: set[str] = set()

    for etf in DISCOVERY_ETFS:
        constituents: list[str] = []

        # Attempt live fetch
        try:
            resp = requests.get(
                f"{_MASSIVE_BASE}/v3/reference/tickers/{etf}/holdings",
                params={"apiKey": massive_key},
                timeout=15,
            )
            time.sleep(_MASSIVE_RATE_SLEEP)
            if resp.status_code == 200:
                data = resp.json()
                holdings = data.get("holdings", data.get("results", []))
                if holdings:
                    constituents = [
                        h.get("ticker") or h.get("symbol") or ""
                        for h in holdings
                        if h.get("ticker") or h.get("symbol")
                    ]
        except Exception as exc:
            logger.warning("ETF %s: holdings fetch failed: %s — using fallback", etf, exc)

        if not constituents:
            constituents = ETF_FALLBACK_UNIVERSE.get(etf, [])

        etf_counts[etf] = len(constituents)
        combined.update(constituents)

    # Remove known parents and empty strings
    combined.discard("")
    for parent in KNOWN_PARENTS:
        combined.discard(parent)

    # Filter already promoted or active in graph
    active_in_graph = set(get_all_tickers()) | set(MEGA_CAPS)
    promoted = {
        r[0]
        for r in conn.execute(
            "SELECT ticker FROM discovery_candidates WHERE status = 'PROMOTED'"
        ).fetchall()
    }
    combined -= active_in_graph
    combined -= promoted

    # Price and ADV pre-filter using prices table (skip slow Massive calls here)
    filtered: list[str] = []
    for ticker in sorted(combined):
        db_result = _price_adv_from_db(ticker, conn)
        if db_result is not None:
            latest_close, avg_dv, oldest_date = db_result
            if latest_close < _MIN_PRICE or avg_dv < _MIN_ADV:
                continue
            try:
                oldest = date.fromisoformat(oldest_date)
                if (date.today() - oldest).days < _MIN_LIST_DAYS:
                    continue
            except ValueError:
                pass
        filtered.append(ticker)

    counts_str = " ".join(f"{etf}:{etf_counts.get(etf, 0)}" for etf in DISCOVERY_ETFS)
    logger.info(
        "ETF universe: %d candidates (%s deduped and filtered: %d)",
        len(combined) + len(active_in_graph & combined) + len(promoted & combined),
        counts_str,
        len(filtered),
    )
    return filtered


# ---------- price / ADV helpers (unchanged) --------------------------------------

def _price_adv_from_db(ticker: str, conn) -> Optional[tuple[float, float, str]]:
    """Return (latest_close, avg_daily_value, oldest_date) from prices table, or None."""
    rows = conn.execute(
        """
        SELECT close, volume, date
        FROM prices
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT 21
        """,
        [ticker],
    ).fetchall()

    if not rows:
        return None

    latest_close = rows[0][0]
    if latest_close is None:
        return None

    dollar_vols = [r[0] * r[1] for r in rows if r[0] is not None and r[1] is not None]
    avg_daily_value = sum(dollar_vols) / len(dollar_vols) if dollar_vols else 0.0

    oldest_date = str(rows[-1][2])
    return latest_close, avg_daily_value, oldest_date


def _price_adv_from_massive(ticker: str, api_key: str) -> Optional[tuple[float, float]]:
    """Fetch last 25 daily bars from Massive to compute price and ADV."""
    to_date = date.today().isoformat()
    from_date = (date.today() - timedelta(days=40)).isoformat()
    url = (
        f"{_MASSIVE_BASE}/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
        f"?adjusted=true&sort=desc&limit=25&apiKey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        time.sleep(_MASSIVE_RATE_SLEEP)
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None
        latest_close = results[0].get("c")
        if latest_close is None:
            return None
        dollar_vols = [r["c"] * r["v"] for r in results if r.get("c") and r.get("v")]
        avg_dv = sum(dollar_vols) / len(dollar_vols) if dollar_vols else 0.0
        return float(latest_close), avg_dv
    except Exception as exc:
        logger.warning("%s: Massive aggs fetch failed: %s", ticker, exc)
        return None


def _list_date_from_massive(ticker: str, api_key: str) -> Optional[str]:
    """Fetch ticker detail from Massive to get list_date. Returns ISO date string or None."""
    url = f"{_MASSIVE_BASE}/v3/reference/tickers/{ticker}?apiKey={api_key}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        time.sleep(_MASSIVE_RATE_SLEEP)
        data = resp.json()
        result = data.get("results", {})
        return result.get("list_date")
    except Exception as exc:
        logger.warning("%s: Massive ticker detail fetch failed: %s", ticker, exc)
        return None


# ---------- 10-K concentration extraction ----------------------------------------

def _classify_customer_name(
    customer_name: str,
) -> Optional[str]:
    """Return KNOWN_PARENT ticker if customer_name matches an alias, else None."""
    name_lower = customer_name.lower().strip()
    for parent, aliases in PARENT_ALIASES.items():
        for alias in aliases:
            if alias.lower() in name_lower or name_lower in alias.lower():
                return parent
    return None


def _is_placeholder(customer_name: str) -> bool:
    """Return True if customer_name looks like an anonymised placeholder."""
    name = customer_name.strip()
    for pattern in PLACEHOLDER_PATTERNS:
        if re.match(pattern, name, re.IGNORECASE):
            return True
    return False


def _extract_concentration(
    ticker: str, massive_key: str, conn
) -> tuple[list[tuple[str, float, str]], int]:
    """Fetch ticker's own 10-K and extract >10% revenue concentration in a known parent.

    Returns (results, placeholder_attempts) where:
      results            — list of (parent_ticker, revenue_pct, resolved_by)
      placeholder_attempts — count of placeholders detected (resolved or not)
    resolved_by is one of: 'explicit', 'regex', 'haiku'.
    """
    sections: dict[str, str] = {}
    filing_year = ""  # derived from filing_date in Massive response — stable across runs

    for section in ("business", "risk_factors"):
        # Skip risk_factors if business came back empty — no point burning an API call
        if section == "risk_factors" and not sections:
            break
        try:
            resp = requests.get(
                f"{_MASSIVE_BASE}/stocks/filings/10-K/vX/sections",
                params={"ticker": ticker, "section": section, "apiKey": massive_key},
                timeout=30,
            )
            resp.raise_for_status()
            time.sleep(_MASSIVE_RATE_SLEEP)
            results = resp.json().get("results", [])
            # Derive a stable cache key from the filing date year (accession_number is
            # never returned by Massive; date-based keys break the cache daily).
            if results and not filing_year:
                fd = results[0].get("filing_date") or results[0].get("filed_at") or ""
                filing_year = f"10k_{str(fd)[:4]}" if fd else ""
            text = ""
            for result in results[:2]:
                candidate = result.get("text", "")
                if len(candidate) >= 100:
                    text = candidate
                    break
            if text:
                sections[section] = text
            else:
                logger.debug("%s: 10-K section '%s' empty or too short — skipping", ticker, section)
        except Exception as exc:
            logger.warning("%s: 10-K section '%s' fetch failed: %s", ticker, section, exc)
            time.sleep(_MASSIVE_RATE_SLEEP)

    if not sections:
        return [], 0  # FIX: was bare `return []` — caused unpack error in discover()

    if not filing_year:
        filing_year = f"10k_{date.today().year}"

    business_text = sections.get("business", "")
    risk_factors_text = sections.get("risk_factors", "")
    combined_text = "\n\n".join(sections.values())

    found: list[tuple[str, float, str]] = []
    seen_parents: set[str] = set()
    placeholder_attempts: list[str] = []  # names attempted through resolution path

    def _process_match(name_str: str, pct_str: str) -> None:
        """Classify one (customer_name, revenue_pct) hit and append to found."""
        try:
            revenue_pct = float(pct_str)
        except ValueError:
            return
        if revenue_pct < 10.0:
            return
        customer_name = re.sub(r'\s+', ' ', name_str).strip()

        parent = _classify_customer_name(customer_name)
        if parent and parent not in seen_parents:
            seen_parents.add(parent)
            logger.info("%s: EXPLICIT %s = %.0f%% of revenue ✅", ticker, parent, revenue_pct)
            found.append((parent, revenue_pct, "explicit"))
            return

        if _is_placeholder(customer_name) and customer_name not in seen_parents:
            placeholder_attempts.append(customer_name)
            resolved = _resolve_placeholder(
                ticker, filing_year, customer_name,
                business_text, risk_factors_text,
                revenue_pct, massive_key, conn,
            )
            r_parent, _conf, resolved_by = resolved
            if r_parent and r_parent not in seen_parents:
                seen_parents.add(r_parent)
                found.append((r_parent, revenue_pct, resolved_by))

    try:
        # Pass 1 — table rows: "| Customer A | 20 % | ..."
        # Table rows don't end with .!? so the sentence splitter misses them entirely.
        # Scan raw combined_text line by line for pipe-delimited rows.
        _TABLE_ROW = re.compile(
            r'^\s*\|\s*([^|]{3,60}?)\s*\|\s*(\d+(?:\.\d+)?)\s*%',
            re.IGNORECASE,
        )
        for line in combined_text.splitlines():
            m = _TABLE_ROW.match(line)
            if m and len(m.groups()) >= 2 and m.group(1) and m.group(2):
                _process_match(m.group(1), m.group(2))

        # Pass 2 — prose sentences
        sentences = re.split(r'(?<=[.!?])\s+', combined_text)
        for sentence in sentences:
            for i, pattern in enumerate(CONCENTRATION_PATTERNS):
                m = re.search(pattern, sentence, re.IGNORECASE)
                if not m:
                    continue
                if len(m.groups()) < 2 or not m.group(1) or not m.group(2):
                    continue
                # Patterns 0,2,3: group1=name, group2=pct; pattern 1,4: group1=pct, group2=name
                if i in (1, 4):
                    pct_str, name_str = m.group(1), m.group(2)
                else:
                    name_str, pct_str = m.group(1), m.group(2)
                _process_match(name_str, pct_str)
                break  # only first matching pattern per sentence

    except Exception as exc:
        logger.error("%s: extraction parsing failed: %s", ticker, exc)

    return found, len(placeholder_attempts)


# ---------- placeholder resolution -----------------------------------------------

def _resolve_placeholder(
    ticker: str,
    filing_year: str,
    placeholder: str,
    business_text: str,
    risk_factors_text: str,
    revenue_pct: float,
    massive_key: str,
    conn,
) -> tuple[Optional[str], float, str]:
    """Resolve a placeholder customer name to a known parent ticker.

    filing_year is the stable cache key (e.g. '10k_2026') derived from the
    Massive filing_date field — avoids daily key churn of date.today()-based keys.

    Returns (parent, confidence, resolved_by) — parent is None if unresolved.
    resolved_by: 'cache', 'regex', 'haiku', or 'unresolved'.
    """
    logger.info("%s: %s = %.0f%% of revenue — resolving...", ticker, placeholder, revenue_pct)

    # Step 1 — check cache (only trust resolved entries; skip stale 'unresolved' rows
    # so a broken prior run with no API key doesn't permanently block haiku)
    cached = conn.execute(
        """
        SELECT resolved_parent, confidence, resolved_by
        FROM placeholder_resolutions
        WHERE ticker = ? AND accession_number = ? AND placeholder = ?
          AND resolved_by != 'unresolved'
        """,
        [ticker, filing_year, placeholder],
    ).fetchone()
    if cached:
        parent, conf, resolved_by = cached
        logger.info("  → Cache hit: %s (conf: %.2f, via %s)", parent, conf or 0.0, resolved_by)
        return parent, conf or 0.0, resolved_by or "unresolved"

    # Step 2 — regex context clue resolution
    all_text = f"{business_text}\n\n{risk_factors_text}"
    sentences = re.split(r'(?<=[.!?])\s+', all_text)
    context_sentences: list[str] = []
    for idx, s in enumerate(sentences):
        if placeholder.lower() in s.lower():
            start = max(0, idx - 1)
            end = min(len(sentences), idx + 4)
            context_sentences = sentences[start:end]
            break
    context_window = " ".join(context_sentences).lower()

    clue_hits: dict[str, int] = {}
    for parent, clues in CONTEXT_CLUES.items():
        hits = sum(1 for clue in clues if clue.lower() in context_window)
        if hits > 0:
            clue_hits[parent] = hits

    best_parent = max(clue_hits, key=lambda p: clue_hits[p]) if clue_hits else None
    if best_parent and clue_hits[best_parent] >= 2:
        logger.info(
            "  → Regex: '%s' (%d clue hits) → %s",
            placeholder, clue_hits[best_parent], best_parent,
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO placeholder_resolutions
                (ticker, accession_number, placeholder, resolved_parent,
                 confidence, resolved_by, created_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [ticker, filing_year, placeholder, best_parent, 0.85, "regex"],
        )
        return best_parent, 0.85, "regex"

    # Step 3 — Haiku fallback
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        if _SETTINGS_PATH.exists():
            with _SETTINGS_PATH.open() as f:
                settings = yaml.safe_load(f) or {}
            # settings.yaml may store the key as lowercase 'anthropic_api_key'
            api_key = (
                settings.get("ANTHROPIC_API_KEY")
                or settings.get("anthropic_api_key")
                or ""
            ).strip()

    if not api_key:
        # Don't cache — next run with a working key should retry
        logger.warning("  → No ANTHROPIC_API_KEY — placeholder unresolved (not cached)")
        return None, 0.0, "unresolved"

    # Race condition guard — check cache again before calling haiku
    cached = conn.execute(
        """
        SELECT resolved_parent, confidence, resolved_by
        FROM placeholder_resolutions
        WHERE ticker = ? AND accession_number = ? AND placeholder = ?
          AND resolved_by != 'unresolved'
        """,
        [ticker, filing_year, placeholder],
    ).fetchone()
    if cached:
        parent, conf, resolved_by = cached
        return parent, conf or 0.0, resolved_by or "unresolved"

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Format known companies as "TICKER (Name1/Name2)" so haiku answers in tickers
        known_list_str = ", ".join(
            f"{t} ({'/'.join(aliases[:2])})"
            for t, aliases in PARENT_ALIASES.items()
        )

        def _targeted_excerpt(text: str, needle: str, window: int = 2000) -> str:
            """Return a window centred on the first occurrence of needle in text."""
            if not text:
                return ""
            idx = text.lower().find(needle.lower())
            if idx == -1:
                return text[:window]
            start = max(0, idx - window // 4)
            end = min(len(text), idx + window * 3 // 4)
            return text[start:end]

        biz_excerpt = _targeted_excerpt(business_text, placeholder)
        risk_excerpt = _targeted_excerpt(risk_factors_text, placeholder)

        user_msg = (
            f"A company uses '{placeholder}' as a customer pseudonym in their 10-K filing. "
            f"This customer accounts for {revenue_pct:.0f}% of their revenue.\n\n"
            f"Based on these filing excerpts, which company from this list is most likely "
            f"'{placeholder}'? Answer using the ticker symbol.\n\n"
            f"Known companies (ticker: common names):\n{known_list_str}\n\n"
            f"Business section excerpt (centred on '{placeholder}'):\n{biz_excerpt}\n\n"
            f"Risk factors excerpt (centred on '{placeholder}'):\n{risk_excerpt}\n\n"
            'Respond in JSON only:\n'
            '{"resolved_parent": "TSM" or null, "confidence": 0.0-1.0, '
            '"evidence": "max 150 char quote"}'
        )

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=(
                "You are a financial analyst reading SEC 10-K filings. "
                "Respond only in JSON. No preamble. No markdown."
            ),
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = message.content[0].text.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        import json
        parsed = json.loads(raw)
        resolved_parent = parsed.get("resolved_parent")
        confidence = float(parsed.get("confidence", 0.0))
        evidence = str(parsed.get("evidence", ""))[:150]

        # Always log what Haiku returned so failures are visible
        logger.info(
            "  → Haiku: parent=%s conf=%.2f evidence=%s",
            resolved_parent, confidence, evidence,
        )

        if resolved_parent and confidence >= 0.70:
            conn.execute(
                """
                INSERT OR REPLACE INTO placeholder_resolutions
                    (ticker, accession_number, placeholder, resolved_parent,
                     confidence, resolved_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [ticker, filing_year, placeholder, resolved_parent, confidence, "haiku"],
            )
            logger.info(
                "  → RESOLVED: %s %s → %s (conf: %.2f, cached as %s)",
                ticker, placeholder, resolved_parent, confidence, filing_year,
            )
            return resolved_parent, confidence, "haiku"
        else:
            logger.info(
                "  → Below threshold (%.2f < 0.70) — caching as haiku_low_conf",
                confidence,
            )
            # Use 'haiku_low_conf' (not 'unresolved') so this IS returned from cache
            # on subsequent runs — prevents re-calling Haiku every run for the same
            # ticker/placeholder. 'unresolved' is reserved for broken-run entries that
            # should be retried (no API key, etc.) and is filtered out by the cache query.
            _cache_haiku_low_conf(conn, ticker, filing_year, placeholder)
            return None, 0.0, "unresolved"

    except Exception as exc:
        # Don't cache exceptions — allow retry on next run
        logger.warning("%s %s: haiku call failed: %s — not cached, will retry", ticker, placeholder, exc)
        return None, 0.0, "unresolved"


def _cache_haiku_low_conf(conn, ticker: str, filing_year: str, placeholder: str) -> None:
    """Cache a genuine Haiku low-confidence result. NOT filtered out by cache query."""
    conn.execute(
        """
        INSERT OR REPLACE INTO placeholder_resolutions
            (ticker, accession_number, placeholder, resolved_parent,
             confidence, resolved_by, created_at)
        VALUES (?, ?, ?, NULL, 0.0, 'haiku_low_conf', CURRENT_TIMESTAMP)
        """,
        [ticker, filing_year, placeholder],
    )


# ---------- quality gate ---------------------------------------------------------

def _already_in_graph() -> set[str]:
    tickers = set(get_all_tickers())
    tickers.update(MEGA_CAPS)
    return tickers


def _build_exclude_sets(conn) -> tuple[set[str], set[str]]:
    """Return (active_in_discovery, recently_failed) sets for fast gate checks."""
    cutoff = (date.today() - timedelta(days=_FAILED_REELIGIBLE_DAYS)).isoformat()

    recently_failed = {
        r[0]
        for r in conn.execute(
            "SELECT ticker FROM discovery_candidates WHERE status = 'FAILED' AND discovered_date >= ?",
            [cutoff],
        ).fetchall()
    }
    active_in_discovery = {
        r[0]
        for r in conn.execute(
            "SELECT ticker FROM discovery_candidates WHERE status != 'FAILED'",
        ).fetchall()
    }
    return active_in_discovery, recently_failed


def _passes_quality_gate(
    ticker: str,
    parent: str,
    conn,
    massive_key: str,
    in_graph: set[str],
    active_in_discovery: set[str],
    recently_failed: set[str],
) -> tuple[bool, str]:
    """Return (passes, skip_reason). skip_reason is empty on pass."""
    if parent not in KNOWN_PARENTS:
        return False, f"parent {parent} not in KNOWN_PARENTS"
    if ticker in in_graph:
        return False, "already active"
    if ticker in active_in_discovery:
        return False, "already in discovery"
    if ticker in recently_failed:
        return False, f"failed within {_FAILED_REELIGIBLE_DAYS}d"

    # Price + ADV gate — prices table first, Massive fallback
    db_result = _price_adv_from_db(ticker, conn)
    if db_result is not None:
        latest_close, avg_dv, oldest_date = db_result
        if latest_close < _MIN_PRICE:
            return False, f"price ${latest_close:.2f} below ${_MIN_PRICE}"
        if avg_dv < _MIN_ADV:
            return False, f"ADV ${avg_dv:,.0f} below ${_MIN_ADV:,.0f}"
        try:
            oldest = date.fromisoformat(oldest_date)
            days_listed = (date.today() - oldest).days
            if days_listed < _MIN_LIST_DAYS:
                return False, f"listed ~{days_listed}d ago (need {_MIN_LIST_DAYS}d)"
        except ValueError:
            pass
    else:
        massive_result = _price_adv_from_massive(ticker, massive_key)
        if massive_result is None:
            return False, "no price data available"
        latest_close, avg_dv = massive_result
        if latest_close < _MIN_PRICE:
            return False, f"price ${latest_close:.2f} below ${_MIN_PRICE}"
        if avg_dv < _MIN_ADV:
            return False, f"ADV ${avg_dv:,.0f} below ${_MIN_ADV:,.0f}"
        list_date_str = _list_date_from_massive(ticker, massive_key)
        if list_date_str:
            try:
                list_dt = date.fromisoformat(list_date_str)
                days_listed = (date.today() - list_dt).days
                if days_listed < _MIN_LIST_DAYS:
                    return False, f"listed {days_listed}d ago (need {_MIN_LIST_DAYS}d)"
            except ValueError:
                pass

    return True, ""


# ---------- dependency_graph sync -----------------------------------------------

def sync_dependency_graph(conn) -> None:
    """Upsert all static edges from dependencies.yaml into dependency_graph table."""
    try:
        with open(_DEPS_YAML_PATH) as f:
            data = yaml.safe_load(f) or {}
        edges = data.get("edges", [])
        rows = [
            (e["child"], e["parent"], "static_yaml", date.today().isoformat(),
             None, None, None, "ACTIVE")
            for e in edges
            if e.get("child") and e.get("parent")
        ]
        conn.executemany(
            """
            INSERT OR IGNORE INTO dependency_graph
                (ticker, parent, source, validated_date, ic_h1, ic_h2, ic_full, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        logger.info("dependency_graph: synced %d static edges from dependencies.yaml", len(rows))
    except Exception as exc:
        logger.error("dependency_graph sync failed: %s", exc)


# ---------- discover() main ------------------------------------------------------

def discover(conn=None) -> int:
    """ETF universe → candidate 10-K concentration → quality gate → insert ACCUMULATING.

    Returns count of new insertions.
    """
    _own_conn = conn is None
    if _own_conn:
        conn = get_connection()

    try:
        api_key = _get_massive_key()
        if not api_key:
            logger.error("MASSIVE_API_KEY not set — discovery cannot run")
            return 0

        logger.info("Running discovery pipeline...")

        sync_dependency_graph(conn)

        candidates = _get_etf_universe(api_key, conn)
        logger.info("Universe: %d candidates", len(candidates))

        in_graph = _already_in_graph()
        active_in_discovery, recently_failed = _build_exclude_sets(conn)

        new_pairs: list[tuple[str, str, float, str]] = []
        total = len(candidates)

        for idx, ticker in enumerate(candidates, 1):
            logger.info("[%d/%d] %s: fetching 10-K...", idx, total, ticker)
            try:
                results, placeholder_count = _extract_concentration(ticker, api_key, conn)
            except Exception as exc:
                logger.error("%s: extraction failed: %s — skipping", ticker, exc)
                continue

            if not results:
                if placeholder_count:
                    logger.info(
                        "  → %d placeholder(s) found but all unresolved — skip",
                        placeholder_count,
                    )
                else:
                    logger.info("  → no concentration >= 10%% found — skip")
                continue

            for parent, pct, resolved_by in results:
                passes, reason = _passes_quality_gate(
                    ticker, parent, conn, api_key,
                    in_graph, active_in_discovery, recently_failed,
                )
                if not passes:
                    logger.info("  → SKIP: %s — %s", ticker, reason)
                    continue

                strength = "STRONG" if pct >= 20.0 else "MEDIUM"
                try:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO discovery_candidates
                            (ticker, parent, dependency_strength, claude_confidence,
                             revenue_pct, evidence, source_accession,
                             discovered_date, status)
                        VALUES (?, ?, ?, NULL, ?, NULL,
                                ?, CURRENT_DATE, 'ACCUMULATING')
                        """,
                        [ticker, parent, strength, pct, f"etf_10k_{resolved_by}"],
                    )
                    active_in_discovery.add(ticker)
                    new_pairs.append((ticker, parent, pct, resolved_by))
                    logger.info(
                        "NEW: %s → %s (%s, %.0f%% revenue)",
                        ticker, parent, resolved_by, pct,
                    )
                except Exception as exc:
                    logger.error("%s → %s: insert failed: %s", ticker, parent, exc)

        logger.info("Discovery complete: %d new candidates", len(new_pairs))

        if new_pairs:
            logger.info("Backfilling RS for %d new candidates...", len(new_pairs))
            _backfill_rs_pairs([(t, p) for t, p, _, _ in new_pairs], conn)
            _advance_accumulating_status(conn)

        return len(new_pairs)

    finally:
        if _own_conn:
            conn.close()


# ---------- RS accumulation ------------------------------------------------------

def _advance_accumulating_status(conn) -> None:
    """Transition ACCUMULATING candidates to READY_FOR_BACKTEST at 60+ RS days."""
    conn.execute(
        """
        UPDATE discovery_candidates
        SET status = 'READY_FOR_BACKTEST'
        WHERE status = 'ACCUMULATING'
          AND (ticker, parent) IN (
              SELECT ticker, parent FROM rs_accumulation
              GROUP BY ticker, parent
              HAVING COUNT(DISTINCT date) >= ?
          )
        """,
        [_RS_MIN_DAYS],
    )


def _load_price_series(ticker: str, conn) -> dict[str, float]:
    """Return {date_str: adj_close} for all available prices for ticker."""
    rows = conn.execute(
        "SELECT date, adj_close FROM prices WHERE ticker = ? ORDER BY date",
        [ticker],
    ).fetchall()
    return {str(r[0]): r[1] for r in rows if r[1] is not None}


def _compute_rs_series(
    child_prices: dict[str, float],
    parent_prices: dict[str, float],
    already_accumulated: set[str],
) -> list[tuple[str, float]]:
    """Return [(date, rs_score)] for all dates not yet accumulated."""
    child_dates = sorted(child_prices)
    parent_dates_set = set(parent_prices)

    results: list[tuple[str, float]] = []

    for i, d in enumerate(child_dates):
        if d in already_accumulated:
            continue
        if d not in parent_dates_set:
            continue

        child_window = child_dates[max(0, i - _RS_LOOKBACK): i + 1]
        if len(child_window) < _RS_LOOKBACK + 1:
            continue

        parent_window = sorted(dt for dt in parent_dates_set if dt <= d)
        if len(parent_window) < _RS_LOOKBACK + 1:
            continue
        parent_window = parent_window[-(  _RS_LOOKBACK + 1):]

        child_latest = child_prices[child_window[-1]]
        child_oldest = child_prices[child_window[0]]
        parent_latest = parent_prices[parent_window[-1]]
        parent_oldest = parent_prices[parent_window[0]]

        if not child_oldest or not parent_oldest:
            continue

        child_ret = (child_latest - child_oldest) / child_oldest
        parent_ret = (parent_latest - parent_oldest) / parent_oldest
        results.append((d, child_ret - parent_ret))

    return results


def _backfill_rs_pairs(pairs: list[tuple[str, str]], conn) -> int:
    """Backfill RS scores for the given (ticker, parent) pairs.

    Returns total rows inserted.
    """
    if not pairs:
        return 0

    parent_cache: dict[str, dict[str, float]] = {}
    inserted = 0

    for ticker, parent in pairs:
        child_prices = _load_price_series(ticker, conn)
        if not child_prices:
            continue

        if parent not in parent_cache:
            parent_cache[parent] = _load_price_series(parent, conn)
        parent_prices = parent_cache[parent]
        if not parent_prices:
            continue

        already = {
            str(r[0])
            for r in conn.execute(
                "SELECT date FROM rs_accumulation WHERE ticker = ? AND parent = ?",
                [ticker, parent],
            ).fetchall()
        }

        new_rows = _compute_rs_series(child_prices, parent_prices, already)
        if not new_rows:
            continue

        conn.executemany(
            "INSERT OR IGNORE INTO rs_accumulation (ticker, parent, date, rs_score) VALUES (?, ?, ?, ?)",
            [(ticker, parent, d, score) for d, score in new_rows],
        )
        inserted += len(new_rows)
        logger.info("%s → %s: %d RS rows inserted", ticker, parent, len(new_rows))

    return inserted


def accumulate_rs(conn=None) -> int:
    """Backfill and extend RS scores for all ACCUMULATING candidates.

    Returns total rows inserted.
    """
    _own_conn = conn is None
    if _own_conn:
        conn = get_connection()

    try:
        candidates = conn.execute(
            """
            SELECT ticker, parent FROM discovery_candidates
            WHERE status IN ('ACCUMULATING', 'READY_FOR_BACKTEST')
            """
        ).fetchall()

        if not candidates:
            return 0

        inserted = _backfill_rs_pairs(list(candidates), conn)
        _advance_accumulating_status(conn)
        return inserted

    finally:
        if _own_conn:
            conn.close()


# ---------- auto-backtest + promote ----------------------------------------------

def _spearman_ic(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 5:
        return float("nan")

    def _rank(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        for r, i in enumerate(order):
            ranks[i] = float(r + 1)
        return ranks

    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n)) / n
    std_x = math.sqrt(sum((v - mx) ** 2 for v in rx) / n)
    std_y = math.sqrt(sum((v - my) ** 2 for v in ry) / n)
    if std_x == 0 or std_y == 0:
        return float("nan")
    return cov / (std_x * std_y)


def _run_ic_backtest(ticker: str, parent: str, conn) -> dict:
    """Return IC metrics from rs_accumulation data for (ticker, parent)."""
    rows = conn.execute(
        """
        SELECT date, rs_score FROM rs_accumulation
        WHERE ticker = ? AND parent = ?
          AND rs_score IS NOT NULL
        ORDER BY date
        """,
        [ticker, parent],
    ).fetchall()

    if len(rows) < 10:
        return {"ic_h1": float("nan"), "ic_h2": float("nan"), "ic_full": float("nan"), "passed": False}

    dates = [r[0] for r in rows]
    scores = [r[1] for r in rows]

    pairs: list[tuple[float, float]] = []
    for d, score in zip(dates, scores):
        fwd_rows = conn.execute(
            """
            SELECT adj_close FROM prices
            WHERE ticker = ? AND date > ?
            ORDER BY date LIMIT 6
            """,
            [ticker, str(d)],
        ).fetchall()
        if len(fwd_rows) < 5:
            continue
        entry = fwd_rows[0][0]
        exit_ = fwd_rows[4][0]
        if not entry or entry == 0:
            continue
        fwd_ret = (exit_ - entry) / entry
        pairs.append((score, fwd_ret))

    if len(pairs) < 10:
        return {"ic_h1": float("nan"), "ic_h2": float("nan"), "ic_full": float("nan"), "passed": False}

    mid = len(pairs) // 2
    h1_pairs, h2_pairs = pairs[:mid], pairs[mid:]

    ic_full = _spearman_ic([p[0] for p in pairs], [p[1] for p in pairs])
    ic_h1 = (
        _spearman_ic([p[0] for p in h1_pairs], [p[1] for p in h1_pairs])
        if len(h1_pairs) >= 5 else float("nan")
    )
    ic_h2 = (
        _spearman_ic([p[0] for p in h2_pairs], [p[1] for p in h2_pairs])
        if len(h2_pairs) >= 5 else float("nan")
    )

    passed = (
        not math.isnan(ic_h1) and not math.isnan(ic_h2)
        and ic_h1 > _IC_THRESHOLD and ic_h2 > _IC_THRESHOLD
    )

    return {"ic_h1": ic_h1, "ic_h2": ic_h2, "ic_full": ic_full, "passed": passed}


def _append_to_dependencies_yaml(ticker: str, parent: str, evidence: str) -> None:
    """Append a new edge to dependencies.yaml."""
    with open(_DEPS_YAML_PATH) as f:
        data = yaml.safe_load(f)

    new_edge = {
        "parent": parent,
        "child": ticker,
        "type": "systems",
        "weight": 0.3,
        "notes": (
            f"Auto-promoted via discovery pipeline: {evidence[:80]}"
            if evidence else "Auto-promoted via discovery pipeline"
        ),
    }
    data["edges"].append(new_edge)

    with open(_DEPS_YAML_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    _load_edges.cache_clear()
    logger.info("PROMOTED: %s → %s appended to dependencies.yaml", ticker, parent)


def run_discovery_backtests(conn=None) -> int:
    """Run IC backtest for READY_FOR_BACKTEST candidates. Returns count processed."""
    _own_conn = conn is None
    if _own_conn:
        conn = get_connection()

    try:
        rows = conn.execute(
            """
            SELECT ticker, parent, evidence
            FROM discovery_candidates
            WHERE status = 'READY_FOR_BACKTEST'
            """
        ).fetchall()

        if not rows:
            return 0

        today = date.today().isoformat()
        processed = 0

        for ticker, parent, evidence in rows:
            try:
                metrics = _run_ic_backtest(ticker, parent, conn)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO discovery_backtest_results
                        (ticker, parent, run_date, ic_h1, ic_h2, ic_full, passed, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        ticker, parent, today,
                        metrics["ic_h1"], metrics["ic_h2"], metrics["ic_full"],
                        metrics["passed"], None,
                    ],
                )

                if metrics["passed"]:
                    conn.execute(
                        "UPDATE discovery_candidates SET status = 'PROMOTED' WHERE ticker = ? AND parent = ?",
                        [ticker, parent],
                    )
                    _append_to_dependencies_yaml(ticker, parent, evidence or "")
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO dependency_graph
                            (ticker, parent, source, validated_date,
                             ic_h1, ic_h2, ic_full, status)
                        VALUES (?, ?, 'discovery_promoted', ?, ?, ?, ?, 'ACTIVE')
                        """,
                        [ticker, parent, today,
                         metrics["ic_h1"], metrics["ic_h2"], metrics["ic_full"]],
                    )
                    logger.info(
                        "PROMOTED: %s → %s  IC H1:%.3f H2:%.3f",
                        ticker, parent, metrics["ic_h1"], metrics["ic_h2"],
                    )
                else:
                    conn.execute(
                        "UPDATE discovery_candidates SET status = 'FAILED' WHERE ticker = ? AND parent = ?",
                        [ticker, parent],
                    )
                    logger.info(
                        "FAILED: %s → %s  IC H1:%s H2:%s",
                        ticker, parent,
                        f"{metrics['ic_h1']:.3f}" if not math.isnan(metrics["ic_h1"]) else "N/A",
                        f"{metrics['ic_h2']:.3f}" if not math.isnan(metrics["ic_h2"]) else "N/A",
                    )

                processed += 1

            except Exception as exc:
                logger.error("%s → %s: backtest failed: %s", ticker, parent, exc)
                continue

        return processed

    finally:
        if _own_conn:
            conn.close()


# ---------- display data helpers -------------------------------------------------

def get_discovery_display(conn) -> dict:
    """Return discovery pipeline data for UI rendering."""
    thirty_days_ago = (date.today() - timedelta(days=30)).isoformat()
    seven_days_ago = (date.today() - timedelta(days=7)).isoformat()

    accumulating_rows = conn.execute(
        """
        SELECT dc.ticker, dc.parent, dc.dependency_strength,
               COUNT(DISTINCT ra.date) AS days_accumulated,
               AVG(ra.rs_score) AS avg_rs
        FROM discovery_candidates dc
        LEFT JOIN rs_accumulation ra ON dc.ticker = ra.ticker AND dc.parent = ra.parent
        WHERE dc.status = 'ACCUMULATING'
        GROUP BY dc.ticker, dc.parent, dc.dependency_strength
        ORDER BY days_accumulated DESC
        """
    ).fetchall()

    accumulating = []
    for ticker, parent, strength, days, avg_rs in accumulating_rows:
        last5 = conn.execute(
            """
            SELECT AVG(rs_score) FROM (
                SELECT rs_score FROM rs_accumulation
                WHERE ticker = ? AND parent = ?
                ORDER BY date DESC LIMIT 5
            )
            """,
            [ticker, parent],
        ).fetchone()[0]

        prior5 = conn.execute(
            """
            SELECT AVG(rs_score) FROM (
                SELECT rs_score FROM rs_accumulation
                WHERE ticker = ? AND parent = ?
                ORDER BY date DESC LIMIT 10
            ) t WHERE rs_score NOT IN (
                SELECT rs_score FROM rs_accumulation
                WHERE ticker = ? AND parent = ?
                ORDER BY date DESC LIMIT 5
            )
            """,
            [ticker, parent, ticker, parent],
        ).fetchone()[0]

        if last5 is not None and prior5 is not None:
            trend = "↑" if last5 > prior5 + 0.002 else ("↓" if last5 < prior5 - 0.002 else "→")
        else:
            trend = "→"

        accumulating.append({
            "ticker": ticker,
            "parent": parent,
            "strength": strength,
            "days": days or 0,
            "rs": avg_rs,
            "trend": trend,
        })

    ready_rows = conn.execute(
        """
        SELECT dc.ticker, dc.parent, COUNT(DISTINCT ra.date) AS days
        FROM discovery_candidates dc
        LEFT JOIN rs_accumulation ra ON dc.ticker = ra.ticker AND dc.parent = ra.parent
        WHERE dc.status = 'READY_FOR_BACKTEST'
        GROUP BY dc.ticker, dc.parent
        """
    ).fetchall()
    ready = [{"ticker": r[0], "parent": r[1], "days": r[2]} for r in ready_rows]

    promoted_rows = conn.execute(
        """
        SELECT dbr.ticker, dbr.parent, dbr.ic_h1, dbr.ic_h2, dbr.run_date
        FROM discovery_backtest_results dbr
        JOIN discovery_candidates dc ON dbr.ticker = dc.ticker AND dbr.parent = dc.parent
        WHERE dc.status = 'PROMOTED' AND dbr.run_date >= ?
        ORDER BY dbr.run_date DESC
        """,
        [thirty_days_ago],
    ).fetchall()
    promoted = [
        {"ticker": r[0], "parent": r[1], "ic_h1": r[2], "ic_h2": r[3], "run_date": str(r[4])}
        for r in promoted_rows
    ]

    failed_rows = conn.execute(
        """
        SELECT dbr.ticker, dbr.parent, dbr.ic_h1, dbr.ic_h2
        FROM discovery_backtest_results dbr
        JOIN discovery_candidates dc ON dbr.ticker = dc.ticker AND dbr.parent = dc.parent
        WHERE dc.status = 'FAILED' AND dbr.run_date >= ?
        ORDER BY dbr.run_date DESC
        """,
        [thirty_days_ago],
    ).fetchall()
    failed = [
        {"ticker": r[0], "parent": r[1], "ic_h1": r[2], "ic_h2": r[3]}
        for r in failed_rows
    ]

    new_rows = conn.execute(
        """
        SELECT ticker, parent, claude_confidence
        FROM discovery_candidates
        WHERE discovered_date >= ?
        ORDER BY discovered_date DESC
        """,
        [seven_days_ago],
    ).fetchall()
    new_candidates = [
        {"ticker": r[0], "parent": r[1], "confidence": r[2]}
        for r in new_rows
    ]

    return {
        "accumulating": accumulating,
        "ready": ready,
        "promoted": promoted,
        "failed": failed,
        "new_candidates": new_candidates,
    }
