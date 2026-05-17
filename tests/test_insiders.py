"""Tests for insider ingest filter logic and enrichment queries.

No network calls — all EDGAR HTTP is bypassed by testing pure functions directly.
"""

from datetime import date
from pathlib import Path

import duckdb
import pytest

_FIXTURES = Path(__file__).parent / "fixtures"

from unittest.mock import MagicMock

from scanner.ingest.insiders import (
    _EDGAR_ARCHIVES,
    _is_xsl_path,
    _resolve_raw_xml,
    check_offering_keywords,
    check_prospectus_reference,
    parse_form4_xml,
)
from scanner.enrichment.insiders import get_insider_summary

# ---------------------------------------------------------------------------
# Minimal Form 4 XML templates
# ---------------------------------------------------------------------------

_XML_BASE = """\
<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>{name}</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isOfficer>{is_officer}</isOfficer>
      <isDirector>{is_director}</isDirector>
      <isTenPercentOwner>0</isTenPercentOwner>
      <officerTitle>{title}</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    {transactions}
  </nonDerivativeTable>
  {footnotes_block}
</ownershipDocument>"""

_TXN_TEMPLATE = """\
<nonDerivativeTransaction>
  <transactionDate><value>{txn_date}</value></transactionDate>
  <transactionCoding>
    <transactionCode>{code}</transactionCode>
  </transactionCoding>
  <transactionAmounts>
    <transactionShares><value>{shares}</value></transactionShares>
    <transactionPricePerShare><value>{price}</value>{fn_link}</transactionPricePerShare>
    <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
  </transactionAmounts>
  <postTransactionAmounts>
    <sharesOwnedFollowingTransaction><value>{owned_after}</value></sharesOwnedFollowingTransaction>
  </postTransactionAmounts>
</nonDerivativeTransaction>"""


def _make_xml(
    code: str = "P",
    shares: float = 10000,
    price: float = 50.0,
    owned_after: float = 110000,
    txn_date: str = "2025-04-01",
    footnote_id: str = "",
    footnote_text: str = "",
    name: str = "John Smith",
    title: str = "CEO",
    is_officer: int = 1,
    is_director: int = 0,
) -> str:
    fn_link = f'<footnoteId id="{footnote_id}"/>' if footnote_id else ""
    txn = _TXN_TEMPLATE.format(
        code=code, shares=shares, price=price, owned_after=owned_after,
        txn_date=txn_date, fn_link=fn_link,
    )
    if footnote_id and footnote_text:
        fn_block = f'<footnotes><footnote id="{footnote_id}">{footnote_text}</footnote></footnotes>'
    else:
        fn_block = ""
    return _XML_BASE.format(
        name=name, title=title, is_officer=is_officer, is_director=is_director,
        transactions=txn, footnotes_block=fn_block,
    )


# ---------------------------------------------------------------------------
# check_offering_keywords
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Shares purchased pursuant to an underwritten offering", True),
    ("In connection with a public offering of common stock", True),
    ("Acquired in a registered direct offering", True),
    ("follow-on offering participation", True),
    ("secondary offering shares", True),
    ("private placement transaction", True),
    ("pursuant to a PIPE transaction", True),
    # case-insensitive
    ("Underwritten Offering", True),
    ("PUBLIC OFFERING", True),
    # ESPP — plan-driven at predetermined prices, not discretionary
    ("Common Shares purchased by the administrator of the issuer's Employee Stock Purchase Plan on behalf of the filer", True),
    ("Shares purchased pursuant to ESPP terms", True),
    ("employee stock purchase plan participation", True),
    # should NOT match
    ("Open-market purchase on NYSE", False),
    ("Shares purchased at market price", False),
    ("10b5-1 plan purchase", False),
    ("", False),
])
def test_check_offering_keywords(text, expected):
    assert check_offering_keywords(text) == expected


# ---------------------------------------------------------------------------
# check_prospectus_reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("See S-1 registration statement filed 2025-01-10", True),
    ("prospectus filed on Form S-3", True),
    ("424B5 prospectus supplement", True),
    ("424b3 base prospectus", True),
    # should NOT match
    ("Open-market purchase", False),
    ("10b5-1 plan", False),
    ("", False),
])
def test_check_prospectus_reference(text, expected):
    assert check_prospectus_reference(text) == expected


# ---------------------------------------------------------------------------
# parse_form4_xml — clean open-market purchase
# ---------------------------------------------------------------------------

def test_parse_clean_buy():
    xml = _make_xml(code="P", shares=5000, price=100.0, owned_after=55000,
                    txn_date="2025-03-15")
    txns, reviews = parse_form4_xml(xml, "ACC-001", "AAPL", "2025-03-15", "https://example.com/form4.xml")
    assert len(txns) == 1
    assert len(reviews) == 0
    t = txns[0]
    assert t["transaction_code"] == "P"
    assert t["is_open_market"] is True
    assert t["shares"] == 5000
    assert t["price_per_share"] == 100.0
    assert t["total_value"] == pytest.approx(500_000.0)
    assert t["insider_name"] == "John Smith"
    assert t["insider_title"] == "CEO"
    assert t["is_officer"] is True
    assert t["is_director"] is False
    assert t["transaction_seq"] == 0


def test_parse_clean_buy_officer_flag():
    xml = _make_xml(code="P", is_officer=1, is_director=0, title="CFO")
    txns, _ = parse_form4_xml(xml, "ACC-002", "MSFT", "2025-04-01", "https://example.com/f.xml")
    assert txns[0]["is_officer"] is True
    assert txns[0]["insider_title"] == "CFO"


# ---------------------------------------------------------------------------
# parse_form4_xml — offering-participation purchase → is_open_market=False
# ---------------------------------------------------------------------------

def test_parse_offering_participation_buy():
    xml = _make_xml(
        code="P", footnote_id="F1",
        footnote_text="Purchased in connection with an underwritten offering of common stock",
    )
    txns, reviews = parse_form4_xml(xml, "ACC-003", "NVDA", "2025-04-01", "https://example.com/f.xml")
    assert len(txns) == 1
    assert len(reviews) == 0
    assert txns[0]["is_open_market"] is False
    assert "underwritten offering" in txns[0]["footnote_text"].lower()


# ---------------------------------------------------------------------------
# parse_form4_xml — prospectus reference → review table, excluded from transactions
# ---------------------------------------------------------------------------

def test_parse_prospectus_reference_routes_to_review():
    xml = _make_xml(
        code="P", footnote_id="F1",
        footnote_text="See S-3 registration statement for full details",
    )
    txns, reviews = parse_form4_xml(xml, "ACC-004", "AMD", "2025-04-01", "https://example.com/f.xml")
    assert len(txns) == 0
    assert len(reviews) == 1
    assert reviews[0]["reason"] == "prospectus_reference"
    assert reviews[0]["ticker"] == "AMD"


# ---------------------------------------------------------------------------
# parse_form4_xml — sale transaction
# ---------------------------------------------------------------------------

def test_parse_sale():
    xml = _make_xml(code="S", shares=2000, price=200.0, owned_after=8000)
    txns, reviews = parse_form4_xml(xml, "ACC-005", "GOOGL", "2025-04-10", "https://example.com/f.xml")
    assert len(txns) == 1
    assert txns[0]["transaction_code"] == "S"
    assert txns[0]["is_open_market"] is None


# ---------------------------------------------------------------------------
# parse_form4_xml — non-P/S codes are ignored
# ---------------------------------------------------------------------------

def test_parse_ignores_award():
    xml = _make_xml(code="A")
    txns, reviews = parse_form4_xml(xml, "ACC-006", "META", "2025-04-01", "https://example.com/f.xml")
    assert txns == []
    assert reviews == []


# ---------------------------------------------------------------------------
# parse_form4_xml — malformed XML returns empty lists
# ---------------------------------------------------------------------------

def test_parse_bad_xml():
    txns, reviews = parse_form4_xml("<broken<xml", "ACC-007", "AAPL", "2025-04-01", "url")
    assert txns == []
    assert reviews == []


# ---------------------------------------------------------------------------
# get_insider_summary — in-memory DuckDB
# ---------------------------------------------------------------------------

def _setup_db():
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE insider_transactions (
            accession_number     VARCHAR  NOT NULL,
            transaction_seq      INTEGER  NOT NULL,
            ticker               VARCHAR  NOT NULL,
            filed_date           DATE     NOT NULL,
            transaction_date     DATE,
            insider_name         VARCHAR,
            insider_title        VARCHAR,
            is_officer           BOOLEAN,
            is_director          BOOLEAN,
            is_ten_percent_owner BOOLEAN,
            transaction_code     VARCHAR(2),
            shares               DOUBLE,
            price_per_share      DOUBLE,
            total_value          DOUBLE,
            shares_owned_after   DOUBLE,
            is_open_market       BOOLEAN,
            footnote_text        VARCHAR,
            filing_url           VARCHAR,
            PRIMARY KEY (accession_number, transaction_seq)
        )
    """)
    return conn


def test_summary_empty_when_no_data():
    conn = _setup_db()
    summary = get_insider_summary("AAPL", date(2025, 4, 30), conn)
    assert summary["buy_count"] == 0
    assert summary["total_dollar_value"] == 0.0
    assert summary["has_cluster_buy"] is False
    conn.close()


def test_summary_single_officer_buy():
    conn = _setup_db()
    conn.execute("""
        INSERT INTO insider_transactions VALUES
        ('ACC-A', 0, 'AAPL', '2025-04-15', '2025-04-14',
         'Tim Cook', 'CEO', true, false, false,
         'P', 10000, 200.0, 2000000.0, 110000, true, NULL, 'http://example.com')
    """)
    as_of = date(2025, 4, 30)
    summary = get_insider_summary("AAPL", as_of, conn)
    assert summary["buy_count"] == 1
    assert summary["distinct_insiders"] == 1
    assert summary["total_dollar_value"] == pytest.approx(2_000_000.0)
    assert summary["has_officer_buys"] is True
    assert summary["has_cluster_buy"] is False
    assert summary["largest_single_buy"] == pytest.approx(2_000_000.0)
    assert summary["days_since_last_buy"] == 16
    conn.close()


def test_summary_cluster_buy():
    conn = _setup_db()
    for i, name in enumerate(["Alice", "Bob", "Carol"]):
        conn.execute("""
            INSERT INTO insider_transactions VALUES
            (?, 0, 'NVDA', '2025-04-10', '2025-04-09',
             ?, 'Director', false, true, false,
             'P', 5000, 100.0, 500000.0, 50000, true, NULL, 'http://example.com')
        """, [f"ACC-{i}", name])
    summary = get_insider_summary("NVDA", date(2025, 4, 30), conn)
    assert summary["buy_count"] == 3
    assert summary["distinct_insiders"] == 3
    assert summary["has_cluster_buy"] is True
    conn.close()


def test_summary_notable_selling():
    conn = _setup_db()
    conn.execute("""
        INSERT INTO insider_transactions VALUES
        ('ACC-S', 0, 'META', '2025-04-20', '2025-04-18',
         'CFO Name', 'CFO', true, false, false,
         'S', 3000, 600.0, 1800000.0, 7000, NULL, NULL, 'http://example.com')
    """)
    summary = get_insider_summary("META", date(2025, 4, 30), conn)
    assert summary["buy_count"] == 0
    assert summary["total_sell_value"] == pytest.approx(1_800_000.0)
    assert summary["has_notable_selling"] is True
    conn.close()


def test_summary_outside_window_excluded():
    conn = _setup_db()
    conn.execute("""
        INSERT INTO insider_transactions VALUES
        ('ACC-OLD', 0, 'AAPL', '2025-03-01', '2025-02-28',
         'Old Buy', 'VP', true, false, false,
         'P', 1000, 150.0, 150000.0, 10000, true, NULL, 'http://example.com')
    """)
    # as_of=2025-04-30, window=30d → start=2025-03-31; trade on 2025-02-28 is outside window
    summary = get_insider_summary("AAPL", date(2025, 4, 30), conn, window_days=30)
    assert summary["buy_count"] == 0
    conn.close()


def test_summary_offering_participation_excluded():
    """is_open_market=False rows must not be counted as buys."""
    conn = _setup_db()
    conn.execute("""
        INSERT INTO insider_transactions VALUES
        ('ACC-OFF', 0, 'AMD', '2025-04-15', '2025-04-14',
         'Lisa Su', 'CEO', true, false, false,
         'P', 50000, 50.0, 2500000.0, 500000, false, 'underwritten offering', 'http://example.com')
    """)
    summary = get_insider_summary("AMD", date(2025, 4, 30), conn)
    assert summary["buy_count"] == 0
    assert summary["has_officer_buys"] is False
    conn.close()


# ---------------------------------------------------------------------------
# _is_xsl_path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("document,expected", [
    ("xslF345X06/form4.xml", True),
    ("xslF345X05/form4-04282026.xml", True),
    ("XslF345X06/form4.xml", True),    # case-insensitive
    ("form4.xml", False),
    ("ownership.xml", False),
    ("primary_doc.xml", False),
    ("0001234567-26-000001.xml", False),
])
def test_is_xsl_path(document, expected):
    assert _is_xsl_path(document) == expected


# ---------------------------------------------------------------------------
# _resolve_raw_xml — non-XSL path returned as-is without any network call
# ---------------------------------------------------------------------------

def test_resolve_raw_xml_non_xsl_no_network():
    session = MagicMock()
    result = _resolve_raw_xml(12345, "0001234567-26-000001", "form4.xml", session)
    assert result == "form4.xml"
    session.get.assert_not_called()


# ---------------------------------------------------------------------------
# _resolve_raw_xml — XSL path resolved via index.json
# ---------------------------------------------------------------------------

def test_resolve_raw_xml_uses_index_json():
    session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "directory": {
            "item": [
                {"name": "form4-04282026_090449.xml", "type": "4", "size": "3456"},
                {"name": "xslF345X06", "type": "folder.gif", "size": "1"},
            ]
        }
    }
    session.get.return_value = mock_resp

    result = _resolve_raw_xml(
        1678660, "0001231919-26-000422",
        "xslF345X06/form4-04282026_090449.xml", session,
    )
    assert result == "form4-04282026_090449.xml"

    expected_index_url = f"{_EDGAR_ARCHIVES}/1678660/000123191926000422/index.json"
    session.get.assert_called_once_with(expected_index_url, timeout=30)


# ---------------------------------------------------------------------------
# _resolve_raw_xml — falls back to stripping xsl prefix if index.json fails
# ---------------------------------------------------------------------------

def test_resolve_raw_xml_fallback_strips_xsl_dir():
    session = MagicMock()
    session.get.side_effect = Exception("network error")

    result = _resolve_raw_xml(
        12345, "0001234567-26-000001",
        "xslF345X06/form4.xml", session,
    )
    assert result == "form4.xml"


# ---------------------------------------------------------------------------
# Fixture regression test: real PRLD / OrbiMed–Bonita David April 2026 Form 4
#
# This filing (accession 0000947871-26-000462) contains two P transactions
# where footnote F1 reads "These securities were purchased in an underwritten
# public offering."  The footnote is referenced via <footnoteId id="F1"/> — the
# real EDGAR element name.  This test exists to prevent regression of the bug
# where the parser searched for <footnoteLink> instead of <footnoteId>, causing
# footnote_text to come back None and offering-participation to be misclassified
# as is_open_market=True.
# ---------------------------------------------------------------------------

def test_real_prld_offering_participation_classified_correctly():
    xml = (_FIXTURES / "prld_orbimed_bonita_apr2026.xml").read_text(encoding="utf-8")
    txns, reviews = parse_form4_xml(
        xml,
        "0000947871-26-000462",
        "PRLD",
        "2026-04-23",
        "https://www.sec.gov/Archives/edgar/data/1678660/000094787126000462/ownership.xml",
    )

    # The filing has exactly 2 nonDerivativeTransaction elements (1 nonDerivativeHolding
    # is ignored by the parser).
    assert len(txns) == 2, f"expected 2 transactions, got {len(txns)}"
    assert len(reviews) == 0

    for t in txns:
        assert t["transaction_code"] == "P"
        # Footnote text must be resolved — not None/empty
        assert t["footnote_text"], f"footnote_text is empty for seq={t['transaction_seq']}"
        assert "underwritten public offering" in t["footnote_text"].lower(), (
            f"offering keyword missing from footnote_text: {t['footnote_text']!r}"
        )
        # Must be classified as NOT open-market (offering participation)
        assert t["is_open_market"] is False, (
            f"is_open_market should be False but got {t['is_open_market']!r}"
        )

    assert txns[0]["insider_name"] == "Bonita David P"
    assert txns[0]["is_director"] is True
    assert txns[0]["is_officer"] is False
