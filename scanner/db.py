"""DuckDB connection and schema management."""

import os
from pathlib import Path
import duckdb

_DB_PATH = Path(os.environ.get("DATABASE_PATH", str(Path(__file__).parent.parent / "data" / "scanner.duckdb")))

_DDL = """
CREATE TABLE IF NOT EXISTS prices (
    ticker   VARCHAR NOT NULL,
    date     DATE    NOT NULL,
    open     DOUBLE,
    high     DOUBLE,
    low      DOUBLE,
    close    DOUBLE NOT NULL,
    volume   BIGINT,
    adj_close DOUBLE NOT NULL,
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

CREATE TABLE IF NOT EXISTS insider_transactions (
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
);

CREATE INDEX IF NOT EXISTS idx_insider_ticker_date
    ON insider_transactions(ticker, transaction_date);

CREATE TABLE IF NOT EXISTS insider_filter_review (
    accession_number  VARCHAR  NOT NULL,
    transaction_seq   INTEGER  NOT NULL,
    ticker            VARCHAR  NOT NULL,
    filed_date        DATE     NOT NULL,
    reason            VARCHAR  NOT NULL,
    footnote_text     VARCHAR,
    filing_url        VARCHAR,
    PRIMARY KEY (accession_number, transaction_seq)
);

CREATE TABLE IF NOT EXISTS estimates (
    ticker          VARCHAR NOT NULL,
    date            DATE    NOT NULL,
    eps_current_q   DOUBLE,
    eps_next_q      DOUBLE,
    analysts_up     INTEGER,
    analysts_down   INTEGER,
    revision_score  DOUBLE,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS scheduler_activity_log (
    logged_at  TIMESTAMP NOT NULL,
    job_name   VARCHAR   NOT NULL,
    status     VARCHAR   NOT NULL,
    message    VARCHAR,
    PRIMARY KEY (logged_at, job_name)
);

CREATE TABLE IF NOT EXISTS earnings_dates (
    ticker             VARCHAR   NOT NULL,
    next_earnings_date DATE,
    ingested_at        TIMESTAMP NOT NULL,
    PRIMARY KEY (ticker)
);

CREATE TABLE IF NOT EXISTS journal (
    scan_date          DATE      NOT NULL,
    ticker             VARCHAR   NOT NULL,
    parent             VARCHAR   NOT NULL,
    parent_regime      VARCHAR,
    price              DOUBLE,
    rs_score           DOUBLE,
    rank               VARCHAR,
    action_label       VARCHAR,
    est_rev            DOUBLE,
    rvol               DOUBLE,
    confirm            INTEGER,
    insider_annotation VARCHAR,
    earnings_days      INTEGER,
    parent_20d_return  DOUBLE,
    notes              VARCHAR,
    PRIMARY KEY (scan_date, ticker, parent)
);

CREATE TABLE IF NOT EXISTS macro_observations (
    id               INTEGER   NOT NULL,
    observation_date DATE      NOT NULL,
    type             VARCHAR   NOT NULL,
    primary_ticker   VARCHAR   NOT NULL,
    affected_tickers VARCHAR,
    note             TEXT      NOT NULL,
    link             VARCHAR,
    created_at       TIMESTAMP NOT NULL,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS filings_8k (
    id               INTEGER   NOT NULL,
    ticker           VARCHAR   NOT NULL,
    filed_date       DATE      NOT NULL,
    accession_number VARCHAR   NOT NULL,
    form_type        VARCHAR   NOT NULL,
    item_numbers     VARCHAR,
    title            VARCHAR,
    description      TEXT,
    filing_url       VARCHAR,
    saved_to_journal BOOLEAN   NOT NULL,
    ingested_at      TIMESTAMP NOT NULL,
    impact           VARCHAR,
    impact_source    VARCHAR,
    PRIMARY KEY (id),
    UNIQUE (accession_number)
);

CREATE INDEX IF NOT EXISTS idx_filings_8k_ticker_date
    ON filings_8k(ticker, filed_date);

CREATE TABLE IF NOT EXISTS scheduler_runs (
    job_name  VARCHAR   NOT NULL,
    run_time  TIMESTAMP NOT NULL,
    status    VARCHAR   NOT NULL,
    message   VARCHAR,
    PRIMARY KEY (job_name, run_time)
);

CREATE TABLE IF NOT EXISTS scan_results (
    scan_date   DATE      NOT NULL,
    result_json TEXT      NOT NULL,
    computed_at TIMESTAMP NOT NULL,
    PRIMARY KEY (scan_date)
);
"""


def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(_DB_PATH), read_only=read_only)
    if not read_only:
        conn.execute(_DDL)
        # Add columns that may be missing on tables created before the schema was updated.
        for _stmt in [
            "ALTER TABLE journal ADD COLUMN est_rev DOUBLE",
            "ALTER TABLE journal ADD COLUMN rvol DOUBLE",
            "ALTER TABLE journal ADD COLUMN confirm INTEGER",
            "ALTER TABLE journal ADD COLUMN earnings_days INTEGER",
            "ALTER TABLE filings_8k ADD COLUMN summary TEXT",
        ]:
            try:
                conn.execute(_stmt)
            except Exception:
                pass
    return conn


def get_read_connection() -> duckdb.DuckDBPyConnection:
    return get_connection(read_only=False)
