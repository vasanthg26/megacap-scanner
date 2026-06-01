# Mega-Cap Scanner — Claude Code Conventions

## Project Structure

```
scanner/
  cli.py          — Typer entry point; all user-facing commands live here
  db.py           — DuckDB schema DDL and connection factory
  ingest/
    prices.py     — OHLCV fetch (yfinance) + upsert into DuckDB
    insiders.py   — Form 4 ingest via SEC EDGAR REST API
  enrichment/
    insiders.py   — get_insider_summary() aggregation query (display only)
  graph/
    loader.py     — Reads config/dependencies.yaml; provides get_dependents(),
                    get_parents(), get_edge_weight()
  signals/
    base.py       — Signal protocol + weighted_sum() combiner
    relative_strength.py — Rolling return differential, edge-weight scaled
  backtest/
    runner.py     — Walk-forward engine; weekly rebalance, long top quintile
tests/            — pytest; no network calls, mock yfinance in all ingest tests
config/
  dependencies.yaml — Typed directed edges (parent → child)
data/             — DuckDB file + caches (gitignored)
```

## How to Add a New Signal

1. Create `scanner/signals/my_signal.py`.
2. Implement the `Signal` protocol from `signals/base.py`:
   ```python
   class MySignal:
       name = "my_signal"
       def compute(self, ticker: str, date: str, conn: duckdb.DuckDBPyConnection) -> float:
           ...  # return nan on insufficient data
   ```
3. **Enforce no-lookahead**: every SQL query MUST use `WHERE date <= ?` with the
   signal date as the bound. Never read prices after the as-of date.
4. Register the signal name in `SIGNAL_REGISTRY` in `backtest/runner.py`.
5. Add it as an option in the `backtest` CLI command.
6. Write tests in `tests/test_signals.py` using in-memory DuckDB — do not hit yfinance.

## How to Extend the Dependency Graph

Edit `config/dependencies.yaml`. Each edge requires:
```yaml
- parent: TICKER
  child: TICKER
  type: <cooling|power|semiconductor|foundry|rf|optical|networking|systems|materials|audio>
  weight: 0.0–1.0   # reflects revenue exposure / criticality
  notes: one-line description of the relationship
```

The graph loader caches parsed edges via `@lru_cache`. After editing the YAML,
restart the process (the cache is process-scoped; no hot-reload).

## Data Ingest Conventions

- `ingest_ticker()` uses `INSERT OR REPLACE` — idempotent by design.
- Swap yfinance for Polygon.io by replacing `_fetch_ohlcv()` in `ingest/prices.py`.
  The function contract: return a DataFrame with columns
  `[ticker, date, open, high, low, close, volume, adj_close]`.
- Always log errors with `logger.error(...)` — silent failures are forbidden.

## Backtest Conventions

- Walk-forward only. Never report in-sample numbers.
- **IC (Spearman rank correlation of score vs forward return) is the headline metric.**
  Sharpe is secondary. A signal with IC > 0.05 and IC IR > 0.5 is worth pursuing.
- `horizon` is measured in trading days, not calendar days.
- Portfolio is equal-weight long of top quintile per parent, rebalanced weekly (every 5 trading days).

## Insider Enrichment Layer

Insider data is **enrichment only** — it annotates the `scan` display with an
"Insiders (30d)" column but does NOT modify signal scores, composite scores, action
labels, or the regime gate. Never route insider data into signal computation.

### CLI commands
- `scanner ingest-insiders [--lookback-days N] [tickers...]` — fetches Form 4 XML from
  SEC EDGAR REST API for the universe (or subset) and writes to DuckDB.
- `scanner insider-recent [--days N] [--ticker T] [--min-value V]` — lists recent
  open-market insider purchases; ★ = officer, ◆ = director.

### Schema
- `insider_transactions(accession_number, transaction_seq, ticker, filed_date, ...)` — PK is
  composite `(accession_number, transaction_seq)` because a single Form 4 XML can contain
  multiple nonDerivativeTransaction elements.
- `insider_filter_review(...)` — P transactions with S-1/S-3/424B footnote references;
  routed here for human review rather than auto-included or auto-excluded.

### Offering-participation filter (applied in `ingest/insiders.py`)
1. Transaction code must be `P`.
2. Check footnote text (case-insensitive) for `_OFFERING_KEYWORDS`:
   `underwritten offering`, `public offering`, `registered direct`,
   `follow-on offering`, `secondary offering`, `private placement`, `pipe`.
   Match → `is_open_market=False` (stored but excluded from buy-signal queries).
3. Check for `_PROSPECTUS_PATTERNS`: `s-1`, `s-3`, `424b`.
   Match (no offering keyword) → `insider_filter_review` only; excluded from transactions.

### `get_insider_summary(ticker, as_of, conn, window_days=30)`
Returns a dict with: `buy_count`, `distinct_insiders`, `total_dollar_value`,
`has_officer_buys`, `has_cluster_buy` (3+ distinct insiders), `largest_single_buy`,
`most_recent_buy_date`, `days_since_last_buy`, `total_sell_value`, `has_notable_selling`
(>=$500K), `top_seller_title`. Only counts rows where `is_open_market=TRUE`.

### Rate limiting
SEC EDGAR requires ≤10 req/s. `_RATE_SLEEP = 0.11s` is applied after every HTTP call.
User-Agent header must identify the application (set in `_USER_AGENT` constant).

### Form 4/A amendments
Stored with a different accession number from the original Form 4. Both the original and
the amendment are ingested; `ON CONFLICT DO NOTHING` prevents double-counting within the
same accession, but amended + original filings for the same economic transaction will
appear as separate rows. Known caveat — acceptable for display purposes.

### Foreign private issuers
Some tickers in the universe (e.g. TSM) are foreign private issuers and may not file
Form 4 on EDGAR. If a ticker has no CIK in the company_tickers.json map, `ingest_form4`
logs a warning and returns 0 (no error raised).

## ⚠️ CRITICAL: Form 4 Offering-Participation Bug

**Do not implement Form 4 signals without this filter.**

Many data providers (OpenInsider, some commercial feeds) classify offering-participation
purchases as open-market insider buys. This creates **strong false-positive signals**:
a CEO buying in a secondary offering looks identical to a conviction open-market buy in raw
Form 4 data, but has zero predictive value (price is set by bankers, not the market).

### Correct implementation:
1. Transaction code must be `P` (open-market purchase).
2. Then check ALL footnote text (case-insensitive) for:
   - `"underwritten offering"`
   - `"public offering"`
   - `"registered direct"`
   - `"follow-on offering"`
   - `"secondary offering"`
3. If any match: set `offering_participation = TRUE` and **exclude** from buy signals.

See `ingest/insiders.py` for the full schema design and Phase 2 implementation plan.

## Regime-Gating (scan command)

Regime-gating is Phase 1 discovery based on backtested evidence that vol-normalized RS
(`rs_normalized`) has positive IC for MSFT/META dependents **only** when the parent is in
the -15% to 0% drawdown range (CORRECTION and MILD_PULLBACK regimes).

- **CORRECTION** (-15% to -5%): IC +0.13 pooled, +0.062 / +0.217 in H1/H2 out-of-sample split.
- **MILD_PULLBACK** (-5% to 0%): IC +0.085 pooled; H1/H2 split is mixed — treat with caution.
- **UP** (> 0%) and **DRAWDOWN** (< -15%): no demonstrated edge; scan suppresses action labels.

**Do not remove the regime gate without backtest evidence supporting expansion.**
Expanding to UP or DRAWDOWN regimes requires a fresh walk-forward showing IC > 0.05 in those
buckets. Use `--force-all-regimes` only for inspection, never for live signal generation.

The `classify_regime()` function and `TRADEABLE_REGIMES` constant live in
`scanner/signals/base.py` — edit thresholds there if evidence changes.

## Confirmation Score Backtest Results (May 2026)

Signal: Cross-signal confirmation score (0-4, Est Rev pending)
Scope: MSFT/META dependents in CORRECTION regime only
Date range: 2023-08-01 to 2026-04-14

### Findings
- 10d: IC +0.122, H1 +0.122, H2 +0.122 — PASS
- 20d: IC +0.133, H1 +0.254, H2 +0.064 — PASS
- 5d: IC +0.101, H1 -0.009, H2 +0.155 — FAIL (H1 negative)
- Confirmation score beats RS alone at all horizons
- Bucket spread (20d): 0-1/4 -> +4.96%, 2/4 -> +6.65%, 3-4/4 -> +10.74%

### Caveats
- 3-4 bucket has only 27 observations — directional only, not statistically robust until 100+
- Est Rev excluded from scoring — max score currently 4, not 5
- QQQ RS fallback used — own 20d return > 0 as proxy
- Survivorship bias caveat applies

### Current Status
- Informational display only — does NOT modify action labels yet
- Revisit when 3-4 bucket reaches 100+ observations
- Revisit when Est Rev backtest unlocks (~May 30th)

### Promotion Rules (same evidentiary bar as RS signal)
- Only promote to composite modifier after 3-4 bucket reaches 100+ observations
- Must pass IC > 0.05 with both halves positive at that point
- Add for MSFT/META dependents only first
- Do not apply to unvalidated parents without separate backtest evidence

## Rotation Backtest Results (May 2026)

Signal: XLK sector rotation status (rank among XLF, XLE, XLV by 20d+60d composite)
Scope: MSFT/META dependents in CORRECTION regime only

### Findings
- Tech LEADING rotation REDUCES IC at 20d (-0.023 vs baseline +0.084)
- Tech NOT-leading rotation IMPROVES IC at 20d (+0.115)
- H1/H2 split for LEADING bucket inconsistent — not reliable as gate
- Counterintuitive finding: macro-driven corrections produce cleaner RS signals
  than company-specific corrections

### Decision
- Rotation banner stays display only — do NOT add rotation as signal gate or composite modifier
- Do not use rotation status to suppress or amplify action labels
- Revisit only with significantly larger sample size

## Earnings Proximity Warning

Earnings proximity is **display-only enrichment** — it annotates the `scan` and `scan-megacap`
displays with an "Earnings" column but does NOT suppress action labels, modify signal scores,
or interact with the regime gate. It is a warning only.

- Source: yfinance `earnings_dates` DataFrame, filtered to dates >= today, taking the minimum.
- yfinance earnings dates are sometimes missing or unreliable. **Always prefer NULL over a wrong date.**
  If the fetch fails, log a warning, store NULL, and continue — never crash the scan.
- Stored in `earnings_dates(ticker, next_earnings_date, ingested_at)`, PK `(ticker)`. One row per ticker.
- Refreshed daily at 9:15 AM ET by the `ingest_earnings` APScheduler job.
- Display thresholds: >14d → blank; 8-14d → "EARNINGS 10d" (yellow); 3-7d → "EARNINGS 5d" (orange);
  1-2d → "EARNINGS TOMORROW" (red); 0d → "EARNINGS TODAY" (red).
- Never use earnings proximity as a signal gate or composite modifier.

## Insider Hybrid Architecture

- **Massive API** = primary source for Form 4 transaction listing (uses `issuer_cik` param + `filing_date.gte` filter + `next_url` pagination)
- **SEC EDGAR XML** = footnotes only, fetched only for accessions that contain at least one P (purchase) transaction
- S-only accessions insert directly from Massive data — no XML fetch, no rate limit hit on EDGAR
- CIK cached in `ticker_cik_map` table (populated from EDGAR `company_tickers.json` via `_load_cik_map`)
- `SEC_USER_AGENT` still required for all EDGAR XML calls; `MASSIVE_API_KEY` required for Massive calls
- Fallback to EDGAR-only when `MASSIVE_API_KEY` is absent
- **PRLD canonical test must pass** after any changes to `insiders.py`: `python -m scanner ingest-insiders PRLD` and verify OrbiMed/Baker Bros buys are classified as `is_open_market = False`
- `parse_form4_xml()` and the offering-participation filter are unchanged from the EDGAR-only implementation — do not modify without re-running PRLD test

## Discovery Pipeline Invariants

- `discovery_candidates` is append-only for audit trail (never DELETE rows; use status transitions)
- Promotion requires `ic_h1 > 0.05` AND `ic_h2 > 0.05` — both halves positive (same bar as RS signal)
- Failed tickers re-eligible for re-discovery after 90 days only
- Claude confidence threshold: `>= 0.70`; dependency strength: `STRONG` or `MEDIUM` only
- Dependency parents: must be in `MEGA_CAPS` list from `graph/loader.py`
- **10-K sections**: use `business` + `risk_factors` via Massive `/stocks/filings/10-K/vX/sections`
  (`customer_concentration` section is not available — returns empty results)
- **Universe fetch**: Massive `/v3/reference/tickers?exchange=XNAS&type=CS&active=true` (no market_cap sort available)
- Claude model: `claude-haiku-4-5-20251001` for 10-K analysis — Haiku only, never Sonnet for discovery
- Scheduler: weekly discovery Sunday 6 AM ET; daily RS accumulation 9:45 AM ET
- `_load_edges` is `@lru_cache` — `_append_to_dependencies_yaml` calls `_load_edges.cache_clear()` after write; restart process to pick up new edges in live server
- Auto-promotion writes to `dependencies.yaml` directly — validate YAML format after any manual edits
- IC backtest uses 5-day forward returns from `prices` table; minimum 10 observation pairs required
- n=60 RS observations is the minimum to trigger backtest, not a statistical guarantee — treat promoted tickers as Watch candidates until 100+ observations accumulate in production

## Survivorship-Bias Warning

yfinance only returns currently-listed tickers. Any backtest result using yfinance data
overstates Sharpe and understates drawdowns because delisted/bankrupt tickers are absent.
Document this clearly. Migrate to Polygon.io (which provides delisted ticker history)
before trusting any live performance numbers.

## Test Conventions

```bash
pip install -e ".[dev]"
pytest
pytest -v tests/test_signals.py   # run a specific module
pytest --cov=scanner              # with coverage
```

- Never make real network calls in tests. Mock `yfinance.Ticker` and `_fetch_ohlcv`.
- Use in-memory DuckDB (`duckdb.connect(":memory:")`) in all tests — never the data/ file.
- `scanner.graph.loader._load_edges` is `@lru_cache` — use `_load_edges.cache_clear()` if a
  test modifies the YAML (prefer not to; test with the real YAML).

## Railway Deployment Notes

### DATABASE_PATH
- Must be set as environment variable: `/data/scanner.duckdb`
- Leading slash required — points to persistent volume
- Variable name must be `DATABASE_PATH` not `DATABASE_URL`

### ANTHROPIC_API_KEY
- Must be set in Railway Variables tab
- Must NOT have quotes or extra spaces
- Starts with `sk-ant-`
- 401 error means key is wrong or has extra characters

### Ingestion
- `ingest-filings` default 7 days too narrow for weekends; use `--days-back 14` for initial backfill on fresh deployment
- Scan uses `MAX(date)` from prices not `CURRENT_DATE` — works correctly on weekends and holidays

### Fresh Deployment Checklist
1. Set `DATABASE_PATH=/data/scanner.duckdb`
2. Set `SEC_USER_AGENT=name email@domain.com`
3. Set `ANTHROPIC_API_KEY=sk-ant-...`
4. Add persistent volume mounted at `/data`
5. Deploy
6. Click Refresh All from Scheduler page
7. Wait ~3 minutes for full completion
8. Scanner page loads with all data

### Admin Endpoints (temporary, for maintenance)
- `GET /api/admin/clear-all-filings`
- `GET /api/admin/clear-high-filings`
- `GET /api/admin/reingest-filings-14d`
- `GET /api/admin/generate-summaries`
- `GET /api/admin/test-anthropic`
- `GET /api/debug`
