"""Divergence flag detection — surfaces tickers rising while market/benchmark falls."""

import logging

import duckdb

logger = logging.getLogger(__name__)


def _batch_1d_return(
    tickers: list[str], as_of: str, conn: duckdb.DuckDBPyConnection
) -> dict[str, float]:
    """1-day return for each ticker: (today - yesterday) / yesterday."""
    if not tickers:
        return {}
    placeholders = ",".join("?" for _ in tickers)
    rows = conn.execute(
        f"""
        SELECT ticker, adj_close, rn FROM (
            SELECT ticker, adj_close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices WHERE ticker IN ({placeholders}) AND date <= ?
        ) t WHERE rn <= 2
        """,
        tickers + [as_of],
    ).fetchall()
    by_ticker: dict[str, list] = {}
    for tkr, c, rn in rows:
        if tkr not in by_ticker:
            by_ticker[tkr] = [None, None]
        if rn == 1:
            by_ticker[tkr][0] = float(c) if c is not None else None
        elif rn == 2:
            by_ticker[tkr][1] = float(c) if c is not None else None
    result: dict[str, float] = {}
    for tkr, closes in by_ticker.items():
        today, yesterday = closes
        if today is None or yesterday is None or yesterday == 0:
            continue
        result[tkr] = (today - yesterday) / yesterday
    return result


def _classify_divergence(
    ticker: str,
    ticker_1d: float,
    spy_1d: float,
    benchmark: str,
    benchmark_1d: float,
    theme: str | None,
    parent: str | None,
    adaptive: bool,
) -> dict | None:
    """Return a flag dict if the ticker qualifies, else None."""
    divergence_vs_spy = ticker_1d - spy_1d
    divergence_vs_benchmark = ticker_1d - benchmark_1d

    strength: str | None = None

    if adaptive:
        # SPY > +1%: require ticker >= SPY + 3% relative outperformance
        if divergence_vs_spy >= 0.03 and divergence_vs_benchmark >= 0.02:
            strength = "STRONG"
        elif divergence_vs_spy >= 0.03 and divergence_vs_benchmark >= 0.03:
            strength = "MODERATE"
        elif ticker_1d >= 0.03 and benchmark_1d <= -0.01:
            strength = "BENCHMARK"
    else:
        if ticker_1d >= 0.02 and spy_1d <= 0.0 and divergence_vs_benchmark >= 0.02:
            strength = "STRONG"
        elif ticker_1d >= 0.02 and spy_1d <= 0.0 and divergence_vs_benchmark >= 0.03:
            strength = "MODERATE"
        elif ticker_1d >= 0.03 and benchmark_1d <= -0.01:
            strength = "BENCHMARK"

    if strength is None:
        return None

    return {
        "ticker": ticker,
        "ticker_1d": round(ticker_1d * 100, 2),
        "spy_1d": round(spy_1d * 100, 2),
        "benchmark": benchmark,
        "benchmark_1d": round(benchmark_1d * 100, 2),
        "divergence_vs_spy": round(divergence_vs_spy * 100, 2),
        "divergence_vs_benchmark": round(divergence_vs_benchmark * 100, 2),
        "strength": strength,
        "theme": theme,
        "parent": parent,
    }


def find_divergences(
    conn: duckdb.DuckDBPyConnection, as_of: str | None = None
) -> list[dict]:
    """Find tickers that rose while their benchmark fell on the given date.

    For dependency tickers, the benchmark is the primary parent.
    For theme tickers not in the dependency graph, the benchmark is the theme ETF.

    Returns list of flag dicts sorted by divergence_vs_benchmark descending.
    """
    if as_of is None:
        row = conn.execute(
            "SELECT MAX(date) FROM prices WHERE ticker = 'SPY'"
        ).fetchone()
        if not row or row[0] is None:
            return []
        as_of = str(row[0])

    try:
        from scanner.graph.loader import _load_edges
        edges = _load_edges()
    except Exception as exc:
        logger.error("find_divergences: failed to load edges: %s", exc)
        edges = []

    # Pick highest-weight parent per child
    dep_parents: dict[str, tuple[str, float]] = {}
    for edge in edges:
        child = edge.child
        if child not in dep_parents or edge.weight > dep_parents[child][1]:
            dep_parents[child] = (edge.parent, edge.weight)

    # Theme tickers with their ETF benchmarks and labels
    try:
        theme_rows = conn.execute(
            "SELECT ticker, benchmark_etf, theme_label FROM themes WHERE is_active = true"
        ).fetchall()
    except Exception as exc:
        logger.error("find_divergences: failed to query themes: %s", exc)
        theme_rows = []
    theme_map: dict[str, tuple[str, str]] = {}
    for tkr, bench, label in theme_rows:
        theme_map[tkr] = (bench, label)

    all_universe = list(set(list(dep_parents.keys()) + list(theme_map.keys())))
    parent_benchmarks = list({p for p, _ in dep_parents.values()})
    etf_benchmarks = list({b for b, _ in theme_map.values()})
    all_fetch = list(set(all_universe + parent_benchmarks + etf_benchmarks + ["SPY"]))

    returns_1d = _batch_1d_return(all_fetch, as_of, conn)

    spy_1d = returns_1d.get("SPY")
    if spy_1d is None:
        logger.warning("find_divergences: no SPY 1d return for %s — skipping", as_of)
        return []

    # SPY > +1%: switch to relative mode
    adaptive = spy_1d > 0.01

    flags: list[dict] = []
    processed: set[str] = set()

    for tkr, (parent, _) in dep_parents.items():
        processed.add(tkr)
        ticker_1d = returns_1d.get(tkr)
        if ticker_1d is None:
            continue
        benchmark_1d = returns_1d.get(parent)
        if benchmark_1d is None:
            continue
        theme_label = theme_map.get(tkr, (None, None))[1]
        flag = _classify_divergence(
            ticker=tkr,
            ticker_1d=ticker_1d,
            spy_1d=spy_1d,
            benchmark=parent,
            benchmark_1d=benchmark_1d,
            theme=theme_label,
            parent=parent,
            adaptive=adaptive,
        )
        if flag:
            flags.append(flag)

    # Theme-only tickers (not in dependency graph)
    for tkr, (bench, label) in theme_map.items():
        if tkr in processed:
            continue
        ticker_1d = returns_1d.get(tkr)
        if ticker_1d is None:
            continue
        benchmark_1d = returns_1d.get(bench)
        if benchmark_1d is None:
            continue
        flag = _classify_divergence(
            ticker=tkr,
            ticker_1d=ticker_1d,
            spy_1d=spy_1d,
            benchmark=bench,
            benchmark_1d=benchmark_1d,
            theme=label,
            parent=None,
            adaptive=adaptive,
        )
        if flag:
            flags.append(flag)

    flags.sort(key=lambda f: f["divergence_vs_benchmark"], reverse=True)
    return flags[:7]


def store_divergences(
    flags: list[dict], flag_date: str, conn: duckdb.DuckDBPyConnection
) -> None:
    """Upsert divergence flags into divergence_flags table and prune rows >30 days old."""
    for f in flags:
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO divergence_flags
                    (ticker, flag_date, ticker_1d, spy_1d, benchmark, benchmark_1d,
                     divergence_vs_spy, divergence_vs_benchmark, strength, theme, parent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    f["ticker"], flag_date, f["ticker_1d"], f["spy_1d"],
                    f["benchmark"], f["benchmark_1d"], f["divergence_vs_spy"],
                    f["divergence_vs_benchmark"], f["strength"], f["theme"], f["parent"],
                ],
            )
        except Exception as exc:
            logger.error("store_divergences: failed for %s/%s: %s", f["ticker"], flag_date, exc)

    try:
        conn.execute(
            "DELETE FROM divergence_flags WHERE flag_date < CURRENT_DATE - INTERVAL 30 DAYS"
        )
    except Exception as exc:
        logger.error("store_divergences: prune failed: %s", exc)
