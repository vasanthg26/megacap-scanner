"""Insider Transaction IC Backtest.

Measures whether Form 4 insider buying/selling predicts forward returns
across the full scanner universe (dependency graph + theme basket).

Usage:
    python -m scanner.backtest.insider_runner
    python -m scanner.backtest.insider_runner --start 2022-01-01
    python -m scanner.backtest.insider_runner --start 2023-01-01 --end 2025-12-31

Signal classification (30-day lookback):
    CLUSTER_BUY (+2) — 3+ distinct insiders with open-market purchases
    BUY_SIGNAL  (+1) — 1-2 insiders with open-market purchases
    NEUTRAL      (0) — no transactions
    SELL_SIGNAL (-1) — open-market sales only, no purchases

Only transaction_code='P' AND is_open_market=TRUE counts as a buy.
Offering-participation purchases (is_open_market=FALSE) are excluded.

PASS criteria: IC > 0.05 pooled, both H1 and H2 positive, n >= 100.
"""

import bisect
import logging
import math
import argparse
from datetime import date as _date, timedelta

from scanner.db import get_connection
from scanner.graph.loader import get_all_tickers, get_dependents, get_parents
from scanner.signals.base import VALIDATED_PARENTS, classify_regime

logger = logging.getLogger(__name__)

_HORIZONS = (5, 10, 20)
_IC_THRESHOLD = 0.05
_MIN_OBS = 100
_WINDOW_DAYS = 30
_STEP_DAYS = 5  # weekly resampling
_SIGNAL_LABELS = ["CLUSTER_BUY", "BUY_SIGNAL", "NEUTRAL", "SELL_SIGNAL"]


# ---------- data loading --------------------------------------------------------

def _get_universe(conn) -> list[str]:
    """All tickers with both insider transaction data and price history."""
    graph_tickers: set[str] = set(get_all_tickers())
    try:
        theme_rows = conn.execute(
            "SELECT DISTINCT ticker FROM themes WHERE is_active = TRUE"
        ).fetchall()
        theme_tickers = {r[0] for r in theme_rows}
    except Exception:
        theme_tickers = set()
    candidates = graph_tickers | theme_tickers

    with_insiders = {
        r[0] for r in conn.execute("SELECT DISTINCT ticker FROM insider_transactions").fetchall()
    }
    with_prices = {
        r[0] for r in conn.execute("SELECT DISTINCT ticker FROM prices").fetchall()
    }
    return sorted(candidates & with_insiders & with_prices)


def _validated_parent_dependents() -> set[str]:
    tickers: set[str] = set()
    for parent in VALIDATED_PARENTS:
        for edge in get_dependents(parent):
            tickers.add(edge.child)
    return tickers


def _ticker_regime_parent(ticker: str) -> str:
    """Return the VALIDATED_PARENTS parent of a ticker, or MSFT as market proxy."""
    for edge in get_parents(ticker):
        if edge.parent in VALIDATED_PARENTS:
            return edge.parent
    return "MSFT"


def _load_prices(
    conn, tickers: list[str], start: str, end: str
) -> dict[str, tuple[list[str], list[float]]]:
    """Return {ticker: ([dates_asc], [closes_asc])} extended 35 cal-days past end."""
    end_ext = (_date.fromisoformat(end) + timedelta(days=35)).isoformat()
    placeholders = ",".join("?" for _ in tickers)
    rows = conn.execute(
        f"SELECT ticker, date, adj_close FROM prices "
        f"WHERE ticker IN ({placeholders}) AND date >= ? AND date <= ? "
        f"ORDER BY ticker, date",
        tickers + [start, end_ext],
    ).fetchall()

    from collections import defaultdict
    raw: dict[str, list] = defaultdict(list)
    for ticker, d, c in rows:
        raw[ticker].append((str(d), float(c) if c is not None else float("nan")))

    result: dict[str, tuple[list[str], list[float]]] = {}
    for ticker, pairs in raw.items():
        dates = [p[0] for p in pairs]
        closes = [p[1] for p in pairs]
        result[ticker] = (dates, closes)
    return result


def _load_transactions(conn, start: str, end: str) -> dict[str, list[dict]]:
    """Return {ticker: [{date, code, is_open_market, insider_name}]} for buy/sell codes.

    Loaded from 35 calendar days before start so the first signal date's
    30-day lookback window is fully covered.
    """
    start_ext = (_date.fromisoformat(start) - timedelta(days=35)).isoformat()
    rows = conn.execute(
        """
        SELECT ticker, transaction_date, transaction_code, is_open_market, insider_name
        FROM insider_transactions
        WHERE transaction_date IS NOT NULL
          AND transaction_date >= ?
          AND transaction_date <= ?
          AND transaction_code IN ('P', 'S')
        ORDER BY ticker, transaction_date
        """,
        [start_ext, end],
    ).fetchall()

    from collections import defaultdict
    txns: dict[str, list[dict]] = defaultdict(list)
    for ticker, d, code, is_om, name in rows:
        txns[ticker].append(
            {
                "date": str(d),
                "code": code,
                "is_open_market": bool(is_om) if is_om is not None else None,
                "insider_name": name or "",
            }
        )
    return dict(txns)


def _get_signal_dates(conn, start: str, end: str) -> list[str]:
    """All trading dates at weekly (5-day) intervals in [start, end]."""
    rows = conn.execute(
        "SELECT DISTINCT date FROM prices WHERE date >= ? AND date <= ? ORDER BY date",
        [start, end],
    ).fetchall()
    dates = [str(r[0]) for r in rows]
    return dates[::_STEP_DAYS]


def _precompute_parent_returns(
    signal_dates: list[str],
    parents: list[str],
    prices: dict[str, tuple[list[str], list[float]]],
) -> dict[tuple[str, str], float]:
    """Precompute 20d return for each (parent, signal_date) pair."""
    out: dict[tuple[str, str], float] = {}
    for parent in parents:
        if parent not in prices:
            continue
        date_list, close_list = prices[parent]
        for sd in signal_dates:
            idx = bisect.bisect_right(date_list, sd) - 1  # last date <= sd
            if idx < 20:
                out[(parent, sd)] = float("nan")
                continue
            today_c = close_list[idx]
            past_c = close_list[idx - 20]
            if not past_c or math.isnan(today_c) or math.isnan(past_c):
                out[(parent, sd)] = float("nan")
                continue
            out[(parent, sd)] = (today_c - past_c) / past_c
    return out


# ---------- per-observation computation ----------------------------------------

def _insider_signal(
    ticker: str, signal_date: str, txns_by_ticker: dict[str, list[dict]]
) -> tuple[str, int]:
    """Classify insider signal as of signal_date (30-day lookback)."""
    txns = txns_by_ticker.get(ticker)
    if not txns:
        return "NEUTRAL", 0

    cutoff = (_date.fromisoformat(signal_date) - timedelta(days=_WINDOW_DAYS)).isoformat()

    buyers: set[str] = set()
    has_sell = False

    for t in txns:
        if t["date"] > signal_date or t["date"] <= cutoff:
            continue
        if t["code"] == "P" and t["is_open_market"] is True:
            buyers.add(t["insider_name"])
        elif t["code"] == "S":
            has_sell = True

    n_buyers = len(buyers)
    if n_buyers >= 3:
        return "CLUSTER_BUY", 2
    if n_buyers >= 1:
        return "BUY_SIGNAL", 1
    if has_sell:
        return "SELL_SIGNAL", -1
    return "NEUTRAL", 0


def _forward_return(
    ticker: str, signal_date: str, horizon: int, prices: dict[str, tuple[list[str], list[float]]]
) -> float | None:
    """Forward return: entry = day+1 close, exit = day+horizon close."""
    entry_data = prices.get(ticker)
    if not entry_data:
        return None
    date_list, close_list = entry_data
    idx = bisect.bisect_right(date_list, signal_date)  # first date > signal_date
    if idx + horizon - 1 >= len(date_list):
        return None
    entry = close_list[idx]
    exit_ = close_list[idx + horizon - 1]
    if not entry or math.isnan(entry) or math.isnan(exit_):
        return None
    return (exit_ - entry) / entry


# ---------- IC and output -------------------------------------------------------

def _spearman_ic(pairs: list[tuple[float, float]]) -> float:
    """Spearman rank correlation for (signal, return) pairs."""
    n = len(pairs)
    if n < 5:
        return float("nan")

    def _rank(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        for r, i in enumerate(order):
            ranks[i] = float(r + 1)
        return ranks

    signals = [p[0] for p in pairs]
    returns = [p[1] for p in pairs]
    rs = _rank(signals)
    rr = _rank(returns)
    mean_s = sum(rs) / n
    mean_r = sum(rr) / n
    cov = sum((rs[i] - mean_s) * (rr[i] - mean_r) for i in range(n)) / n
    std_s = math.sqrt(sum((x - mean_s) ** 2 for x in rs) / n)
    std_r = math.sqrt(sum((x - mean_r) ** 2 for x in rr) / n)
    if std_s == 0 or std_r == 0:
        return float("nan")
    return cov / (std_s * std_r)


def _compute_ic(obs: list[dict], horizon: int) -> tuple[float, int]:
    pairs = [
        (o["score"], o[f"fwd_{horizon}d"])
        for o in obs
        if o.get(f"fwd_{horizon}d") is not None
    ]
    return _spearman_ic(pairs), len(pairs)


def _h1_h2_split(obs: list[dict]) -> tuple[list[dict], list[dict]]:
    dates = sorted({o["date"] for o in obs})
    if not dates:
        return [], []
    mid = dates[len(dates) // 2]
    return [o for o in obs if o["date"] < mid], [o for o in obs if o["date"] >= mid]


def _signal_distribution(obs: list[dict]) -> dict[str, int]:
    counts = {label: 0 for label in _SIGNAL_LABELS}
    for o in obs:
        counts[o["label"]] += 1
    return counts


def _pass_fail(ic_pool: float, ic_h1: float, ic_h2: float, n: int) -> str:
    if n < _MIN_OBS:
        return f"INSUF(n={n})"
    if any(math.isnan(v) for v in (ic_pool, ic_h1, ic_h2)):
        return "N/A"
    if ic_pool > _IC_THRESHOLD and ic_h1 > 0 and ic_h2 > 0:
        return "PASS"
    return "FAIL"


def _fmt_ic(v: float) -> str:
    return f"{v:+.3f}" if not math.isnan(v) else "   N/A"


def _print_version(title: str, obs: list[dict]) -> None:
    print(f"\n{title}")
    print("-" * 62)
    if not obs:
        print("  No observations.")
        return

    dist = _signal_distribution(obs)
    total = len(obs)
    h1, h2 = _h1_h2_split(obs)

    print(f"Total observations: {total:,}")
    print("Signal distribution:")
    for label in _SIGNAL_LABELS:
        n = dist[label]
        pct = n / total * 100 if total else 0
        print(f"  {label:<12}: {n:>5} ({pct:4.1f}%)")

    print()
    print(f" {'Horizon':<8} {'IC pooled':>10} {'H1':>8} {'H2':>8}    Status")

    for h in _HORIZONS:
        ic_pool, n_pool = _compute_ic(obs, h)
        ic_h1, _ = _compute_ic(h1, h)
        ic_h2, _ = _compute_ic(h2, h)
        status = _pass_fail(ic_pool, ic_h1, ic_h2, n_pool)
        print(f" {h}d       {_fmt_ic(ic_pool):>10} {_fmt_ic(ic_h1):>8} {_fmt_ic(ic_h2):>8}   {status}")


# ---------- main ----------------------------------------------------------------

def run(start: str = "2023-01-01", end: str | None = None) -> None:
    if end is None:
        end = _date.today().isoformat()

    conn = get_connection()

    print("\nInsider Transaction IC Backtest")
    print("=" * 34)
    print(f"Period      : {start} to {end}")
    print(f"Lookback    : {_WINDOW_DAYS} days")
    print(f"Resampling  : every {_STEP_DAYS} trading days (weekly)")
    print(f"PASS criteria: IC > {_IC_THRESHOLD} pooled, H1 > 0, H2 > 0, n >= {_MIN_OBS}")

    universe = _get_universe(conn)
    vp_dependents = _validated_parent_dependents()
    vp_in_universe = set(universe) & vp_dependents

    print(f"\nUniverse    : {len(universe)} tickers (with insider + price data)")
    print(f"VP dependents in universe: {len(vp_in_universe)} tickers")

    signal_dates = _get_signal_dates(conn, start, end)
    print(f"Signal dates: {len(signal_dates)}")

    # Preload all data into memory
    print("\nLoading prices...")
    all_price_tickers = list(set(universe) | set(VALIDATED_PARENTS) | {"MSFT"})
    prices = _load_prices(conn, all_price_tickers, start, end)

    print("Loading insider transactions...")
    txns = _load_transactions(conn, start, end)

    # Precompute per-ticker regime parents and parent 20d returns
    print("Precomputing parent regime returns...")
    unique_regime_parents = list(set(VALIDATED_PARENTS) | {"MSFT"})
    parent_returns = _precompute_parent_returns(signal_dates, unique_regime_parents, prices)

    regime_parent_map = {ticker: _ticker_regime_parent(ticker) for ticker in universe}

    # Build all observations
    print("Building observations...")
    all_obs: list[dict] = []

    for i, ticker in enumerate(universe):
        if i % 20 == 0:
            logger.info("  %s (%d/%d)", ticker, i + 1, len(universe))

        rp = regime_parent_map[ticker]

        for signal_date in signal_dates:
            label, score = _insider_signal(ticker, signal_date, txns)

            fwds = {h: _forward_return(ticker, signal_date, h, prices) for h in _HORIZONS}
            if all(v is None for v in fwds.values()):
                continue

            parent_ret = parent_returns.get((rp, signal_date), float("nan"))
            regime = classify_regime(parent_ret)

            obs: dict = {
                "ticker": ticker,
                "date": signal_date,
                "label": label,
                "score": score,
                "regime": regime,
                "in_vp": ticker in vp_dependents,
            }
            for h in _HORIZONS:
                obs[f"fwd_{h}d"] = fwds[h]
            all_obs.append(obs)

    print(f"Total observations: {len(all_obs):,}")

    # VERSION A: Full universe, all regimes
    _print_version("VERSION A — Full Universe", all_obs)

    # VERSION B: CORRECTION regime only (parent 20d return in [-15%, -5%])
    correction_obs = [o for o in all_obs if o["regime"] == "CORRECTION"]
    _print_version("VERSION B — CORRECTION Regime Only", correction_obs)

    # VERSION C: Cluster buys only (signal = +2)
    cluster_obs = [o for o in all_obs if o["score"] == 2]
    _print_version("VERSION C — Cluster Buys Only (signal=+2)", cluster_obs)

    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Insider transaction IC backtest")
    parser.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD, default: today)")
    args = parser.parse_args()
    run(start=args.start, end=args.end)
