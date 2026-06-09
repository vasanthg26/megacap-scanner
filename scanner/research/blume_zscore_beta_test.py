"""Research script: Blume beta Z-score theme signal enhancement test.

Tests whether multiplying the cross-sectional Z-score signal by Z_beta or
Z_beta_blume (Blume-adjusted) improves IC over the current Z_RS x RVOL baseline.

Usage:
    python scanner/research/blume_zscore_beta_test.py

Read-only. Does NOT modify any production table or code.
"""

import bisect
import math
import os
from collections import defaultdict
from pathlib import Path

import duckdb

DB_PATH = Path(
    os.environ.get(
        "DATABASE_PATH",
        str(Path(__file__).parent.parent.parent / "data" / "scanner.duckdb"),
    )
)

# -- Signal parameters ----------------------------------------------------------
_RS_LOOKBACK = 20    # days for cross-sectional return used in Z_RS
_BETA_WINDOW = 60    # trailing days for rolling beta (cov/var)
_FWD_HORIZON = 20   # forward return horizon for IC
_MIN_BASKET = 3      # minimum tickers with data on a signal date
_MIN_OBS_HALF = 30   # minimum observations per half for IC to be reported

# -- Baskets --------------------------------------------------------------------
BASKETS = [
    {
        "name": "AI Power Semiconductors",
        "tickers": ["BE", "FPS", "CEG", "VST", "HUT"],
        "benchmark": "GRID",
    },
    {
        "name": "Developer Platforms & Edge Compute",
        "tickers": ["DOCN", "NET", "AKAM", "DDOG"],
        "benchmark": "WCLD",
    },
    {
        "name": "Crypto / Bitcoin Infra",
        "tickers": ["HUT", "WGMI"],
        "benchmark": "WGMI",
    },
    {
        "name": "Critical Minerals & Battery Materials",
        "tickers": ["SQM", "ALB", "ALTM"],
        "benchmark": "LIT",
    },
]

# -- Utilities ------------------------------------------------------------------

def _mean(lst: list[float]) -> float:
    return sum(lst) / len(lst) if lst else float("nan")


def _std(lst: list[float]) -> float:
    if len(lst) < 2:
        return float("nan")
    m = _mean(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / len(lst))


def _spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 5:
        return float("nan")

    def _rank(lst: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: lst[i])
        r = [0.0] * n
        for rk, i in enumerate(order):
            r[i] = float(rk + 1)
        return r

    rs, rr = _rank(xs), _rank(ys)
    ms, mr = _mean(rs), _mean(rr)
    cov = sum((rs[i] - ms) * (rr[i] - mr) for i in range(n)) / n
    ss = math.sqrt(sum((x - ms) ** 2 for x in rs) / n)
    sr = math.sqrt(sum((x - mr) ** 2 for x in rr) / n)
    if ss == 0 or sr == 0:
        return float("nan")
    return cov / (ss * sr)


def _fmt(v: float, width: int = 7) -> str:
    if math.isnan(v):
        return f"{'n/a':>{width}}"
    return f"{v:>+{width}.3f}"


def _stable(h1: float, h2: float) -> str:
    if math.isnan(h1) or math.isnan(h2):
        return "?"
    if h1 > 0.05 and h2 > 0.05:
        return "YES"
    if h1 > 0 and h2 > 0:
        return "weak"
    return "NO"


# -- Price data loader ----------------------------------------------------------

def _load_prices(
    conn: duckdb.DuckDBPyConnection, tickers: list[str]
) -> tuple[
    dict[str, list[str]],     # ticker -> sorted date strings
    dict[str, list[float]],   # ticker -> closes (same order)
    dict[str, list[float]],   # ticker -> volumes (same order)
]:
    placeholders = ",".join("?" for _ in tickers)
    rows = conn.execute(
        f"SELECT ticker, CAST(date AS VARCHAR), adj_close, volume "
        f"FROM prices WHERE ticker IN ({placeholders}) ORDER BY ticker, date",
        tickers,
    ).fetchall()

    dates: dict[str, list[str]] = defaultdict(list)
    closes: dict[str, list[float]] = defaultdict(list)
    vols: dict[str, list[float]] = defaultdict(list)
    for tkr, dt, c, v in rows:
        dates[tkr].append(dt)
        closes[tkr].append(float(c))
        vols[tkr].append(float(v) if v is not None else 0.0)
    return dict(dates), dict(closes), dict(vols)


# -- Per-ticker beta at a signal date ------------------------------------------

def _rolling_beta(
    ticker: str,
    signal_date: str,
    ticker_dates: dict[str, list[str]],
    ticker_closes: dict[str, list[float]],
    spy_dates: list[str],
    spy_closes: list[float],
    window: int = _BETA_WINDOW,
) -> float | None:
    """Compute OLS beta of ticker vs SPY using `window` trailing daily returns.

    Returns None if insufficient data.  Uses sample covariance (n-1 denominator),
    consistent with beta_rs.py.
    """
    t_dates = ticker_dates.get(ticker, [])
    t_closes = ticker_closes.get(ticker, [])

    # Find position of signal_date in each series
    t_idx = bisect.bisect_right(t_dates, signal_date) - 1
    s_idx = bisect.bisect_right(spy_dates, signal_date) - 1

    if t_idx < 0 or t_dates[t_idx] != signal_date:
        return None
    if s_idx < 0 or spy_dates[s_idx] != signal_date:
        return None

    # Need window+1 prices for window daily returns
    if t_idx < window or s_idx < window:
        return None

    t_px = t_closes[t_idx - window : t_idx + 1]   # window+1 prices, oldest first
    s_px = spy_closes[s_idx - window : s_idx + 1]

    t_rets = [(t_px[i + 1] - t_px[i]) / t_px[i] for i in range(window) if t_px[i] != 0]
    s_rets = [(s_px[i + 1] - s_px[i]) / s_px[i] for i in range(window) if s_px[i] != 0]

    if len(t_rets) < window or len(s_rets) < window:
        return None

    n = window
    mean_t, mean_s = _mean(t_rets), _mean(s_rets)
    cov = sum((t_rets[i] - mean_t) * (s_rets[i] - mean_s) for i in range(n)) / (n - 1)
    var_s = sum((r - mean_s) ** 2 for r in s_rets) / (n - 1)
    if var_s == 0:
        return None
    return cov / var_s


# -- Observation builder --------------------------------------------------------

def _build_observations(basket: dict, conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """Build one observation dict per (signal_date, ticker) for a basket.

    Each dict contains: date, ticker, z_rs, rvol, beta_raw, beta_blume, fwd_20d.
    Observations with missing forward return are excluded.
    """
    tickers = basket["tickers"]
    benchmark = basket["benchmark"]
    all_needed = list(dict.fromkeys(tickers + [benchmark, "SPY"]))

    t_dates, t_closes, t_vols = _load_prices(conn, all_needed)

    spy_dates = t_dates.get("SPY", [])
    spy_closes = t_closes.get("SPY", [])

    # Union of dates where >=1 basket ticker has data
    union_dates = sorted({dt for tkr in tickers for dt in t_dates.get(tkr, [])})

    observations: list[dict] = []

    for signal_date in union_dates:
        # -- Z_RS: 20-day return cross-sectional Z-score within basket ----------
        basket_rets_20: dict[str, float] = {}
        basket_rvols: dict[str, float] = {}
        basket_betas_raw: dict[str, float] = {}

        for tkr in tickers:
            dates = t_dates.get(tkr, [])
            closes = t_closes.get(tkr, [])
            vols = t_vols.get(tkr, [])

            idx = bisect.bisect_right(dates, signal_date) - 1
            if idx < 0 or dates[idx] != signal_date:
                continue
            if idx < _RS_LOOKBACK:
                continue

            c_now = closes[idx]
            c_20 = closes[idx - _RS_LOOKBACK]
            if c_20 == 0:
                continue
            basket_rets_20[tkr] = (c_now - c_20) / c_20

            # RVOL: today / 20-day avg
            avg_vol = _mean(vols[idx - _RS_LOOKBACK : idx])
            if avg_vol > 0:
                basket_rvols[tkr] = vols[idx] / avg_vol

            # Beta
            beta = _rolling_beta(tkr, signal_date, t_dates, t_closes, spy_dates, spy_closes)
            if beta is not None:
                basket_betas_raw[tkr] = beta

        # For baskets with < 3 tickers allow n=2, but Z-score will be degenerate (+/-1)
        min_needed = min(_MIN_BASKET, len(tickers))
        if len(basket_rets_20) < min_needed:
            continue

        # Cross-sectional Z-score of 20d return
        ret_vals = list(basket_rets_20.values())
        ret_mean = _mean(ret_vals)
        ret_std = _std(ret_vals)
        if ret_std == 0 or math.isnan(ret_std):
            continue

        # Cross-sectional Z-score of raw beta (only tickers with beta data)
        beta_vals = list(basket_betas_raw.values())
        beta_mean = _mean(beta_vals) if beta_vals else float("nan")
        beta_std = _std(beta_vals) if len(beta_vals) >= 2 else float("nan")

        blume_vals = [(0.67 * b + 0.33 * 1.0) for b in basket_betas_raw.values()]
        blume_mean = _mean(blume_vals) if blume_vals else float("nan")
        blume_std = _std(blume_vals) if len(blume_vals) >= 2 else float("nan")

        for tkr in tickers:
            if tkr not in basket_rets_20:
                continue

            z_rs = (basket_rets_20[tkr] - ret_mean) / ret_std
            rvol = basket_rvols.get(tkr, float("nan"))
            combined_base = z_rs * rvol if not math.isnan(rvol) else float("nan")

            # Beta Z-scores for this ticker
            beta_raw = basket_betas_raw.get(tkr)
            if beta_raw is not None and not math.isnan(beta_std) and beta_std > 0:
                z_beta = (beta_raw - beta_mean) / beta_std
                beta_blume = 0.67 * beta_raw + 0.33 * 1.0
                z_beta_blume = (beta_blume - blume_mean) / blume_std if not math.isnan(blume_std) and blume_std > 0 else float("nan")
            else:
                z_beta = float("nan")
                z_beta_blume = float("nan")
                beta_raw = float("nan")

            # Forward return
            dates = t_dates.get(tkr, [])
            closes = t_closes.get(tkr, [])
            idx = bisect.bisect_right(dates, signal_date) - 1
            exit_idx = idx + _FWD_HORIZON
            if exit_idx >= len(closes) or closes[idx] == 0:
                continue
            fwd_20d = (closes[exit_idx] - closes[idx]) / closes[idx]

            observations.append({
                "date": signal_date,
                "ticker": tkr,
                "z_rs": z_rs,
                "rvol": rvol,
                "combined_base": combined_base,                                  # Z_RS x RVOL
                "combined_beta": combined_base * z_beta if not math.isnan(combined_base) and not math.isnan(z_beta) else float("nan"),
                "combined_blume": combined_base * z_beta_blume if not math.isnan(combined_base) and not math.isnan(z_beta_blume) else float("nan"),
                "beta_raw": beta_raw,
                "beta_blume": 0.67 * beta_raw + 0.33 if not math.isnan(beta_raw) else float("nan"),
                "z_beta": z_beta,
                "z_beta_blume": z_beta_blume,
                "fwd_20d": fwd_20d,
            })

    return observations


# -- IC computation -------------------------------------------------------------

def _compute_ic_table(
    obs: list[dict],
    basket_name: str,
    n_tickers: int,
) -> dict:
    """Compute IC for all three variants.  Returns dict of results."""
    all_dates = sorted({o["date"] for o in obs})
    mid_date = all_dates[len(all_dates) // 2] if all_dates else ""

    def _pairs(signal_key: str, half: str = "full") -> list[tuple[float, float]]:
        result = []
        for o in obs:
            if half == "h1" and o["date"] >= mid_date:
                continue
            if half == "h2" and o["date"] < mid_date:
                continue
            s = o[signal_key]
            f = o["fwd_20d"]
            if not math.isnan(s):
                result.append((s, f))
        return result

    def _ic_safe(pairs: list[tuple[float, float]]) -> float:
        if len(pairs) < _MIN_OBS_HALF:
            return float("nan")
        xs, ys = zip(*pairs)
        return _spearman(list(xs), list(ys))

    variants = [
        ("Z_RS x RVOL",           "combined_base"),
        ("Z_RS x RVOL x Z_beta",  "combined_beta"),
        ("Z_RS x RVOL x Z_blume", "combined_blume"),
    ]

    rows = []
    for label, key in variants:
        full_ic = _ic_safe(_pairs(key, "full"))
        h1_ic   = _ic_safe(_pairs(key, "h1"))
        h2_ic   = _ic_safe(_pairs(key, "h2"))
        rows.append({
            "label": label,
            "key": key,
            "ic_full": full_ic,
            "h1": h1_ic,
            "h2": h2_ic,
            "stable": _stable(h1_ic, h2_ic),
            "n_full": len(_pairs(key, "full")),
        })

    return {
        "basket_name": basket_name,
        "n_tickers": n_tickers,
        "mid_date": mid_date,
        "start_date": all_dates[0] if all_dates else "",
        "end_date": all_dates[-1] if all_dates else "",
        "total_obs": len(obs),
        "variants": rows,
    }


# -- Beta diagnostics (current values) -----------------------------------------

def _beta_diagnostics(
    basket: dict, conn: duckdb.DuckDBPyConnection
) -> list[dict]:
    """Return current beta values for each ticker (as of latest shared date)."""
    tickers = basket["tickers"]
    all_needed = list(dict.fromkeys(tickers + ["SPY"]))
    t_dates, t_closes, _ = _load_prices(conn, all_needed)

    spy_dates = t_dates.get("SPY", [])
    spy_closes = t_closes.get("SPY", [])

    # Use latest date where all tickers (and SPY) have data
    common_dates = set(spy_dates)
    for tkr in tickers:
        common_dates &= set(t_dates.get(tkr, []))
    if not common_dates:
        return []
    as_of = sorted(common_dates)[-1]

    raw_betas: dict[str, float] = {}
    for tkr in tickers:
        b = _rolling_beta(tkr, as_of, t_dates, t_closes, spy_dates, spy_closes)
        if b is not None:
            raw_betas[tkr] = b

    beta_vals = list(raw_betas.values())
    beta_mean = _mean(beta_vals)
    beta_std = _std(beta_vals)
    blume_vals = [0.67 * b + 0.33 for b in beta_vals]
    blume_mean = _mean(blume_vals)
    blume_std = _std(blume_vals)

    rows = []
    for tkr in tickers:
        if tkr not in raw_betas:
            rows.append({"ticker": tkr, "beta_raw": float("nan"), "beta_blume": float("nan"),
                         "z_beta": float("nan"), "z_blume": float("nan")})
            continue
        br = raw_betas[tkr]
        bl = 0.67 * br + 0.33
        z_b = (br - beta_mean) / beta_std if beta_std > 0 else float("nan")
        z_bl = (bl - blume_mean) / blume_std if blume_std > 0 else float("nan")
        rows.append({"ticker": tkr, "beta_raw": br, "beta_blume": bl,
                     "z_beta": z_b, "z_blume": z_bl})
    return rows


# -- Printing helpers ----------------------------------------------------------

def _print_ic_table(result: dict) -> None:
    name = result["basket_name"]
    n = result["n_tickers"]
    print(f"\nBASKET: {name}  (n={n} tickers, {result['total_obs']} obs)")
    if n == 2:
        print("  !  Only 2 tickers - cross-sectional Z-score is unstable (n=2 always gives +/-1)")
    print(f"  Date range: {result['start_date']} to {result['end_date']}  |  H1/H2 split: {result['mid_date']}")
    print("=" * 68)
    print(f"{'Signal Variant':<28} {'IC Full':>8} {'H1':>8} {'H2':>8}  {'Stable?':<8} {'n'}")
    print("-" * 68)
    for r in result["variants"]:
        print(
            f"{r['label']:<28} {_fmt(r['ic_full'], 8)} {_fmt(r['h1'], 8)} {_fmt(r['h2'], 8)}"
            f"  {r['stable']:<8} {r['n_full']}"
        )
    print("-" * 68)

    # Determine winner: best IC Full among the three
    valid = [r for r in result["variants"] if not math.isnan(r["ic_full"])]
    if valid:
        winner = max(valid, key=lambda r: r["ic_full"])
        baseline = result["variants"][0]["ic_full"]
        delta = winner["ic_full"] - baseline
        print(f"Winner: {winner['label']}   (d vs baseline: {delta:+.3f})")
    else:
        print("Winner: insufficient data")


def _print_beta_diagnostics(basket_name: str, rows: list[dict]) -> None:
    print(f"\nBETA DIAGNOSTICS - {basket_name}  (trailing {_BETA_WINDOW}d, as of latest common date)")
    print(f"{'Ticker':<8} {'Beta_raw':>10} {'Beta_blume':>11} {'Z_beta':>8} {'Z_blume':>9}")
    print("-" * 52)
    for r in rows:
        print(
            f"{r['ticker']:<8}"
            f" {_fmt(r['beta_raw'], 10)}"
            f" {_fmt(r['beta_blume'], 11)}"
            f" {_fmt(r['z_beta'], 8)}"
            f" {_fmt(r['z_blume'], 9)}"
        )


def _print_recommendation(summary: list[dict]) -> None:
    print("\n\nRECOMMENDATION")
    print("=" * 58)
    print(f"{'Basket':<32} {'Best Signal':<26} {'IC d vs current':>16}")
    print("-" * 58)

    improvements: list[float] = []
    h2_improved = 0

    for s in summary:
        name = s["basket_name"]
        variants = s["variants"]
        baseline = variants[0]
        valid = [r for r in variants if not math.isnan(r["ic_full"])]
        if not valid:
            print(f"{name:<32} {'n/a':<26} {'n/a':>16}")
            continue
        winner = max(valid, key=lambda r: r["ic_full"])
        delta = winner["ic_full"] - baseline["ic_full"]
        improvements.append(delta)

        # Check H2 stability improvement vs baseline
        b_h2 = baseline["h2"]
        w_h2 = winner["h2"]
        if not math.isnan(b_h2) and not math.isnan(w_h2) and w_h2 > b_h2:
            h2_improved += 1

        print(f"{name:<32} {winner['label']:<26} {delta:>+16.3f}")

    print("-" * 58)

    valid_improvements = [d for d in improvements if not math.isnan(d)]
    if not valid_improvements:
        print("\nOverall: Add Blume Z_beta to themes? INSUFFICIENT DATA")
        return

    avg_improvement = _mean(valid_improvements)
    n_baskets = len(valid_improvements)

    # Promotion criteria
    avg_pass = avg_improvement > 0.01
    h2_pass = h2_improved >= (n_baskets / 2)
    verdict = "YES" if (avg_pass and h2_pass) else "NO"

    reasons = []
    if avg_pass:
        reasons.append(f"avg IC improvement {avg_improvement:+.3f} > 0.01 threshold")
    else:
        reasons.append(f"avg IC improvement {avg_improvement:+.3f} <= 0.01 threshold")
    if h2_pass:
        reasons.append(f"H2 stability improved in {h2_improved}/{n_baskets} baskets")
    else:
        reasons.append(f"H2 stability improved in only {h2_improved}/{n_baskets} baskets")

    print(f"\nOverall: Add Blume Z_beta to themes? {verdict}")
    print("Reason: " + "; ".join(reasons))
    print("\nPromotion criteria applied:")
    print(f"  Average IC improvement > 0.01 -> {'PASS' if avg_pass else 'FAIL'} ({avg_improvement:+.3f})")
    print(f"  H2 stability improves          -> {'PASS' if h2_pass else 'FAIL'} ({h2_improved}/{n_baskets} baskets)")


# -- Main -----------------------------------------------------------------------

def main() -> None:
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        print("=" * 68)
        print("BLUME BETA Z-SCORE THEME SIGNAL TEST")
        print(f"DB: {DB_PATH}")
        print(f"Signal horizon: {_FWD_HORIZON}d  |  Beta window: {_BETA_WINDOW}d  |  RS window: {_RS_LOOKBACK}d")
        print("=" * 68)

        summary: list[dict] = []

        for basket in BASKETS:
            print(f"\n{'-'*68}")
            print(f"Processing: {basket['name']} ...")
            obs = _build_observations(basket, conn)
            if not obs:
                print(f"  No observations - skipping (prices may be absent).")
                continue

            ic_result = _compute_ic_table(obs, basket["name"], len(basket["tickers"]))
            _print_ic_table(ic_result)
            summary.append(ic_result)

            diag_rows = _beta_diagnostics(basket, conn)
            _print_beta_diagnostics(basket["name"], diag_rows)

        _print_recommendation(summary)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
