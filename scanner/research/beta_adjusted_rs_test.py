"""
Research script: beta-adjusted RS vs raw RS comparison.

Usage:
    python scanner/research/beta_adjusted_rs_test.py

Reads from the production DuckDB (or DATABASE_PATH env var).
Does NOT modify any production code or tables.
"""

import math
import os
from pathlib import Path

import duckdb
import numpy as np
from scipy import stats

DB_PATH = Path(os.environ.get("DATABASE_PATH", str(Path(__file__).parent.parent.parent / "data" / "scanner.duckdb")))

TEST_PAIRS = [
    ("ALAB", "NVDA"),
    ("STM", "TSLA"),
    ("MRVL", "AMZN"),
    ("DELL", "NVDA"),
    ("ON", "TSLA"),
]

LOOKBACK = 20           # days for rolling return window
BETA_WINDOW = 60        # days for beta estimation
FORWARD = 20            # days for forward return (IC target)
MIN_REGIME_OBS = 30     # minimum observations to report a regime bucket

REGIME_ORDER = ["STRONG_UP", "UP", "FLAT", "MILD_PULLBACK", "CORRECTION", "CRASH"]


def classify_regime(parent_20d: float) -> str:
    if parent_20d > 0.10:
        return "STRONG_UP"
    if parent_20d > 0.02:
        return "UP"
    if parent_20d >= -0.02:
        return "FLAT"
    if parent_20d > -0.05:
        return "MILD_PULLBACK"
    if parent_20d >= -0.15:
        return "CORRECTION"
    return "CRASH"


def load_prices(conn: duckdb.DuckDBPyConnection, ticker: str) -> dict:
    rows = conn.execute("""
        SELECT date::VARCHAR, adj_close
        FROM prices
        WHERE ticker = ?
        ORDER BY date ASC
    """, [ticker]).fetchall()
    return {r[0]: r[1] for r in rows}


def daily_returns(price_map: dict) -> tuple[list, list]:
    dates = sorted(price_map.keys())
    rets = []
    ret_dates = []
    for i in range(1, len(dates)):
        p0 = price_map[dates[i - 1]]
        p1 = price_map[dates[i]]
        if p0 and p0 != 0:
            rets.append((p1 - p0) / p0)
            ret_dates.append(dates[i])
    return ret_dates, rets


def rolling_return(price_map: dict, date: str, window: int) -> float:
    dates = sorted(d for d in price_map if d <= date)
    if len(dates) < window + 1:
        return float("nan")
    oldest = price_map[dates[-(window + 1)]]
    latest = price_map[dates[-1]]
    if not oldest or oldest == 0:
        return float("nan")
    return (latest - oldest) / oldest


def rolling_beta(
    child_rets: dict,
    spy_rets: dict,
    date: str,
    window: int,
) -> float:
    dates = sorted(d for d in child_rets if d <= date and d in spy_rets)
    if len(dates) < window:
        return float("nan")
    window_dates = dates[-window:]
    c = np.array([child_rets[d] for d in window_dates])
    s = np.array([spy_rets[d] for d in window_dates])
    var_s = np.var(s, ddof=1)
    if var_s == 0:
        return float("nan")
    return float(np.cov(c, s, ddof=1)[0, 1] / var_s)


def spearman_ic(scores: list, fwd_returns: list) -> float:
    if len(scores) < 5:
        return float("nan")
    corr, _ = stats.spearmanr(scores, fwd_returns)
    return float(corr) if not math.isnan(corr) else float("nan")


def half_ics(pairs_data: list) -> tuple[float, float]:
    mid = len(pairs_data) // 2
    h1 = [(s, f) for s, f in pairs_data[:mid] if not math.isnan(s) and not math.isnan(f)]
    h2 = [(s, f) for s, f in pairs_data[mid:] if not math.isnan(s) and not math.isnan(f)]
    ic1 = spearman_ic([x[0] for x in h1], [x[1] for x in h1]) if h1 else float("nan")
    ic2 = spearman_ic([x[0] for x in h2], [x[1] for x in h2]) if h2 else float("nan")
    return ic1, ic2


def ic_winner(raw_ic: float, adj_ic: float, raw_stab: float, adj_stab: float) -> str:
    """Primary: higher abs IC. Tiebreaker: smaller stability gap if diff < 0.01."""
    if math.isnan(raw_ic) or math.isnan(adj_ic):
        return "n/a"
    diff = abs(adj_ic) - abs(raw_ic)
    if abs(diff) < 0.01:
        if not math.isnan(raw_stab) and not math.isnan(adj_stab):
            return "Adj" if adj_stab < raw_stab else "Raw"
        return "Tie"
    return "Adj" if diff > 0 else "Raw"


def evaluate_pair(
    conn: duckdb.DuckDBPyConnection,
    child: str,
    parent: str,
    spy_prices: dict,
    spy_rets_map: dict,
) -> dict | None:
    print(f"  Loading prices for {child}, {parent}...")

    child_prices = load_prices(conn, child)
    parent_prices = load_prices(conn, parent)

    if not child_prices:
        print(f"  WARNING: no prices for {child}")
        return None
    if not parent_prices:
        print(f"  WARNING: no prices for {parent}")
        return None

    child_ret_dates, child_ret_vals = daily_returns(child_prices)
    parent_ret_dates, parent_ret_vals = daily_returns(parent_prices)

    child_rets_map = dict(zip(child_ret_dates, child_ret_vals))
    parent_rets_map = dict(zip(parent_ret_dates, parent_ret_vals))

    all_child_dates = sorted(child_prices.keys())
    all_dates = sorted(set(all_child_dates) & set(parent_prices.keys()) & set(spy_prices.keys()))

    # Each obs: (date, raw_rs, adj_rs, fwd_return, parent_20d, beta_child)
    obs = []
    betas = []

    for date in all_dates:
        future_dates = [d for d in all_child_dates if d > date]
        if len(future_dates) < FORWARD:
            continue
        fwd_date = future_dates[FORWARD - 1]
        p_now = child_prices.get(date)
        p_fwd = child_prices.get(fwd_date)
        if not p_now or not p_fwd or p_now == 0:
            continue
        fwd_ret = (p_fwd - p_now) / p_now

        raw_child = rolling_return(child_prices, date, LOOKBACK)
        raw_parent = rolling_return(parent_prices, date, LOOKBACK)
        if math.isnan(raw_child) or math.isnan(raw_parent):
            continue

        raw_rs = raw_child - raw_parent

        beta_child = rolling_beta(child_rets_map, spy_rets_map, date, BETA_WINDOW)
        beta_parent = rolling_beta(parent_rets_map, spy_rets_map, date, BETA_WINDOW)
        spy_roll = rolling_return(spy_prices, date, LOOKBACK)

        if math.isnan(beta_child) or math.isnan(beta_parent) or math.isnan(spy_roll):
            adj_rs = float("nan")
        else:
            betas.append(beta_child)
            resid_child = raw_child - beta_child * spy_roll
            resid_parent = raw_parent - beta_parent * spy_roll
            adj_rs = resid_child - resid_parent

        obs.append((date, raw_rs, adj_rs, fwd_ret, raw_parent))

    if not obs:
        print(f"  WARNING: insufficient observations for {child}->{parent}")
        return None

    raw_valid = [(d, r, a, f, p) for d, r, a, f, p in obs if not math.isnan(r)]
    adj_valid = [(d, r, a, f, p) for d, r, a, f, p in obs if not math.isnan(a)]

    if not raw_valid or not adj_valid:
        print(f"  WARNING: insufficient valid observations for {child}->{parent}")
        return None

    raw_scores = [x[1] for x in raw_valid]
    raw_fwds   = [x[3] for x in raw_valid]
    adj_scores = [x[2] for x in adj_valid]
    adj_fwds   = [x[3] for x in adj_valid]

    raw_ic = spearman_ic(raw_scores, raw_fwds)
    adj_ic = spearman_ic(adj_scores, adj_fwds)

    raw_h1, raw_h2 = half_ics(list(zip(raw_scores, raw_fwds)))
    adj_h1, adj_h2 = half_ics(list(zip(adj_scores, adj_fwds)))

    raw_stab = abs(raw_h2 - raw_h1) if not math.isnan(raw_h1) and not math.isnan(raw_h2) else float("nan")
    adj_stab = abs(adj_h2 - adj_h1) if not math.isnan(adj_h1) and not math.isnan(adj_h2) else float("nan")

    # Signal correlation (raw vs adj, aligned by date)
    raw_by_date = {x[0]: x[1] for x in raw_valid}
    adj_by_date = {x[0]: x[2] for x in adj_valid}
    shared = sorted(set(raw_by_date) & set(adj_by_date))
    sig_corr = float("nan")
    if len(shared) >= 5:
        sig_corr, _ = stats.pearsonr(
            [raw_by_date[d] for d in shared],
            [adj_by_date[d] for d in shared],
        )

    # Regime-conditional IC buckets using parent_20d (= raw_parent col)
    # adj_valid has parent_20d at index 4
    regime_buckets: dict[str, list] = {r: [] for r in REGIME_ORDER}
    for d, r, a, f, p in adj_valid:
        regime = classify_regime(p)
        regime_buckets[regime].append((r, a, f))

    regime_ics: dict[str, dict] = {}
    for regime, bucket in regime_buckets.items():
        n = len(bucket)
        if n < MIN_REGIME_OBS:
            regime_ics[regime] = {"n": n, "raw_ic": float("nan"), "adj_ic": float("nan")}
            continue
        raw_ic_r = spearman_ic([x[0] for x in bucket], [x[2] for x in bucket])
        adj_ic_r = spearman_ic([x[1] for x in bucket], [x[2] for x in bucket])
        regime_ics[regime] = {"n": n, "raw_ic": raw_ic_r, "adj_ic": adj_ic_r}

    # Per-date signal dict for macro shift analysis
    signals_by_date = {
        d: {"raw": r, "adj": a}
        for d, r, a, f, p in obs
        if not math.isnan(r) and not math.isnan(a)
    }

    return {
        "child": child,
        "parent": parent,
        "n_raw": len(raw_valid),
        "n_adj": len(adj_valid),
        "raw_ic": raw_ic,
        "adj_ic": adj_ic,
        "raw_h1": raw_h1,
        "raw_h2": raw_h2,
        "adj_h1": adj_h1,
        "adj_h2": adj_h2,
        "raw_stability": raw_stab,
        "adj_stability": adj_stab,
        "sig_corr": sig_corr,
        "avg_beta": float(np.mean(betas)) if betas else float("nan"),
        "regime_ics": regime_ics,
        "signals_by_date": signals_by_date,
    }


def fmt(v: float, width: int = 7) -> str:
    if math.isnan(v):
        return f"{'n/a':>{width}}"
    return f"{v:>{width}.3f}"


def print_ic_comparison(results: list) -> None:
    print("\n" + "=" * 95)
    print("BETA-ADJUSTED RS vs RAW RS — IC COMPARISON")
    print("=" * 95)
    hdr = f"{'Pair':<14} {'Raw IC':>7} {'Adj IC':>7} {'Raw H1':>7} {'Raw H2':>7} {'Adj H1':>7} {'Adj H2':>7} {'N':>6}"
    print(hdr)
    print("-" * 95)
    for r in results:
        pair = f"{r['child']}->{r['parent']}"
        print(
            f"{pair:<14}"
            f"{fmt(r['raw_ic'])}"
            f"{fmt(r['adj_ic'])}"
            f"{fmt(r['raw_h1'])}"
            f"{fmt(r['raw_h2'])}"
            f"{fmt(r['adj_h1'])}"
            f"{fmt(r['adj_h2'])}"
            f"{r['n_raw']:>7}"
        )


def print_diagnostics(results: list) -> None:
    print("\n" + "=" * 80)
    print("SIGNAL DIAGNOSTICS")
    print("=" * 80)
    hdr = f"{'Pair':<14} {'Avg Beta':>9} {'Sig Corr':>9} {'Raw Stab':>9} {'Adj Stab':>9} {'Winner':>8}"
    print(hdr)
    print("-" * 80)
    for r in results:
        pair = f"{r['child']}->{r['parent']}"
        winner = ic_winner(r["raw_ic"], r["adj_ic"], r["raw_stability"], r["adj_stability"])
        print(
            f"{pair:<14}"
            f"{fmt(r['avg_beta'], 9)}"
            f"{fmt(r['sig_corr'], 9)}"
            f"{fmt(r['raw_stability'], 9)}"
            f"{fmt(r['adj_stability'], 9)}"
            f"{winner:>8}"
        )

    valid_raw = [r["raw_ic"] for r in results if not math.isnan(r["raw_ic"])]
    valid_adj = [r["adj_ic"] for r in results if not math.isnan(r["adj_ic"])]
    valid_corrs = [r["sig_corr"] for r in results if not math.isnan(r["sig_corr"])]

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    if valid_raw:
        print(f"  Avg raw IC:      {np.mean(valid_raw):.3f}")
    if valid_adj:
        print(f"  Avg adj IC:      {np.mean(valid_adj):.3f}")
    if valid_corrs:
        avg_corr = float(np.mean(valid_corrs))
        note = "  <- signals nearly identical; beta adjustment not worth complexity" if avg_corr > 0.85 else "  <- signals diverge; beta adjustment adds information"
        print(f"  Avg signal corr: {avg_corr:.3f}{note}")

    ic_adj_wins = sum(
        1 for r in results
        if not math.isnan(r["raw_ic"]) and not math.isnan(r["adj_ic"])
        and abs(r["adj_ic"]) > abs(r["raw_ic"])
    )
    ic_raw_wins = sum(
        1 for r in results
        if not math.isnan(r["raw_ic"]) and not math.isnan(r["adj_ic"])
        and abs(r["raw_ic"]) > abs(r["adj_ic"])
    )
    print(f"  IC winner (abs magnitude): Raw={ic_raw_wins}, Adj={ic_adj_wins}")

    stab_raw_wins = sum(
        1 for r in results
        if not math.isnan(r["raw_stability"]) and not math.isnan(r["adj_stability"])
        and r["raw_stability"] < r["adj_stability"]
    )
    stab_adj_wins = sum(
        1 for r in results
        if not math.isnan(r["raw_stability"]) and not math.isnan(r["adj_stability"])
        and r["adj_stability"] < r["raw_stability"]
    )
    print(f"  Stability winner (smaller H1-H2 gap): Raw={stab_raw_wins}, Adj={stab_adj_wins}")


def print_regime_breakdown(results: list) -> None:
    for r in results:
        pair = f"{r['child']}->{r['parent']}"
        print(f"\nREGIME-CONDITIONAL IC -- {pair}")
        print("=" * 62)
        hdr = f"{'Regime':<16} {'N':>5}  {'Raw IC':>7}  {'Adj IC':>7}  {'Winner':<8}  Notes"
        print(hdr)
        print("-" * 62)
        for regime in REGIME_ORDER:
            bucket = r["regime_ics"].get(regime, {})
            n = bucket.get("n", 0)
            raw_ic = bucket.get("raw_ic", float("nan"))
            adj_ic = bucket.get("adj_ic", float("nan"))

            if n < MIN_REGIME_OBS:
                print(f"{regime:<16} {n:>5}  {'--':>7}  {'--':>7}  {'--':<8}  insufficient")
                continue

            winner = ic_winner(raw_ic, adj_ic, float("nan"), float("nan"))
            note = " <- key" if regime == "CORRECTION" else ""
            print(f"{regime:<16} {n:>5}  {fmt(raw_ic):>7}  {fmt(adj_ic):>7}  {winner:<8}{note}")
        print("-" * 62)


def print_macro_shift_days(results: list, spy_rets_map: dict) -> None:
    if not spy_rets_map:
        return

    top10 = sorted(spy_rets_map.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
    top10_dates = sorted(d for d, _ in top10)

    pair_labels = [f"{r['child']}" for r in results]

    col_w = 10
    header = f"{'Date':<12} {'SPY%':>7}"
    for label in pair_labels:
        header += f"  {label+'_raw':>{col_w}} {label+'_adj':>{col_w}} {'Diff':>{col_w}}"
    print("\n" + "=" * (12 + 7 + len(results) * (col_w * 3 + 6)))
    print("MACRO SHIFT DAYS -- RAW vs ADJUSTED RS")
    print("=" * (12 + 7 + len(results) * (col_w * 3 + 6)))
    print(header)
    print("-" * (12 + 7 + len(results) * (col_w * 3 + 6)))

    for date in top10_dates:
        spy_ret = spy_rets_map.get(date, float("nan"))
        row = f"{date:<12} {spy_ret * 100:>+6.2f}%"
        for r in results:
            sigs = r["signals_by_date"].get(date)
            if sigs:
                raw_v = sigs["raw"]
                adj_v = sigs["adj"]
                diff_v = raw_v - adj_v
                row += (
                    f"  {raw_v:>{col_w}.4f}"
                    f" {adj_v:>{col_w}.4f}"
                    f" {diff_v:>{col_w}.4f}"
                )
            else:
                dash = f"{'--':>{col_w}}"
                row += f"  {dash} {dash} {dash}"
        print(row)


def print_revenue_gate(results: list) -> None:
    print("\n" + "=" * 55)
    print("REVENUE GATE ANALYSIS")
    print("=" * 55)
    hdr = f"{'Pair':<14} {'Adj IC':>7}  {'Corr IC':>8}  Keep?"
    print(hdr)
    print("-" * 55)
    for r in results:
        pair = f"{r['child']}->{r['parent']}"
        adj_ic = r["adj_ic"]
        corr_bucket = r["regime_ics"].get("CORRECTION", {})
        corr_ic = corr_bucket.get("adj_ic", float("nan"))

        if math.isnan(adj_ic):
            verdict = "INSUFFICIENT DATA"
        elif adj_ic > 0.10:
            verdict = "KEEP [OK]"
        elif corr_ic > 0.10 if not math.isnan(corr_ic) else False:
            verdict = "KEEP (regime value) [OK]"
        elif adj_ic < 0.05 and (math.isnan(corr_ic) or corr_ic < 0.05):
            verdict = "CONSIDER DROPPING [!]"
        else:
            verdict = "MARGINAL"

        print(f"{pair:<14} {fmt(adj_ic):>7}  {fmt(corr_ic, 8):>8}  {verdict}")
    print("-" * 55)


def main() -> None:
    print(f"Connecting to {DB_PATH}")
    if not DB_PATH.exists():
        print(f"ERROR: database not found at {DB_PATH}")
        return

    conn = duckdb.connect(str(DB_PATH), read_only=True)

    print("\nLoading SPY prices...")
    spy_prices = load_prices(conn, "SPY")
    if not spy_prices:
        print("ERROR: no SPY prices in database")
        conn.close()
        return
    spy_ret_dates, spy_ret_vals = daily_returns(spy_prices)
    spy_rets_map = dict(zip(spy_ret_dates, spy_ret_vals))

    results = []
    for child, parent in TEST_PAIRS:
        print(f"\nEvaluating {child}->{parent}...")
        r = evaluate_pair(conn, child, parent, spy_prices, spy_rets_map)
        if r:
            results.append(r)

    conn.close()

    if not results:
        print("\nNo results to display.")
        return

    print_ic_comparison(results)
    print_diagnostics(results)
    print_regime_breakdown(results)
    print_macro_shift_days(results, spy_rets_map)
    print_revenue_gate(results)
    print()


if __name__ == "__main__":
    main()
