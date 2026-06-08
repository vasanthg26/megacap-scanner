"""Research script: Critical Minerals & Battery Materials basket IC backtest.

Signal: 20-day cross-sectional Z-score x RVOL vs LIT benchmark.
Outputs per-ticker IC(RS), IC(Z), IC(Z*RVOL) and H1/H2 split at 20d horizon.

Usage:
    python scanner/research/battery_materials_backtest.py

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

BASKET = ["SQM", "ALB", "ALTM", "MP"]
BENCHMARK = "LIT"
_LOOKBACK = 20
_FWD_HORIZON = 20
_MIN_BASKET = 3
_MIN_OBS_HALF = 20


def _mean(lst):
    return sum(lst) / len(lst) if lst else float("nan")


def _std(lst):
    if len(lst) < 2:
        return float("nan")
    m = _mean(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / len(lst))


def _spearman(xs, ys):
    n = len(xs)
    if n < 5:
        return float("nan")

    def _rank(lst):
        order = sorted(range(n), key=lambda i: lst[i])
        r = [0.0] * n
        for rk, i in enumerate(order):
            r[i] = float(rk + 1)
        return r

    rs, rr = _rank(xs), _rank(ys)
    ms, mr = _mean(rs), _mean(rr)
    cov = sum((rs[i] - ms) * (rr[i] - mr) for i in range(n)) / n
    vs = math.sqrt(sum((x - ms) ** 2 for x in rs) / n)
    vr = math.sqrt(sum((x - mr) ** 2 for x in rr) / n)
    if vs == 0 or vr == 0:
        return float("nan")
    return cov / (vs * vr)


def _ic(pairs):
    if len(pairs) < 5:
        return float("nan")
    xs, ys = zip(*pairs)
    return _spearman(list(xs), list(ys))


def _fmt(v):
    if math.isnan(v):
        return "   n/a"
    return f"{v:+.3f}"


def main():
    conn = duckdb.connect(str(DB_PATH), read_only=True)

    all_tickers = BASKET + [BENCHMARK]
    placeholders = ",".join("?" for _ in all_tickers)
    rows = conn.execute(
        f"SELECT ticker, CAST(date AS VARCHAR), adj_close, volume "
        f"FROM prices WHERE ticker IN ({placeholders}) ORDER BY ticker, date",
        all_tickers,
    ).fetchall()

    ticker_dates = defaultdict(list)
    ticker_closes = defaultdict(list)
    ticker_vols = defaultdict(list)

    for tkr, dt, close, vol in rows:
        ticker_dates[tkr].append(dt)
        ticker_closes[tkr].append(float(close))
        ticker_vols[tkr].append(float(vol) if vol is not None else 0.0)

    conn.close()

    bench_dates = ticker_dates.get(BENCHMARK, [])
    bench_closes = ticker_closes.get(BENCHMARK, [])

    all_dates = sorted({dt for tkr in BASKET for dt in ticker_dates.get(tkr, [])})
    if len(all_dates) < _LOOKBACK + _FWD_HORIZON + 5:
        print("Insufficient price history.")
        return

    # per-ticker accumulators: (signal_value, fwd_return)
    # keys: rs, z, combined
    obs: dict[str, dict[str, list]] = {
        tkr: {"rs": [], "z": [], "combined": []} for tkr in BASKET
    }

    for signal_date in all_dates:
        # basket 20d returns
        basket_rets: dict[str, float] = {}
        basket_rvols: dict[str, float] = {}

        for tkr in BASKET:
            dates = ticker_dates.get(tkr, [])
            closes = ticker_closes.get(tkr, [])
            vols = ticker_vols.get(tkr, [])
            idx = bisect.bisect_right(dates, signal_date) - 1
            if idx < _LOOKBACK or idx < 0 or dates[idx] != signal_date:
                continue
            c0, c20 = closes[idx], closes[idx - _LOOKBACK]
            if c20 == 0:
                continue
            basket_rets[tkr] = (c0 - c20) / c20
            avg_vol = _mean(vols[max(0, idx - _LOOKBACK):idx])
            if avg_vol > 0:
                basket_rvols[tkr] = vols[idx] / avg_vol

        if len(basket_rets) < _MIN_BASKET:
            continue

        # benchmark 20d return for RS
        bidx = bisect.bisect_right(bench_dates, signal_date) - 1
        bench_ret = float("nan")
        if bidx >= _LOOKBACK and bench_dates[bidx] == signal_date:
            bc0, bc20 = bench_closes[bidx], bench_closes[bidx - _LOOKBACK]
            if bc20 != 0:
                bench_ret = (bc0 - bc20) / bc20

        # cross-sectional Z-score
        ret_vals = list(basket_rets.values())
        bm = _mean(ret_vals)
        bstd = math.sqrt(sum((r - bm) ** 2 for r in ret_vals) / len(ret_vals))

        for tkr, ret_20d in basket_rets.items():
            dates = ticker_dates[tkr]
            closes = ticker_closes[tkr]
            idx = bisect.bisect_right(dates, signal_date) - 1
            exit_idx = idx + _FWD_HORIZON
            if exit_idx >= len(closes) or closes[idx] == 0:
                continue
            fwd = (closes[exit_idx] - closes[idx]) / closes[idx]

            rs = ret_20d - bench_ret if not math.isnan(bench_ret) else float("nan")
            z = (ret_20d - bm) / bstd if bstd > 0 else float("nan")
            rvol = basket_rvols.get(tkr, float("nan"))
            combined = z * rvol if not (math.isnan(z) or math.isnan(rvol)) else float("nan")

            if not math.isnan(rs):
                obs[tkr]["rs"].append((rs, fwd))
            if not math.isnan(z):
                obs[tkr]["z"].append((z, fwd))
            if not math.isnan(combined):
                obs[tkr]["combined"].append((combined, fwd))

    # H1/H2 split by signal date
    all_sig_dates = sorted({
        signal_date for signal_date in all_dates
        for tkr in BASKET if obs[tkr]["combined"]
    })
    mid_date = all_sig_dates[len(all_sig_dates) // 2] if all_sig_dates else ""

    def _h1h2_ic(pairs, signal_dates_obs=None):
        # We'll split by index since we don't track signal_date per pair here.
        # Use index split as approximation (same as blume script).
        n = len(pairs)
        if n < _MIN_OBS_HALF * 2:
            return float("nan"), float("nan")
        mid = n // 2
        return _ic(pairs[:mid]), _ic(pairs[mid:])

    print()
    print(f"Theme: Critical Minerals & Battery Materials")
    print("=" * 62)
    print(f"Basket: {', '.join(BASKET)}  |  Benchmark: {BENCHMARK}  |  Horizon: {_FWD_HORIZON}d")
    print()
    print(f"{'Ticker':<7}  {'IC(RS)':>7}  {'IC(Z)':>7}  {'IC(ZxRVOL)':>11}  {'H1':>7}  {'H2':>7}  {'n':>5}")
    print("-" * 62)

    basket_rs, basket_z, basket_comb = [], [], []

    for tkr in BASKET:
        rs_pairs = obs[tkr]["rs"]
        z_pairs = obs[tkr]["z"]
        comb_pairs = obs[tkr]["combined"]

        ic_rs = _ic(rs_pairs)
        ic_z = _ic(z_pairs)
        ic_comb = _ic(comb_pairs)
        h1, h2 = _h1h2_ic(comb_pairs)

        basket_rs.extend(rs_pairs)
        basket_z.extend(z_pairs)
        basket_comb.extend(comb_pairs)

        n = len(comb_pairs)
        print(
            f"{tkr:<7}  {_fmt(ic_rs):>7}  {_fmt(ic_z):>7}  {_fmt(ic_comb):>11}  "
            f"{_fmt(h1):>7}  {_fmt(h2):>7}  {n:>5}"
        )

    print("-" * 62)
    ic_rs_b = _ic(basket_rs)
    ic_z_b = _ic(basket_z)
    ic_comb_b = _ic(basket_comb)
    h1_b, h2_b = _h1h2_ic(basket_comb)
    n_b = len(basket_comb)
    print(
        f"{'Basket':<7}  {_fmt(ic_rs_b):>7}  {_fmt(ic_z_b):>7}  {_fmt(ic_comb_b):>11}  "
        f"{_fmt(h1_b):>7}  {_fmt(h2_b):>7}  {n_b:>5}"
    )
    print()

    passes = (
        not math.isnan(h1_b) and not math.isnan(h2_b) and
        h1_b > 0.05 and h2_b > 0.05
    )
    status = "VALIDATED" if passes else "WATCH"
    print(f"Promotion decision: {'PASS -> ' if passes else 'FAIL -> '}{status}")
    if not passes:
        h1_str = f"{h1_b:+.3f}" if not math.isnan(h1_b) else "n/a"
        h2_str = f"{h2_b:+.3f}" if not math.isnan(h2_b) else "n/a"
        print(f"  Basket H1={h1_str}, H2={h2_str} — both must exceed +0.05")
    print()

    # ALTM history note
    altm_n = len(ticker_dates.get("ALTM", []))
    if altm_n < 252:
        print(f"NOTE: ALTM has only {altm_n} trading days of history (merger Jan 2024).")
        print("      Results will improve as history accumulates.")
    print(f"Date range: {all_dates[_LOOKBACK]} to {all_dates[-(1 + _FWD_HORIZON)]}")


if __name__ == "__main__":
    main()
