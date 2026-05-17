"""Walk-forward backtest for the scan-megacap composite signal.

Methodology:
  - Daily composite scores computed from price history in DuckDB (+ QQQ/XLK).
  - Forward returns at 5d, 10d, 20d use day+1 entry / day+H exit (same
    convention as the dependent-scan backtest).
  - IC is Spearman rank correlation computed per-date (cross-section of 10
    mega-caps), then averaged — giving a meaningful IC IR.
  - H1/H2 split is by date to test out-of-sample consistency.
  - Label spread and per-signal isolation use pooled observations.

Survivorship-bias caveat: yfinance / DuckDB only contain currently-listed
tickers. Results overstate performance. Use Polygon.io for production.
"""

import math
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

from scanner.db import get_connection
from scanner.signals.megacap import MEGACAP_UNIVERSE

logger = logging.getLogger(__name__)

_HORIZONS: tuple[int, ...] = (5, 10, 20)
_RS_TIMEFRAMES: tuple[int, ...] = (20, 60, 120)
_HIST_WINDOW = 252
_MA_SHORT = 50
_MA_LONG = 200
_VOL_LOOKBACK = 20
# Warmup: enough bars for hist_pct_120 (252 + 120 + 1 = 373)
_MIN_WARMUP = _HIST_WINDOW + max(_RS_TIMEFRAMES) + 1
_MIN_VALID_TICKERS = 6   # need at least 6 to form 3 LEADING + 3 WEAKENING


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LabelSpread:
    horizon: int
    leading_avg: float
    neutral_avg: float
    weakening_avg: float
    spread: float           # leading_avg - weakening_avg
    n_leading: int
    n_neutral: int
    n_weakening: int


@dataclass
class SignalIC:
    name: str
    ic_5d: float
    ic_10d: float
    ic_20d: float
    n: int                  # number of pooled (ticker, date) pairs


@dataclass
class MegaCapBacktestResult:
    start_date: str
    end_date: str
    mid_date: str
    n_obs: int              # total (ticker, date) observations
    n_dates: int            # distinct backtest dates
    # Per-date IC averaged across all dates
    ic_mean: dict[int, float]
    ic_std: dict[int, float]
    ic_ir: dict[int, float]
    # H1/H2 out-of-sample split
    ic_h1_mean: dict[int, float]
    ic_h2_mean: dict[int, float]
    n_h1: int
    n_h2: int
    label_spreads: list[LabelSpread]
    signal_ics: list[SignalIC]


# ---------------------------------------------------------------------------
# Signal primitives (operate on numpy arrays already sliced to ≤ date)
# ---------------------------------------------------------------------------

def _raw_return(closes: np.ndarray, lookback: int) -> float:
    if len(closes) < lookback + 1:
        return float("nan")
    base = closes[-(lookback + 1)]
    if base == 0:
        return float("nan")
    return float((closes[-1] - base) / base)


def _rs(ticker_arr: np.ndarray, bench_arr: np.ndarray, lookback: int) -> float:
    t = _raw_return(ticker_arr, lookback)
    b = _raw_return(bench_arr, lookback)
    if math.isnan(t) or math.isnan(b):
        return float("nan")
    return t - b


def _hist_pct(closes: np.ndarray, lookback: int, window: int = _HIST_WINDOW) -> float:
    if len(closes) < window + lookback + 1:
        return float("nan")
    base = closes[-(lookback + 1)]
    if base == 0:
        return float("nan")
    current_ret = float((closes[-1] - base) / base)
    n = len(closes)
    hist: list[float] = []
    for i in range(1, window + 1):
        idx = n - 1 - i
        prev = idx - lookback
        if prev < 0:
            break
        if closes[prev] == 0:
            continue
        hist.append(float((closes[idx] - closes[prev]) / closes[prev]))
    if not hist:
        return float("nan")
    return sum(1 for r in hist if r < current_ret) / len(hist)


def _trend(closes: np.ndarray) -> tuple[int, float]:
    if len(closes) < _MA_LONG:
        return (0, float("nan"))
    price = closes[-1]
    score = (1 if price > closes[-_MA_SHORT:].mean() else 0) + (
        1 if price > closes[-_MA_LONG:].mean() else 0
    )
    return (score, score / 2.0)


def _vol_conviction(closes: np.ndarray, volumes: np.ndarray) -> float:
    if len(closes) < _VOL_LOOKBACK + 1 or len(volumes) < _VOL_LOOKBACK:
        return float("nan")
    c = closes[-(_VOL_LOOKBACK + 1):]
    v = volumes[-_VOL_LOOKBACK:]
    up_mask = c[1:] > c[:-1]
    up_vol = float(v[up_mask].sum())
    down_vol = float(v[~up_mask].sum())
    total = up_vol + down_vol
    if total == 0:
        return float("nan")
    return up_vol / total


def _minmax_norm(values: list[float]) -> list[float]:
    valid = [v for v in values if not math.isnan(v)]
    if not valid:
        return values
    lo, hi = min(valid), max(valid)
    if hi == lo:
        return [0.5 if not math.isnan(v) else float("nan") for v in values]
    return [(v - lo) / (hi - lo) if not math.isnan(v) else float("nan") for v in values]


def _eq_avg(values: list[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else float("nan")


def _spearman(scores: list[float], fwds: list[float]) -> float | None:
    pairs = [(s, f) for s, f in zip(scores, fwds) if not math.isnan(s) and not math.isnan(f)]
    if len(pairs) < 4:
        return None
    s_arr, f_arr = zip(*pairs)
    ic, _ = spearmanr(s_arr, f_arr)
    return float(ic) if not math.isnan(ic) else None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_db_prices(conn) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[pd.Timestamp]]:
    """Load adj_close and volume for all mega-caps from DuckDB.

    Returns aligned arrays (same index = same date) and the common date list.
    """
    series_adj: dict[str, pd.Series] = {}
    series_vol: dict[str, pd.Series] = {}

    for ticker in MEGACAP_UNIVERSE:
        rows = conn.execute(
            "SELECT date, adj_close, volume FROM prices WHERE ticker = ? ORDER BY date",
            [ticker],
        ).fetchall()
        if not rows:
            logger.warning("No price data for %s — run 'scanner ingest' first", ticker)
            continue
        idx = pd.to_datetime([str(r[0]) for r in rows])
        series_adj[ticker] = pd.Series([float(r[1]) for r in rows], index=idx)
        series_vol[ticker] = pd.Series(
            [float(r[2]) if r[2] is not None else 0.0 for r in rows], index=idx
        )

    if not series_adj:
        raise RuntimeError("No price data found in DuckDB. Run 'scanner ingest' first.")

    common_idx: pd.DatetimeIndex = series_adj[next(iter(series_adj))].index
    for s in series_adj.values():
        common_idx = common_idx.intersection(s.index)
    common_idx = common_idx.sort_values()

    adj_arrays: dict[str, np.ndarray] = {}
    vol_arrays: dict[str, np.ndarray] = {}
    for ticker in series_adj:
        adj_arrays[ticker] = series_adj[ticker].reindex(common_idx).ffill().to_numpy(dtype=float)
        vol_arrays[ticker] = series_vol[ticker].reindex(common_idx).ffill().fillna(0.0).to_numpy(dtype=float)

    return adj_arrays, vol_arrays, list(common_idx)


def _load_bench(ticker: str, common_dates: list[pd.Timestamp], conn) -> np.ndarray:
    """Load benchmark adj_close from DuckDB if available, else fetch from yfinance."""
    rows = conn.execute(
        "SELECT date, adj_close FROM prices WHERE ticker = ? ORDER BY date",
        [ticker],
    ).fetchall()

    if rows:
        idx = pd.to_datetime([str(r[0]) for r in rows])
        series = pd.Series([float(r[1]) for r in rows], index=idx)
    else:
        logger.warning(
            "%s not found in DuckDB — fetching from yfinance. "
            "Run 'scanner ingest %s' to cache it.",
            ticker, ticker,
        )
        try:
            df = yf.Ticker(ticker).history(period="max", auto_adjust=True)
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            series = df["Close"].rename(ticker)
        except Exception as exc:
            logger.error("Failed to fetch %s: %s — RS vs %s will be nan", ticker, exc, ticker)
            return np.full(len(common_dates), float("nan"))

    dt_index = pd.DatetimeIndex(common_dates)
    return series.reindex(dt_index).ffill().to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Per-date score computation
# ---------------------------------------------------------------------------

def _scores_on_date(
    i: int,
    adj: dict[str, np.ndarray],
    vol: dict[str, np.ndarray],
    qqq: np.ndarray,
    xlk: np.ndarray,
    tickers: list[str],
) -> dict[str, dict] | None:
    """Compute all signals for all mega-caps at array position i.

    Returns None if fewer than _MIN_VALID_TICKERS have valid composite scores.
    """
    qqq_slice = qqq[: i + 1]
    xlk_slice = xlk[: i + 1]

    raw: dict[str, dict] = {}
    for ticker in tickers:
        tc = adj[ticker][: i + 1]
        tv = vol[ticker][: i + 1]
        if len(tc) < _MIN_WARMUP:
            continue
        r: dict = {}
        for tf in _RS_TIMEFRAMES:
            r[f"rs_qqq_{tf}"] = _rs(tc, qqq_slice, tf)
            r[f"rs_xlk_{tf}"] = _rs(tc, xlk_slice, tf)
            r[f"hist_pct_{tf}"] = _hist_pct(tc, tf)
        _, tr_norm = _trend(tc)
        r["trend"] = tr_norm
        r["vol"] = _vol_conviction(tc, tv)
        raw[ticker] = r

    if len(raw) < _MIN_VALID_TICKERS:
        return None

    ordered = list(raw.keys())

    # Cross-sectionally normalize RS differentials per timeframe
    for tf in _RS_TIMEFRAMES:
        for key in (f"rs_qqq_{tf}", f"rs_xlk_{tf}"):
            vals = [raw[t][key] for t in ordered]
            normed = _minmax_norm(vals)
            for t, nv in zip(ordered, normed):
                raw[t][f"norm_{key}"] = nv

    # Build composite RS (3 timeframes × 3 benchmarks) then final composite
    for ticker in ordered:
        r = raw[ticker]
        tf_scores = [
            _eq_avg([r[f"norm_rs_qqq_{tf}"], r[f"norm_rs_xlk_{tf}"], r[f"hist_pct_{tf}"]])
            for tf in _RS_TIMEFRAMES
        ]
        r["composite_rs"] = _eq_avg(tf_scores)
        r["composite"] = _eq_avg([r["composite_rs"], r["trend"], r["vol"]])

        # Isolation signals (raw, rank-invariant for Spearman)
        rs_raws = [r[f"rs_qqq_{tf}"] for tf in _RS_TIMEFRAMES] + [
            r[f"rs_xlk_{tf}"] for tf in _RS_TIMEFRAMES
        ]
        r["iso_rs_multi"] = _eq_avg(rs_raws)
        r["iso_hist_pct"] = _eq_avg([r[f"hist_pct_{tf}"] for tf in _RS_TIMEFRAMES])

    # Rank and label (fixed top-3 / bottom-3)
    ranked = sorted(ordered, key=lambda t: raw[t]["composite"] if not math.isnan(raw[t]["composite"]) else -999, reverse=True)
    n = len(ranked)
    for idx, ticker in enumerate(ranked):
        rank = idx + 1
        raw[ticker]["label"] = "LEADING" if rank <= 3 else ("WEAKENING" if rank > n - 3 else "NEUTRAL")

    return raw


# ---------------------------------------------------------------------------
# Main backtest entry point
# ---------------------------------------------------------------------------

def run_megacap_backtest() -> MegaCapBacktestResult:
    conn = get_connection()
    adj, vol, common_dates = _load_db_prices(conn)

    missing = set(MEGACAP_UNIVERSE) - set(adj)
    if missing:
        raise RuntimeError(f"Missing price data for {missing}. Run 'scanner ingest' first.")

    qqq_arr = _load_bench("QQQ", common_dates, conn)
    xlk_arr = _load_bench("XLK", common_dates, conn)
    conn.close()

    tickers = list(adj.keys())
    n_dates = len(common_dates)
    max_h = max(_HORIZONS)

    if n_dates < _MIN_WARMUP + max_h + 2:
        raise RuntimeError(
            f"Need at least {_MIN_WARMUP + max_h + 2} trading days in DuckDB. "
            f"Only {n_dates} found. Run 'scanner ingest'."
        )

    # Determine H1/H2 midpoint index
    start_i = _MIN_WARMUP
    end_i = n_dates - max_h - 2    # last valid index (entry=i+1, exit=i+max_h)
    mid_i = (start_i + end_i) // 2

    # --- Main iteration ---
    observations: list[dict] = []
    per_date_ic: dict[int, list[float]] = {h: [] for h in _HORIZONS}
    per_date_ic_h1: dict[int, list[float]] = {h: [] for h in _HORIZONS}
    per_date_ic_h2: dict[int, list[float]] = {h: [] for h in _HORIZONS}

    for i in range(start_i, end_i + 1):
        scores = _scores_on_date(i, adj, vol, qqq_arr, xlk_arr, tickers)
        if scores is None:
            continue

        # Forward returns: entry = adj[i+1], exit = adj[i+H]
        fwds: dict[str, dict[int, float | None]] = {}
        for ticker in scores:
            arr = adj[ticker]
            entry = arr[i + 1] if i + 1 < n_dates else None
            fwds[ticker] = {}
            for h in _HORIZONS:
                if entry is None or entry == 0 or i + h >= n_dates:
                    fwds[ticker][h] = None
                else:
                    fwds[ticker][h] = float((arr[i + h] - entry) / entry)

        # Per-date IC across the cross-section
        in_h1 = i < mid_i
        for h in _HORIZONS:
            date_scores = [scores[t]["composite"] for t in scores]
            date_fwds_raw = [fwds[t][h] for t in scores]
            date_fwds = [f if f is not None else float("nan") for f in date_fwds_raw]
            ic = _spearman(date_scores, date_fwds)
            if ic is not None:
                per_date_ic[h].append(ic)
                if in_h1:
                    per_date_ic_h1[h].append(ic)
                else:
                    per_date_ic_h2[h].append(ic)

        for ticker, r in scores.items():
            observations.append({
                "date_idx": i,
                "in_h1": i < mid_i,
                "ticker": ticker,
                "composite": r["composite"],
                "iso_rs_multi": r["iso_rs_multi"],
                "iso_trend": r["trend"],
                "iso_vol": r["vol"],
                "iso_hist_pct": r["iso_hist_pct"],
                "label": r["label"],
                "fwd_5": fwds[ticker][5],
                "fwd_10": fwds[ticker][10],
                "fwd_20": fwds[ticker][20],
            })

    if not observations:
        raise RuntimeError("No valid observations produced. Check price history coverage.")

    n_obs = len(observations)
    n_h1 = sum(1 for o in observations if o["in_h1"])
    n_h2 = n_obs - n_h1

    # --- IC summary ---
    def _avg_ic(ic_list: list[float]) -> float:
        return float(np.mean(ic_list)) if ic_list else float("nan")

    def _std_ic(ic_list: list[float]) -> float:
        return float(np.std(ic_list, ddof=1)) if len(ic_list) > 1 else float("nan")

    ic_mean = {h: _avg_ic(per_date_ic[h]) for h in _HORIZONS}
    ic_std = {h: _std_ic(per_date_ic[h]) for h in _HORIZONS}
    ic_ir = {
        h: (ic_mean[h] / ic_std[h] if not math.isnan(ic_std[h]) and ic_std[h] > 0 else float("nan"))
        for h in _HORIZONS
    }
    ic_h1_mean = {h: _avg_ic(per_date_ic_h1[h]) for h in _HORIZONS}
    ic_h2_mean = {h: _avg_ic(per_date_ic_h2[h]) for h in _HORIZONS}

    # --- Label spread ---
    label_spreads: list[LabelSpread] = []
    for h in _HORIZONS:
        fwd_key = f"fwd_{h}"
        groups: dict[str, list[float]] = {"LEADING": [], "NEUTRAL": [], "WEAKENING": []}
        for o in observations:
            v = o[fwd_key]
            if v is not None and not math.isnan(v):
                groups[o["label"]].append(v)
        def _gavg(lst: list[float]) -> float:
            return float(np.mean(lst)) if lst else float("nan")
        leading_avg = _gavg(groups["LEADING"])
        neutral_avg = _gavg(groups["NEUTRAL"])
        weakening_avg = _gavg(groups["WEAKENING"])
        spread = (leading_avg - weakening_avg
                  if not math.isnan(leading_avg) and not math.isnan(weakening_avg)
                  else float("nan"))
        label_spreads.append(LabelSpread(
            horizon=h,
            leading_avg=leading_avg,
            neutral_avg=neutral_avg,
            weakening_avg=weakening_avg,
            spread=spread,
            n_leading=len(groups["LEADING"]),
            n_neutral=len(groups["NEUTRAL"]),
            n_weakening=len(groups["WEAKENING"]),
        ))

    # --- Per-signal isolation IC (pooled across all observations) ---
    signal_ics: list[SignalIC] = []
    isolation_signals = [
        ("Multi-TF RS (QQQ+XLK)", "iso_rs_multi"),
        ("Trend Structure",        "iso_trend"),
        ("Volume Conviction",      "iso_vol"),
        ("Historical Percentile",  "iso_hist_pct"),
    ]
    for name, key in isolation_signals:
        ic_by_h: dict[int, float] = {}
        n_valid = 0
        for h in _HORIZONS:
            fwd_key = f"fwd_{h}"
            sig_vals = [o[key] for o in observations]
            fwd_vals_raw = [o[fwd_key] for o in observations]
            fwd_vals = [f if f is not None else float("nan") for f in fwd_vals_raw]
            ic = _spearman(sig_vals, fwd_vals)
            ic_by_h[h] = ic if ic is not None else float("nan")
            n_valid = sum(
                1 for s, f in zip(sig_vals, fwd_vals)
                if not math.isnan(s) and not math.isnan(f)
            )
        signal_ics.append(SignalIC(
            name=name,
            ic_5d=ic_by_h[5],
            ic_10d=ic_by_h[10],
            ic_20d=ic_by_h[20],
            n=n_valid,
        ))

    start_date = common_dates[start_i].strftime("%Y-%m-%d")
    end_date = common_dates[end_i].strftime("%Y-%m-%d")
    mid_date = common_dates[mid_i].strftime("%Y-%m-%d")
    n_backtest_dates = end_i - start_i + 1

    return MegaCapBacktestResult(
        start_date=start_date,
        end_date=end_date,
        mid_date=mid_date,
        n_obs=n_obs,
        n_dates=n_backtest_dates,
        ic_mean=ic_mean,
        ic_std=ic_std,
        ic_ir=ic_ir,
        ic_h1_mean=ic_h1_mean,
        ic_h2_mean=ic_h2_mean,
        n_h1=n_h1,
        n_h2=n_h2,
        label_spreads=label_spreads,
        signal_ics=signal_ics,
    )
