"""Mega-cap composite signal: multi-timeframe RS vs triple benchmark, Trend, Volume Conviction."""

import math
import logging
from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

MEGACAP_UNIVERSE = ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "TSM"]

_RS_TIMEFRAMES = [20, 60, 120]
_HIST_WINDOW = 252   # trading days of own-history for percentile
_VOL_LOOKBACK = 20
_MA_SHORT = 50
_MA_LONG = 200


@dataclass
class MegaCapResult:
    ticker: str
    price: float
    rs_20: float            # 20d return vs QQQ (raw, for display)
    rs_60: float            # 60d return vs QQQ (raw, for display)
    rs_120: float           # 120d return vs QQQ (raw, for display)
    hist_pct: float         # percentile of 20d return in own 252d history (0–1, for display)
    trend_raw: int          # 0, 1, or 2 (above 50d MA, above 200d MA)
    trend: float            # normalized 0.0–1.0
    vol_conviction: float   # 0.0–1.0
    rvol: float             # today's volume / 20d avg volume
    composite: float        # equal-weight: composite_rs + trend + vol_conviction
    rank: int = 0
    label: str = ""
    headlines: list[str] = field(default_factory=list)


def _fetch_ohlcv(tickers: list[str]) -> dict[str, pd.DataFrame]:
    prices: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = yf.Ticker(t).history(period="2y", auto_adjust=True)
            if df.empty:
                logger.error("empty OHLCV for %s", t)
                continue
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            prices[t] = df
        except Exception as exc:
            logger.error("OHLCV fetch failed for %s: %s", t, exc)
    return prices


def _rolling_return(df: pd.DataFrame, lookback: int) -> float:
    """Return of df over last `lookback` trading days. nan if insufficient data."""
    if len(df) < lookback + 1:
        return float("nan")
    closes = df["Close"]
    base = closes.iloc[-(lookback + 1)]
    if base == 0:
        return float("nan")
    return float((closes.iloc[-1] - base) / base)


def _rs_signal(df: pd.DataFrame, benchmark: pd.DataFrame, lookback: int) -> float:
    """Raw return differential: ticker lookback-day return minus benchmark lookback-day return."""
    t_ret = _rolling_return(df, lookback)
    b_ret = _rolling_return(benchmark, lookback)
    if math.isnan(t_ret) or math.isnan(b_ret):
        return float("nan")
    return t_ret - b_ret


def _hist_pct_signal(df: pd.DataFrame, lookback: int, history: int = _HIST_WINDOW) -> float:
    """Percentile of current lookback-day return vs prior `history` daily lookback-day returns."""
    required = history + lookback + 1
    if len(df) < required:
        return float("nan")
    closes = df["Close"].values
    n = len(closes)
    base = closes[-(lookback + 1)]
    if base == 0:
        return float("nan")
    current_ret = (closes[-1] - base) / base
    historical: list[float] = []
    for i in range(1, history + 1):
        idx = n - 1 - i
        prev_idx = idx - lookback
        if prev_idx < 0:
            break
        prev_base = closes[prev_idx]
        if prev_base == 0:
            continue
        historical.append((closes[idx] - prev_base) / prev_base)
    if not historical:
        return float("nan")
    return sum(1 for r in historical if r < current_ret) / len(historical)


def _trend_signal(df: pd.DataFrame) -> tuple[int, float]:
    """Return (raw_score 0-2, normalized 0.0-1.0)."""
    if len(df) < _MA_LONG:
        return (0, float("nan"))
    price = float(df["Close"].iloc[-1])
    ma50 = float(df["Close"].iloc[-_MA_SHORT:].mean())
    ma200 = float(df["Close"].iloc[-_MA_LONG:].mean())
    score = (1 if price > ma50 else 0) + (1 if price > ma200 else 0)
    return (score, score / 2.0)


def _rvol_signal(df: pd.DataFrame, lookback: int = 20) -> float:
    """Today's volume divided by prior `lookback`-session average volume."""
    if len(df) < lookback + 2:
        return float("nan")
    today_vol = float(df["Volume"].iloc[-1])
    avg_vol = float(df["Volume"].iloc[-(lookback + 1):-1].mean())
    if avg_vol == 0:
        return float("nan")
    return today_vol / avg_vol


def _vol_conviction_signal(df: pd.DataFrame) -> float:
    """Up-day volume share over last _VOL_LOOKBACK sessions."""
    if len(df) < _VOL_LOOKBACK + 1:
        return float("nan")
    window = df.iloc[-(_VOL_LOOKBACK + 1):].copy()
    window["prev_close"] = window["Close"].shift(1)
    window = window.iloc[1:]
    up_vol = float(window.loc[window["Close"] > window["prev_close"], "Volume"].sum())
    down_vol = float(window.loc[window["Close"] <= window["prev_close"], "Volume"].sum())
    total = up_vol + down_vol
    if total == 0:
        return float("nan")
    return up_vol / total


def _minmax_normalize(values: dict[str, float]) -> dict[str, float]:
    valid = {k: v for k, v in values.items() if not math.isnan(v)}
    if not valid:
        return dict(values)
    lo, hi = min(valid.values()), max(valid.values())
    if hi == lo:
        return {k: (0.5 if not math.isnan(v) else float("nan")) for k, v in values.items()}
    return {
        k: ((v - lo) / (hi - lo) if not math.isnan(v) else float("nan"))
        for k, v in values.items()
    }


def _equal_weight_avg(values: list[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else float("nan")


def compute_megacap_scores() -> list[MegaCapResult]:
    """Fetch prices, compute multi-timeframe triple-benchmark RS + trend + vol, rank all 10."""
    tickers_to_fetch = MEGACAP_UNIVERSE + ["QQQ", "XLK"]
    prices = _fetch_ohlcv(tickers_to_fetch)

    qqq = prices.get("QQQ")
    xlk = prices.get("XLK")
    if qqq is None:
        logger.error("QQQ data unavailable — RS vs QQQ will be nan")
    if xlk is None:
        logger.error("XLK data unavailable — RS vs XLK will be nan")

    # --- Pass 1: compute raw per-ticker values ---
    raw: dict[str, dict] = {}
    for ticker in MEGACAP_UNIVERSE:
        df = prices.get(ticker)
        r: dict = {}
        if df is None or df.empty:
            for tf in _RS_TIMEFRAMES:
                r[f"rs_qqq_{tf}"] = float("nan")
                r[f"rs_xlk_{tf}"] = float("nan")
                r[f"hist_pct_{tf}"] = float("nan")
            r["trend_raw"] = 0
            r["trend"] = float("nan")
            r["vol"] = float("nan")
            r["rvol"] = float("nan")
            r["price"] = float("nan")
            raw[ticker] = r
            continue

        r["price"] = float(df["Close"].iloc[-1])
        for tf in _RS_TIMEFRAMES:
            r[f"rs_qqq_{tf}"] = _rs_signal(df, qqq, tf) if qqq is not None else float("nan")
            r[f"rs_xlk_{tf}"] = _rs_signal(df, xlk, tf) if xlk is not None else float("nan")
            r[f"hist_pct_{tf}"] = _hist_pct_signal(df, tf)
        tr_raw, tr_norm = _trend_signal(df)
        r["trend_raw"] = tr_raw
        r["trend"] = tr_norm
        r["vol"] = _vol_conviction_signal(df)
        r["rvol"] = _rvol_signal(df)
        raw[ticker] = r

    # --- Pass 2: cross-sectionally normalize the return-differential signals ---
    for tf in _RS_TIMEFRAMES:
        for benchmark_key in (f"rs_qqq_{tf}", f"rs_xlk_{tf}"):
            vals = {t: raw[t][benchmark_key] for t in MEGACAP_UNIVERSE}
            normed = _minmax_normalize(vals)
            for t in MEGACAP_UNIVERSE:
                raw[t][f"norm_{benchmark_key}"] = normed[t]

    # --- Pass 3: build composite RS per timeframe, then average across timeframes ---
    for ticker in MEGACAP_UNIVERSE:
        r = raw[ticker]
        tf_composites: list[float] = []
        for tf in _RS_TIMEFRAMES:
            tf_score = _equal_weight_avg([
                r[f"norm_rs_qqq_{tf}"],
                r[f"norm_rs_xlk_{tf}"],
                r[f"hist_pct_{tf}"],
            ])
            r[f"composite_rs_{tf}"] = tf_score
            tf_composites.append(tf_score)
        r["composite_rs"] = _equal_weight_avg(tf_composites)

    # --- Pass 4: build final composite and results list ---
    results: list[MegaCapResult] = []
    for ticker in MEGACAP_UNIVERSE:
        r = raw[ticker]
        composite = _equal_weight_avg([r["composite_rs"], r["trend"], r["vol"]])
        results.append(MegaCapResult(
            ticker=ticker,
            price=r["price"],
            rs_20=r["rs_qqq_20"],
            rs_60=r["rs_qqq_60"],
            rs_120=r["rs_qqq_120"],
            hist_pct=r["hist_pct_20"],
            trend_raw=r["trend_raw"],
            trend=r["trend"],
            vol_conviction=r["vol"],
            rvol=r["rvol"],
            composite=composite,
        ))

    results.sort(key=lambda r: r.composite if not math.isnan(r.composite) else -999, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1
        if r.rank <= 3:
            r.label = "LEADING"
        elif r.rank >= 8:
            r.label = "WEAKENING"
        else:
            r.label = "NEUTRAL"

    return results


def fetch_headlines(ticker: str, max_headlines: int = 3) -> list[str]:
    """Fetch recent news headlines for a ticker via yfinance."""
    try:
        news_items = yf.Ticker(ticker).news or []
        headlines = []
        for item in news_items:
            content = item.get("content", {})
            title = (
                (content.get("title") if isinstance(content, dict) else None)
                or item.get("title", "")
            )
            if title:
                headlines.append(title)
            if len(headlines) >= max_headlines:
                break
        return headlines
    except Exception as exc:
        logger.error("news fetch failed for %s: %s", ticker, exc)
        return []
