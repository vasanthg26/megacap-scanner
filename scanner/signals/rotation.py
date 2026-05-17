"""Sector rotation score: XLK rank vs XLF, XLE, XLV peer ETFs.

compute_rotation(as_of, conn) -> RotationResult
  Ranks the 4 sector ETFs by equal-weight 20d+60d composite return.
  XLK status: LEADING (rank 1-2), NEUTRAL (rank 3), LAGGING (rank 4).
  Degrades gracefully if any ETF lacks sufficient data.

fetch_vix() -> (level: float, label: str)
  Fetches ^VIX current level from yfinance (no DuckDB storage).
  Labels: LOW (<15), CALM (15-20), ELEVATED (20-30), FEAR (>30).
  Returns (nan, "") on failure.
"""

import logging
import math
from dataclasses import dataclass, field

import duckdb

logger = logging.getLogger(__name__)

_SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV"]
_MIN_ROWS = 65  # need 61 rows for 60d return + 4 extra buffer


@dataclass
class RotationResult:
    xlk_rank: int        # 1-4 among the 4 sector ETFs
    xlk_status: str      # "LEADING", "NEUTRAL", "LAGGING"
    xlk_20d: float       # XLK raw 20d return
    xlk_60d: float       # XLK raw 60d return
    spy_20d: float       # SPY 20d return (nan if not in prices)
    xlk_vs_spy: float    # xlk_20d - spy_20d (nan if either missing)
    data_available: bool # False if any of the 4 ETFs lacks sufficient data
    vix_level: float = field(default_factory=lambda: float("nan"))
    vix_label: str = ""  # "LOW", "CALM", "ELEVATED", "FEAR", or "" if unavailable


def fetch_vix() -> tuple[float, str]:
    """Fetch current VIX level from yfinance. Returns (nan, '') on any failure."""
    try:
        import yfinance as yf
        fi = yf.Ticker("^VIX").fast_info
        level = getattr(fi, "last_price", None) or getattr(fi, "previous_close", None)
        if level is None:
            return float("nan"), ""
        level = float(level)
        if level < 15:
            label = "LOW"
        elif level < 20:
            label = "CALM"
        elif level < 30:
            label = "ELEVATED"
        else:
            label = "FEAR"
        return level, label
    except Exception as exc:
        logger.warning("VIX fetch failed: %s", exc)
        return float("nan"), ""


def _rolling_return(closes: list[float], lookback: int) -> float:
    if len(closes) < lookback + 1:
        return float("nan")
    base = closes[-(lookback + 1)]
    if base == 0:
        return float("nan")
    return (closes[-1] - base) / base


def compute_rotation(as_of: str, conn: duckdb.DuckDBPyConnection) -> RotationResult:
    """Compute XLK rotation status relative to XLF, XLE, XLV as of a given date.

    Queries the prices table (adj_close column) for up to 65 rows per ticker
    on or before `as_of`. Returns data_available=False if any sector ETF has
    insufficient history.
    """
    all_tickers = _SECTOR_ETFS + ["SPY"]

    rows = conn.execute(
        """
        WITH ranked AS (
            SELECT ticker, adj_close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
            WHERE ticker = ANY(?)
              AND date <= ?
        )
        SELECT ticker, adj_close
        FROM ranked
        WHERE rn <= ?
        ORDER BY ticker, rn DESC
        """,
        [all_tickers, as_of, _MIN_ROWS],
    ).fetchall()

    by_ticker: dict[str, list[float]] = {t: [] for t in all_tickers}
    for tkr, close in rows:
        if tkr in by_ticker:
            by_ticker[tkr].append(float(close))

    # Check all 4 sector ETFs have enough data
    data_available = all(len(by_ticker[t]) >= _MIN_ROWS for t in _SECTOR_ETFS)

    etf_composites: dict[str, float] = {}
    xlk_20d = float("nan")
    xlk_60d = float("nan")

    for etf in _SECTOR_ETFS:
        closes = by_ticker[etf]
        r20 = _rolling_return(closes, 20)
        r60 = _rolling_return(closes, 60)
        if etf == "XLK":
            xlk_20d = r20
            xlk_60d = r60
        valid = [v for v in (r20, r60) if not math.isnan(v)]
        etf_composites[etf] = sum(valid) / len(valid) if valid else float("nan")

    # Rank 1 = best composite; rank 4 = worst
    valid_etfs = [(e, c) for e, c in etf_composites.items() if not math.isnan(c)]
    valid_etfs.sort(key=lambda x: x[1], reverse=True)
    rank_map = {e: i + 1 for i, (e, _) in enumerate(valid_etfs)}
    # ETFs with nan get rank len+1
    for e in _SECTOR_ETFS:
        if e not in rank_map:
            rank_map[e] = len(_SECTOR_ETFS)

    xlk_rank = rank_map.get("XLK", len(_SECTOR_ETFS))
    if xlk_rank <= 2:
        xlk_status = "LEADING"
    elif xlk_rank == 3:
        xlk_status = "NEUTRAL"
    else:
        xlk_status = "LAGGING"

    spy_closes = by_ticker["SPY"]
    spy_20d = _rolling_return(spy_closes, 20) if len(spy_closes) >= 21 else float("nan")

    if not math.isnan(xlk_20d) and not math.isnan(spy_20d):
        xlk_vs_spy = xlk_20d - spy_20d
    else:
        xlk_vs_spy = float("nan")

    vix_level, vix_label = fetch_vix()

    return RotationResult(
        xlk_rank=xlk_rank,
        xlk_status=xlk_status,
        xlk_20d=xlk_20d,
        xlk_60d=xlk_60d,
        spy_20d=spy_20d,
        xlk_vs_spy=xlk_vs_spy,
        data_available=data_available,
        vix_level=vix_level,
        vix_label=vix_label,
    )
