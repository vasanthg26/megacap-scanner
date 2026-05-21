"""Signal protocol definition, action classification, and compositing utilities."""

import logging
import math
import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Protocol, Sequence
import duckdb

logger = logging.getLogger(__name__)


class Action(StrEnum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


def classify_action(score: float, universe_scores: Sequence[float]) -> Action:
    """
    Convert a raw score to a discrete action via cross-sectional quintile rank.

    Percentile is computed as the fraction of universe scores strictly below
    `score` (i.e. scipy-style 'strict' rank), so ties always round down.
    NaN scores in universe_scores are ignored; a NaN `score` returns HOLD.

    Q5 (pct >= 0.80) → STRONG_BUY
    Q4 (0.60–0.80)   → BUY
    Q2-Q3 (0.20–0.60) → HOLD
    Q1 (< 0.20)      → SELL
    """
    if math.isnan(score):
        return Action.HOLD
    valid = [s for s in universe_scores if not math.isnan(s)]
    if not valid:
        return Action.HOLD
    pct = sum(s < score for s in valid) / len(valid)
    if pct >= 0.80:
        return Action.STRONG_BUY
    elif pct >= 0.60:
        return Action.BUY
    elif pct >= 0.20:
        return Action.HOLD
    else:
        return Action.SELL


def score_rank(score: float, universe_scores: Sequence[float]) -> tuple[int, int]:
    """
    Return (ordinal_rank, total_n) where rank 1 = highest score.
    NaN scores in universe_scores are excluded from the count.
    """
    valid = [s for s in universe_scores if not math.isnan(s)]
    if not valid:
        return (1, 1)
    rank = sum(s > score for s in valid) + 1
    return (rank, len(valid))


TRADEABLE_REGIMES: frozenset[str] = frozenset({"CORRECTION", "MILD_PULLBACK"})

# Parents with demonstrated positive backtest IC (CORRECTION + MILD_PULLBACK regimes only).
# All other mega-caps are unvalidated — scan shows WAIT labels for their dependents.
VALIDATED_PARENTS: frozenset[str] = frozenset({"MSFT", "META"})


def classify_regime(parent_ret_20d: float) -> str:
    """Classify a parent's 20-day return into a regime label for signal gating.

    CORRECTION and MILD_PULLBACK are the tradeable regimes (positive backtest IC).
    UP and DRAWDOWN suppress action labels — signal has no demonstrated edge there.
    """
    if math.isnan(parent_ret_20d):
        return "UNKNOWN"
    if parent_ret_20d > 0:
        return "UP"
    if parent_ret_20d >= -0.05:
        return "MILD_PULLBACK"
    if parent_ret_20d >= -0.15:
        return "CORRECTION"
    return "DRAWDOWN"


class Signal(Protocol):
    name: str

    def compute(self, ticker: str, date: str, conn: duckdb.DuckDBPyConnection) -> float:
        """
        Return a scalar score for ticker as of date.
        MUST only use data available strictly before or on date (no lookahead).
        Returns float('nan') when insufficient data.
        """
        ...


def weighted_sum(
    signals: list[tuple[Signal, float]],
    ticker: str,
    date: str,
    conn: duckdb.DuckDBPyConnection,
) -> float:
    """Compute a weighted composite score from multiple signals."""
    import math

    total_weight = 0.0
    total_score = 0.0
    for signal, weight in signals:
        score = signal.compute(ticker, date, conn)
        if not math.isnan(score):
            total_score += score * weight
            total_weight += weight

    return total_score / total_weight if total_weight > 0 else float("nan")


def calc_rvol_trend(ticker: str, as_of: str, conn: duckdb.DuckDBPyConnection) -> str:
    """RVOL direction arrow: ↑ building, ↓ fading, → flat, — if unknown."""
    rows = conn.execute(
        """
        SELECT volume, rn FROM (
            SELECT volume, ROW_NUMBER() OVER (ORDER BY date DESC) AS rn
            FROM prices WHERE ticker = ? AND date <= ?
        ) t WHERE rn <= 26
        """,
        [ticker, as_of],
    ).fetchall()
    if len(rows) < 21:
        return "—"
    rows_sorted = sorted(rows, key=lambda x: x[1])
    vols = [float(r[0]) for r in rows_sorted if r[0] is not None]
    if len(vols) < 21:
        return "—"
    today_avg = sum(vols[1:21]) / 20
    if today_avg == 0:
        return "—"
    today_rvol = vols[0] / today_avg
    if len(vols) < 26:
        return "→"
    ago_avg = sum(vols[6:26]) / 20
    if ago_avg == 0:
        return "→"
    ago_rvol = vols[5] / ago_avg
    if today_rvol > ago_rvol * 1.10:
        return "↑"
    if today_rvol < ago_rvol * 0.90:
        return "↓"
    return "→"


def calc_rs_trend(
    ticker: str, parent: str, as_of: str, conn: duckdb.DuckDBPyConnection
) -> str:
    """RS direction arrow vs yesterday: ↑ improving, ↓ weakening, → flat, — if unknown."""
    rows = conn.execute(
        """
        SELECT ticker, adj_close, rn FROM (
            SELECT ticker, adj_close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices WHERE ticker IN (?, ?) AND date <= ?
        ) t WHERE rn <= 22
        """,
        [ticker, parent, as_of],
    ).fetchall()
    closes: dict[str, list[float | None]] = {}
    for tkr, c, rn in rows:
        if tkr not in closes:
            closes[tkr] = [None] * 22
        if rn <= 22:
            closes[tkr][rn - 1] = float(c)
    cc = closes.get(ticker, [None] * 22)
    pc = closes.get(parent, [None] * 22)

    def _rs(i: int) -> float:
        if cc[i] is None or cc[i + 20] is None or pc[i] is None or pc[i + 20] is None:
            return float("nan")
        if cc[i + 20] == 0 or pc[i + 20] == 0:
            return float("nan")
        return (cc[i] - cc[i + 20]) / cc[i + 20] - (pc[i] - pc[i + 20]) / pc[i + 20]

    today_rs = _rs(0)
    yest_rs = _rs(1)
    if math.isnan(today_rs) or math.isnan(yest_rs):
        return "—"
    if today_rs > yest_rs + 0.01:
        return "↑"
    if today_rs < yest_rs - 0.01:
        return "↓"
    return "→"


def calc_days_above_50ma(
    ticker: str, as_of: str, conn: duckdb.DuckDBPyConnection
) -> int:
    """Count consecutive trading days (from today back) where close > 50d MA."""
    rows = conn.execute(
        """
        SELECT adj_close, rn FROM (
            SELECT adj_close, ROW_NUMBER() OVER (ORDER BY date DESC) AS rn
            FROM prices WHERE ticker = ? AND date <= ?
        ) t WHERE rn <= 101
        """,
        [ticker, as_of],
    ).fetchall()
    rows_sorted = sorted(rows, key=lambda x: x[1])
    closes = [float(r[0]) for r in rows_sorted if r[0] is not None]
    if len(closes) < 51:
        return 0
    streak = 0
    for i in range(min(51, len(closes) - 50)):
        ma50 = sum(closes[i + 1 : i + 51]) / 50
        if closes[i] > ma50:
            streak += 1
        else:
            break
    return streak


def calc_price_range_pct(
    ticker: str, as_of: str, conn: duckdb.DuckDBPyConnection
) -> float:
    """Position within 20-day high-low range as a 0–100 percentage."""
    rows = conn.execute(
        """
        SELECT adj_close FROM prices
        WHERE ticker = ? AND date <= ?
        ORDER BY date DESC LIMIT 20
        """,
        [ticker, as_of],
    ).fetchall()
    if len(rows) < 20:
        return float("nan")
    closes = [float(r[0]) for r in rows]
    today_close = closes[0]
    low_20d = min(closes)
    high_20d = max(closes)
    if high_20d == low_20d:
        return float("nan")
    return (today_close - low_20d) / (high_20d - low_20d) * 100


def generate_signal_explanation(
    ticker: str,
    parent: str,
    scan_date: str,
    signal_data: dict,
    conn: duckdb.DuckDBPyConnection,
) -> str | None:
    """Generate a plain-text explanation for a BUY/STRONG_BUY signal using Haiku→Sonnet cascade.

    Checks DB cache first — returns cached explanation if available, no API call.
    Only call for action_label in (BUY, STRONG_BUY).
    """
    action_label = signal_data.get("action_label", "")
    if action_label not in ("BUY", "STRONG_BUY"):
        return None

    cached = conn.execute(
        "SELECT explanation FROM signal_explanations WHERE ticker=? AND parent=? AND scan_date=?",
        [ticker, parent, scan_date],
    ).fetchone()
    if cached is not None:
        return cached[0]

    from scanner.ingest.filings import _get_haiku_client, _get_sonnet_client

    haiku = _get_haiku_client()
    sonnet = _get_sonnet_client()
    if not haiku or not sonnet:
        return None

    def _fmt(v, suffix="", decimals=2, pct=False):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        if pct:
            return f"{v * 100:.1f}%"
        return f"{v:.{decimals}f}{suffix}"

    facts_input = (
        f"Ticker: {ticker} | Parent: {parent} | Action: {action_label}\n"
        f"RS Score: {_fmt(signal_data.get('rs_score'), decimals=4)} | RS Trend: {signal_data.get('rs_trend', 'N/A')}\n"
        f"RVOL: {_fmt(signal_data.get('rvol'), decimals=2)} | RVOL Trend: {signal_data.get('rvol_trend', 'N/A')}\n"
        f"Confirmation Score: {signal_data.get('confirm', 'N/A')}\n"
        f"Days Above 50MA: {signal_data.get('days_above_50ma', 'N/A')}\n"
        f"20-day Price Range Position: {_fmt(signal_data.get('price_range_pct'), decimals=1)}%\n"
        f"Parent Regime: {signal_data.get('parent_regime', 'N/A')} | Parent 20d Return: {_fmt(signal_data.get('parent_20d_return'), pct=True)}\n"
        f"Insider Activity: {signal_data.get('insider_annotation', 'None')}\n"
        f"Earnings Days: {signal_data.get('earnings_days', 'N/A')}"
    )

    try:
        h_msg = haiku.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": (
                    f"Format these signal metrics as 3-4 concise bullet points for a trader:\n{facts_input}"
                ),
            }],
        )
        bullets = h_msg.content[0].text.strip() if h_msg.content else facts_input
    except Exception as exc:
        logger.warning("Haiku signal formatting failed for %s: %s", ticker, exc)
        bullets = facts_input

    try:
        s_msg = sonnet.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=350,
            messages=[{
                "role": "user",
                "content": (
                    f"Write exactly 3 bullet points (max 150 words total) explaining why {ticker} "
                    f"shows a {action_label} signal relative to its parent {parent}. "
                    f"Use these metrics:\n{bullets}\n\n"
                    f"Each bullet: 1-2 sentences. No labels, no numbering, no 'Paragraph X:' prefix. "
                    f"Start each bullet with •. Cover: RS/momentum, confirmation/volume, risks/caveats. "
                    f"Plain text only, no markdown."
                ),
            }],
        )
        explanation = s_msg.content[0].text.strip() if s_msg.content else None
    except Exception as exc:
        logger.warning("Sonnet signal explanation failed for %s: %s", ticker, exc)
        return None

    if explanation:
        try:
            conn.execute(
                """
                INSERT INTO signal_explanations (ticker, parent, scan_date, action_label, explanation, generated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker, parent, scan_date) DO UPDATE SET
                    explanation = excluded.explanation,
                    generated_at = excluded.generated_at
                """,
                [ticker, parent, scan_date, action_label, explanation, datetime.now(timezone.utc)],
            )
        except Exception as exc:
            logger.error("Failed to cache signal explanation for %s/%s/%s: %s", ticker, parent, scan_date, exc)

    return explanation
