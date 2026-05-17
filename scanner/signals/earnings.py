"""Earnings proximity — days to next earnings date and warning labels."""

from datetime import date


def days_to_earnings(ticker: str, as_of_date: date, conn) -> int | None:
    """Return calendar days from as_of_date to next earnings, or None if unknown/past."""
    try:
        row = conn.execute(
            "SELECT next_earnings_date FROM earnings_dates WHERE ticker = ?",
            [ticker],
        ).fetchone()
        if row is None or row[0] is None:
            return None
        nd = row[0]
        if hasattr(nd, "date"):
            nd = nd.date()
        delta = (nd - as_of_date).days
        return delta if delta >= 0 else None
    except Exception:
        return None


def format_earnings_flag(days: int | None) -> str:
    """Return a warning label for earnings proximity; empty string if >14d or unknown."""
    if days is None or days > 14:
        return ""
    if days == 0:
        return "EARNINGS TODAY"
    if days <= 2:
        return "EARNINGS TOMORROW"
    return f"EARNINGS {days}d"


def earnings_flag_severity(days: int | None) -> str:
    """Return CSS severity class for earnings proximity: red/orange/yellow/empty."""
    if days is None or days > 14:
        return ""
    if days <= 2:
        return "red"
    if days <= 7:
        return "orange"
    return "yellow"
