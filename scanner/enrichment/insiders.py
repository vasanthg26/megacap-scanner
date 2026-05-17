"""Insider transaction enrichment queries.

Reads from insider_transactions (which already has offering-participation filtered out
via is_open_market). Never modifies signal scores or action labels.
"""

from datetime import date, timedelta
from typing import Optional


def get_insider_summary(
    ticker: str,
    as_of: date,
    conn,
    window_days: int = 30,
) -> dict:
    """Aggregate insider buy/sell activity for ticker in the window ending as_of.

    Only counts open-market purchases (is_open_market=TRUE, transaction_code='P').
    Sale aggregates use transaction_code='S' rows.
    Returns a dict suitable for display enrichment — never affects signal scores.
    """
    start = as_of - timedelta(days=window_days)

    buys = conn.execute(
        """
        SELECT
            COUNT(*)                                      AS buy_count,
            COUNT(DISTINCT insider_name)                  AS distinct_insiders,
            COALESCE(SUM(total_value), 0.0)               AS total_dollar_value,
            BOOL_OR(is_officer)                           AS has_officer_buys,
            MAX(total_value)                              AS largest_single_buy,
            MAX(transaction_date)                         AS most_recent_buy_date
        FROM insider_transactions
        WHERE ticker = ?
          AND transaction_code = 'P'
          AND is_open_market = TRUE
          AND transaction_date > ?
          AND transaction_date <= ?
        """,
        [ticker, str(start), str(as_of)],
    ).fetchone()

    sells = conn.execute(
        """
        SELECT
            COALESCE(SUM(total_value), 0.0) AS total_sell_value,
            (
                SELECT insider_title
                FROM insider_transactions it2
                WHERE it2.ticker = ?
                  AND it2.transaction_code = 'S'
                  AND it2.transaction_date > ?
                  AND it2.transaction_date <= ?
                  AND it2.total_value IS NOT NULL
                ORDER BY it2.total_value DESC
                LIMIT 1
            ) AS top_seller_title
        FROM insider_transactions
        WHERE ticker = ?
          AND transaction_code = 'S'
          AND transaction_date > ?
          AND transaction_date <= ?
        """,
        [ticker, str(start), str(as_of), ticker, str(start), str(as_of)],
    ).fetchone()

    buy_count = buys[0] or 0
    distinct_insiders = buys[1] or 0
    total_dollar_value = float(buys[2] or 0.0)
    has_officer_buys = bool(buys[3]) if buys[3] is not None else False
    largest_single_buy = float(buys[4]) if buys[4] is not None else None
    most_recent_buy_date: Optional[date] = buys[5]

    days_since_last_buy: Optional[int] = None
    if most_recent_buy_date is not None:
        if isinstance(most_recent_buy_date, str):
            most_recent_buy_date = date.fromisoformat(most_recent_buy_date)
        days_since_last_buy = (as_of - most_recent_buy_date).days

    total_sell_value = float(sells[0] or 0.0)
    top_seller_title: Optional[str] = sells[1]

    return {
        "window_days": window_days,
        "as_of": as_of,
        "buy_count": buy_count,
        "distinct_insiders": distinct_insiders,
        "total_dollar_value": total_dollar_value,
        "has_officer_buys": has_officer_buys,
        "has_cluster_buy": distinct_insiders >= 3,
        "largest_single_buy": largest_single_buy,
        "most_recent_buy_date": most_recent_buy_date,
        "days_since_last_buy": days_since_last_buy,
        "total_sell_value": total_sell_value,
        "has_notable_selling": total_sell_value >= 500_000,
        "top_seller_title": top_seller_title,
    }
