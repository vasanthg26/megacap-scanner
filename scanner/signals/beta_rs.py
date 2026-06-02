"""Beta-adjusted relative strength — removes market beta to isolate supply chain signal.

Formula:
    residual_child  = return_child  - (beta_child  * return_SPY)
    residual_parent = return_parent - (beta_parent * return_SPY)
    rs_adj = residual_child - residual_parent

Beta is estimated as cov(ticker, SPY) / var(SPY) over a rolling window of daily returns.
No lookahead: all prices fetched as of `as_of` or earlier.
"""

_RS_WINDOW = 20
_BETA_WINDOW = 60


def compute_beta_adjusted_rs(
    child: str,
    parent: str,
    as_of: str,
    conn,
    window: int = _RS_WINDOW,
    beta_window: int = _BETA_WINDOW,
) -> float | None:
    """Return beta-adjusted RS score, or None if insufficient data.

    Needs beta_window + 1 prices for daily returns used in beta estimation.
    The rolling return window (window days) is a subset of that range.
    Returns None rather than raising if SPY is absent or history is too short.
    """
    needed = beta_window + 1  # enough for beta_window daily returns + rolling return

    def _fetch(ticker: str) -> list[float]:
        rows = conn.execute(
            """
            SELECT adj_close FROM prices
            WHERE ticker = ? AND date <= ?
            ORDER BY date DESC LIMIT ?
            """,
            [ticker, as_of, needed],
        ).fetchall()
        return [r[0] for r in rows]

    child_px = _fetch(child)
    parent_px = _fetch(parent)
    spy_px = _fetch("SPY")

    # Prices are DESC (index 0 = most recent). Need at least needed prices.
    if len(child_px) < needed or len(parent_px) < needed or len(spy_px) < needed:
        return None

    # Guard: window must fit inside the fetched series
    if window >= len(child_px) or window >= len(parent_px) or window >= len(spy_px):
        return None

    def _rolling(px: list[float], w: int) -> float | None:
        if not px[w]:
            return None
        return (px[0] - px[w]) / px[w]

    ret_child = _rolling(child_px, window)
    ret_parent = _rolling(parent_px, window)
    ret_spy = _rolling(spy_px, window)

    if ret_child is None or ret_parent is None or ret_spy is None:
        return None

    # Daily returns for beta: px[i] is more recent than px[i+1]
    def _daily_rets(px: list[float]) -> list[float] | None:
        rets = []
        for i in range(beta_window):
            denom = px[i + 1]
            if not denom:
                return None
            rets.append((px[i] - denom) / denom)
        return rets

    dc = _daily_rets(child_px)
    dp = _daily_rets(parent_px)
    ds = _daily_rets(spy_px)

    if dc is None or dp is None or ds is None:
        return None

    def _beta(ticker_rets: list[float], spy_rets: list[float]) -> float | None:
        n = len(spy_rets)
        if n < 2:
            return None
        mean_t = sum(ticker_rets) / n
        mean_s = sum(spy_rets) / n
        cov = sum((ticker_rets[i] - mean_t) * (spy_rets[i] - mean_s) for i in range(n)) / (n - 1)
        var_s = sum((r - mean_s) ** 2 for r in spy_rets) / (n - 1)
        if var_s == 0:
            return None
        return cov / var_s

    beta_child = _beta(dc, ds)
    beta_parent = _beta(dp, ds)

    if beta_child is None or beta_parent is None:
        return None

    resid_child = ret_child - beta_child * ret_spy
    resid_parent = ret_parent - beta_parent * ret_spy
    return resid_child - resid_parent
