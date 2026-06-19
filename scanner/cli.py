"""CLI entry point — `python -m scanner <command>`."""

import logging
import math
import sys
from datetime import date
from typing import Annotated, Optional

# Ensure Unicode output works on Windows terminals with cp1252 default encoding.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

app = typer.Typer(
    name="scanner",
    help="Mega-cap dependency graph stock scanner.",
    no_args_is_help=True,
)
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


@app.command()
def ingest(
    tickers: Annotated[
        Optional[list[str]],
        typer.Argument(help="Specific tickers to ingest (default: full universe)"),
    ] = None,
    force_full: Annotated[
        bool,
        typer.Option(
            "--force-full",
            help="Force full 2-year backfill even if prices exist (use once after Massive API migration)",
            is_flag=True,
        ),
    ] = False,
) -> None:
    """Pull OHLCV for the universe into DuckDB (Massive API or yfinance fallback)."""
    from scanner.ingest.prices import ingest_all

    if force_full:
        console.print("[bold yellow]--force-full: fetching full 2-year history for all tickers...[/bold yellow]")
    console.print("[bold cyan]Starting ingest...[/bold cyan]")
    results = ingest_all(tickers or None, force_full=force_full)

    ok = sum(1 for s in results.values() if s == "ok")
    empty = sum(1 for s in results.values() if s == "empty")
    errors = sum(1 for s in results.values() if s == "error")

    table = Table(title="Ingest Summary", box=box.SIMPLE)
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")
    table.add_row("[green]ok[/green]", str(ok))
    table.add_row("[yellow]empty[/yellow]", str(empty))
    table.add_row("[red]error[/red]", str(errors))
    console.print(table)

    if errors:
        failed = [t for t, s in results.items() if s == "error"]
        console.print(f"[red]Failed tickers:[/red] {', '.join(failed)}")
        sys.exit(1)


_ACTION_STYLE = {
    "STRONG_BUY": "bold green",
    "BUY": "green",
    "HOLD": "yellow",
    "SELL": "red",
}

_REGIME_STYLE = {
    "UP":            "dim",
    "MILD_PULLBACK": "yellow",
    "CORRECTION":    "bold green",
    "DRAWDOWN":      "red",
    "UNKNOWN":       "dim",
}

_REGIME_NOTES = {
    "UP":            "high-confidence signals",
    "MILD_PULLBACK": "moderate signals",
    "CORRECTION":    "high-confidence signals",
    "DRAWDOWN":      "signal suppressed",
    "UNKNOWN":       "insufficient data",
}


def _rolling_return_pct(ticker: str, lookback: int, as_of: str, conn) -> float:
    rows = conn.execute(
        """
        SELECT adj_close
        FROM prices
        WHERE ticker = ? AND date <= ?
        ORDER BY date DESC
        LIMIT ?
        """,
        [ticker, as_of, lookback + 1],
    ).fetchall()
    if len(rows) < lookback + 1:
        return float("nan")
    latest, oldest = rows[0][0], rows[-1][0]
    if oldest == 0:
        return float("nan")
    return (latest - oldest) / oldest


def _fmt_return(r: float) -> str:
    if math.isnan(r):
        return "[dim]n/a[/dim]"
    color = "green" if r > 0 else "red"
    return f"[{color}]{r:+.1%}[/{color}]"


def _fmt_revision(v: float | None) -> str:
    if v is None or math.isnan(v):
        return "[dim]n/a[/dim]"
    color = "green" if v > 0 else ("red" if v < 0 else "dim")
    return f"[{color}]{v:+.2f}[/{color}]"


def _fmt_rvol(v: float) -> str:
    if math.isnan(v):
        return "[dim]n/a[/dim]"
    if v >= 2.0:
        return f"[bold yellow]{v:.2f}x HIGH[/bold yellow]"
    if v < 0.5:
        return f"[dim]{v:.2f}x LOW[/dim]"
    return f"{v:.2f}x"


def _batch_rvol(tickers: list[str], as_of: str, conn) -> dict[str, float]:
    """Compute RVOL for each ticker using the most recent 21 prices up to as_of."""
    if not tickers:
        return {}
    placeholders = ", ".join(["?" for _ in tickers])
    rows = conn.execute(
        f"""
        SELECT ticker, volume, rn
        FROM (
            SELECT ticker, volume,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
            WHERE ticker IN ({placeholders}) AND date <= ?
        ) t
        WHERE rn <= 21
        """,
        tickers + [as_of],
    ).fetchall()
    by_ticker: dict[str, list[tuple[int, float]]] = {}
    for tkr, vol, rn in rows:
        by_ticker.setdefault(tkr, []).append((rn, float(vol) if vol is not None else 0.0))
    result: dict[str, float] = {}
    for tkr, series in by_ticker.items():
        series.sort(key=lambda x: x[0])
        vols = [v for _, v in series]
        if len(vols) < 2:
            result[tkr] = float("nan")
            continue
        today_vol = vols[0]
        avg_vol = sum(vols[1:]) / len(vols[1:])
        result[tkr] = today_vol / avg_vol if avg_vol > 0 else float("nan")
    return result


def _batch_above_50ma(tickers: list[str], as_of: str, conn) -> dict[str, bool]:
    """Return True for each ticker whose latest close is above its 50d MA as of as_of."""
    if not tickers:
        return {}
    placeholders = ", ".join(["?" for _ in tickers])
    rows = conn.execute(
        f"""
        SELECT ticker, adj_close, rn
        FROM (
            SELECT ticker, adj_close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
            WHERE ticker IN ({placeholders}) AND date <= ?
        ) t
        WHERE rn <= 51
        """,
        tickers + [as_of],
    ).fetchall()
    by_ticker: dict[str, list[tuple[int, float]]] = {}
    for tkr, close, rn in rows:
        by_ticker.setdefault(tkr, []).append((rn, float(close)))
    result: dict[str, bool] = {}
    for tkr, series in by_ticker.items():
        series.sort(key=lambda x: x[0])
        closes = [c for _, c in series]
        if len(closes) < 51:
            result[tkr] = False
            continue
        latest = closes[0]
        ma50 = sum(closes[1:51]) / 50
        result[tkr] = latest > ma50
    return result


def _batch_rvol_trend(tickers: list[str], as_of: str, conn) -> dict[str, str]:
    """RVOL direction arrow per ticker: compares today_rvol vs 5d-ago rvol."""
    if not tickers:
        return {}
    placeholders = ", ".join(["?" for _ in tickers])
    rows = conn.execute(
        f"""
        SELECT ticker, volume, rn
        FROM (
            SELECT ticker, volume,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
            WHERE ticker IN ({placeholders}) AND date <= ?
        ) t
        WHERE rn <= 26
        """,
        tickers + [as_of],
    ).fetchall()
    by_ticker: dict[str, list[tuple[int, float]]] = {}
    for tkr, vol, rn in rows:
        by_ticker.setdefault(tkr, []).append((rn, float(vol) if vol is not None else 0.0))
    result: dict[str, str] = {}
    for tkr, series in by_ticker.items():
        series.sort(key=lambda x: x[0])
        vols = [v for _, v in series]
        if len(vols) < 21:
            result[tkr] = "—"
            continue
        today_avg = sum(vols[1:21]) / 20
        if today_avg == 0:
            result[tkr] = "—"
            continue
        today_rvol = vols[0] / today_avg
        if len(vols) < 26:
            result[tkr] = "→"
            continue
        ago_avg = sum(vols[6:26]) / 20
        if ago_avg == 0:
            result[tkr] = "→"
            continue
        ago_rvol = vols[5] / ago_avg
        if today_rvol > ago_rvol * 1.10:
            result[tkr] = "↑"
        elif today_rvol < ago_rvol * 0.90:
            result[tkr] = "↓"
        else:
            result[tkr] = "→"
    return result


def _batch_rs_trend(
    ticker_parent_pairs: list[tuple[str, str]], as_of: str, conn
) -> dict[tuple[str, str], str]:
    """RS direction arrow per (ticker, parent): compares today's raw RS vs yesterday's."""
    if not ticker_parent_pairs:
        return {}
    all_syms = list({t for t, _ in ticker_parent_pairs} | {p for _, p in ticker_parent_pairs})
    placeholders = ", ".join(["?" for _ in all_syms])
    rows = conn.execute(
        f"""
        SELECT ticker, adj_close, rn
        FROM (
            SELECT ticker, adj_close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
            WHERE ticker IN ({placeholders}) AND date <= ?
        ) t
        WHERE rn <= 22
        """,
        all_syms + [as_of],
    ).fetchall()
    closes: dict[str, list] = {}
    for tkr, c, rn in rows:
        if tkr not in closes:
            closes[tkr] = [None] * 22
        if rn <= 22:
            closes[tkr][rn - 1] = float(c)

    def _rs(cc: list, pc: list, i: int) -> float:
        if cc[i] is None or cc[i + 20] is None or pc[i] is None or pc[i + 20] is None:
            return float("nan")
        if cc[i + 20] == 0 or pc[i + 20] == 0:
            return float("nan")
        return (cc[i] - cc[i + 20]) / cc[i + 20] - (pc[i] - pc[i + 20]) / pc[i + 20]

    result: dict[tuple[str, str], str] = {}
    for ticker, parent in ticker_parent_pairs:
        cc = closes.get(ticker, [None] * 22)
        pc = closes.get(parent, [None] * 22)
        today_rs = _rs(cc, pc, 0)
        yest_rs = _rs(cc, pc, 1)
        if math.isnan(today_rs) or math.isnan(yest_rs):
            result[(ticker, parent)] = "—"
        elif today_rs > yest_rs + 0.01:
            result[(ticker, parent)] = "↑"
        elif today_rs < yest_rs - 0.01:
            result[(ticker, parent)] = "↓"
        else:
            result[(ticker, parent)] = "→"
    return result


def _batch_days_above_50ma_count(tickers: list[str], as_of: str, conn) -> dict[str, int]:
    """Consecutive trading days from today back where close > 50d MA."""
    if not tickers:
        return {}
    placeholders = ", ".join(["?" for _ in tickers])
    rows = conn.execute(
        f"""
        SELECT ticker, adj_close, rn
        FROM (
            SELECT ticker, adj_close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
            WHERE ticker IN ({placeholders}) AND date <= ?
        ) t
        WHERE rn <= 101
        """,
        tickers + [as_of],
    ).fetchall()
    by_ticker: dict[str, list[tuple[int, float]]] = {}
    for tkr, c, rn in rows:
        by_ticker.setdefault(tkr, []).append((rn, float(c)))
    result: dict[str, int] = {}
    for tkr, series in by_ticker.items():
        series.sort(key=lambda x: x[0])
        closes = [c for _, c in series]
        if len(closes) < 51:
            result[tkr] = 0
            continue
        streak = 0
        for i in range(min(51, len(closes) - 50)):
            ma50 = sum(closes[i + 1 : i + 51]) / 50
            if closes[i] > ma50:
                streak += 1
            else:
                break
        result[tkr] = streak
    return result


def _batch_price_range_pct(tickers: list[str], as_of: str, conn) -> dict[str, float]:
    """Position within 20-day high-low range as a 0–100 float."""
    if not tickers:
        return {}
    placeholders = ", ".join(["?" for _ in tickers])
    rows = conn.execute(
        f"""
        SELECT ticker, adj_close, rn
        FROM (
            SELECT ticker, adj_close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
            WHERE ticker IN ({placeholders}) AND date <= ?
        ) t
        WHERE rn <= 20
        """,
        tickers + [as_of],
    ).fetchall()
    by_ticker: dict[str, list[tuple[int, float]]] = {}
    for tkr, c, rn in rows:
        by_ticker.setdefault(tkr, []).append((rn, float(c)))
    result: dict[str, float] = {}
    for tkr, series in by_ticker.items():
        series.sort(key=lambda x: x[0])
        closes = [c for _, c in series]
        if len(closes) < 20:
            result[tkr] = float("nan")
            continue
        today_close = closes[0]
        low_20d = min(closes)
        high_20d = max(closes)
        if high_20d == low_20d:
            result[tkr] = float("nan")
            continue
        result[tkr] = (today_close - low_20d) / (high_20d - low_20d) * 100
    return result


def _fmt_rvol_trend(v: float, trend: str) -> str:
    base = _fmt_rvol(v)
    if trend == "↑":
        return f"{base} [green](+)[/green]"
    if trend == "↓":
        return f"{base} [red](-)[/red]"
    if trend == "→":
        return f"{base} [dim](=)[/dim]"
    return base


def _fmt_rs_with_trend(s: float, trend: str) -> str:
    if trend == "↑":
        arrow = "[green](+)[/green]"
    elif trend == "↓":
        arrow = "[red](-)[/red]"
    elif trend == "→":
        arrow = "[dim](=)[/dim]"
    else:
        arrow = "[dim]--[/dim]"
    return f"{s:+.4f} {arrow}"


def _fmt_days_above_50ma(days: int) -> str:
    if days == 0:
        return "[dim]—[/dim]"
    return f"[green]{days}d[/green]"


def _fmt_price_range_pct(pct: float) -> str:
    if math.isnan(pct):
        return "[dim]—[/dim]"
    pct_int = round(pct)
    if pct >= 70:
        return f"[green]{pct_int}%[/green]"
    if pct >= 30:
        return f"[yellow]{pct_int}%[/yellow]"
    return f"[red]{pct_int}%[/red]"


def _fmt_confirm(score: int, max_score: int) -> str:
    pct = score / max_score if max_score > 0 else 0
    if pct >= 0.8:
        return f"[bold green]{score}/{max_score}[/bold green]"
    if pct >= 0.6:
        return f"[green]{score}/{max_score}[/green]"
    if pct >= 0.4:
        return f"[yellow]{score}/{max_score}[/yellow]"
    return f"[dim]{score}/{max_score}[/dim]"


def _insider_category(summary: dict) -> str:
    """Return plain token for insider activity: cluster/officer/single/selling/none."""
    if not summary:
        return "none"
    buys = summary.get("buy_count", 0)
    has_cluster = summary.get("has_cluster_buy", False)
    has_officer = summary.get("has_officer_buys", False)
    has_selling = summary.get("has_notable_selling", False)
    if buys == 0 and has_selling:
        return "selling"
    if buys == 0:
        return "none"
    if has_cluster:
        return "cluster"
    if has_officer:
        return "officer"
    return "single"


def _is_market_open() -> bool:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    mins = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= mins < 16 * 60


def _fetch_current_prices(tickers: list[str]) -> dict[str, float]:
    import yfinance as yf

    use_live = _is_market_open()
    logger = logging.getLogger(__name__)
    prices: dict[str, float] = {}
    for ticker in tickers:
        try:
            fi = yf.Ticker(ticker).fast_info
            live = getattr(fi, "last_price", None)
            prev = getattr(fi, "previous_close", None)
            price = (live or prev) if use_live else (prev or live)
            if price and float(price) > 0:
                prices[ticker] = float(price)
        except Exception as exc:
            logger.warning("price fetch failed for %s: %s", ticker, exc)
    return prices


def _fmt_insider(summary: dict) -> str:
    """Format insider activity summary for scan table display."""
    cat = _insider_category(summary)
    if cat == "selling":
        return "[yellow]! selling[/yellow]"
    if cat == "none":
        return "[dim]-[/dim]"
    if cat == "cluster":
        return "[bold green]++ cluster[/bold green]"
    if cat == "officer":
        return "[green]+ officer[/green]"
    return "[green]+[/green]"


def _render_divergence_panel(flags: list[dict], as_of: str) -> None:
    """Render the divergence flags section. Silently skips if no flags."""
    if not flags:
        return

    _STRENGTH_ICON = {"STRONG": "⚡⚡", "MODERATE": "⚡", "BENCHMARK": "📈", "SECTOR_OUTPERFORM": "🚀"}
    _STRENGTH_LABEL = {"SECTOR_OUTPERFORM": "OUTPERFORM"}

    table = Table(
        title=f"[bold orange1]DIVERGENCE FLAGS — {as_of}[/bold orange1]",
        box=box.SIMPLE_HEAD,
        border_style="orange1",
    )
    table.add_column("Ticker", width=8)
    table.add_column("Theme/Parent", width=22)
    table.add_column("1d%", justify="right", width=8)
    table.add_column("SPY%", justify="right", width=8)
    table.add_column("Bench%", justify="right", width=8)
    table.add_column("Div vs Bench", justify="right", width=13)
    table.add_column("Strength", width=15)

    for f in flags:
        icon = _STRENGTH_ICON.get(f["strength"], "")
        strength_text = _STRENGTH_LABEL.get(f["strength"], f["strength"])
        label = f["theme"] or f["parent"] or f["benchmark"]
        if label and len(label) > 22:
            label = label[:21] + "…"

        def _pct_cell(v: float) -> str:
            color = "green" if v > 0 else "red"
            return f"[{color}]{v:+.1f}%[/{color}]"

        div_color = "bold green" if f["strength"] == "STRONG" else "green"
        table.add_row(
            f["ticker"],
            label or "—",
            _pct_cell(f["ticker_1d"]),
            _pct_cell(f["spy_1d"]),
            _pct_cell(f["benchmark_1d"]),
            f"[{div_color}]{f['divergence_vs_benchmark']:+.1f}%[/{div_color}]",
            f"{icon} {strength_text}",
        )

    console.print(table)
    console.print("[dim]Tip: Check news and filings for these tickers — divergence often precedes a catalyst.[/dim]")


def _render_rotation_panel(rot) -> None:
    """Render the XLK sector rotation banner. Silently skips if rot is None."""
    if rot is None:
        return

    if not rot.data_available:
        console.print(
            Panel(
                "[yellow]XLK rotation data insufficient — run:[/yellow]\n"
                "  [dim]scanner ingest XLK XLF XLE XLV SPY[/dim]",
                title="Sector Rotation",
                border_style="dim",
            )
        )
        return

    status_color = {"LEADING": "green", "NEUTRAL": "yellow", "LAGGING": "red"}.get(
        rot.xlk_status, "white"
    )
    xlk_20d_str = f"{rot.xlk_20d:+.1%}" if not math.isnan(rot.xlk_20d) else "n/a"
    xlk_60d_str = f"{rot.xlk_60d:+.1%}" if not math.isnan(rot.xlk_60d) else "n/a"
    vs_spy_str = (
        f"vs SPY: {rot.xlk_vs_spy:+.1%}" if not math.isnan(rot.xlk_vs_spy) else ""
    )

    vix_str = ""
    if rot.vix_label:
        vix_color = {"LOW": "green", "CALM": "green", "ELEVATED": "yellow", "FEAR": "red"}.get(
            rot.vix_label, "white"
        )
        vix_level_str = f"{rot.vix_level:.1f}" if not math.isnan(rot.vix_level) else "n/a"
        vix_str = f"  |  VIX: [{vix_color}]{vix_level_str} {rot.vix_label}[/{vix_color}]"

    body = (
        f"[{status_color}][bold]XLK {rot.xlk_status}[/bold][/{status_color}]"
        f"  rank {rot.xlk_rank}/4  |  20d: {xlk_20d_str}  |  60d: {xlk_60d_str}"
        + (f"  |  {vs_spy_str}" if vs_spy_str else "")
        + vix_str
    )
    if rot.xlk_status == "LAGGING":
        body += (
            "\n[red]XLK is the weakest sector ETF — tech tailwind absent.[/red]"
        )

    console.print(Panel(body, title="Sector Rotation", border_style=status_color))


@app.command("ingest-insiders")
def ingest_insiders(
    tickers: Annotated[
        Optional[list[str]],
        typer.Argument(help="Specific tickers to ingest (default: full universe)"),
    ] = None,
    lookback_days: Annotated[int, typer.Option(help="Lookback window in calendar days")] = 365,
) -> None:
    """Pull Form 4 insider transactions from SEC EDGAR into DuckDB."""
    import time as _time

    import requests as _requests

    from scanner.db import get_connection
    from scanner.graph.loader import get_all_tickers
    from scanner.ingest.insiders import _load_cik_map, _make_edgar_session, ingest_form4

    if tickers:
        universe = tickers
    else:
        _conn = get_connection()
        try:
            theme_tickers = [r[0] for r in _conn.execute("SELECT DISTINCT ticker FROM themes").fetchall()]
        except Exception:
            theme_tickers = []
        finally:
            _conn.close()
        universe = list(dict.fromkeys(get_all_tickers() + theme_tickers))
    console.print(
        f"[bold cyan]Ingesting Form 4 filings for {len(universe)} tickers "
        f"({lookback_days}d lookback)...[/bold cyan]"
    )

    try:
        edgar_session = _make_edgar_session()
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        sys.exit(1)
    try:
        cik_map_cache = _load_cik_map(edgar_session)
    except Exception as exc:
        console.print(f"[red]Failed to load CIK map:[/red] {exc}")
        sys.exit(1)

    ok_count = 0
    error_count = 0
    total_rows = 0

    conn = get_connection()
    try:
        for ticker in universe:
            try:
                rows = ingest_form4(
                    ticker, lookback_days=lookback_days, conn=conn,
                    cik_map_cache=cik_map_cache, edgar_session=edgar_session,
                )
                total_rows += rows
                ok_count += 1
                console.print(f"  {ticker}: {rows} rows")
            except Exception as exc:
                error_count += 1
                console.print(f"  [red]{ticker}: {exc}[/red]")
    finally:
        conn.close()
        edgar_session.close()

    table = Table(title="Insider Ingest Summary", box=box.SIMPLE)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Tickers processed", str(ok_count))
    table.add_row("[red]Errors[/red]", str(error_count))
    table.add_row("Total rows written", str(total_rows))
    console.print(table)

    if error_count:
        sys.exit(1)


@app.command("ingest-estimates")
def ingest_estimates_cmd(
    tickers: Annotated[
        Optional[list[str]],
        typer.Argument(help="Specific tickers to ingest (default: full universe)"),
    ] = None,
) -> None:
    """Snapshot current analyst estimate revision counts into DuckDB (run daily)."""
    from scanner.ingest.estimates import ingest_estimates

    label = ", ".join(tickers) if tickers else "full universe"
    console.print(f"[bold cyan]Ingesting estimates for {label}...[/bold cyan]")

    results = ingest_estimates(tickers or None)

    ok = sum(1 for s in results.values() if s == "ok")
    no_data = sum(1 for s in results.values() if s == "no_data")
    errors = sum(1 for s in results.values() if s == "error")

    table = Table(title="Estimates Ingest Summary", box=box.SIMPLE)
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")
    table.add_row("[green]ok[/green]", str(ok))
    table.add_row("[yellow]no_data[/yellow]", str(no_data))
    table.add_row("[red]error[/red]", str(errors))
    console.print(table)
    console.print(
        "[dim]Note: this stores a daily snapshot. Run ingest-estimates each day "
        "to accumulate history for backtest-estimates.[/dim]"
    )
    if errors:
        sys.exit(1)


@app.command("ingest-earnings")
def ingest_earnings_cmd(
    tickers: Annotated[
        Optional[list[str]],
        typer.Argument(help="Specific tickers to ingest (default: full universe)"),
    ] = None,
) -> None:
    """Fetch next earnings dates from yfinance into DuckDB (run daily)."""
    from scanner.ingest.earnings import ingest_earnings

    label = ", ".join(tickers) if tickers else "full universe"
    console.print(f"[bold cyan]Ingesting earnings dates for {label}...[/bold cyan]")

    results = ingest_earnings(tickers or None)

    ok = sum(1 for s in results.values() if s == "ok")
    null = sum(1 for s in results.values() if s == "null")
    errors = sum(1 for s in results.values() if s == "error")

    table = Table(title="Earnings Ingest Summary", box=box.SIMPLE)
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")
    table.add_row("[green]ok[/green]", str(ok))
    table.add_row("[yellow]null (no date)[/yellow]", str(null))
    table.add_row("[red]error[/red]", str(errors))
    console.print(table)
    if errors:
        sys.exit(1)


@app.command("ingest-filings")
def ingest_filings_cmd(
    tickers: Annotated[
        Optional[list[str]],
        typer.Argument(help="Specific tickers to ingest (default: full universe)"),
    ] = None,
    days_back: Annotated[int, typer.Option(help="How many calendar days back to fetch")] = 7,
) -> None:
    """Fetch 8-K filings from SEC EDGAR into DuckDB (run daily)."""
    from scanner.ingest.filings import ingest_filings

    label = ", ".join(tickers) if tickers else "full universe"
    console.print(f"[bold cyan]Ingesting 8-K filings for {label} (last {days_back}d)...[/bold cyan]")

    results = ingest_filings(tickers or None, days_back=days_back)

    ok = sum(1 for s in results.values() if s == "ok")
    skipped = sum(1 for s in results.values() if s == "skipped")
    errors = sum(1 for s in results.values() if s == "error")

    table = Table(title="8-K Filings Ingest Summary", box=box.SIMPLE)
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")
    table.add_row("[green]ok[/green]", str(ok))
    table.add_row("[yellow]skipped (no CIK)[/yellow]", str(skipped))
    table.add_row("[red]error[/red]", str(errors))
    console.print(table)
    if errors:
        sys.exit(1)


@app.command("ingest-short")
def ingest_short_cmd(
    tickers: Annotated[
        Optional[list[str]],
        typer.Argument(help="Specific tickers to ingest (default: full universe)"),
    ] = None,
) -> None:
    """Fetch short interest data from Massive API into DuckDB."""
    from scanner.ingest.short_interest import ingest_short_interest

    label = ", ".join(tickers) if tickers else "full universe"
    console.print(f"[bold cyan]Ingesting short interest for {label}...[/bold cyan]")

    results = ingest_short_interest(tickers or None)

    if not results:
        console.print("[yellow]No data fetched — MASSIVE_API_KEY may not be set.[/yellow]")
        return

    ok = sum(1 for s in results.values() if s == "ok")
    empty = sum(1 for s in results.values() if s == "empty")
    errors = sum(1 for s in results.values() if s == "error")

    table = Table(title="Short Interest Ingest Summary", box=box.SIMPLE)
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")
    table.add_row("[green]ok[/green]", str(ok))
    table.add_row("[yellow]empty[/yellow]", str(empty))
    table.add_row("[red]error[/red]", str(errors))
    console.print(table)
    if errors:
        sys.exit(1)


@app.command("insider-recent")
def insider_recent(
    days: Annotated[int, typer.Option(help="Lookback window in calendar days")] = 30,
    ticker: Annotated[Optional[str], typer.Option(help="Filter to a specific ticker")] = None,
    min_value: Annotated[
        float, typer.Option(help="Minimum transaction value in dollars")
    ] = 0.0,
) -> None:
    """List recent open-market insider purchases."""
    from datetime import date as _date, timedelta as _timedelta

    from scanner.db import get_connection

    as_of = _date.today()
    start = as_of - _timedelta(days=days)

    conn = get_connection(read_only=False)
    try:
        where = "transaction_code = 'P' AND is_open_market = TRUE AND transaction_date > ? AND transaction_date <= ?"
        params: list = [str(start), str(as_of)]
        if ticker:
            where += " AND ticker = ?"
            params.append(ticker.upper())
        if min_value > 0:
            where += " AND total_value >= ?"
            params.append(min_value)

        rows = conn.execute(
            f"""
            SELECT ticker, transaction_date, insider_name, insider_title,
                   is_officer, is_director, shares, price_per_share, total_value, filing_url
            FROM insider_transactions
            WHERE {where}
            ORDER BY transaction_date DESC, total_value DESC
            """,
            params,
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        console.print("[yellow]No insider purchases found for the selected filters.[/yellow]")
        return

    table = Table(
        title=f"Open-Market Insider Purchases — last {days} days",
        box=box.SIMPLE_HEAD,
    )
    table.add_column("Ticker", width=7)
    table.add_column("Date", width=12)
    table.add_column("Insider", width=22)
    table.add_column("Title", width=20)
    table.add_column("Shares", justify="right", width=12)
    table.add_column("Price", justify="right", width=10)
    table.add_column("Value", justify="right", width=14)

    for row in rows:
        tkr, txn_date, name, title, is_off, is_dir, shares, price, value, url = row
        badge = ""
        if is_off:
            badge = " [bold yellow]★[/bold yellow]"
        elif is_dir:
            badge = " [cyan]◆[/cyan]"
        name_cell = f"{name or '—'}{badge}"
        shares_str = f"{shares:,.0f}" if shares is not None else "—"
        price_str = f"${price:,.2f}" if price is not None else "—"
        value_str = f"${value:,.0f}" if value is not None else "—"
        table.add_row(tkr, str(txn_date), name_cell, title or "—", shares_str, price_str, value_str)

    console.print(table)
    console.print("[dim]★ = officer  ◆ = director[/dim]")


@app.command()
def scan(
    scan_date: Annotated[
        Optional[str],
        typer.Option("--date", help="Scan date YYYY-MM-DD (default: today)"),
    ] = None,
    lookback: Annotated[int, typer.Option(help="Rolling return lookback in trading days")] = 20,
    top_n: Annotated[int, typer.Option(help="Number of top/bottom children to show per parent")] = 3,
    force_all_regimes: Annotated[
        bool,
        typer.Option("--force-all-regimes", help="Show action labels for all regimes (override gate)", is_flag=True),
    ] = False,
    journal: Annotated[
        bool,
        typer.Option("--journal", help="Append scan output to data/scan_journal.csv", is_flag=True),
    ] = False,
    force_journal: Annotated[
        bool,
        typer.Option("--force-journal", help="Overwrite today's existing journal entry", is_flag=True),
    ] = False,
) -> None:
    """Rank dependents by relative-strength score for each mega-cap."""
    from scanner.db import get_connection
    from scanner.graph.loader import MEGA_CAPS, get_dependents
    from scanner.signals.relative_strength import RelativeStrengthVsParent
    from scanner.enrichment.insiders import get_insider_summary
    from scanner.signals.base import Action, classify_action, score_rank, classify_regime, REGIME_MAP, VALIDATED_PARENTS, calc_basket_zscore

    as_of = scan_date or str(date.today())
    as_of_date = date.fromisoformat(as_of)
    console.print(f"[bold cyan]Scanning universe as of {as_of}[/bold cyan]")

    conn = get_connection()

    # Check whether insider data has been ingested — if not, skip the column entirely.
    insider_row_count = conn.execute("SELECT COUNT(*) FROM insider_transactions").fetchone()[0]
    show_insiders = insider_row_count > 0

    # Check whether estimates data has been ingested.
    estimates_row_count = conn.execute("SELECT COUNT(*) FROM estimates").fetchone()[0]
    show_estimates = estimates_row_count > 0

    # Mega-cap context panel — compute 20d return once, classify regime, display both.
    mc_table = Table(title=f"Mega-caps as of {as_of}", box=box.SIMPLE_HEAD)
    mc_table.add_column("Ticker", width=7)
    mc_table.add_column("1-day", justify="right", width=9)
    mc_table.add_column("5-day", justify="right", width=9)
    mc_table.add_column("20-day", justify="right", width=9)
    mc_table.add_column("Regime", width=16)

    parent_regimes: dict[str, str] = {}
    parent_returns: dict[str, float] = {}
    for mc in MEGA_CAPS:
        ret_20d = _rolling_return_pct(mc, 20, as_of, conn)
        regime = classify_regime(ret_20d)
        parent_regimes[mc] = regime
        parent_returns[mc] = ret_20d
        r_style = _REGIME_STYLE.get(regime, "")
        regime_cell = f"[{r_style}]{regime}[/{r_style}]" if r_style else regime
        mc_table.add_row(
            mc,
            _fmt_return(_rolling_return_pct(mc, 1, as_of, conn)),
            _fmt_return(_rolling_return_pct(mc, 5, as_of, conn)),
            _fmt_return(ret_20d),
            regime_cell,
        )
    console.print(mc_table)

    try:
        from scanner.signals.rotation import compute_rotation as _compute_rotation
        _rot = _compute_rotation(as_of, conn)
    except Exception:
        _rot = None
    _render_rotation_panel(_rot)

    # Divergence flags — tickers rising while market/benchmark fell
    try:
        from scanner.signals.divergence import find_divergences, store_divergences
        _div_flags = find_divergences(conn, as_of, source="dependency")
        if _div_flags:
            store_divergences(_div_flags, as_of, conn)
        _render_divergence_panel(_div_flags, as_of)
    except Exception as _div_exc:
        logger.warning("Divergence flag computation failed: %s", _div_exc)

    # Pass 1 — collect all scores across every parent to build the cross-sectional universe.
    per_parent: dict[str, list[tuple[str, float]]] = {}
    for parent in MEGA_CAPS:
        edges = get_dependents(parent)
        if not edges:
            continue
        signal = RelativeStrengthVsParent(parent=parent, lookback=lookback)
        seen: dict[str, float] = {}
        for edge in edges:
            if edge.child in seen:
                continue
            s = signal.compute(edge.child, as_of, conn)
            if not math.isnan(s):
                seen[edge.child] = s
        per_parent[parent] = sorted(seen.items(), key=lambda x: x[1], reverse=True)

    universe_scores = [s for rows in per_parent.values() for _, s in rows]

    all_scan_tickers = list({t for scored in per_parent.values() for t, _ in scored})

    # Batch-load most recent revision scores for all tickers in scope.
    est_scores: dict[str, float | None] = {}
    if show_estimates:
        for t in all_scan_tickers:
            row = conn.execute(
                "SELECT revision_score FROM estimates WHERE ticker = ? ORDER BY date DESC LIMIT 1",
                [t],
            ).fetchone()
            est_scores[t] = row[0] if row else None

    # Batch-load RVOL for all dependent tickers from DuckDB prices.
    rvol_scores: dict[str, float] = _batch_rvol(all_scan_tickers, as_of, conn)

    # Batch-load above-50d-MA flag for all dependent tickers.
    above_50ma_flags: dict[str, bool] = _batch_above_50ma(all_scan_tickers, as_of, conn)

    # Batch-load new signal indicators.
    rvol_trend_flags: dict[str, str] = _batch_rvol_trend(all_scan_tickers, as_of, conn)
    ticker_parent_pairs = [(t, p) for p, scored in per_parent.items() for t, _ in scored]
    rs_trend_flags: dict[tuple[str, str], str] = _batch_rs_trend(ticker_parent_pairs, as_of, conn)
    days_50ma_counts: dict[str, int] = _batch_days_above_50ma_count(all_scan_tickers, as_of, conn)
    price_range_pcts: dict[str, float] = _batch_price_range_pct(all_scan_tickers, as_of, conn)

    # Pre-compute insider summaries for all tickers in scope (one pass, reused across tables).
    insider_summaries: dict[str, dict] = {}
    if show_insiders:
        for t in all_scan_tickers:
            insider_summaries[t] = get_insider_summary(t, as_of_date, conn)

    # Batch-load earnings dates — optional, silently skip if table is empty.
    earnings_next_dates: dict[str, object] = {}
    try:
        ec = conn.execute("SELECT COUNT(*) FROM earnings_dates").fetchone()[0]
        if ec > 0:
            placeholders = ", ".join(["?" for _ in all_scan_tickers])
            for tkr, nd in conn.execute(
                f"SELECT ticker, next_earnings_date FROM earnings_dates WHERE ticker IN ({placeholders})",
                all_scan_tickers,
            ).fetchall():
                earnings_next_dates[tkr] = nd
    except Exception:
        pass

    # Batch-load short interest — optional, silently skip if table is empty.
    short_pct: dict[str, float | None] = {}
    show_short = False
    try:
        si_count = conn.execute("SELECT COUNT(*) FROM short_interest").fetchone()[0]
        if si_count > 0:
            show_short = True
            placeholders = ", ".join(["?" for _ in all_scan_tickers])
            for tkr, sp in conn.execute(
                f"""
                SELECT ticker, short_percent_of_float
                FROM (
                    SELECT ticker, short_percent_of_float,
                           ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY settlement_date DESC) AS rn
                    FROM short_interest
                    WHERE ticker IN ({placeholders})
                ) t WHERE rn = 1
                """,
                all_scan_tickers,
            ).fetchall():
                short_pct[tkr] = sp
    except Exception:
        pass

    console.print(f"[dim]Fetching prices for {len(all_scan_tickers)} tickers...[/dim]")
    ticker_prices = _fetch_current_prices(all_scan_tickers)

    # Journal setup — DuckDB idempotency check before the render loop.
    journal_rows: list[dict] = []

    if journal:
        today_exists = conn.execute(
            "SELECT 1 FROM journal WHERE scan_date = ? LIMIT 1", [as_of]
        ).fetchone()
        if today_exists and not force_journal:
            console.print(f"[yellow]Journal already contains an entry for {as_of}. Use --force-journal to overwrite.[/yellow]")
            journal = False
        elif today_exists and force_journal:
            conn.execute("DELETE FROM journal WHERE scan_date = ?", [as_of])

    # Pass 2 — render per-parent tables with regime header and gated action labels.
    for parent in MEGA_CAPS:
        scored = per_parent.get(parent)
        if scored is None:
            continue
        if not scored:
            console.print(f"[yellow]{parent}[/yellow]: no data")
            continue

        regime = parent_regimes.get(parent, "UNKNOWN")
        ret_20d = parent_returns.get(parent, float("nan"))
        is_tradeable = (regime in REGIME_MAP.get(parent, [])) or force_all_regimes
        is_validated = parent in VALIDATED_PARENTS
        regime_note = (
            "gate overridden" if force_all_regimes
            else _REGIME_NOTES.get(regime, "") if is_tradeable
            else "signal suppressed"
        )

        table = Table(
            title=f"{parent} dependents - regime: {regime} ({regime_note})",
            box=box.SIMPLE_HEAD,
        )
        table.add_column("Ticker", width=7)
        table.add_column("Parent", width=7)
        table.add_column("Price", justify="right", width=10)
        table.add_column("RS Score", justify="right", width=12)
        table.add_column("RS Adj", justify="right", width=10)
        table.add_column("Action", width=18)
        table.add_column("Rank", justify="right", width=10)
        if show_estimates:
            table.add_column("Est Rev", justify="right", width=9)
        table.add_column("RVOL", justify="right", width=16)
        table.add_column("Confirm", justify="right", width=9)
        table.add_column("50MA", justify="right", width=6)
        table.add_column("Range", justify="right", width=8)
        table.add_column("Z", justify="right", width=8)
        if show_short:
            table.add_column("Short%", justify="right", width=8)
        if show_insiders:
            table.add_column("Insiders (30d)", width=14)

        def _add_row(
            ticker: str,
            s: float,
            _tradeable: bool = is_tradeable,
            _validated: bool = is_validated,
            _parent: str = parent,
            _regime: str = regime,
            _ret_20d: float = ret_20d,
            _prices: dict = ticker_prices,
        ) -> None:
            from scanner.signals.confirmation import compute_confirmation_dependent
            from scanner.signals.earnings import format_earnings_flag
            from scanner.signals.beta_rs import compute_beta_adjusted_rs

            action = classify_action(s, universe_scores)
            rank, total = score_rank(s, universe_scores)
            if not _validated:
                action_cell = "[dim]WAIT - unvalidated[/dim]"
                action_label = "WAIT - unvalidated"
            elif _tradeable:
                style = _ACTION_STYLE[action]
                action_cell = f"[{style}]{action}[/{style}]"
                action_label = str(action)
            else:
                action_cell = "[dim]WAIT[/dim]"
                action_label = "WAIT"

            # Earnings proximity flag — informational only, never suppresses action.
            nd = earnings_next_dates.get(ticker)
            days_out: int | None = None
            if nd is not None:
                nd_date = nd.date() if hasattr(nd, "date") else nd
                delta = (nd_date - as_of_date).days
                if delta >= 0:
                    days_out = delta
            eflag = format_earnings_flag(days_out)
            if eflag:
                action_cell = action_cell + f"  [yellow]{eflag}[/yellow]"

            ins = insider_summaries.get(ticker, {}) if show_insiders else {}
            raw_price = _prices.get(ticker)
            price_cell = f"${raw_price:,.2f}" if raw_price is not None else "[dim]n/a[/dim]"

            rvol_val = rvol_scores.get(ticker, float("nan"))
            rev_score_val = est_scores.get(ticker) if show_estimates else None
            insider_buying = ins.get("buy_count", 0) > 0
            above_50ma = above_50ma_flags.get(ticker, False)
            confirm_score = compute_confirmation_dependent(
                rs_score=s,
                rvol=rvol_val,
                rev_score=rev_score_val,
                insider_buying=insider_buying,
                above_50ma=above_50ma,
            )

            rs_trend_val = rs_trend_flags.get((ticker, _parent), "—")
            rvol_trend_val = rvol_trend_flags.get(ticker, "—")

            try:
                rs_adj_val = compute_beta_adjusted_rs(ticker, _parent, as_of, conn)
            except Exception:
                rs_adj_val = None
            if rs_adj_val is not None:
                adj_color = "green" if rs_adj_val >= 0 else "red"
                rs_adj_cell = f"[{adj_color}]{rs_adj_val:+.4f}[/{adj_color}]"
            else:
                rs_adj_cell = "[dim]—[/dim]"

            base_cells = [ticker, _parent, price_cell, _fmt_rs_with_trend(s, rs_trend_val), rs_adj_cell, action_cell, f"({rank}/{total})"]
            if show_estimates:
                base_cells.append(_fmt_revision(rev_score_val))
            base_cells.append(_fmt_rvol_trend(rvol_val, rvol_trend_val))
            base_cells.append(_fmt_confirm(confirm_score, 5))
            base_cells.append(_fmt_days_above_50ma(days_50ma_counts.get(ticker, 0)))
            base_cells.append(_fmt_price_range_pct(price_range_pcts.get(ticker, float("nan"))))
            z = calc_basket_zscore(ticker, as_of, conn)
            z_str = f"[green]{z:+.2f}[/green]" if z is not None and z >= 0 else (f"[red]{z:+.2f}[/red]" if z is not None else "[dim]—[/dim]")
            base_cells.append(z_str)
            if show_short:
                sp = short_pct.get(ticker)
                if sp is not None:
                    short_cell = f"[orange1]{sp:.1f}%[/orange1]" if sp >= 10.0 else f"{sp:.1f}%"
                else:
                    short_cell = "[dim]—[/dim]"
                base_cells.append(short_cell)
            if show_insiders:
                base_cells.append(_fmt_insider(ins))
            table.add_row(*base_cells)
            if journal:
                cat = _insider_category(ins)
                if cat == "cluster":
                    n_buyers = ins.get("distinct_insiders", 0)
                    note = (
                        f"{n_buyers} insiders bought {ticker} with their own money in the last 30 days. "
                        f"Not actionable yet -- scanner only trades {ticker} when {_parent} is pulling back, not trending up."
                    )
                else:
                    note = ""
                journal_rows.append({
                    "scan_date": as_of,
                    "parent": _parent,
                    "parent_regime": _regime,
                    "ticker": ticker,
                    "price": raw_price,
                    "rs_score": s,
                    "rank": f"({rank}/{total})",
                    "action_label": action_label,
                    "earnings_days": days_out,
                    "est_rev": rev_score_val,
                    "rvol": rvol_val if not math.isnan(rvol_val) else None,
                    "confirm": confirm_score,
                    "insider_annotation": cat,
                    "parent_20d_return": _ret_20d if not math.isnan(_ret_20d) else None,
                    "notes": note,
                })

        shown: set[str] = set()
        for ticker, s in scored[:top_n]:
            shown.add(ticker)
            _add_row(ticker, s)

        bottom = [(t, s) for t, s in scored[-top_n:] if t not in shown]
        if bottom:
            table.add_section()
            for ticker, s in bottom:
                _add_row(ticker, s)

        console.print(table)

    if journal and journal_rows:
        for row in journal_rows:
            conn.execute(
                """INSERT OR REPLACE INTO journal
                   (scan_date, ticker, parent, parent_regime, price, rs_score,
                    rank, action_label, est_rev, rvol, confirm, insider_annotation,
                    earnings_days, parent_20d_return, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    row["scan_date"], row["ticker"], row["parent"], row["parent_regime"],
                    row["price"], row["rs_score"], row["rank"], row["action_label"],
                    row["est_rev"], row["rvol"], row["confirm"], row["insider_annotation"],
                    row["earnings_days"], row["parent_20d_return"], row["notes"],
                ],
            )
        console.print(f"[dim]Journal updated: {len(journal_rows)} rows written to DuckDB[/dim]")

    conn.close()


@app.command("scan-themes")
def scan_themes(
    scan_date: Annotated[
        Optional[str],
        typer.Option("--date", help="Scan date YYYY-MM-DD (default: latest price date)"),
    ] = None,
) -> None:
    """Run theme basket scanner — RS vs benchmark ETF for all active themes."""
    from scanner.db import get_connection
    from scanner.signals.theme_scanner import compute_theme_scan

    conn = get_connection()
    try:
        as_of = scan_date
        if not as_of:
            row = conn.execute("SELECT MAX(date) FROM prices").fetchone()
            as_of = str(row[0]) if row and row[0] else str(date.today())

        console.print(f"[bold cyan]Theme scan as of {as_of}[/bold cyan]")
        results = compute_theme_scan(conn, as_of)
    finally:
        conn.close()

    if not results:
        console.print("[yellow]No theme results — run ingest first.[/yellow]")
        return

    _THEME_ACTIVE_STYLE = {"ACTIVE": "bold green", "NEUTRAL": "yellow", "INACTIVE": "red", "UNKNOWN": "dim"}

    _VALIDATED_FOOTNOTES = {
        "HUT": "[dim]  Validated: IC +0.062 at 10d (WGMI benchmark)[/dim]",
        "NOK": "[dim]  Validated post-pivot (2024+): IC +0.124 at 20d (XLK benchmark) -- limited history[/dim]",
        "BB":  "[dim]  Validated post-pivot (2024+): IC +0.239 at 20d (XLK benchmark)[/dim]",
    }

    _ACTION_LABEL_STYLE = {
        "STRONG_BUY": "bold green",
        "BUY": "green",
        "HOLD": "yellow",
        "SELL": "red",
        "WATCH": "dim",
    }

    current_theme = None
    table: Table | None = None
    pending_footnotes: list[str] = []

    def _print_table():
        if table is not None:
            console.print(table)
        for fn in pending_footnotes:
            console.print(fn)
        pending_footnotes.clear()

    for r in sorted(results, key=lambda x: (x["theme_name"], -(x["rs_score"] or -999))):
        if r["theme_name"] != current_theme:
            _print_table()
            current_theme = r["theme_name"]
            theme_label = r.get("theme_label", r["theme_name"])
            theme_active = r["theme_active"]
            ta_style = _THEME_ACTIVE_STYLE.get(theme_active, "")
            table = Table(
                title=f"{theme_label} [benchmark: {r['benchmark_etf']}]  [{ta_style}]{theme_active}[/{ta_style}]",
                box=box.SIMPLE_HEAD,
            )
            table.add_column("Ticker", width=7)
            table.add_column("Signal", width=11)
            table.add_column("Price", justify="right", width=9)
            table.add_column("RS vs ETF", justify="right", width=10)
            table.add_column("RS Trend", justify="center", width=9)
            table.add_column("RVOL", justify="right", width=7)
            table.add_column("RVOL Trnd", justify="center", width=10)
            table.add_column("Confirm", justify="right", width=9)
            table.add_column("50MA Days", justify="right", width=10)
            table.add_column("Range%", justify="right", width=8)
            table.add_column("Est Rev", justify="right", width=9)
            table.add_column("Insiders", width=14)
            table.add_column("Earnings", width=14)

        def _fmt(v, decimals=2, pct=False, suffix=""):
            if v is None:
                return "—"
            if pct:
                return f"{v * 100:.1f}%"
            return f"{v:.{decimals}f}{suffix}"

        earnings_str = ""
        ed = r.get("earnings_days")
        if ed is not None:
            if ed == 0:
                earnings_str = "[red bold]TODAY[/red bold]"
            elif ed <= 2:
                earnings_str = f"[red]~{ed}d[/red]"
            elif ed <= 7:
                earnings_str = f"[orange1]~{ed}d[/orange1]"
            elif ed <= 14:
                earnings_str = f"[yellow]~{ed}d[/yellow]"
            else:
                earnings_str = f"[dim]~{ed}d[/dim]"

        est_rev = r.get("est_rev")
        est_rev_str = "—" if est_rev is None else f"{est_rev:+.2f}"

        rs = r.get("rs_score")
        rs_str = "—" if rs is None else f"{rs * 100:+.2f}%"

        action = r.get("action_label", "WATCH")
        action_style = _ACTION_LABEL_STYLE.get(action, "dim")
        action_cell = f"[{action_style}]{action}[/{action_style}]"

        table.add_row(
            r["ticker"],
            action_cell,
            _fmt(r.get("price"), decimals=2),
            rs_str,
            r.get("rs_trend") or "—",
            _fmt(r.get("rvol"), decimals=2),
            r.get("rvol_trend") or "—",
            str(r.get("confirm", "—")),
            str(r.get("days_above_50ma", "—")),
            _fmt(r.get("price_range_pct"), decimals=1),
            est_rev_str,
            r.get("insider_annotation") or "",
            earnings_str,
        )

        if r["ticker"] in _VALIDATED_FOOTNOTES:
            pending_footnotes.append(_VALIDATED_FOOTNOTES[r["ticker"]])

    _print_table()
    console.print(
        "\n[dim]Validated: HUT (WGMI), NOK (XLK 2024+), BB (XLK 2024+)\n"
        "Exploratory: BE, AEHR, FPS, YSS -- insufficient edge or data[/dim]"
    )

    # Theme divergence flags — open a fresh connection (original was closed above)
    try:
        from scanner.db import get_connection as _gc
        from scanner.signals.divergence import find_divergences, store_divergences
        _div_conn = _gc()
        try:
            _tdiv = find_divergences(_div_conn, as_of, source="theme")
            if _tdiv:
                store_divergences(_tdiv, as_of, _div_conn)
        finally:
            _div_conn.close()
        _render_divergence_panel(_tdiv, as_of)
    except Exception as _tdiv_exc:
        logging.getLogger(__name__).warning("Theme divergence flag computation failed: %s", _tdiv_exc)


_MEGACAP_LABEL_STYLE = {
    "LEADING": "bold green",
    "NEUTRAL": "yellow",
    "WEAKENING": "red",
}


@app.command("scan-megacap")
def scan_megacap() -> None:
    """Rank the 10 mega-caps by composite score (RS vs QQQ, Trend, Volume Conviction)."""
    from datetime import date as _date

    from scanner.db import get_connection as _get_conn
    from scanner.enrichment.insiders import get_insider_summary
    from scanner.signals.confirmation import compute_confirmation_megacap
    from scanner.signals.megacap import compute_megacap_scores, fetch_headlines

    console.print("[bold cyan]Fetching prices and computing mega-cap composite scores...[/bold cyan]")

    results = compute_megacap_scores()

    as_of_date = _date.today()
    as_of_str = str(as_of_date)

    _conn = _get_conn()
    revision_scores: dict[str, float | None] = {}
    insider_summaries_mc: dict[str, dict] = {}
    earnings_next_dates_mc: dict[str, object] = {}
    has_insider_data = False
    _rot_mc = None
    rvol_trend_mc: dict[str, str] = {}
    rs_trend_mc: dict[tuple[str, str], str] = {}
    days_50ma_mc: dict[str, int] = {}
    range_pct_mc: dict[str, float] = {}
    try:
        insider_row_count = _conn.execute("SELECT COUNT(*) FROM insider_transactions").fetchone()[0]
        has_insider_data = insider_row_count > 0

        for r in results:
            row = _conn.execute(
                "SELECT revision_score FROM estimates WHERE ticker = ? ORDER BY date DESC LIMIT 1",
                [r.ticker],
            ).fetchone()
            revision_scores[r.ticker] = row[0] if row else None
            if has_insider_data:
                insider_summaries_mc[r.ticker] = get_insider_summary(r.ticker, as_of_date, _conn)

        try:
            ec = _conn.execute("SELECT COUNT(*) FROM earnings_dates").fetchone()[0]
            if ec > 0:
                mc_tickers = [r.ticker for r in results]
                placeholders = ", ".join(["?" for _ in mc_tickers])
                for tkr, nd in _conn.execute(
                    f"SELECT ticker, next_earnings_date FROM earnings_dates WHERE ticker IN ({placeholders})",
                    mc_tickers,
                ).fetchall():
                    earnings_next_dates_mc[tkr] = nd
        except Exception:
            pass

        try:
            from scanner.signals.rotation import compute_rotation as _compute_rot_mc
            _rot_mc = _compute_rot_mc(as_of_str, _conn)
        except Exception:
            pass

        mc_tickers_list = [r.ticker for r in results]
        rvol_trend_mc = _batch_rvol_trend(mc_tickers_list, as_of_str, _conn)
        rs_trend_mc = _batch_rs_trend([(r.ticker, "QQQ") for r in results], as_of_str, _conn)
        days_50ma_mc = _batch_days_above_50ma_count(mc_tickers_list, as_of_str, _conn)
        range_pct_mc = _batch_price_range_pct(mc_tickers_list, as_of_str, _conn)
    finally:
        _conn.close()

    _render_rotation_panel(_rot_mc)

    table = Table(title="Mega-Cap Rankings", box=box.SIMPLE_HEAD)
    table.add_column("Rank", justify="right", width=6)
    table.add_column("Ticker", width=7)
    table.add_column("Price", justify="right", width=11)
    table.add_column("RS-20", justify="right", width=12)
    table.add_column("RS-60", justify="right", width=9)
    table.add_column("RS-120", justify="right", width=9)
    table.add_column("Trend", justify="right", width=7)
    table.add_column("Vol Conv", justify="right", width=10)
    table.add_column("Est Rev", justify="right", width=9)
    table.add_column("RVOL", justify="right", width=16)
    table.add_column("Confirm", justify="right", width=9)
    table.add_column("50MA", justify="right", width=6)
    table.add_column("Range", justify="right", width=8)
    table.add_column("Hist Pct", justify="right", width=10)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Label", width=12)
    table.add_column("Earnings", width=18)

    def _fmt_rs(v: float) -> str:
        if math.isnan(v):
            return "[dim]n/a[/dim]"
        color = "green" if v > 0 else "red"
        return f"[{color}]{v:+.1%}[/{color}]"

    for r in results:
        from scanner.signals.earnings import format_earnings_flag, earnings_flag_severity

        label_style = _MEGACAP_LABEL_STYLE.get(r.label, "")
        label_cell = f"[{label_style}]{r.label}[/{label_style}]" if label_style else r.label

        price_str = f"${r.price:,.2f}" if not math.isnan(r.price) else "[dim]n/a[/dim]"
        hist_cell = f"{r.hist_pct:.0%}" if not math.isnan(r.hist_pct) else "[dim]n/a[/dim]"
        trend_cell = f"{r.trend_raw}/2" if not math.isnan(r.trend) else "[dim]n/a[/dim]"
        vol_cell = f"{r.vol_conviction:.1%}" if not math.isnan(r.vol_conviction) else "[dim]n/a[/dim]"
        score_cell = f"{r.composite:.3f}" if not math.isnan(r.composite) else "[dim]n/a[/dim]"

        rev_score = revision_scores.get(r.ticker)
        rev_cell = _fmt_revision(rev_score)
        rvol_cell = _fmt_rvol_trend(r.rvol, rvol_trend_mc.get(r.ticker, "—"))

        ins_summary = insider_summaries_mc.get(r.ticker, {})
        insider_buying = ins_summary.get("buy_count", 0) > 0
        confirm_score = compute_confirmation_megacap(
            rs_20=r.rs_20,
            rvol=r.rvol,
            rev_score=rev_score,
            insider_buying=insider_buying,
            trend_raw=r.trend_raw,
        )
        confirm_cell = _fmt_confirm(confirm_score, 5)

        rs20_trend = rs_trend_mc.get((r.ticker, "QQQ"), "—")
        days_50ma_cell = _fmt_days_above_50ma(days_50ma_mc.get(r.ticker, 0))
        range_pct_cell = _fmt_price_range_pct(range_pct_mc.get(r.ticker, float("nan")))

        nd = earnings_next_dates_mc.get(r.ticker)
        days_out: int | None = None
        if nd is not None:
            nd_date = nd.date() if hasattr(nd, "date") else nd
            delta = (nd_date - as_of_date).days
            if delta >= 0:
                days_out = delta
        eflag = format_earnings_flag(days_out)
        sev = earnings_flag_severity(days_out)
        if eflag and sev:
            earnings_cell = f"[{sev}]{eflag}[/{sev}]"
        else:
            earnings_cell = "[dim]—[/dim]"

        table.add_row(
            str(r.rank), r.ticker, price_str,
            _fmt_rs_with_trend(r.rs_20, rs20_trend), _fmt_rs(r.rs_60), _fmt_rs(r.rs_120),
            trend_cell, vol_cell, rev_cell, rvol_cell, confirm_cell,
            days_50ma_cell, range_pct_cell,
            hist_cell, score_cell, label_cell, earnings_cell,
        )

    console.print(table)
    console.print(
        "[dim]RS-20/60/120: return vs QQQ over that window  |  "
        "Trend: price above 50d/200d MA (out of 2)  |  "
        "Vol Conv: up-day volume share (20 sessions)  |  "
        "Est Rev: analyst revision score (up-down)/(up+down) last 30d, -1 to +1  |  "
        "RVOL: today volume / 20d avg volume (HIGH>=2.0x, LOW<0.5x)  |  "
        "Confirm: cross-signal confirmation score 0-5 (RS-20+RVOL+EstRev+Insider+Trend) -- informational only  |  "
        "Hist Pct: 20d return percentile vs own 252d history  |  "
        "Score: composite of multi-timeframe RS (QQQ+XLK+own history) + Trend + Vol[/dim]"
    )

    console.print(
        "\n[bold]Recent Headlines[/bold]  "
        "[dim][CONTEXT ONLY — not a signal][/dim]\n"
    )
    seen_headlines: set[str] = set()
    for r in results:
        console.print(f"  [bold]{r.ticker}[/bold]  [dim]{r.label}[/dim]")
        all_headlines = fetch_headlines(r.ticker)
        unique = [h for h in all_headlines if h.lower() not in seen_headlines]
        seen_headlines.update(h.lower() for h in unique)
        if unique:
            for h in unique:
                console.print(f"    • {h}")
        else:
            console.print("    [dim]No unique headlines available[/dim]")
        console.print()


@app.command("backtest-megacap")
def backtest_megacap() -> None:
    """Walk-forward backtest of the scan-megacap composite signal."""
    from scanner.backtest.megacap_runner import run_megacap_backtest

    console.print("[bold cyan]Running mega-cap walk-forward backtest...[/bold cyan]")
    console.print("[dim]This may take a minute — loading price history from DuckDB.[/dim]\n")

    try:
        result = run_megacap_backtest()
    except RuntimeError as exc:
        console.print(f"[red]Backtest failed:[/red] {exc}")
        sys.exit(1)

    def _ic_color(v: float) -> str:
        if math.isnan(v):
            return "dim"
        return "green" if v > 0.05 else ("yellow" if v > 0 else "red")

    def _fmt_ic(v: float) -> str:
        if math.isnan(v):
            return "[dim]n/a[/dim]"
        c = _ic_color(v)
        return f"[{c}]{v:+.4f}[/{c}]"

    def _fmt_ret(v: float) -> str:
        if math.isnan(v):
            return "[dim]n/a[/dim]"
        color = "green" if v > 0 else "red"
        return f"[{color}]{v:+.2%}[/{color}]"

    # --- Table 1: Overall IC summary ---
    ic_table = Table(
        title=f"Mega-Cap Composite IC  ({result.start_date} to {result.end_date}  |  {result.n_dates} dates  |  {result.n_obs} obs)",
        box=box.SIMPLE_HEAD,
    )
    ic_table.add_column("Horizon", width=10)
    ic_table.add_column("IC Mean", justify="right", width=11)
    ic_table.add_column("IC Std", justify="right", width=11)
    ic_table.add_column("IC IR", justify="right", width=11)

    for h in (5, 10, 20):
        ic_table.add_row(
            f"{h}d",
            _fmt_ic(result.ic_mean[h]),
            f"{result.ic_std[h]:.4f}" if not math.isnan(result.ic_std[h]) else "[dim]n/a[/dim]",
            _fmt_ic(result.ic_ir[h]),
        )
    console.print(ic_table)

    # --- Table 2: H1/H2 out-of-sample split ---
    split_table = Table(
        title=(
            f"Out-of-Sample H1/H2 Split  "
            f"(H1: {result.start_date} to {result.mid_date} n={result.n_h1}  |  "
            f"H2: {result.mid_date} to {result.end_date} n={result.n_h2})"
        ),
        box=box.SIMPLE_HEAD,
    )
    split_table.add_column("Horizon", width=10)
    split_table.add_column("H1 IC Mean", justify="right", width=12)
    split_table.add_column("H2 IC Mean", justify="right", width=12)
    split_table.add_column("Both +?", width=10)

    for h in (5, 10, 20):
        h1 = result.ic_h1_mean[h]
        h2 = result.ic_h2_mean[h]
        both_pos = (not math.isnan(h1) and h1 > 0) and (not math.isnan(h2) and h2 > 0)
        consistent = "[green]Yes[/green]" if both_pos else "[red]No[/red]"
        split_table.add_row(f"{h}d", _fmt_ic(h1), _fmt_ic(h2), consistent)
    console.print(split_table)

    # --- Table 3: Label spread ---
    spread_table = Table(title="LEADING vs WEAKENING Average Forward Return", box=box.SIMPLE_HEAD)
    spread_table.add_column("Horizon", width=10)
    spread_table.add_column("LEADING", justify="right", width=12)
    spread_table.add_column("NEUTRAL", justify="right", width=12)
    spread_table.add_column("WEAKENING", justify="right", width=12)
    spread_table.add_column("Spread (L-W)", justify="right", width=14)
    spread_table.add_column("n (L/N/W)", justify="right", width=12)

    for ls in result.label_spreads:
        spread_color = "green" if not math.isnan(ls.spread) and ls.spread > 0 else "red"
        spread_cell = (
            f"[{spread_color}]{ls.spread:+.2%}[/{spread_color}]"
            if not math.isnan(ls.spread) else "[dim]n/a[/dim]"
        )
        spread_table.add_row(
            f"{ls.horizon}d",
            _fmt_ret(ls.leading_avg),
            _fmt_ret(ls.neutral_avg),
            _fmt_ret(ls.weakening_avg),
            spread_cell,
            f"{ls.n_leading}/{ls.n_neutral}/{ls.n_weakening}",
        )
    console.print(spread_table)

    # --- Table 4: Per-signal isolation IC ---
    sig_table = Table(title="Per-Signal Isolation IC (pooled Spearman)", box=box.SIMPLE_HEAD)
    sig_table.add_column("Signal", width=26)
    sig_table.add_column("IC 5d", justify="right", width=11)
    sig_table.add_column("IC 10d", justify="right", width=11)
    sig_table.add_column("IC 20d", justify="right", width=11)
    sig_table.add_column("n", justify="right", width=8)

    for sig in result.signal_ics:
        sig_table.add_row(
            sig.name,
            _fmt_ic(sig.ic_5d),
            _fmt_ic(sig.ic_10d),
            _fmt_ic(sig.ic_20d),
            str(sig.n),
        )
    console.print(sig_table)

    # --- Validation bar ---
    console.print("[bold]Validation[/bold]")
    for h in (5, 10, 20):
        ic = result.ic_mean[h]
        h1 = result.ic_h1_mean[h]
        h2 = result.ic_h2_mean[h]
        ic_ok = not math.isnan(ic) and ic > 0.05
        both_ok = (not math.isnan(h1) and h1 > 0) and (not math.isnan(h2) and h2 > 0)
        if ic_ok and both_ok:
            status = "[bold green]PASS[/bold green]"
            note = "IC > 0.05 and both halves positive"
        elif not math.isnan(ic) and ic > 0 and both_ok:
            status = "[yellow]MARGINAL[/yellow]"
            note = "IC positive but < 0.05 threshold"
        else:
            status = "[red]FAIL[/red]"
            note = "IC <= 0 or inconsistent across halves"
        console.print(f"  {h}d: {status}  [dim]{note}[/dim]")

    console.print(
        "\n[dim]"
        "Autocorrelation caveat: overlapping forward returns (10d, 20d) create serial correlation "
        "in per-date IC — IC IR is understated. Use 5d IC IR as the primary metric.\n"
        "Survivorship-bias caveat: DuckDB/yfinance only contain currently-listed tickers. "
        "Results overstate performance. Migrate to Polygon.io before trusting live numbers."
        "[/dim]"
    )


@app.command("backtest-estimates")
def backtest_estimates() -> None:
    """Walk-forward backtest of the earnings revision momentum signal."""
    from scanner.backtest.estimates_runner import run_estimates_backtest, _MIN_DATES, _MIN_OBS

    console.print("[bold cyan]Running earnings revision backtest...[/bold cyan]")

    result = run_estimates_backtest()

    if result is None:
        # Check how many dates are actually stored to give a helpful message.
        from scanner.db import get_connection as _gc

        _c = _gc()
        try:
            n = _c.execute(
                "SELECT COUNT(DISTINCT date) FROM estimates WHERE revision_score IS NOT NULL"
            ).fetchone()[0]
        finally:
            _c.close()

        console.print(
            f"\n[yellow]Insufficient history for a meaningful backtest.[/yellow]\n"
            f"  Dates with analyst coverage stored: [bold]{n}[/bold] / {_MIN_DATES} required\n"
            f"  Total observations needed at 5d horizon: {_MIN_OBS}\n\n"
            "[dim]Run [bold]scanner ingest-estimates[/bold] once per trading day.\n"
            f"The backtest will unlock automatically after ~{max(0, _MIN_DATES - n)} more days.[/dim]"
        )
        return

    def _fmt_ic(v: float) -> str:
        if math.isnan(v):
            return "[dim]n/a[/dim]"
        c = "green" if v > 0.05 else ("yellow" if v > 0 else "red")
        return f"[{c}]{v:+.4f}[/{c}]"

    def _fmt_ret(v: float) -> str:
        if math.isnan(v):
            return "[dim]n/a[/dim]"
        color = "green" if v > 0 else "red"
        return f"[{color}]{v:+.2%}[/{color}]"

    # Table 1: IC summary
    ic_table = Table(
        title=(
            f"Earnings Revision IC  "
            f"({result.start_date} to {result.end_date}  |  "
            f"{result.n_dates} dates  |  {result.n_obs} obs)"
        ),
        box=box.SIMPLE_HEAD,
    )
    ic_table.add_column("Horizon", width=10)
    ic_table.add_column("IC Mean", justify="right", width=11)
    ic_table.add_column("IC Std", justify="right", width=11)
    ic_table.add_column("IC IR", justify="right", width=11)
    for h in (5, 10, 20):
        ic_table.add_row(
            f"{h}d",
            _fmt_ic(result.ic_mean[h]),
            f"{result.ic_std[h]:.4f}" if not math.isnan(result.ic_std[h]) else "[dim]n/a[/dim]",
            _fmt_ic(result.ic_ir[h]),
        )
    console.print(ic_table)

    # Table 2: H1/H2 split
    split_table = Table(
        title=(
            f"Out-of-Sample H1/H2 Split  "
            f"(H1: {result.start_date} to {result.mid_date} n={result.n_h1}  |  "
            f"H2: {result.mid_date} to {result.end_date} n={result.n_h2})"
        ),
        box=box.SIMPLE_HEAD,
    )
    split_table.add_column("Horizon", width=10)
    split_table.add_column("H1 IC Mean", justify="right", width=12)
    split_table.add_column("H2 IC Mean", justify="right", width=12)
    split_table.add_column("Both +?", width=10)
    for h in (5, 10, 20):
        h1 = result.ic_h1_mean[h]
        h2 = result.ic_h2_mean[h]
        both_pos = (not math.isnan(h1) and h1 > 0) and (not math.isnan(h2) and h2 > 0)
        split_table.add_row(
            f"{h}d", _fmt_ic(h1), _fmt_ic(h2),
            "[green]Yes[/green]" if both_pos else "[red]No[/red]",
        )
    console.print(split_table)

    # Table 3: positive vs negative revision spread
    spread_table = Table(
        title="Avg Forward Return: Positive vs Negative Revision Score",
        box=box.SIMPLE_HEAD,
    )
    spread_table.add_column("Horizon", width=10)
    spread_table.add_column("Rev > 0", justify="right", width=12)
    spread_table.add_column("Rev < 0", justify="right", width=12)
    spread_table.add_column("Spread", justify="right", width=12)
    for h in (5, 10, 20):
        sp = result.spread[h]
        sp_color = "green" if not math.isnan(sp) and sp > 0 else "red"
        sp_cell = f"[{sp_color}]{sp:+.2%}[/{sp_color}]" if not math.isnan(sp) else "[dim]n/a[/dim]"
        spread_table.add_row(f"{h}d", _fmt_ret(result.positive_avg[h]), _fmt_ret(result.negative_avg[h]), sp_cell)
    console.print(spread_table)

    # Validation bar
    console.print("[bold]Validation[/bold]")
    for h in (5, 10, 20):
        ic = result.ic_mean[h]
        h1 = result.ic_h1_mean[h]
        h2 = result.ic_h2_mean[h]
        ic_ok = not math.isnan(ic) and ic > 0.05
        both_ok = (not math.isnan(h1) and h1 > 0) and (not math.isnan(h2) and h2 > 0)
        n_ok = result.n_obs >= _MIN_OBS
        if ic_ok and both_ok and n_ok:
            status = "[bold green]PASS[/bold green]"
            note = "IC > 0.05, both halves positive, n >= threshold"
        elif not math.isnan(ic) and ic > 0 and both_ok:
            status = "[yellow]MARGINAL[/yellow]"
            note = "IC positive but < 0.05 threshold"
        else:
            status = "[red]FAIL[/red]"
            note = "IC <= 0 or inconsistent across halves"
        console.print(f"  {h}d: {status}  [dim]{note}[/dim]")

    console.print(
        "\n[dim]"
        "Data-availability caveat: revision_score is a current-day snapshot -- no historical "
        "revision series. IC accumulates only as daily snapshots are stored. "
        "Results improve as history grows.\n"
        "Survivorship-bias caveat: DuckDB/yfinance only contain currently-listed tickers."
        "[/dim]"
    )


@app.command("backtest-rvol")
def backtest_rvol() -> None:
    """Walk-forward backtest of the Relative Volume (RVOL) signal."""
    from scanner.backtest.rvol_runner import _MIN_OBS, run_rvol_backtest

    console.print("[bold cyan]Running RVOL walk-forward backtest...[/bold cyan]")
    console.print("[dim]Loading price+volume history from DuckDB...[/dim]\n")

    result = run_rvol_backtest()

    if result is None:
        console.print(
            f"\n[yellow]Insufficient data for RVOL backtest.[/yellow]\n"
            f"  Need at least {_MIN_OBS} (date x ticker) 5d-horizon observations.\n"
            "[dim]Run [bold]scanner ingest[/bold] to populate price+volume history.[/dim]"
        )
        return

    def _fmt_ic(v: float) -> str:
        if math.isnan(v):
            return "[dim]n/a[/dim]"
        c = "green" if v > 0.05 else ("yellow" if v > 0 else "red")
        return f"[{c}]{v:+.4f}[/{c}]"

    def _fmt_ret(v: float) -> str:
        if math.isnan(v):
            return "[dim]n/a[/dim]"
        color = "green" if v > 0 else "red"
        return f"[{color}]{v:+.2%}[/{color}]"

    # Table 1: Signed RVOL IC (headline)
    ic_table = Table(
        title=(
            f"RVOL Signed IC (rvol * day_direction)  "
            f"({result.start_date} to {result.end_date}  |  "
            f"{result.n_dates} dates  |  {result.n_obs} obs at 5d)"
        ),
        box=box.SIMPLE_HEAD,
    )
    ic_table.add_column("Horizon", width=10)
    ic_table.add_column("IC Mean", justify="right", width=11)
    ic_table.add_column("IC Std", justify="right", width=11)
    ic_table.add_column("IC IR", justify="right", width=11)
    for h in (5, 10, 20):
        ic_table.add_row(
            f"{h}d",
            _fmt_ic(result.ic_mean[h]),
            f"{result.ic_std[h]:.4f}" if not math.isnan(result.ic_std[h]) else "[dim]n/a[/dim]",
            _fmt_ic(result.ic_ir[h]),
        )
    console.print(ic_table)

    # Table 2: H1/H2 out-of-sample split
    split_table = Table(
        title=(
            f"Out-of-Sample H1/H2 Split  "
            f"(H1: {result.start_date} to {result.mid_date} n={result.n_h1}  |  "
            f"H2: {result.mid_date} to {result.end_date} n={result.n_h2})"
        ),
        box=box.SIMPLE_HEAD,
    )
    split_table.add_column("Horizon", width=10)
    split_table.add_column("H1 IC Mean", justify="right", width=12)
    split_table.add_column("H2 IC Mean", justify="right", width=12)
    split_table.add_column("Both +?", width=10)
    for h in (5, 10, 20):
        h1 = result.ic_h1_mean[h]
        h2 = result.ic_h2_mean[h]
        both_pos = (not math.isnan(h1) and h1 > 0) and (not math.isnan(h2) and h2 > 0)
        split_table.add_row(
            f"{h}d", _fmt_ic(h1), _fmt_ic(h2),
            "[green]Yes[/green]" if both_pos else "[red]No[/red]",
        )
    console.print(split_table)

    # Table 3: Interaction decomposition (signed vs magnitude vs direction)
    inter_table = Table(
        title="Interaction Decomposition: Pooled IC (same obs set)",
        box=box.SIMPLE_HEAD,
    )
    inter_table.add_column("Horizon", width=10)
    inter_table.add_column("Signed RVOL", justify="right", width=13)
    inter_table.add_column("Magnitude only", justify="right", width=16)
    inter_table.add_column("Direction only", justify="right", width=15)
    inter_table.add_column("Combo wins?", width=13)
    for h in (5, 10, 20):
        s_ic = result.ic_mean[h]
        u_ic = result.ic_unsigned[h]
        d_ic = result.ic_direction[h]
        combo_wins = (
            not math.isnan(s_ic) and not math.isnan(u_ic) and not math.isnan(d_ic)
            and abs(s_ic) > abs(u_ic) and abs(s_ic) > abs(d_ic)
        )
        inter_table.add_row(
            f"{h}d", _fmt_ic(s_ic), _fmt_ic(u_ic), _fmt_ic(d_ic),
            "[green]Yes[/green]" if combo_wins else "[dim]No[/dim]",
        )
    console.print(inter_table)

    # Table 4: Mega-cap vs dependent split
    split2_table = Table(
        title=(
            f"Mega-cap vs Dependent Split  "
            f"(mega-cap obs 5d: {result.n_megacap_obs}  |  "
            f"dependent obs 5d: {result.n_dependent_obs})"
        ),
        box=box.SIMPLE_HEAD,
    )
    split2_table.add_column("Horizon", width=10)
    split2_table.add_column("Mega-cap IC", justify="right", width=13)
    split2_table.add_column("Dependent IC", justify="right", width=14)
    for h in (5, 10, 20):
        split2_table.add_row(f"{h}d", _fmt_ic(result.ic_megacap[h]), _fmt_ic(result.ic_dependent[h]))
    console.print(split2_table)

    # Table 5: High-RVOL segment analysis
    seg_table = Table(
        title=(
            f"High-RVOL Segment (>=2.0x) Avg Forward Return  "
            f"(up-day n={result.n_up_high}  |  down-day n={result.n_down_high})"
        ),
        box=box.SIMPLE_HEAD,
    )
    seg_table.add_column("Horizon", width=10)
    seg_table.add_column("Up-day avg ret", justify="right", width=16)
    seg_table.add_column("Down-day avg ret", justify="right", width=17)
    seg_table.add_column("Spread", justify="right", width=12)
    for h in (5, 10, 20):
        up = result.up_high_rvol_avg[h]
        dn = result.down_high_rvol_avg[h]
        spread = up - dn if not (math.isnan(up) or math.isnan(dn)) else float("nan")
        sp_color = "green" if not math.isnan(spread) and spread > 0 else "red"
        sp_cell = f"[{sp_color}]{spread:+.2%}[/{sp_color}]" if not math.isnan(spread) else "[dim]n/a[/dim]"
        seg_table.add_row(f"{h}d", _fmt_ret(up), _fmt_ret(dn), sp_cell)
    console.print(seg_table)

    # Validation summary
    console.print("[bold]Validation[/bold]")
    for h in (5, 10, 20):
        ic = result.ic_mean[h]
        h1 = result.ic_h1_mean[h]
        h2 = result.ic_h2_mean[h]
        ic_ok = not math.isnan(ic) and ic > 0.05
        both_ok = (not math.isnan(h1) and h1 > 0) and (not math.isnan(h2) and h2 > 0)
        n_ok = result.n_obs >= _MIN_OBS
        if ic_ok and both_ok and n_ok:
            status = "[bold green]PASS[/bold green]"
            note = "IC > 0.05, both halves positive, n >= threshold"
        elif not math.isnan(ic) and ic > 0 and both_ok:
            status = "[yellow]MARGINAL[/yellow]"
            note = "IC positive but < 0.05 threshold"
        else:
            status = "[red]FAIL[/red]"
            note = "IC <= 0 or inconsistent across halves"
        console.print(f"  {h}d: {status}  [dim]{note}[/dim]")

    console.print(
        "\n[dim]"
        "Signed RVOL = rvol * sign(day_return). Positive when high volume on up day "
        "(bullish conviction), negative when high volume on down day (distribution).\n"
        "Survivorship-bias caveat: DuckDB/yfinance only contain currently-listed tickers."
        "[/dim]"
    )


def _backtest_confirmation_regime(run_fn: object, min_obs: int) -> None:
    """Render the CORRECTION-regime focused confirmation backtest output."""
    console.print(
        "[bold cyan]Running CORRECTION-regime confirmation score backtest...[/bold cyan]"
    )
    console.print(
        "[dim]Scope: dependent stocks only, parent 20d return in [-15%, -5%].[/dim]"
    )
    console.print()

    result = run_fn()

    if result is None:
        console.print(
            f"\n[yellow]Insufficient data for regime confirmation backtest.[/yellow]\n"
            f"  Need at least {min_obs} 5d-horizon observations in CORRECTION regime.\n"
            "[dim]Run [bold]scanner ingest[/bold] to populate price+volume history.[/dim]"
        )
        return

    def _fmt_ic(v: float) -> str:
        if math.isnan(v):
            return "[dim]n/a[/dim]"
        c = "green" if v > 0.05 else ("yellow" if v > 0 else "red")
        return f"[{c}]{v:+.4f}[/{c}]"

    def _fmt_ret(v: float) -> str:
        if math.isnan(v):
            return "[dim]n/a[/dim]"
        color = "green" if v > 0 else "red"
        return f"[{color}]{v:+.2%}[/{color}]"

    def _n_flag(n: int) -> str:
        if n < 50:
            return f"[bold red]{n} (!)[/bold red]"
        if n < 100:
            return f"[yellow]{n}[/yellow]"
        return str(n)

    def _render_ic_section(
        label: str,
        n_obs: dict,
        ic_mean: dict,
        ic_std: dict,
        ic_ir: dict,
        ic_h1_mean: dict,
        ic_h2_mean: dict,
        ic_rs_alone: dict,
        ic_confirm_score: dict,
        buckets: list,
        start_date: str,
        end_date: str,
        mid_date: str,
        n_h1: int,
        n_h2: int,
        min_obs_n: int,
    ) -> None:
        # IC summary
        ic_table = Table(
            title=(
                f"{label} -- Confirmation Score IC in CORRECTION  "
                f"({start_date} to {end_date})"
            ),
            box=box.SIMPLE_HEAD,
        )
        ic_table.add_column("Horizon", width=10)
        ic_table.add_column("n obs", justify="right", width=10)
        ic_table.add_column("IC Mean", justify="right", width=11)
        ic_table.add_column("IC Std", justify="right", width=11)
        ic_table.add_column("IC IR", justify="right", width=11)
        for h in (5, 10, 20):
            n = n_obs.get(h, 0)
            m = ic_mean.get(h, float("nan"))
            s = ic_std.get(h, float("nan")) if ic_std else float("nan")
            ir = ic_ir.get(h, float("nan")) if ic_ir else float("nan")
            ic_table.add_row(
                f"{h}d",
                _n_flag(n),
                _fmt_ic(m),
                f"{s:.4f}" if not math.isnan(s) else "[dim]n/a[/dim]",
                _fmt_ic(ir),
            )
        console.print(ic_table)

        # H1/H2 split
        split_table = Table(
            title=(
                f"{label} -- H1/H2 Split  "
                f"(H1: {start_date} to {mid_date} n={n_h1}  |  "
                f"H2: {mid_date} to {end_date} n={n_h2})"
            ),
            box=box.SIMPLE_HEAD,
        )
        split_table.add_column("Horizon", width=10)
        split_table.add_column("H1 IC Mean", justify="right", width=12)
        split_table.add_column("H2 IC Mean", justify="right", width=12)
        split_table.add_column("Both +?", width=10)
        for h in (5, 10, 20):
            h1v = ic_h1_mean.get(h, float("nan"))
            h2v = ic_h2_mean.get(h, float("nan"))
            both_pos = (not math.isnan(h1v) and h1v > 0) and (not math.isnan(h2v) and h2v > 0)
            split_table.add_row(
                f"{h}d", _fmt_ic(h1v), _fmt_ic(h2v),
                "[green]Yes[/green]" if both_pos else "[red]No[/red]",
            )
        console.print(split_table)

        # Bucket spread
        bucket_table = Table(
            title=f"{label} -- Avg Forward Return by Score Bucket (max=4, Est Rev excluded)",
            box=box.SIMPLE_HEAD,
        )
        bucket_table.add_column("Bucket", width=8)
        bucket_table.add_column("n (5d ref)", justify="right", width=12)
        bucket_table.add_column("5d avg ret", justify="right", width=12)
        bucket_table.add_column("10d avg ret", justify="right", width=12)
        bucket_table.add_column("20d avg ret", justify="right", width=12)
        for bk in buckets:
            bucket_table.add_row(
                bk.label,
                _n_flag(bk.n),
                _fmt_ret(bk.avg_ret.get(5, float("nan"))),
                _fmt_ret(bk.avg_ret.get(10, float("nan"))),
                _fmt_ret(bk.avg_ret.get(20, float("nan"))),
            )
        if len(buckets) >= 2:
            top = buckets[-1].avg_ret.get(5, float("nan"))
            bot = buckets[0].avg_ret.get(5, float("nan"))
            spread_5 = top - bot if not (math.isnan(top) or math.isnan(bot)) else float("nan")
            sp_color = "green" if not math.isnan(spread_5) and spread_5 > 0 else "red"
            sp_cell = (
                f"[{sp_color}]{spread_5:+.2%}[/{sp_color}]"
                if not math.isnan(spread_5)
                else "[dim]n/a[/dim]"
            )
            bucket_table.add_section()
            bucket_table.add_row("Spread (top-bot)", "", sp_cell, "", "")
        console.print(bucket_table)

        # RS alone vs confirmation interaction
        inter_table = Table(
            title=f"{label} -- Interaction: RS Alone vs Confirmation Score",
            box=box.SIMPLE_HEAD,
        )
        inter_table.add_column("Horizon", width=10)
        inter_table.add_column("RS alone", justify="right", width=12)
        inter_table.add_column("Confirm score", justify="right", width=14)
        inter_table.add_column("Confirm wins?", width=14)
        for h in (5, 10, 20):
            rs_ic = ic_rs_alone.get(h, float("nan"))
            c_ic = ic_confirm_score.get(h, float("nan"))
            wins = (
                not math.isnan(rs_ic) and not math.isnan(c_ic) and c_ic > rs_ic
            )
            inter_table.add_row(
                f"{h}d", _fmt_ic(rs_ic), _fmt_ic(c_ic),
                "[green]Yes[/green]" if wins else "[dim]No[/dim]",
            )
        console.print(inter_table)

        # Validation bar
        console.print(f"[bold]Validation ({label})[/bold]")
        for h in (5, 10, 20):
            ic = ic_mean.get(h, float("nan"))
            h1v = ic_h1_mean.get(h, float("nan"))
            h2v = ic_h2_mean.get(h, float("nan"))
            n = n_obs.get(h, 0)
            ic_ok = not math.isnan(ic) and ic > 0.05
            both_ok = (not math.isnan(h1v) and h1v > 0) and (not math.isnan(h2v) and h2v > 0)
            n_ok = n >= min_obs_n
            if n < 50:
                status = "[bold red]INSUFFICIENT (n<50)[/bold red]"
                note = f"Only {n} obs at {h}d horizon -- results unreliable"
            elif ic_ok and both_ok and n_ok:
                status = "[bold green]PASS[/bold green]"
                note = "IC > 0.05, both halves positive, n >= threshold"
            elif not math.isnan(ic) and ic > 0 and both_ok:
                status = "[yellow]MARGINAL[/yellow]"
                note = "IC positive but < 0.05 threshold"
            else:
                status = "[red]FAIL[/red]"
                note = "IC <= 0 or inconsistent across halves"
            console.print(f"  {h}d: {status}  [dim]{note}[/dim]")
        console.print()

    # Main section: all CORRECTION-qualifying dependents
    _render_ic_section(
        label="All CORRECTION dependents",
        n_obs=result.n_obs,
        ic_mean=result.ic_mean,
        ic_std=result.ic_std,
        ic_ir=result.ic_ir,
        ic_h1_mean=result.ic_h1_mean,
        ic_h2_mean=result.ic_h2_mean,
        ic_rs_alone=result.ic_rs_alone,
        ic_confirm_score=result.ic_confirm_score,
        buckets=result.buckets,
        start_date=result.start_date,
        end_date=result.end_date,
        mid_date=result.mid_date,
        n_h1=result.n_h1,
        n_h2=result.n_h2,
        min_obs_n=min_obs,
    )

    # Validated sub-analysis: MSFT/META dependents only
    v = result.validated
    console.print(
        "[bold cyan]Validated Parents Sub-analysis (MSFT + META dependents only)[/bold cyan]"
    )
    _render_ic_section(
        label=v.label,
        n_obs=v.n_obs,
        ic_mean=v.ic_mean,
        ic_std={},
        ic_ir={},
        ic_h1_mean=v.ic_h1_mean,
        ic_h2_mean=v.ic_h2_mean,
        ic_rs_alone=v.ic_rs_alone,
        ic_confirm_score=v.ic_confirm_score,
        buckets=v.buckets,
        start_date=result.start_date,
        end_date=result.end_date,
        mid_date=result.mid_date,
        n_h1=result.n_h1,
        n_h2=result.n_h2,
        min_obs_n=min_obs,
    )

    console.print(
        "[dim]"
        "CORRECTION regime: parent 20d return in [-15%, -5%]. "
        "Dependent RS uses own 20d return > 0.\n"
        "Insider lookahead guard: filed_date <= T, transaction_date >= T - 30d.\n"
        "Survivorship-bias caveat: yfinance data excludes delisted tickers."
        "[/dim]"
    )


@app.command("backtest-confirmation")
def backtest_confirmation(
    regime_only: Annotated[
        bool, typer.Option("--regime-only", is_flag=True, help="Run focused CORRECTION-regime backtest (dependents only)")
    ] = False,
) -> None:
    """Walk-forward backtest of the Cross-Signal Confirmation Score (0-4, Est Rev excluded)."""
    from scanner.backtest.confirmation_runner import (
        _MIN_OBS,
        run_confirmation_backtest,
        run_confirmation_regime_backtest,
    )

    if regime_only:
        _backtest_confirmation_regime(run_confirmation_regime_backtest, _MIN_OBS)
        return

    console.print("[bold cyan]Running confirmation score walk-forward backtest...[/bold cyan]")
    console.print("[dim]Loading price, volume, and insider data from DuckDB...[/dim]")
    console.print()
    console.print(
        "[yellow]NOTE: Confirmation score max = 4 in backtest (Est Rev excluded).[/yellow]\n"
        "[dim]Score buckets: 0-1 (low), 2 (mid), 3-4 (high)\n"
        "Est Rev is excluded until 20+ daily snapshots are accumulated "
        "via `scanner ingest-estimates`. Run it daily to unlock the 5th signal.[/dim]"
    )
    console.print()

    result = run_confirmation_backtest()

    if result is None:
        console.print(
            f"\n[yellow]Insufficient data for confirmation backtest.[/yellow]\n"
            f"  Need at least {_MIN_OBS} (date x ticker) 5d-horizon observations.\n"
            "[dim]Run [bold]scanner ingest[/bold] to populate price+volume history.[/dim]"
        )
        return

    if not result.qqq_available:
        console.print(
            "[yellow]QQQ not in prices table -- mega-cap RS uses own 20d return as fallback.[/yellow]\n"
            "[dim]Run [bold]scanner ingest[/bold] (QQQ is in the universe) to fix this.[/dim]\n"
        )

    def _fmt_ic(v: float) -> str:
        if math.isnan(v):
            return "[dim]n/a[/dim]"
        c = "green" if v > 0.05 else ("yellow" if v > 0 else "red")
        return f"[{c}]{v:+.4f}[/{c}]"

    def _fmt_ret(v: float) -> str:
        if math.isnan(v):
            return "[dim]n/a[/dim]"
        color = "green" if v > 0 else "red"
        return f"[{color}]{v:+.2%}[/{color}]"

    # Table 1: Overall IC summary
    ic_table = Table(
        title=(
            f"Confirmation Score IC  "
            f"({result.start_date} to {result.end_date}  |  "
            f"{result.n_dates} dates  |  {result.n_obs} obs at 5d)"
        ),
        box=box.SIMPLE_HEAD,
    )
    ic_table.add_column("Horizon", width=10)
    ic_table.add_column("IC Mean", justify="right", width=11)
    ic_table.add_column("IC Std", justify="right", width=11)
    ic_table.add_column("IC IR", justify="right", width=11)
    for h in (5, 10, 20):
        ic_table.add_row(
            f"{h}d",
            _fmt_ic(result.ic_mean[h]),
            f"{result.ic_std[h]:.4f}" if not math.isnan(result.ic_std[h]) else "[dim]n/a[/dim]",
            _fmt_ic(result.ic_ir[h]),
        )
    console.print(ic_table)

    # Table 2: H1/H2 out-of-sample split
    split_table = Table(
        title=(
            f"Out-of-Sample H1/H2 Split  "
            f"(H1: {result.start_date} to {result.mid_date} n={result.n_h1}  |  "
            f"H2: {result.mid_date} to {result.end_date} n={result.n_h2})"
        ),
        box=box.SIMPLE_HEAD,
    )
    split_table.add_column("Horizon", width=10)
    split_table.add_column("H1 IC Mean", justify="right", width=12)
    split_table.add_column("H2 IC Mean", justify="right", width=12)
    split_table.add_column("Both +?", width=10)
    for h in (5, 10, 20):
        h1 = result.ic_h1_mean[h]
        h2 = result.ic_h2_mean[h]
        both_pos = (not math.isnan(h1) and h1 > 0) and (not math.isnan(h2) and h2 > 0)
        split_table.add_row(
            f"{h}d", _fmt_ic(h1), _fmt_ic(h2),
            "[green]Yes[/green]" if both_pos else "[red]No[/red]",
        )
    console.print(split_table)

    # Table 3: Score bucket spread
    bucket_table = Table(
        title="Avg Forward Return by Confirmation Score Bucket (max score = 4, Est Rev excluded)",
        box=box.SIMPLE_HEAD,
    )
    bucket_table.add_column("Bucket", width=8)
    bucket_table.add_column("n (5d ref)", justify="right", width=12)
    bucket_table.add_column("5d avg ret", justify="right", width=12)
    bucket_table.add_column("10d avg ret", justify="right", width=12)
    bucket_table.add_column("20d avg ret", justify="right", width=12)
    for bk in result.buckets:
        bucket_table.add_row(
            bk.label,
            str(bk.n),
            _fmt_ret(bk.avg_ret.get(5, float("nan"))),
            _fmt_ret(bk.avg_ret.get(10, float("nan"))),
            _fmt_ret(bk.avg_ret.get(20, float("nan"))),
        )
    # Top vs bottom spread row
    if len(result.buckets) >= 2:
        top = result.buckets[-1].avg_ret.get(5, float("nan"))
        bot = result.buckets[0].avg_ret.get(5, float("nan"))
        spread_5 = top - bot if not (math.isnan(top) or math.isnan(bot)) else float("nan")
        sp_color = "green" if not math.isnan(spread_5) and spread_5 > 0 else "red"
        sp_cell = f"[{sp_color}]{spread_5:+.2%}[/{sp_color}]" if not math.isnan(spread_5) else "[dim]n/a[/dim]"
        bucket_table.add_section()
        bucket_table.add_row("Spread (top-bot)", "", sp_cell, "", "")
    console.print(bucket_table)

    # Table 4: Mega-cap vs dependent split
    split2_table = Table(
        title=(
            f"Mega-cap vs Dependent Split  "
            f"(mega-cap obs 5d: {result.n_megacap_obs}  |  "
            f"dependent obs 5d: {result.n_dependent_obs})"
        ),
        box=box.SIMPLE_HEAD,
    )
    split2_table.add_column("Horizon", width=10)
    split2_table.add_column("Mega-cap IC", justify="right", width=13)
    split2_table.add_column("Dependent IC", justify="right", width=14)
    for h in (5, 10, 20):
        split2_table.add_row(f"{h}d", _fmt_ic(result.ic_megacap[h]), _fmt_ic(result.ic_dependent[h]))
    console.print(split2_table)

    # Table 5: Interaction — RS alone vs confirmation score
    inter_table = Table(
        title="Interaction: RS Alone vs Confirmation Score (pooled Spearman IC)",
        box=box.SIMPLE_HEAD,
    )
    inter_table.add_column("Horizon", width=10)
    inter_table.add_column("RS alone", justify="right", width=12)
    inter_table.add_column("Confirm score", justify="right", width=14)
    inter_table.add_column("Confirm wins?", width=14)
    for h in (5, 10, 20):
        rs_ic = result.ic_rs_alone[h]
        c_ic = result.ic_confirm_score[h]
        wins = (
            not math.isnan(rs_ic) and not math.isnan(c_ic)
            and c_ic > rs_ic
        )
        inter_table.add_row(
            f"{h}d", _fmt_ic(rs_ic), _fmt_ic(c_ic),
            "[green]Yes[/green]" if wins else "[dim]No[/dim]",
        )
    console.print(inter_table)

    # Validation bar
    console.print("[bold]Validation[/bold]")
    for h in (5, 10, 20):
        ic = result.ic_mean[h]
        h1 = result.ic_h1_mean[h]
        h2 = result.ic_h2_mean[h]
        ic_ok = not math.isnan(ic) and ic > 0.05
        both_ok = (not math.isnan(h1) and h1 > 0) and (not math.isnan(h2) and h2 > 0)
        n_ok = result.n_obs >= _MIN_OBS
        if ic_ok and both_ok and n_ok:
            status = "[bold green]PASS[/bold green]"
            note = "IC > 0.05, both halves positive, n >= threshold"
        elif not math.isnan(ic) and ic > 0 and both_ok:
            status = "[yellow]MARGINAL[/yellow]"
            note = "IC positive but < 0.05 threshold"
        else:
            status = "[red]FAIL[/red]"
            note = "IC <= 0 or inconsistent across halves"
        console.print(f"  {h}d: {status}  [dim]{note}[/dim]")

    console.print(
        "\n[dim]"
        "Confirmation score 0-4 (backtest): RS + RVOL + Insider + Above-50MA. "
        "Est Rev excluded until 20+ daily snapshots exist.\n"
        "Insider lookahead guard: uses filed_date <= T, not transaction_date, "
        "to respect the 2-business-day Form 4 filing window.\n"
        "Survivorship-bias caveat: DuckDB/yfinance only contain currently-listed tickers. "
        "Results overstate performance. Migrate to Polygon.io before trusting live numbers."
        "[/dim]"
    )


@app.command()
def backtest(
    signal: Annotated[str, typer.Option(help="Signal name (rs, rs_normalized)")] = "rs",
    horizon: Annotated[int, typer.Option(help="Forward return horizon in trading days")] = 5,
    lookback: Annotated[int, typer.Option(help="Signal lookback in trading days")] = 20,
    start: Annotated[Optional[str], typer.Option(help="Start date YYYY-MM-DD")] = None,
    end: Annotated[Optional[str], typer.Option(help="End date YYYY-MM-DD")] = None,
    parents: Annotated[
        Optional[str],
        typer.Option(help="Comma-separated mega-cap tickers to restrict universe (e.g. NVDA,MSFT)"),
    ] = None,
    tickers: Annotated[
        Optional[str],
        typer.Option(help="Comma-separated child tickers to restrict backtest universe (e.g. STM,SQM,ALB)"),
    ] = None,
) -> None:
    """Walk-forward backtest. IC is the headline metric."""
    from scanner.backtest.runner import run_backtest

    parent_list = [p.strip().upper() for p in parents.split(",")] if parents else None
    ticker_list = [t.strip().upper() for t in tickers.split(",")] if tickers else None
    scope = f"  parents={','.join(parent_list)}" if parent_list else ""
    if ticker_list:
        scope += f"  tickers={','.join(ticker_list)}"
    console.print(
        f"[bold cyan]Running walk-forward backtest[/bold cyan]  "
        f"signal={signal}  horizon={horizon}d  lookback={lookback}d{scope}"
    )

    try:
        result = run_backtest(
            signal_name=signal,
            horizon=horizon,
            lookback=lookback,
            start=start,
            end=end,
            parents=parent_list,
            tickers=ticker_list,
        )
    except Exception as exc:
        console.print(f"[red]Backtest failed:[/red] {exc}")
        sys.exit(1)

    table = Table(title="Walk-Forward Backtest Results", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    ic_color = "green" if result.ic_mean > 0.05 else ("yellow" if result.ic_mean > 0 else "red")
    sharpe_color = "green" if result.sharpe > 1.0 else ("yellow" if result.sharpe > 0 else "red")

    table.add_row(
        "IC (Mean) [headline]",
        f"[{ic_color}]{result.ic_mean:+.4f}[/{ic_color}]",
    )
    table.add_row("IC (Std Dev)", f"{result.ic_std:.4f}")
    table.add_row("IC IR  (IC / IC_std)", f"{result.ic_ir:+.3f}")
    table.add_row(
        "Sharpe Ratio (annualised)",
        f"[{sharpe_color}]{result.sharpe:+.3f}[/{sharpe_color}]",
    )
    table.add_row("Max Drawdown", f"[red]{result.max_drawdown:+.2%}[/red]")
    table.add_row("Rebalance Periods", str(result.n_periods))
    table.add_row("Horizon", f"{result.horizon} trading days")

    console.print(table)

    # --- Per-quintile calibration table ---
    if ticker_list and len(ticker_list) < 3:
        console.print(
            "\n[dim]Note: Single-ticker backtest — quintile bucketing not applicable "
            "(requires 3+ tickers for cross-sectional ranking). "
            "IC and regime breakdown are still valid.[/dim]"
        )
    elif result.quintile_stats:
        q_table = Table(title="Per-Quintile Calibration", box=box.SIMPLE_HEAD)
        q_table.add_column("Bucket", width=13)
        q_table.add_column("n obs", justify="right", width=7)
        q_table.add_column("1d fwd", justify="right", width=9)
        q_table.add_column("5d fwd", justify="right", width=9)
        q_table.add_column("10d fwd", justify="right", width=9)
        q_table.add_column("20d fwd", justify="right", width=9)
        q_table.add_column("Hit Rate", justify="right", width=10)
        q_table.add_column("Vol 20d", justify="right", width=10)

        sb_vol = next((s.avg_vol_20d for s in result.quintile_stats if s.action == "STRONG_BUY"), float("nan"))
        hold_vol = next((s.avg_vol_20d for s in result.quintile_stats if s.action == "HOLD"), float("nan"))

        for stat in result.quintile_stats:
            hr = stat.hit_rate
            if math.isnan(hr):
                hit_str = "[dim]n/a[/dim]"
            else:
                hr_color = "green" if hr >= 0.52 else ("red" if hr < 0.48 else "yellow")
                hit_str = f"[{hr_color}]{hr:.1%}[/{hr_color}]"

            vol_str = f"{stat.avg_vol_20d:.1%}" if not math.isnan(stat.avg_vol_20d) else "[dim]n/a[/dim]"
            style = _ACTION_STYLE.get(stat.action, "")
            label = f"[{style}]{stat.action}[/{style}]" if style else stat.action

            q_table.add_row(
                label,
                str(stat.count),
                _fmt_return(stat.avg_fwd_1d),
                _fmt_return(stat.avg_fwd_5d),
                _fmt_return(stat.avg_fwd_10d),
                _fmt_return(stat.avg_fwd_20d),
                hit_str,
                vol_str,
            )
        console.print(q_table)

        if not (math.isnan(sb_vol) or math.isnan(hold_vol)) and hold_vol > 0:
            vol_ratio = sb_vol / hold_vol
            if vol_ratio >= 2.0:
                console.print(
                    f"[yellow]WARN  Vol ratio STRONG_BUY/HOLD = {vol_ratio:.1f}x -- "
                    "STRONG_BUY bucket carries significantly more risk; Sharpe may be illusory.[/yellow]"
                )

    # --- Per-parent IC table ---
    if result.per_parent_ic:
        from scanner.graph.loader import MEGA_CAPS as _MEGA_CAPS

        pp_table = Table(title="Per-Parent IC (pooled)", box=box.SIMPLE_HEAD)
        pp_table.add_column("Parent", width=8)
        pp_table.add_column("IC", justify="right", width=9)
        pp_table.add_column("n obs", justify="right", width=7)

        for mc in _MEGA_CAPS:
            pair = result.per_parent_ic.get(mc)
            if pair is None:
                pp_table.add_row(mc, "[dim]n/a[/dim]", "[dim]-[/dim]")
                continue
            ic_val, n = pair
            ic_color = "green" if ic_val > 0.05 else ("yellow" if ic_val > 0 else "red")
            pp_table.add_row(mc, f"[{ic_color}]{ic_val:+.4f}[/{ic_color}]", str(n))
        console.print(pp_table)

    # --- Parent-return conditional IC table ---
    if result.parent_conditional_ic:
        cond_table = Table(title="IC Conditional on Parent Return (20d)", box=box.SIMPLE_HEAD)
        cond_table.add_column("Condition", width=22)
        cond_table.add_column("IC", justify="right", width=9)
        cond_table.add_column("n obs", justify="right", width=7)

        labels = {
            "strong": "Parent > 0 (20d)",
            "weak": "Parent <= 0 (20d)",
        }
        for key in ("strong", "weak"):
            pair = result.parent_conditional_ic.get(key)
            if pair is None:
                cond_table.add_row(labels[key], "[dim]n/a[/dim]", "[dim]-[/dim]")
                continue
            ic_val, n = pair
            ic_color = "green" if ic_val > 0.05 else ("yellow" if ic_val > 0 else "red")
            cond_table.add_row(labels[key], f"[{ic_color}]{ic_val:+.4f}[/{ic_color}]", str(n))
        console.print(cond_table)

    # --- Parent-regime IC table ---
    if result.parent_regime_ic:
        regime_table = Table(title="IC by Parent Drawdown Magnitude (20d)", box=box.SIMPLE_HEAD)
        regime_table.add_column("Regime", width=24)
        regime_table.add_column("IC", justify="right", width=9)
        regime_table.add_column("n obs", justify="right", width=7)

        regime_labels = {
            "strong":     "Parent > 0%",
            "mild":       "Parent -5% to 0%",
            "correction": "Parent -15% to -5%",
            "drawdown":   "Parent < -15%",
        }
        for key in ("strong", "mild", "correction", "drawdown"):
            pair = result.parent_regime_ic.get(key)
            if pair is None:
                regime_table.add_row(regime_labels[key], "[dim]n/a[/dim]", "[dim]-[/dim]")
                continue
            ic_val, n = pair
            ic_color = "green" if ic_val > 0.05 else ("yellow" if ic_val > 0 else "red")
            regime_table.add_row(
                regime_labels[key],
                f"[{ic_color}]{ic_val:+.4f}[/{ic_color}]",
                str(n),
            )
        console.print(regime_table)

    # --- Out-of-sample regime IC split table ---
    if result.regime_ic_splits and result.split_dates:
        start_d, mid_d, end_d = result.split_dates
        split_table = Table(
            title=f"Regime IC - Out-of-Sample Split  (H1: {start_d} to {mid_d} | H2: {mid_d} to {end_d})",
            box=box.SIMPLE_HEAD,
        )
        split_table.add_column("Regime", width=24)
        split_table.add_column("H1 IC", justify="right", width=9)
        split_table.add_column("H1 n", justify="right", width=7)
        split_table.add_column("H2 IC", justify="right", width=9)
        split_table.add_column("H2 n", justify="right", width=7)

        regime_labels = {
            "strong":     "Parent > 0%",
            "mild":       "Parent -5% to 0%",
            "correction": "Parent -15% to -5%",
            "drawdown":   "Parent < -15%",
        }
        h1 = result.regime_ic_splits.get("h1", {})
        h2 = result.regime_ic_splits.get("h2", {})
        for key in ("strong", "mild", "correction", "drawdown"):
            h1_pair = h1.get(key)
            h2_pair = h2.get(key)

            if h1_pair is None:
                h1_ic_str, h1_n_str = "[dim]n/a[/dim]", "[dim]-[/dim]"
            else:
                ic_val, n = h1_pair
                c = "green" if ic_val > 0.05 else ("yellow" if ic_val > 0 else "red")
                h1_ic_str, h1_n_str = f"[{c}]{ic_val:+.4f}[/{c}]", str(n)

            if h2_pair is None:
                h2_ic_str, h2_n_str = "[dim]n/a[/dim]", "[dim]-[/dim]"
            else:
                ic_val, n = h2_pair
                c = "green" if ic_val > 0.05 else ("yellow" if ic_val > 0 else "red")
                h2_ic_str, h2_n_str = f"[{c}]{ic_val:+.4f}[/{c}]", str(n)

            split_table.add_row(regime_labels[key], h1_ic_str, h1_n_str, h2_ic_str, h2_n_str)
        console.print(split_table)

    console.print(
        "\n[dim]Note: Backtest uses price history from DuckDB (Massive.com source). "
        "Sharpe/DD figures may be overstated due to survivorship bias in the static universe.[/dim]"
    )


@app.command("backtest-rotation")
def backtest_rotation() -> None:
    """Walk-forward backtest: does XLK rotation status predict MSFT/META dependent returns?"""
    from scanner.backtest.rotation_runner import (
        run_rotation_backtest,
        _INSUFFICIENT_OBS,
        _MARGINAL_OBS,
    )

    console.print("[bold cyan]Running sector rotation backtest...[/bold cyan]")
    console.print(
        "[dim]Scope: CORRECTION regime (-15% to -5% parent 20d return), "
        "MSFT/META dependents only[/dim]"
    )

    result = run_rotation_backtest()

    if result is None:
        console.print(
            "[yellow]No observations found. Ensure prices are ingested:\n"
            "  scanner ingest XLK XLF XLE XLV SPY\n"
            "  scanner ingest  (full universe)[/yellow]"
        )
        return

    if not result.etf_data_available:
        console.print(
            "[yellow]Warning: sector ETF data missing — rotation buckets will be empty.[/yellow]\n"
            "[dim]Run: scanner ingest XLK XLF XLE XLV SPY[/dim]"
        )

    def _fmt_ic(v: float) -> str:
        if math.isnan(v):
            return "[dim]n/a[/dim]"
        c = "green" if v > 0.05 else ("yellow" if v > 0 else "red")
        return f"[{c}]{v:+.4f}[/{c}]"

    def _fmt_n(bucket_n: dict, h: int) -> str:
        n = bucket_n.get(h, 0)
        if n < _INSUFFICIENT_OBS:
            return f"[red]{n}[/red]"
        if n < _MARGINAL_OBS:
            return f"[yellow]{n}[/yellow]"
        return str(n)

    # Table 1: IC by bucket and horizon
    ic_table = Table(
        title=(
            f"Rotation Backtest IC  "
            f"({result.start_date} to {result.end_date}  |  "
            f"{result.n_dates} dates  |  H1: {result.n_h1} / H2: {result.n_h2})"
        ),
        box=box.SIMPLE_HEAD,
    )
    ic_table.add_column("Bucket", width=14)
    ic_table.add_column("Sufficiency", width=13)
    ic_table.add_column("n (5d)", justify="right", width=9)
    for h in (5, 10, 20):
        ic_table.add_column(f"IC {h}d", justify="right", width=11)
    ic_table.add_column("IC H1 (5d)", justify="right", width=12)
    ic_table.add_column("IC H2 (5d)", justify="right", width=12)

    for bucket in (result.baseline, result.leading, result.not_leading):
        suf_color = {"OK": "green", "MARGINAL": "yellow", "INSUFFICIENT": "red"}.get(
            bucket.sufficiency, "white"
        )
        suf_cell = f"[{suf_color}]{bucket.sufficiency}[/{suf_color}]"
        ic_table.add_row(
            bucket.label,
            suf_cell,
            _fmt_n(bucket.n, 5),
            _fmt_ic(bucket.ic_mean.get(5, float("nan"))),
            _fmt_ic(bucket.ic_mean.get(10, float("nan"))),
            _fmt_ic(bucket.ic_mean.get(20, float("nan"))),
            _fmt_ic(bucket.ic_h1_mean.get(5, float("nan"))),
            _fmt_ic(bucket.ic_h2_mean.get(5, float("nan"))),
        )
    console.print(ic_table)

    # Table 2: IC delta (LEADING vs baseline)
    delta_table = Table(title="IC Delta: LEADING vs baseline", box=box.SIMPLE_HEAD)
    delta_table.add_column("Horizon", width=10)
    delta_table.add_column("Baseline IC", justify="right", width=13)
    delta_table.add_column("LEADING IC", justify="right", width=13)
    delta_table.add_column("Delta", justify="right", width=10)

    for h in (5, 10, 20):
        base_ic = result.baseline.ic_mean.get(h, float("nan"))
        lead_ic = result.leading.ic_mean.get(h, float("nan"))
        delta = (
            lead_ic - base_ic
            if not (math.isnan(lead_ic) or math.isnan(base_ic))
            else float("nan")
        )
        delta_str = (
            f"[{'green' if delta > 0 else 'red'}]{delta:+.4f}[/{'green' if delta > 0 else 'red'}]"
            if not math.isnan(delta) else "[dim]n/a[/dim]"
        )
        delta_table.add_row(f"{h}d", _fmt_ic(base_ic), _fmt_ic(lead_ic), delta_str)
    console.print(delta_table)

    # Validation summary
    console.print("[bold]Validation[/bold]")
    baseline_ic_5 = result.baseline.ic_mean.get(5, float("nan"))
    baseline_h1 = result.baseline.ic_h1_mean.get(5, float("nan"))
    baseline_h2 = result.baseline.ic_h2_mean.get(5, float("nan"))
    leading_ic_5 = result.leading.ic_mean.get(5, float("nan"))
    leading_h1 = result.leading.ic_h1_mean.get(5, float("nan"))
    leading_h2 = result.leading.ic_h2_mean.get(5, float("nan"))

    def _val_line(label: str, ic: float, h1: float, h2: float, n: int) -> None:
        if n < _INSUFFICIENT_OBS:
            console.print(f"  {label}: [red]INSUFFICIENT DATA (n={n})[/red]")
            return
        ic_ok = not math.isnan(ic) and ic > 0.05
        both_ok = (not math.isnan(h1) and h1 > 0) and (not math.isnan(h2) and h2 > 0)
        if ic_ok and both_ok:
            status = "[bold green]PASS[/bold green]"
            note = "IC > 0.05 and both halves positive"
        elif not math.isnan(ic) and ic > 0 and both_ok:
            status = "[yellow]MARGINAL[/yellow]"
            note = "IC positive but < 0.05"
        else:
            status = "[red]FAIL[/red]"
            note = "IC <= 0 or inconsistent"
        suf = "" if n >= _MARGINAL_OBS else f"  [yellow](n={n}, MARGINAL)[/yellow]"
        console.print(f"  {label} 5d: {status}  [dim]{note}[/dim]{suf}")

    _val_line("Baseline", baseline_ic_5, baseline_h1, baseline_h2, result.baseline.n.get(5, 0))
    _val_line("LEADING", leading_ic_5, leading_h1, leading_h2, result.leading.n.get(5, 0))

    console.print(
        "\n[dim]"
        "Score: raw 20d return differential (child minus avg qualifying parent) — continuous Spearman IC.\n"
        "Scope: CORRECTION regime (-15% to -5% parent 20d return), MSFT/META dependents only.\n"
        "Survivorship-bias caveat: yfinance only returns currently-listed tickers. "
        "Results overstate performance. Migrate to Polygon.io before trusting live numbers."
        "[/dim]"
    )


@app.command("backtest-themes")
def backtest_themes() -> None:
    """Walk-forward backtest of RS vs benchmark ETF signal for all active theme tickers."""
    from scanner.backtest.theme_runner import MIN_OBS, HORIZONS, run_theme_backtest

    console.print("[bold cyan]Running theme basket backtest...[/bold cyan]")
    console.print(
        "[dim]Signal: 20-day return of ticker minus 20-day return of benchmark ETF\n"
        "Horizons: 5d, 10d, 20d  |  PASS: IC > 0.05 at 10d or 20d, both halves positive[/dim]\n"
    )

    result = run_theme_backtest()

    if not result.ticker_results:
        console.print(
            "[yellow]No theme tickers found. Run 'scanner ingest' first.[/yellow]"
        )
        return

    def _fmt_ic(v: float) -> str:
        if math.isnan(v):
            return "[dim]n/a[/dim]"
        color = "green" if v > 0.05 else ("yellow" if v > 0 else "red")
        return f"[{color}]{v:+.3f}[/{color}]"

    def _fmt_n(n: int) -> str:
        if n < MIN_OBS:
            return f"[red]{n}[/red]"
        return str(n)

    console.print(f"[bold]Theme Basket Backtest Results[/bold]")
    console.print("═" * 70)

    current_theme = None
    for r in result.ticker_results:
        if r.theme_name != current_theme:
            current_theme = r.theme_name
            console.print(f"\n[bold]{r.theme_label.upper()}[/bold]")

        console.print(
            f"  Ticker: [bold]{r.ticker}[/bold]  Benchmark: {r.benchmark_etf}  "
            f"Observations: {_fmt_n(r.n_obs)}"
        )

        if r.n_obs < MIN_OBS:
            console.print(
                f"  [yellow]INSUFFICIENT DATA — need {MIN_OBS}+ observations "
                f"(have {r.n_obs}). Run ingest to accumulate more price history.[/yellow]"
            )
            continue

        t = Table(box=box.SIMPLE, show_header=True, pad_edge=False)
        t.add_column("Horizon", width=9)
        t.add_column("IC (pooled)", justify="right", width=12)
        t.add_column("H1", justify="right", width=10)
        t.add_column("H2", justify="right", width=10)
        t.add_column("Status", width=8)

        for h in HORIZONS:
            ic_full = r.ic.get(h, float("nan"))
            ic_h1 = r.ic_h1.get(h, float("nan"))
            ic_h2 = r.ic_h2.get(h, float("nan"))
            both_pos = (not math.isnan(ic_h1) and ic_h1 > 0) and (not math.isnan(ic_h2) and ic_h2 > 0)
            status = (
                "[green]PASS[/green]"
                if not math.isnan(ic_full) and ic_full > 0.05 and both_pos
                else "[dim]—[/dim]"
            )
            t.add_row(f"{h}d", _fmt_ic(ic_full), _fmt_ic(ic_h1), _fmt_ic(ic_h2), status)

        console.print(t)
        pass_str = "[green]PASS ✓[/green]" if r.passes else "[red]FAIL[/red]"
        console.print(f"  Signal verdict: {pass_str}\n")

    console.print("═" * 70)
    console.print(f"\n[bold]OVERALL THEME BASKET[/bold]")
    t_combined = Table(box=box.SIMPLE, show_header=True, pad_edge=False)
    t_combined.add_column("Horizon", width=9)
    t_combined.add_column("Combined IC", justify="right", width=12)
    for h in HORIZONS:
        t_combined.add_row(f"{h}d", _fmt_ic(result.combined_ic.get(h, float("nan"))))
    console.print(t_combined)

    valid_color = "green" if result.n_valid > 0 else "red"
    console.print(
        f"\n  Valid themes (IC > 0.05, both halves +): "
        f"[{valid_color}]{result.n_valid}/{result.n_total}[/{valid_color}]"
    )
    console.print(
        "\n[dim]Survivorship-bias caveat: yfinance only returns currently-listed tickers. "
        "Results overstate performance. Migrate to Polygon.io before trusting live numbers.[/dim]"
    )


_TICKER_BENCHMARKS: dict[str, str] = {
    "BB": "IGV",
    "AEHR": "SMH",
    "BE": "GRID",
    "NOK": "XLK",
    "HUT": "WGMI",
    "FPS": "GRID",
    "YSS": "ITA",
}

_POST_PIVOT_DATE = "2024-01-01"
_FULL_MIN_OBS = 100
_PIVOT_MIN_OBS = 30


@app.command("backtest-catalyst")
def backtest_catalyst(
    tickers: list[str] = typer.Argument(None, help="Tickers to backtest (e.g. BB AEHR BE)"),
    ticker_opt: str | None = typer.Option(None, "--ticker", "-t", help="Single ticker (alternative to positional args)"),
    start_date: str | None = typer.Option(None, "--start-date", help="Override: use only this single window"),
) -> None:
    """Alternative signal backtest — rvol, rs_vs_xlk, rs_vs_benchmark.

    Runs two windows automatically per ticker: full history (n>=100) and post-2024 (n>=30).
    Benchmarks: BB=IGV, AEHR=SMH, BE=GRID, NOK=XLK, HUT=WGMI. Use --start-date to force one window.
    """
    from scanner.backtest.catalyst_runner import HORIZONS, run_catalyst_backtest

    all_tickers: list[str] = []
    if tickers:
        all_tickers = [t.upper() for t in tickers]
    if ticker_opt:
        all_tickers.append(ticker_opt.upper())
    if not all_tickers:
        all_tickers = ["NOK"]

    def _fmt_ic(v: float) -> str:
        if math.isnan(v):
            return "[dim]n/a[/dim]"
        color = "green" if v > 0.05 else ("yellow" if v > 0 else "red")
        return f"[{color}]{v:+.3f}[/{color}]"

    def _best_signal(signal_results) -> tuple[str, float]:
        """Return (signal_name, best_ic) for the signal with highest passing IC at 10d/20d."""
        best_name, best_ic = "", float("nan")
        for sr in signal_results:
            for h in (10, 20):
                ic = sr.ic.get(h, float("nan"))
                if not math.isnan(ic) and (math.isnan(best_ic) or ic > best_ic):
                    best_ic = ic
                    best_name = sr.name
        return best_name, best_ic

    def _verdict(signal_results, min_obs: int) -> str:
        any_pass = any(sr.passes for sr in signal_results)
        all_insuf = all(sr.n_obs < min_obs for sr in signal_results)
        if all_insuf:
            return "INSUFFICIENT DATA"
        if any_pass:
            return "ACTIVATE"
        any_positive = any(
            not math.isnan(sr.ic.get(h, float("nan"))) and sr.ic.get(h, 0.0) > 0
            for sr in signal_results for h in (10, 20)
        )
        return "WATCH" if any_positive else "WATCH"

    _signal_label_map = {
        "rvol": "RVOL Spike",
        "rs_vs_xlk": "RS vs XLK",
        "rs_vs_benchmark": "RS vs {bm}",
    }

    console.print(
        f"\n[bold cyan]Catalyst Backtest[/bold cyan]  |  "
        f"Signals: RVOL | RS vs XLK | RS vs benchmark  |  Horizons: 5d/10d/20d"
    )
    console.print(
        "[dim]PASS: IC > 0.05 at 10d or 20d, both halves positive, n >= threshold[/dim]\n"
    )

    for tkr in all_tickers:
        benchmark = _TICKER_BENCHMARKS.get(tkr, "XLK")

        if start_date:
            windows = [(start_date, _PIVOT_MIN_OBS, f"from {start_date} (forced)")]
        else:
            windows = [
                (None, _FULL_MIN_OBS, f"Full history  (n >= {_FULL_MIN_OBS})"),
                (_POST_PIVOT_DATE, _PIVOT_MIN_OBS, f"Post-2024     (n >= {_PIVOT_MIN_OBS}, from {_POST_PIVOT_DATE})"),
            ]

        console.print(f"[bold]{'=' * 72}[/bold]")
        console.print(f"[bold]{tkr}[/bold]  vs benchmark: [cyan]{benchmark}[/cyan]")
        console.print(f"[bold]{'=' * 72}[/bold]")

        for win_start, win_min_obs, win_label in windows:
            result = run_catalyst_backtest(
                tkr, benchmark=benchmark, start_date=win_start, min_obs=win_min_obs
            )

            verdict = _verdict(result.signal_results, win_min_obs)
            verdict_style = {
                "ACTIVATE": "bold green",
                "WATCH": "yellow",
                "INSUFFICIENT DATA": "dim",
            }.get(verdict, "white")

            best_name, best_ic = _best_signal(result.signal_results)
            best_label = _signal_label_map.get(best_name, best_name).replace("{bm}", benchmark)
            best_str = f"{best_label} (IC {best_ic:+.3f})" if not math.isnan(best_ic) else "--"

            console.print(f"\n[bold]{win_label}[/bold]")
            console.print(f"  Verdict: [{verdict_style}]{verdict}[/{verdict_style}]   Best signal: {best_str}")

            def _ic_plain(v: float) -> str:
                if math.isnan(v):
                    return "   n/a"
                return f"{v:+.3f}"

            def _col(v: str) -> str:
                try:
                    f = float(v.strip())
                    color = "green" if f > 0.05 else ("yellow" if f > 0 else "red")
                    return f"[{color}]{v}[/{color}]"
                except ValueError:
                    return f"[dim]{v}[/dim]"

            hdr = f"  {'Signal':<16} {'n':>4}  {'IC10':>6} {'IC20':>6}  {'H1/10':>6} {'H2/10':>6}  {'H1/20':>6} {'H2/20':>6}  Res"
            console.print(f"[dim]{hdr}[/dim]")
            console.print(f"[dim]  {'-'*77}[/dim]")
            for sr in result.signal_results:
                sig_label = _signal_label_map.get(sr.name, sr.name).replace("{bm}", benchmark)
                n_color = "red" if sr.n_obs < win_min_obs else "white"
                ic10 = _ic_plain(sr.ic.get(10, float("nan")))
                ic20 = _ic_plain(sr.ic.get(20, float("nan")))
                h1_10 = _ic_plain(sr.ic_h1.get(10, float("nan")))
                h2_10 = _ic_plain(sr.ic_h2.get(10, float("nan")))
                h1_20 = _ic_plain(sr.ic_h1.get(20, float("nan")))
                h2_20 = _ic_plain(sr.ic_h2.get(20, float("nan")))
                pass_str = "[green]PASS[/green]" if sr.passes else "[dim]  --[/dim]"

                console.print(
                    f"  {sig_label:<16} [{n_color}]{sr.n_obs:>4}[/{n_color}]  "
                    f"{_col(ic10)} {_col(ic20)}  "
                    f"{_col(h1_10)} {_col(h2_10)}  "
                    f"{_col(h1_20)} {_col(h2_20)}  {pass_str}"
                )

        console.print()

    console.print(
        "[dim]Survivorship-bias caveat: yfinance only returns currently-listed tickers. "
        "Results overstate performance. Migrate to Polygon.io before trusting live numbers.[/dim]\n"
    )


@app.command("scan-aipower")
def scan_aipower(
    scan_date: Annotated[
        Optional[str],
        typer.Option("--date", help="Scan date YYYY-MM-DD (default: latest price date)"),
    ] = None,
) -> None:
    """Live Z-score × RVOL scan for the AI Power Infrastructure basket (BE, FPS, CEG, VST, HUT)."""
    import bisect
    from collections import defaultdict
    from scanner.db import get_connection

    _BASKET = ["BE", "FPS", "CEG", "VST", "HUT"]
    _LOOKBACK = 20
    _MIN_BASKET = 3

    conn = get_connection()
    try:
        as_of = scan_date
        if not as_of:
            row = conn.execute(
                f"SELECT MAX(date) FROM prices WHERE ticker IN ({','.join('?' for _ in _BASKET)})",
                _BASKET,
            ).fetchone()
            as_of = str(row[0]) if row and row[0] else str(date.today())

        placeholders = ",".join("?" for _ in _BASKET)
        rows = conn.execute(
            f"SELECT ticker, CAST(date AS VARCHAR), adj_close, volume "
            f"FROM prices WHERE ticker IN ({placeholders}) AND date <= ? "
            f"ORDER BY ticker, date",
            _BASKET + [as_of],
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        console.print("[yellow]No price data found for basket. Run ingest first.[/yellow]")
        return

    ticker_dates: dict[str, list[str]] = defaultdict(list)
    ticker_closes: dict[str, list[float]] = defaultdict(list)
    ticker_vols: dict[str, list[float]] = defaultdict(list)
    for tkr, dt, close, vol in rows:
        ticker_dates[tkr].append(dt)
        ticker_closes[tkr].append(float(close))
        ticker_vols[tkr].append(float(vol) if vol is not None else 0.0)

    # --- Compute signals for each ticker on as_of ---
    basket_rets: dict[str, float] = {}
    basket_rvols: dict[str, float] = {}
    ticker_prices: dict[str, float] = {}
    ticker_latest_dates: dict[str, str] = {}

    for tkr in _BASKET:
        dates = ticker_dates.get(tkr, [])
        closes = ticker_closes.get(tkr, [])
        vols = ticker_vols.get(tkr, [])
        if not dates:
            continue

        idx = bisect.bisect_right(dates, as_of) - 1
        if idx < 0:
            continue

        ticker_latest_dates[tkr] = dates[idx]
        ticker_prices[tkr] = closes[idx]

        if idx < _LOOKBACK:
            continue

        close_now = closes[idx]
        close_20d_ago = closes[idx - _LOOKBACK]
        if close_20d_ago == 0:
            continue
        basket_rets[tkr] = (close_now - close_20d_ago) / close_20d_ago

        avg_vol = sum(vols[idx - _LOOKBACK:idx]) / _LOOKBACK
        if avg_vol > 0:
            basket_rvols[tkr] = vols[idx] / avg_vol

    if len(basket_rets) < _MIN_BASKET:
        console.print(
            f"[yellow]Only {len(basket_rets)} tickers have 20d history on {as_of} "
            f"(need {_MIN_BASKET}+). Run ingest first.[/yellow]"
        )
        return

    ret_vals = list(basket_rets.values())
    basket_mean = sum(ret_vals) / len(ret_vals)
    n = len(ret_vals)
    basket_std = math.sqrt(sum((r - basket_mean) ** 2 for r in ret_vals) / n)

    # --- Build per-ticker result rows ---
    _ACTION_THRESHOLDS = [
        (1.5, "STRONG_BUY"),
        (0.5, "BUY"),
        (-0.5, "HOLD"),
    ]
    _ACTION_STYLE = {
        "STRONG_BUY": "bold green",
        "BUY": "green",
        "HOLD": "yellow",
        "SELL": "red",
        "—": "dim",
    }

    def _action(combined: float) -> str:
        if math.isnan(combined):
            return "—"
        for threshold, label in _ACTION_THRESHOLDS:
            if combined > threshold:
                return label
        return "SELL"

    scan_rows = []
    for tkr in _BASKET:
        price = ticker_prices.get(tkr)
        latest_dt = ticker_latest_dates.get(tkr, "—")
        ret_20d = basket_rets.get(tkr)
        rvol = basket_rvols.get(tkr, float("nan"))

        if ret_20d is None or basket_std == 0:
            z = float("nan")
            combined = float("nan")
        else:
            z = (ret_20d - basket_mean) / basket_std
            combined = z * rvol if not math.isnan(rvol) else float("nan")

        scan_rows.append({
            "ticker": tkr,
            "price": price,
            "latest_date": latest_dt,
            "ret_20d": ret_20d,
            "z_score": z,
            "rvol": rvol,
            "combined": combined,
            "action": _action(combined),
        })

    # Sort: STRONG_BUY first, then by combined desc, NaN last
    _order = {"STRONG_BUY": 0, "BUY": 1, "HOLD": 2, "SELL": 3, "—": 4}
    scan_rows.sort(key=lambda r: (_order.get(r["action"], 4), -(r["combined"] if not math.isnan(r.get("combined", float("nan"))) else -999)))

    # --- Render ---
    def _fmt(v, decimals=2, pct=False, sign=False):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        if pct:
            return f"{v * 100:+.2f}%" if sign else f"{v * 100:.2f}%"
        fmt = f"{v:+.{decimals}f}" if sign else f"{v:.{decimals}f}"
        return fmt

    def _color_z(v: float) -> str:
        if math.isnan(v):
            return "[dim]—[/dim]"
        c = "green" if v > 0 else "red"
        return f"[{c}]{v:+.3f}[/{c}]"

    def _color_rvol(v: float) -> str:
        if math.isnan(v):
            return "[dim]—[/dim]"
        c = "green" if v >= 1.5 else ("yellow" if v >= 1.0 else "dim")
        return f"[{c}]{v:.2f}x[/{c}]"

    def _color_combined(v: float) -> str:
        if math.isnan(v):
            return "[dim]—[/dim]"
        c = "bold green" if v > 1.5 else ("green" if v > 0.5 else ("yellow" if v > -0.5 else "red"))
        return f"[{c}]{v:+.3f}[/{c}]"

    std_display = f"{basket_std * 100:.2f}%" if basket_std > 0 else "0%"
    mean_display = f"{basket_mean * 100:+.2f}%"
    table = Table(
        title=(
            f"AI Power Infrastructure — Z-Score × RVOL  |  {as_of}\n"
            f"Basket 20d avg return: {mean_display}  |  basket σ: {std_display}  |  "
            f"n={len(basket_rets)} tickers with 20d history"
        ),
        box=box.SIMPLE_HEAD,
    )
    table.add_column("Ticker", width=7)
    table.add_column("Signal", width=12)
    table.add_column("Price", justify="right", width=9)
    table.add_column("20d Ret", justify="right", width=9)
    table.add_column("Z-Score", justify="right", width=9)
    table.add_column("RVOL", justify="right", width=8)
    table.add_column("Combined", justify="right", width=10)
    table.add_column("Data Date", width=12)

    for r in scan_rows:
        action = r["action"]
        style = _ACTION_STYLE.get(action, "dim")
        table.add_row(
            r["ticker"],
            f"[{style}]{action}[/{style}]",
            _fmt(r["price"], decimals=2),
            _fmt(r["ret_20d"], pct=True, sign=True) if r["ret_20d"] is not None else "—",
            _color_z(r["z_score"]),
            _color_rvol(r["rvol"]),
            _color_combined(r["combined"]),
            r["latest_date"],
        )

    console.print(table)
    console.print(
        "[dim]Thresholds: combined > +1.5 → STRONG_BUY  |  > +0.5 → BUY  |  "
        "> -0.5 → HOLD  |  ≤ -0.5 → SELL\n"
        "Validated: Z-score × RVOL passes IC > 0.05 at 5d and 10d (post-2024 backtest).[/dim]\n"
    )


@app.command("backtest-zscore")
def backtest_zscore() -> None:
    """Cross-sectional Z-score backtest for AI Power Infrastructure basket (BE, FPS, CEG, VST, HUT)."""
    from scanner.backtest.zscore_runner import (
        BASKET,
        HORIZONS,
        _MIN_OBS_FULL,
        _MIN_OBS_POST2024,
        run_zscore_backtest,
    )

    console.print("[bold cyan]Running AI Power Infrastructure Z-Score backtest...[/bold cyan]")
    console.print("[dim]Loading price+volume history from DuckDB...[/dim]\n")

    pair = run_zscore_backtest()

    if pair is None:
        console.print(
            "[yellow]Insufficient data: no price rows found for basket tickers.[/yellow]\n"
            f"[dim]Run [bold]scanner ingest[/bold] for: {', '.join(BASKET)}[/dim]"
        )
        return

    full_result, post2024_result = pair

    def _fmt_ic(v: float) -> str:
        if math.isnan(v):
            return "[dim]  n/a[/dim]"
        c = "green" if v > 0.05 else ("yellow" if v > 0 else "red")
        return f"[{c}]{v:+.4f}[/{c}]"

    def _pass_label(ic_pool: float, ic_h1: float, ic_h2: float, n: int, min_n: int) -> str:
        if n < min_n:
            return "[dim]INSUF DATA[/dim]"
        if math.isnan(ic_pool) or math.isnan(ic_h1) or math.isnan(ic_h2):
            return "[dim]n/a[/dim]"
        passes = ic_pool > 0.05 and ic_h1 > 0 and ic_h2 > 0
        return "[green]PASS[/green]" if passes else "[red]FAIL[/red]"

    def _render_version(result, min_n: int) -> None:
        if result is None:
            console.print("[yellow]No data for this window.[/yellow]\n")
            return

        console.print(
            f"[bold]Ticker observations (5d-horizon):[/bold]  "
            + "  ".join(f"{t}: {result.ticker_obs.get(t, 0)}" for t in BASKET)
        )
        console.print(
            f"Total signal-date pairs: {result.total_obs}  "
            f"({result.start_date} to {result.end_date}  |  "
            f"H1 n={result.n_h1}  H2 n={result.n_h2}  |  mid={result.mid_date})"
        )
        console.print()

        # Z-score alone
        z_table = Table(title="Signal: Z-score alone", box=box.SIMPLE_HEAD)
        z_table.add_column("Horizon", width=10)
        z_table.add_column("IC pooled", justify="right", width=12)
        z_table.add_column("H1", justify="right", width=10)
        z_table.add_column("H2", justify="right", width=10)
        z_table.add_column("Status", width=12)
        for h in HORIZONS:
            ip = result.ic_pooled.get(h, float("nan"))
            h1 = result.ic_h1.get(h, float("nan"))
            h2 = result.ic_h2.get(h, float("nan"))
            z_table.add_row(
                f"{h}d",
                _fmt_ic(ip),
                _fmt_ic(h1),
                _fmt_ic(h2),
                _pass_label(ip, h1, h2, result.total_obs, min_n),
            )
        console.print(z_table)

        # Z-score × RVOL
        c_table = Table(title="Signal: Z-score × RVOL", box=box.SIMPLE_HEAD)
        c_table.add_column("Horizon", width=10)
        c_table.add_column("IC pooled", justify="right", width=12)
        c_table.add_column("H1", justify="right", width=10)
        c_table.add_column("H2", justify="right", width=10)
        c_table.add_column("Status", width=12)
        for h in HORIZONS:
            ip = result.ic_combined_pooled.get(h, float("nan"))
            h1 = result.ic_combined_h1.get(h, float("nan"))
            h2 = result.ic_combined_h2.get(h, float("nan"))
            c_table.add_row(
                f"{h}d",
                _fmt_ic(ip),
                _fmt_ic(h1),
                _fmt_ic(h2),
                _pass_label(ip, h1, h2, result.total_obs, min_n),
            )
        console.print(c_table)

    console.rule("[bold]AI Power Infrastructure — Z-Score Backtest[/bold]")
    console.print(f"Basket: {', '.join(BASKET)}")
    console.print("Method: Cross-sectional Z-score (20d return within basket)\n")

    console.rule("[FULL HISTORY]", style="dim")
    _render_version(full_result, _MIN_OBS_FULL)

    console.rule("[POST-2024 ONLY]", style="dim")
    _render_version(post2024_result, _MIN_OBS_POST2024)

    console.print(
        "[dim]PASS criteria: IC pooled > 0.05 at 10d or 20d, both H1 and H2 positive, "
        f"n >= {_MIN_OBS_FULL} (full) or n >= {_MIN_OBS_POST2024} (post-2024).[/dim]"
    )
    console.print(
        "[dim]Survivorship-bias caveat: yfinance only returns currently-listed tickers. "
        "Results overstate performance.[/dim]\n"
    )


@app.command()
def discover(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show candidate count without inserting", is_flag=True),
    ] = False,
) -> None:
    """Fetch related companies from Massive API for each known parent and insert new ACCUMULATING candidates."""
    from scanner.ingest.discovery import discover as _discover

    if dry_run:
        console.print("[yellow]--dry-run not supported with related-companies source — run without flag to insert[/yellow]")
        return

    console.print("[bold cyan]Running discovery pipeline...[/bold cyan]")
    inserted = _discover()
    console.print(f"[green]Discovery complete: {inserted} new candidates inserted[/green]")


@app.command("accumulate-rs")
def accumulate_rs_cmd() -> None:
    """Compute daily RS score for ACCUMULATING candidates; advance to READY_FOR_BACKTEST at 60 days."""
    from scanner.ingest.discovery import accumulate_rs

    inserted = accumulate_rs()
    console.print(f"[cyan]RS accumulation: {inserted} rows inserted[/cyan]")


@app.command("run-discovery-backtests")
def run_discovery_backtests_cmd() -> None:
    """Run IC backtest for READY_FOR_BACKTEST candidates; auto-promote or mark FAILED."""
    from scanner.ingest.discovery import run_discovery_backtests

    processed = run_discovery_backtests()
    console.print(f"[cyan]Discovery backtests: {processed} candidates processed[/cyan]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
