"""APScheduler jobs: daily price ingest, estimates ingest, weekly insider ingest."""

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Mutable status dict — updated by each job run, read by /api/status endpoint.
scheduler_status: dict[str, Any] = {
    "last_ingest": None,
    "last_estimates": None,
    "last_insiders": None,
    "last_earnings": None,
    "last_filings": None,
    "last_cleanup": None,
    "last_scan": None,
    "last_journal_capture": None,
    "last_ingest_ok": None,
    "last_estimates_ok": None,
    "last_insiders_ok": None,
    "last_earnings_ok": None,
    "last_filings_ok": None,
    "last_cleanup_ok": None,
    "last_scan_ok": None,
    "last_journal_capture_ok": None,
}

# Module-level reference set by lifespan so api_status can query it.
_scheduler_instance = None


def set_scheduler(s) -> None:
    global _scheduler_instance
    _scheduler_instance = s


# Maps APScheduler job id → human label and scheduler_status keys.
_JOB_META = {
    "ingest_prices":    ("ingest",           "last_ingest",     "last_ingest_ok"),
    "ingest_estimates": ("ingest-estimates",  "last_estimates",  "last_estimates_ok"),
    "ingest_insiders":  ("ingest-insiders",   "last_insiders",   "last_insiders_ok"),
    "ingest_earnings":  ("ingest-earnings",   "last_earnings",   "last_earnings_ok"),
    "ingest_filings":   ("ingest-filings",    "last_filings",    "last_filings_ok"),
    "cleanup_filings":  ("cleanup-filings",   "last_cleanup",    "last_cleanup_ok"),
    "run_scan":         ("run-scan",          "last_scan",       "last_scan_ok"),
    "journal_capture":  ("journal-capture",   "last_journal_capture", "last_journal_capture_ok"),
}


def _write_activity_log(job_name: str, status: str, message: str = "") -> None:
    from scanner.db import get_connection

    try:
        conn = get_connection()
        try:
            now = datetime.utcnow()
            conn.execute(
                "INSERT INTO scheduler_runs (job_name, run_time, status, message) VALUES (?, ?, ?, ?)",
                [job_name, now, status, message],
            )
            conn.execute(
                """
                DELETE FROM scheduler_runs
                WHERE (job_name, run_time) NOT IN (
                    SELECT job_name, run_time FROM scheduler_runs
                    ORDER BY run_time DESC
                    LIMIT 200
                )
                """
            )
        finally:
            conn.close()
    except Exception as exc:
        logger.error("Failed to write scheduler run: %s", exc)


def _fmt_dt(dt) -> str:
    """Cross-platform datetime format: 'Mon May 11 9:05 AM ET'."""
    # %-d / %-I are Linux-only; strip leading zeros manually for portability.
    day = str(dt.day)
    hour = str(dt.hour % 12 or 12)
    ampm = "AM" if dt.hour < 12 else "PM"
    return dt.strftime(f"%a %b {day} {hour}:%M {ampm} ET")


def get_scheduler_info() -> dict:
    """Return structured info for /api/status — safe to call even if scheduler is None."""
    from zoneinfo import ZoneInfo
    from scanner.db import get_connection

    et = ZoneInfo("America/New_York")
    running = _scheduler_instance is not None and _scheduler_instance.running

    # Query last run per job from persistent DB — keyed by APScheduler job_id
    db_runs: dict[str, dict] = {}
    try:
        conn = get_connection()
        try:
            for job_id, (name, _lk, _ok) in _JOB_META.items():
                row = conn.execute(
                    "SELECT job_name, run_time, status, message FROM scheduler_runs WHERE job_name = ? ORDER BY run_time DESC LIMIT 1",
                    [name],
                ).fetchone()
                if row:
                    db_runs[job_id] = {"run_time": row[1], "status": row[2]}
        finally:
            conn.close()
    except Exception as exc:
        logger.error("Failed to query scheduler_runs for status: %s", exc)

    # Build next_run_time map from APScheduler (available only while running)
    next_run_map: dict[str, str | None] = {}
    if _scheduler_instance is not None:
        for job in _scheduler_instance.get_jobs():
            if job.next_run_time is not None:
                next_run_map[job.id] = _fmt_dt(job.next_run_time.astimezone(et))

    # Always enumerate all known jobs from _JOB_META so history survives redeploys
    jobs = []
    for job_id, (name, last_key, ok_key) in _JOB_META.items():
        db_entry = db_runs.get(job_id)
        last_run_fmt = None
        status = "never_run"

        if db_entry:
            rt = db_entry["run_time"]
            if rt.tzinfo is None:
                rt = rt.replace(tzinfo=timezone.utc)
            last_run_fmt = _fmt_dt(rt.astimezone(et))
            status = "success" if db_entry["status"] == "success" else "failed"
        else:
            # Fall back to in-memory if DB query failed or returned nothing
            last_run_iso = scheduler_status.get(last_key)
            last_ok = scheduler_status.get(ok_key)
            if last_run_iso is not None:
                dt = datetime.fromisoformat(last_run_iso).replace(tzinfo=timezone.utc).astimezone(et)
                last_run_fmt = _fmt_dt(dt)
                status = "success" if last_ok is True else "failed"

        jobs.append({
            "id": job_id,
            "name": name,
            "next_run_time": next_run_map.get(job_id),
            "last_run_time": last_run_fmt,
            "last_run_status": status,
        })

    return {"running": running, "jobs": jobs}


def _job_ingest() -> None:
    from scanner.ingest.prices import ingest_all

    logger.info("Scheduled job: price ingest starting")
    try:
        results = ingest_all(None)
        errors = [t for t, s in results.items() if s == "error"]
        ok = scheduler_status["last_ingest_ok"] = len(errors) == 0
        scheduler_status["last_ingest"] = datetime.utcnow().isoformat()
        if errors:
            logger.warning("Price ingest completed with errors: %s", errors)
            _write_activity_log("ingest", "error", f"{len(errors)} tickers failed: {errors[:5]}")
        else:
            logger.info("Price ingest completed: %d tickers ok", len(results))
            _write_activity_log("ingest", "success", f"{len(results)} tickers updated")
        # Invalidate cache so next request sees fresh data.
        if ok:
            from scanner.web.app import _cache
            _cache.clear()
            logger.info("Cache cleared after successful ingest")
    except Exception as exc:
        scheduler_status["last_ingest"] = datetime.utcnow().isoformat()
        scheduler_status["last_ingest_ok"] = False
        logger.error("Price ingest job failed: %s", exc)
        _write_activity_log("ingest", "error", str(exc)[:200])


def _job_estimates() -> None:
    from scanner.ingest.estimates import ingest_estimates

    logger.info("Scheduled job: estimates ingest starting")
    try:
        results = ingest_estimates(None)
        errors = [t for t, s in results.items() if s == "error"]
        ok_est = len(errors) == 0
        scheduler_status["last_estimates_ok"] = ok_est
        scheduler_status["last_estimates"] = datetime.utcnow().isoformat()
        logger.info("Estimates ingest completed: %d tickers", len(results))
        if ok_est:
            _write_activity_log("ingest-estimates", "success", f"{len(results)} tickers updated")
        else:
            _write_activity_log("ingest-estimates", "error", f"{len(errors)} tickers failed: {errors[:5]}")
    except Exception as exc:
        scheduler_status["last_estimates"] = datetime.utcnow().isoformat()
        scheduler_status["last_estimates_ok"] = False
        logger.error("Estimates ingest job failed: %s", exc)
        _write_activity_log("ingest-estimates", "error", str(exc)[:200])


def _job_insiders() -> None:
    import requests as _requests
    from scanner.graph.loader import get_all_tickers
    from scanner.ingest.insiders import _load_cik_map, _make_session, ingest_form4
    from scanner.db import get_connection

    logger.info("Scheduled job: insider ingest starting")
    _status = "error"
    _message = ""
    try:
        session = _make_session()
        cik_map = _load_cik_map(session)
        universe = get_all_tickers()
        conn = get_connection()
        total_rows = 0
        errors = 0
        try:
            for ticker in universe:
                try:
                    rows = ingest_form4(
                        ticker, lookback_days=365, conn=conn, cik_map=cik_map, session=session
                    )
                    total_rows += rows
                except Exception as exc:
                    errors += 1
                    logger.error("Insider ingest failed for %s: %s", ticker, exc)
        finally:
            conn.close()
            session.close()
        ok_ins = errors == 0
        scheduler_status["last_insiders_ok"] = ok_ins
        scheduler_status["last_insiders"] = datetime.utcnow().isoformat()
        logger.info("Insider ingest completed: %d rows, %d errors", total_rows, errors)
        if ok_ins:
            _status = "success"
            _message = f"{total_rows} rows ingested"
        else:
            _status = "error"
            _message = f"{errors} tickers failed, {total_rows} rows written"
    except Exception as exc:
        scheduler_status["last_insiders"] = datetime.utcnow().isoformat()
        scheduler_status["last_insiders_ok"] = False
        logger.error("Insider ingest job failed: %s", exc)
        _message = str(exc)[:200]
    finally:
        _write_activity_log("ingest-insiders", _status, _message)


def _job_earnings() -> None:
    from scanner.ingest.earnings import ingest_earnings

    logger.info("Scheduled job: earnings ingest starting")
    try:
        results = ingest_earnings(None)
        errors = [t for t, s in results.items() if s == "error"]
        ok_earn = len(errors) == 0
        scheduler_status["last_earnings_ok"] = ok_earn
        scheduler_status["last_earnings"] = datetime.utcnow().isoformat()
        if ok_earn:
            _write_activity_log("ingest-earnings", "success", f"{len(results)} tickers updated")
        else:
            _write_activity_log("ingest-earnings", "error", f"{len(errors)} tickers failed: {errors[:5]}")
    except Exception as exc:
        scheduler_status["last_earnings"] = datetime.utcnow().isoformat()
        scheduler_status["last_earnings_ok"] = False
        logger.error("Earnings ingest job failed: %s", exc)
        _write_activity_log("ingest-earnings", "error", str(exc)[:200])


def _job_cleanup_filings() -> None:
    from scanner.ingest.filings import cleanup_filings
    from scanner.db import get_connection

    logger.info("Scheduled job: filings cleanup starting")
    _status = "error"
    _message = ""
    try:
        conn = get_connection()
        try:
            counts = cleanup_filings(conn)
        finally:
            conn.close()
        scheduler_status["last_cleanup_ok"] = True
        scheduler_status["last_cleanup"] = datetime.utcnow().isoformat()
        total = counts["low_removed"] + counts["medium_removed"] + counts["high_removed"]
        logger.info("Filings cleanup job completed: %d rows removed", total)
        _status = "success"
        _message = f"LOW -{counts['low_removed']} / MEDIUM -{counts['medium_removed']} / HIGH -{counts['high_removed']} / Protected {counts['protected']}"
    except Exception as exc:
        scheduler_status["last_cleanup"] = datetime.utcnow().isoformat()
        scheduler_status["last_cleanup_ok"] = False
        logger.error("Filings cleanup job failed: %s", exc)
        _message = str(exc)[:200]
    finally:
        _write_activity_log("cleanup-filings", _status, _message)


def _job_filings() -> None:
    from scanner.ingest.filings import ingest_filings

    logger.info("Scheduled job: 8-K filings ingest starting")
    try:
        results = ingest_filings(None, days_back=7)
        errors = [t for t, s in results.items() if s == "error"]
        ok_fil = len(errors) == 0
        scheduler_status["last_filings_ok"] = ok_fil
        scheduler_status["last_filings"] = datetime.utcnow().isoformat()
        if ok_fil:
            _write_activity_log("ingest-filings", "success", f"{len(results)} tickers processed")
        else:
            _write_activity_log("ingest-filings", "error", f"{len(errors)} tickers failed: {errors[:5]}")
    except Exception as exc:
        scheduler_status["last_filings"] = datetime.utcnow().isoformat()
        scheduler_status["last_filings_ok"] = False
        logger.error("8-K filings ingest job failed: %s", exc)
        _write_activity_log("ingest-filings", "error", str(exc)[:200])


def _job_journal_capture() -> None:
    from scanner.web.app import _capture_journal_sync

    logger.info("Scheduled job: journal capture starting")
    try:
        result = _capture_journal_sync()
        ok = result.get("status") in ("success", "exists")
        scheduler_status["last_journal_capture_ok"] = ok
        scheduler_status["last_journal_capture"] = datetime.utcnow().isoformat()
        if result.get("status") == "exists":
            logger.info("Journal capture: already captured for today")
            _write_activity_log("journal-capture", "success", result.get("message", "already captured"))
        else:
            rows = result.get("rows_added", 0)
            logger.info("Journal capture completed: %d rows", rows)
            _write_activity_log("journal-capture", "success", f"{rows} rows captured")
    except Exception as exc:
        scheduler_status["last_journal_capture"] = datetime.utcnow().isoformat()
        scheduler_status["last_journal_capture_ok"] = False
        logger.error("Journal capture job failed: %s", exc)
        _write_activity_log("journal-capture", "error", str(exc)[:200])


def _job_scan() -> None:
    from scanner.db import get_connection
    from scanner.web.app import _run_scan_today_sync

    logger.info("Scheduled job: scan computation starting")
    try:
        conn = get_connection()
        try:
            latest_row = conn.execute("SELECT MAX(date) FROM prices").fetchone()
        finally:
            conn.close()
        latest_date = str(latest_row[0]) if latest_row and latest_row[0] else None
        if not latest_date:
            logger.warning("Scan job: no price data in DB — skipping")
            scheduler_status["last_scan_ok"] = False
            scheduler_status["last_scan"] = datetime.utcnow().isoformat()
            _write_activity_log("run-scan", "error", "no price data")
            return
        _run_scan_today_sync(as_of=latest_date)
        scheduler_status["last_scan_ok"] = True
        scheduler_status["last_scan"] = datetime.utcnow().isoformat()
        logger.info("Scheduled scan complete for %s", latest_date)
        _write_activity_log("run-scan", "success", f"scan computed for {latest_date}")
    except Exception as exc:
        scheduler_status["last_scan"] = datetime.utcnow().isoformat()
        scheduler_status["last_scan_ok"] = False
        logger.error("Scan job failed: %s", exc)
        _write_activity_log("run-scan", "error", str(exc)[:200])


def create_scheduler():
    """Create and return a configured AsyncIOScheduler."""
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = AsyncIOScheduler(timezone="America/New_York")

    # Daily 9:00 AM ET — price ingest
    scheduler.add_job(
        _job_ingest,
        CronTrigger(hour=9, minute=0, timezone="America/New_York"),
        id="ingest_prices",
        name="Daily price ingest",
        replace_existing=True,
    )

    # Daily 9:05 AM ET — estimates ingest
    scheduler.add_job(
        _job_estimates,
        CronTrigger(hour=9, minute=5, timezone="America/New_York"),
        id="ingest_estimates",
        name="Daily estimates ingest",
        replace_existing=True,
    )

    # Weekly Monday 9:10 AM ET — insider ingest
    scheduler.add_job(
        _job_insiders,
        CronTrigger(day_of_week="mon", hour=9, minute=10, timezone="America/New_York"),
        id="ingest_insiders",
        name="Weekly insider ingest",
        replace_existing=True,
    )

    # Daily 9:15 AM ET — earnings dates ingest
    scheduler.add_job(
        _job_earnings,
        CronTrigger(hour=9, minute=15, timezone="America/New_York"),
        id="ingest_earnings",
        name="Daily earnings ingest",
        replace_existing=True,
    )

    # Daily 9:20 AM ET — 8-K filings ingest
    scheduler.add_job(
        _job_filings,
        CronTrigger(hour=9, minute=20, timezone="America/New_York"),
        id="ingest_filings",
        name="Daily 8-K filings ingest",
        replace_existing=True,
    )

    # Weekly Monday 9:25 AM ET — filings retention cleanup
    scheduler.add_job(
        _job_cleanup_filings,
        CronTrigger(day_of_week="mon", hour=9, minute=25, timezone="America/New_York"),
        id="cleanup_filings",
        name="cleanup-filings",
        replace_existing=True,
    )

    # Daily 9:30 AM ET — scan computation (after all ingestion is complete)
    scheduler.add_job(
        _job_scan,
        CronTrigger(hour=9, minute=30, timezone="America/New_York"),
        id="run_scan",
        name="run-scan",
        replace_existing=True,
    )

    # Daily 9:35 AM ET — journal capture (after scan completes)
    scheduler.add_job(
        _job_journal_capture,
        CronTrigger(hour=9, minute=35, timezone="America/New_York"),
        id="journal_capture",
        name="journal-capture",
        replace_existing=True,
    )

    return scheduler
