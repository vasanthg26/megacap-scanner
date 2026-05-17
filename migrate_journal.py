"""Migrate data/scan_journal.csv into DuckDB journal table, then rename CSV to .bak."""

import csv
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "scan_journal.csv"
BAK_PATH = DATA_DIR / "scan_journal.csv.bak"


def _parse_price(s: str) -> float | None:
    if not s:
        return None
    cleaned = s.replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_float(s: str) -> float | None:
    if not s:
        return None
    try:
        return float(s.lstrip("+"))
    except ValueError:
        return None


def _parse_int(s: str) -> int | None:
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def main() -> None:
    if not CSV_PATH.exists():
        print(f"No CSV found at {CSV_PATH} — nothing to migrate.")
        sys.exit(0)

    from scanner.db import get_connection

    conn = get_connection()
    inserted = 0
    skipped = 0

    try:
        with CSV_PATH.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scan_date = row.get("scan_date", "").strip()
                ticker = row.get("ticker", "").strip()
                parent = row.get("parent", "").strip()
                if not scan_date or not ticker or not parent:
                    skipped += 1
                    continue

                existing = conn.execute(
                    "SELECT 1 FROM journal WHERE scan_date=? AND ticker=? AND parent=?",
                    [scan_date, ticker, parent],
                ).fetchone()
                if existing:
                    skipped += 1
                    continue

                conn.execute(
                    """INSERT INTO journal
                       (scan_date, ticker, parent, parent_regime, price, rs_score,
                        rank, action_label, est_rev, rvol, confirm, insider_annotation,
                        earnings_days, parent_20d_return, notes)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        scan_date,
                        ticker,
                        parent,
                        row.get("parent_regime") or None,
                        _parse_price(row.get("price", "")),
                        _parse_float(row.get("rs_score", "")),
                        row.get("rank") or None,
                        row.get("action_label") or None,
                        None,  # est_rev — not in CSV
                        None,  # rvol — not in CSV
                        None,  # confirm — not in CSV
                        row.get("insider_annotation") or None,
                        _parse_int(row.get("earnings_days", "")),
                        _parse_float(row.get("parent_20d_return", "")),
                        row.get("notes") or None,
                    ],
                )
                inserted += 1
    finally:
        conn.close()

    print(f"Migrated {inserted} rows, skipped {skipped} duplicates.")

    if inserted > 0 or skipped > 0:
        CSV_PATH.rename(BAK_PATH)
        print(f"Renamed {CSV_PATH.name} → {BAK_PATH.name}")


if __name__ == "__main__":
    main()
