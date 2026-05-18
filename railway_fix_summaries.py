import duckdb
import os

db_path = os.environ.get("DATABASE_PATH", "data/scanner.duckdb")
conn = duckdb.connect(db_path)

result = conn.execute("""
    DELETE FROM filings_8k
    WHERE impact = 'HIGH'
    AND (summary IS NULL OR summary = '')
""")

print(f"Deleted HIGH impact filings with no summary")
conn.close()
