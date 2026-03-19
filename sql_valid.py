import re
import os
import duckdb
import sqlglot
import sqlglot.expressions as exp
from schema import COLUMNS

DB_PATH = os.environ.get("DUCKDB_PATH", "medicaid_provider_spending.duckdb")
MAX_ROWS = 100

ALLOWED_TABLES = {"data"}
ALLOWED_COLUMNS = {col.upper() for col in COLUMNS}


def validate_and_run(sql: str) -> str:
    sql = re.sub(r"^```(?:sql)?\s*", "", sql.strip(), flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql).strip()

    statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]

    if len(statements) != 1:
        raise ValueError(f"Expected 1 statement, got {len(statements)}")

    parsed = statements[0]

    if not isinstance(parsed, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise ValueError(f"Only SELECT queries are allowed, got {type(parsed).__name__}")

    cte_names = {cte.alias.lower() for cte in parsed.find_all(exp.CTE)}
    for table in parsed.find_all(exp.Table):
        name = table.name.lower()
        if name and name not in cte_names and name not in ALLOWED_TABLES:
            raise ValueError(f"Table '{name}' is not allowed. Allowed: {sorted(ALLOWED_TABLES)}")

    referenced_cols = {
        col.name.upper()
        for col in parsed.find_all(exp.Column)
        if col.name
    }
    bad_cols = referenced_cols - ALLOWED_COLUMNS
    if bad_cols:
        raise ValueError(f"Unknown column(s): {bad_cols}. Allowed: {sorted(ALLOWED_COLUMNS)}")

    clean_sql = f"SELECT * FROM ({parsed.sql(dialect='duckdb')}) AS _sub LIMIT {MAX_ROWS}"

    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        df = con.execute(clean_sql).df()
    finally:
        con.close()

    if df.empty:
        return "Query returned no results."
    return df.to_csv(index=False)
