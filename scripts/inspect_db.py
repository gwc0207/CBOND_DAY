import json
from pathlib import Path
import pyodbc

CONFIG_PATH = Path.home() / ".cbond_daily" / "mssql.json"


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"missing config: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _build_conn_str(cfg: dict) -> str:
    driver = cfg.get("driver", "ODBC Driver 18 for SQL Server")
    database = cfg.get("database") or ""
    db_part = f"DATABASE={database};" if database else ""
    return (
        f"DRIVER={{{driver}}};"
        f"SERVER={cfg['server']};"
        + db_part
        + f"UID={cfg['username']};"
        + f"PWD={cfg['password']};"
        + "Encrypt=yes;"
        + "TrustServerCertificate=yes;"
    )


def _connect():
    cfg = _load_config()
    return pyodbc.connect(_build_conn_str(cfg), timeout=10)


def main() -> None:
    conn = _connect()
    cur = conn.cursor()

    print("== Databases ==")
    cur.execute("SELECT name FROM sys.databases ORDER BY name")
    for row in cur.fetchall():
        print(row.name)

    print("== Tables ==")
    cur.execute(
        """
        SELECT TABLE_SCHEMA, TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE='BASE TABLE'
        ORDER BY TABLE_SCHEMA, TABLE_NAME
        """
    )
    rows = cur.fetchall()
    for r in rows:
        print(f"{r.TABLE_SCHEMA}.{r.TABLE_NAME}")

    sample_tables = [
        "market_cbond.daily_price",
        "market_cbond.daily_twap",
        "market_cbond.daily_vwap",
        "market_cbond.daily_deriv",
        "market_cbond.daily_base",
        "market_cbond.daily_rating",
        "metadata.cbond_info",
        "metadata.trading_calendar",
    ]
    available = {f"{r.TABLE_SCHEMA}.{r.TABLE_NAME}" for r in rows}
    for sample_table in sample_tables:
        if sample_table not in available:
            continue
        schema, table = sample_table.split(".")
        print(f"\n== Sample: {schema}.{table} ==")
        cur.execute(f"SELECT TOP 3 * FROM {schema}.{table}")
        cols = [c[0] for c in cur.description]
        print("columns:", cols)
        for row in cur.fetchall():
            print(row)

    conn.close()


if __name__ == "__main__":
    main()
