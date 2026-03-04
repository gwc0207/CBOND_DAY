from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
import warnings

import pandas as pd

PG_CONFIG_PATH = Path.home() / ".cbond_daily" / "pgsql.json"
MSSQL_CONFIG_PATH = Path.home() / ".cbond_daily" / "mssql.json"

DATE_COLUMNS = {
    "market_cbond.daily_price": "trade_date",
    "market_cbond.daily_twap": "trade_date",
    "market_cbond.daily_vwap": "trade_date",
    "market_cbond.daily_deriv": "trade_date",
    "market_cbond.daily_base": "trade_date",
    "market_cbond.daily_rating": "trade_date",
    "market_stock.daily_price": "trade_date",
    "market_stock.daily_twap": "trade_date",
    "market_stock.daily_vwap": "trade_date",
    "market_index.daily_price": "trade_date",
}


def _normalize_backend_name(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if text in {"postgres", "postgresql", "pgsql", "pg"}:
        return "postgres"
    if text in {"mssql", "sqlserver", "sql_server", "sql-server"}:
        return "mssql"
    return text


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_db_config() -> dict:
    candidates = [PG_CONFIG_PATH, MSSQL_CONFIG_PATH]
    for path in candidates:
        if not path.exists():
            continue
        cfg = _load_json(path)
        backend = _normalize_backend_name(cfg.get("backend"))
        if not backend:
            backend = "postgres" if path == PG_CONFIG_PATH else "mssql"
        if backend not in {"postgres", "mssql"}:
            raise ValueError(f"unsupported backend in {path}: {cfg.get('backend')}")
        cfg["_backend"] = backend
        cfg["_config_path"] = str(path)
        return cfg
    raise FileNotFoundError(f"missing config: {PG_CONFIG_PATH} or {MSSQL_CONFIG_PATH}")


def get_db_backend() -> str:
    return str(load_db_config().get("_backend", "mssql"))


def _normalize_table_name(table: str, *, backend: str, database: str | None = None) -> str:
    # Keep old 3-part table names usable on PostgreSQL by dropping db qualifier.
    if backend == "postgres" and table.count(".") == 2:
        head, schema, name = table.split(".", 2)
        if database and head != database:
            # PostgreSQL does not support cross-database query in one session.
            return f"{schema}.{name}"
        return f"{schema}.{name}"
    return table


def normalize_table_name(table: str) -> str:
    cfg = load_db_config()
    return _normalize_table_name(
        table,
        backend=str(cfg["_backend"]),
        database=cfg.get("database") or cfg.get("dbname"),
    )


def _default_pg_database_for_schema(schema: str, cfg: dict) -> str | None:
    mapping = cfg.get("database_map")
    if isinstance(mapping, dict):
        val = mapping.get(schema)
        if val:
            return str(val)
    return str(cfg.get("database") or cfg.get("dbname") or "") or None


def _resolve_table_target(table: str, cfg: dict) -> tuple[str | None, str]:
    backend = str(cfg.get("_backend", "mssql"))
    if backend != "postgres":
        return None, table
    if table.count(".") == 2:
        db, schema, name = table.split(".", 2)
        return str(db), f"{schema}.{name}"
    if table.count(".") == 1:
        schema, _ = table.split(".", 1)
        return _default_pg_database_for_schema(schema, cfg), table
    return _default_pg_database_for_schema("public", cfg), table


def resolve_table_target(table: str) -> tuple[str | None, str]:
    cfg = load_db_config()
    return _resolve_table_target(table, cfg)


def _build_mssql_conn_str(cfg: dict) -> str:
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


def _parse_host_port(cfg: dict) -> tuple[str, int]:
    host = str(cfg.get("host") or "").strip()
    port = int(cfg.get("port", 5432) or 5432)
    server = str(cfg.get("server") or "").strip()
    if not host and server:
        if "," in server:
            host_part, port_part = server.split(",", 1)
            host = host_part.strip()
            if port_part.strip().isdigit():
                port = int(port_part.strip())
        elif ":" in server:
            host_part, port_part = server.rsplit(":", 1)
            host = host_part.strip()
            if port_part.strip().isdigit():
                port = int(port_part.strip())
        else:
            host = server
    if not host:
        raise KeyError("missing host/server in pgsql config")
    return host, port


def connect(*, database: str | None = None) -> Any:
    cfg = load_db_config()
    backend = str(cfg.get("_backend", "mssql"))
    if backend == "postgres":
        import psycopg2

        host, port = _parse_host_port(cfg)
        dbname = database or cfg.get("database") or cfg.get("dbname")
        if not dbname:
            raise KeyError("missing database/dbname in pgsql config")
        user = cfg.get("username") or cfg.get("user")
        if not user:
            raise KeyError("missing username/user in pgsql config")
        password = cfg.get("password")
        if password is None:
            raise KeyError("missing password in pgsql config")
        kwargs = {
            "host": host,
            "port": int(port),
            "dbname": str(dbname),
            "user": str(user),
            "password": str(password),
            "connect_timeout": int(cfg.get("connect_timeout", 10)),
        }
        sslmode = cfg.get("sslmode")
        if sslmode:
            kwargs["sslmode"] = str(sslmode)
        return psycopg2.connect(**kwargs)

    import pyodbc

    return pyodbc.connect(_build_mssql_conn_str(cfg), timeout=10)


def fetch_table(
    table: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    cfg = load_db_config()
    backend = str(cfg.get("_backend", "mssql"))
    db_override, resolved_table = _resolve_table_target(table, cfg)
    table_name = _normalize_table_name(
        resolved_table,
        backend=backend,
        database=db_override or cfg.get("database") or cfg.get("dbname"),
    )
    marker = "%s" if backend == "postgres" else "?"
    date_col = DATE_COLUMNS.get(table)
    if start and end and date_col:
        sql = f"SELECT * FROM {table_name} WHERE {date_col} >= {marker} AND {date_col} <= {marker}"
        params = [start, end]
    else:
        sql = f"SELECT * FROM {table_name}"
        params = []
    with connect(database=db_override) as conn:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="pandas only supports SQLAlchemy connectable",
                category=UserWarning,
            )
            return pd.read_sql(sql, conn, params=params)
