from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_daily.core.config import load_config_file, parse_date
from cbond_daily.data.dwd_builder import build_dwd_daily
from cbond_daily.data.extract import DATE_COLUMNS, connect, fetch_table
from cbond_daily.data.io import (
    get_latest_dwd_date,
    get_latest_dws_date,
    get_latest_table_date,
    table_has_data,
    write_table_by_date,
)
from cbond_daily.factors.pipeline import run_factor_pipeline
from cbond_daily.backtest.runner import run_backtest_linear
from cbond_daily.core.naming import build_factor_col


def _parse_end(value: str | date) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.lower() == "today":
        return pd.Timestamp.today().date()
    return parse_date(value)


def _sync_raw_data(
    *,
    ods_root: str,
    start: date,
    end: date,
    full_refresh: bool,
    tables: list[str],
) -> None:
    for table in tables:
        last_date = None if full_refresh else get_latest_table_date(ods_root, table)
        date_based = table in DATE_COLUMNS
        if date_based:
            fetch_start = max(start, last_date + timedelta(days=1)) if last_date else start
            if fetch_start > end:
                continue
            df = fetch_table(table, start=str(fetch_start), end=str(end))
        else:
            if not full_refresh and table_has_data(ods_root, table):
                continue
            df = fetch_table(table)
        if df.empty:
            continue
        write_table_by_date(df, ods_root, table, date_col="trade_date")
        print(f"synced {table}: {len(df)}")


def _build_cleaned_data(
    *,
    ods_root: str,
    dwd_root: str,
    start: date,
    end: date,
    full_refresh: bool,
    primary_table: str,
    merge_tables: list[str],
) -> None:
    if not full_refresh:
        last_date = get_latest_dwd_date(dwd_root)
        if last_date is not None:
            start = max(start, last_date + timedelta(days=1))
        if start > end:
            return
    build_dwd_daily(
        ods_root,
        dwd_root,
        start,
        end,
        primary_table=primary_table,
        merge_tables=merge_tables,
        table_schemas=cleaned_cfg.get("table_schemas"),
    )


def _build_factors(
    *,
    dwd_root: str,
    dws_root: str,
    start: date,
    end: date,
    factor_defs: list[dict],
    overwrite: bool,
    update_only: list[str] | None,
) -> None:
    if not overwrite:
        last_date = get_latest_dws_date(dws_root)
        if last_date is not None:
            start = max(start, last_date + timedelta(days=1))
        if start > end:
            return
    run_factor_pipeline(
        ods_root,
        dwd_root,
        dws_root,
        start,
        end,
        factor_defs,
        update_only=update_only,
        overwrite=overwrite,
    )


def _pick_signal(cfg: dict, name: str | None) -> dict:
    signals = cfg.get("signals", [])
    if not signals:
        raise ValueError("backtest_config.json5 missing signals")
    if name:
        for signal in signals:
            if signal.get("name") == name:
                return signal
        raise ValueError(f"signal not found: {name}")
    return signals[0]


def _build_factor_items(items: list[dict]) -> list[dict]:
    factor_items = []
    for it in items:
        col = build_factor_col(it["name"], it.get("params"))
        factor_items.append({"col": col, "w": float(it.get("w", 0.0))})
    return factor_items


def _write_trades(out_dir: Path, positions: pd.DataFrame, trade_day: date) -> None:
    if positions is None or positions.empty:
        return
    trades = positions.copy()
    if "trade_date" in trades.columns:
        trades["trade_date"] = trade_day
    cols = [c for c in ["trade_date", "code", "weight"] if c in trades.columns]
    trades = trades[cols]
    trades.to_csv(out_dir / "trade_list.csv", index=False)


def _write_trades_to_db(
    *,
    trades: pd.DataFrame,
    trade_day: date,
    table: str,
    mode: str = "replace_date",
) -> None:
    if trades is None or trades.empty:
        return
    if "code" not in trades.columns:
        raise ValueError("trade_list missing code column")

    work = trades.copy()
    if "trade_date" in work.columns:
        work["trade_date"] = trade_day

    parts = work["code"].astype(str).str.split(".", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError("code must be in instrument.exchange format, e.g. 110084.SH")
    work["instrument_code"] = parts[0]
    work["exchange_code"] = parts[1]

    if "weight" not in work.columns:
        work["weight"] = None
    if "factor_value" not in work.columns:
        work["factor_value"] = None
    if "rank" not in work.columns:
        work["rank"] = None

    cols = [
        "instrument_code",
        "exchange_code",
        "trade_date",
        "factor_value",
        "weight",
        "rank",
    ]
    payload = work[cols]

    insert_sql = (
        f"INSERT INTO {table} "
        "(instrument_code, exchange_code, trade_date, factor_value, weight, rank) "
        "VALUES (?, ?, ?, ?, ?, ?)"
    )

    with connect() as conn:
        cursor = conn.cursor()
        if mode == "replace_date":
            cursor.execute(f"DELETE FROM {table} WHERE trade_date = ?", trade_day)
        cursor.fast_executemany = True
        cursor.executemany(insert_sql, payload.values.tolist())
        conn.commit()


def main() -> None:
    paths_cfg = load_config_file("paths")
    raw_cfg = load_config_file("raw_data")
    cleaned_cfg = load_config_file("cleaned_data")
    factors_cfg = load_config_file("factor_batch")
    backtest_cfg = load_config_file("backtest")
    live_cfg = load_config_file("live")

    start = parse_date(live_cfg["start"])
    end = _parse_end(live_cfg.get("end", "today"))
    trade_day = end + timedelta(days=1)
    signal_name = live_cfg.get("signal_name")
    batch_id = live_cfg.get("batch_id", "Live")

    ods_root = paths_cfg["ods_root"]
    dwd_root = paths_cfg["dwd_root"]
    dws_root = paths_cfg["dws_root"]

    _sync_raw_data(
        ods_root=ods_root,
        start=start,
        end=end,
        full_refresh=bool(raw_cfg.get("full_refresh", False)),
        tables=raw_cfg.get("sync_tables", []),
    )
    _build_cleaned_data(
        ods_root=ods_root,
        dwd_root=dwd_root,
        start=start,
        end=end,
        full_refresh=bool(cleaned_cfg.get("full_refresh", False)),
        primary_table=cleaned_cfg["primary_table"],
        merge_tables=cleaned_cfg["merge_tables"],
    )
    _build_factors(
        dwd_root=dwd_root,
        dws_root=dws_root,
        start=start,
        end=end,
        factor_defs=factors_cfg.get("factors", []),
        overwrite=bool(factors_cfg.get("overwrite", False)),
        update_only=factors_cfg.get("update_only"),
    )

    signal = _pick_signal(backtest_cfg, signal_name)
    items = signal.get("items", [])
    if not items:
        raise ValueError("signal missing items")
    bin_select = signal.get("bin_select", [])
    if not bin_select:
        raise ValueError("signal missing bin_select")
    factor_items = _build_factor_items(items)

    logs_root = Path(paths_cfg["results"])
    date_dir = f"{trade_day:%Y-%m-%d}"
    out_dir = logs_root / "live" / date_dir / batch_id / (signal.get("name") or "signal")
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_backtest_linear(
        dwd_root=dwd_root,
        dws_root=dws_root,
        start=end,
        end=end,
        factor_items=factor_items,
        buy_twap_col=backtest_cfg["buy_twap_col"],
        sell_twap_col=backtest_cfg["sell_twap_col"],
        min_count=int(backtest_cfg["min_count"]),
        max_weight=float(backtest_cfg["max_weight"]),
        twap_bps=float(backtest_cfg["twap_bps"]) + float(backtest_cfg.get("fee_bps", 0.0)),
        bin_count=int(backtest_cfg.get("ic_bins", 20)),
        bin_select=[int(x) for x in bin_select],
        normalize=signal.get("normalize", "zscore"),
        weight_source=backtest_cfg.get("weight_source", "manual"),
        regression_cfg=backtest_cfg.get("regression_cfg"),
        bin_source=backtest_cfg.get("bin_source", "manual"),
        bin_top_k=int(backtest_cfg.get("bin_top_k", 2)),
        weights_output_dir=out_dir,
    )

    if result.positions is not None:
        _write_trades(out_dir, result.positions, trade_day)
        if live_cfg.get("db_write", False):
            _write_trades_to_db(
                trades=result.positions,
                trade_day=trade_day,
                table=live_cfg["db_table"],
                mode=live_cfg.get("db_mode", "replace_date"),
            )
    if result.diagnostics is not None:
        result.diagnostics.to_csv(out_dir / "diagnostics.csv", index=False)

    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
