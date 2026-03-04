from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import date, datetime, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, render_template, request

from cbond_daily.backtest.execution import apply_twap_bps
from cbond_daily.core.config import load_config_file
from cbond_daily.data.io import read_table_range, read_trading_calendar

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

WIN_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_PROCESS_CACHE: dict = {"items": []}
_PROCESS_CACHE_LOCK = threading.Lock()


def _process_cache_loop() -> None:
    while True:
        items = _list_daemon_processes()
        with _PROCESS_CACHE_LOCK:
            _PROCESS_CACHE["items"] = items
        time.sleep(10)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            creationflags=WIN_NO_WINDOW,
        )
        txt = (out.stdout or "").lower()
        return str(pid) in txt and "no tasks are running" not in txt
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _tail(path: Path, n: int = 120) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-n:])


def _results_live_root() -> Path:
    paths_cfg = load_config_file("paths")
    return Path(paths_cfg["results"]) / "live"


def _to_iso_day_tag(day: str | None = None) -> str:
    if day is None:
        return datetime.now().strftime("%Y-%m-%d")
    text = str(day).strip()
    if not text:
        return datetime.now().strftime("%Y-%m-%d")
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        return datetime.strptime(text, "%Y-%m-%d").strftime("%Y-%m-%d")
    if len(text) == 8 and text.isdigit():
        return datetime.strptime(text, "%Y%m%d").strftime("%Y-%m-%d")
    raise ValueError(f"invalid day: {day}")


def _resolve_log_day_root(day: str | None = None) -> Path:
    root = _results_live_root()
    day_tag = _to_iso_day_tag(day)
    return root / day_tag


def _read_latest_log(day: str | None = None) -> tuple[str, list[str]]:
    day_root = _resolve_log_day_root(day)
    log_root = day_root / "logs"
    if not log_root.exists():
        return "", []
    logs = sorted(log_root.glob("live_scheduler_*.log"))
    if not logs:
        return "", []
    latest = logs[-1]
    lines = latest.read_text(encoding="utf-8", errors="ignore").splitlines()[-400:]
    return str(latest), lines


_TWAP_COL_RE = re.compile(r"^twap_(\d{4})_(\d{4})$")


def _parse_hhmm_token(token: str | None) -> dt_time | None:
    if not token:
        return None
    text = str(token).strip()
    if len(text) != 4 or not text.isdigit():
        return None
    hh = int(text[:2])
    mm = int(text[2:])
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None
    return dt_time(hour=hh, minute=mm)


def _parse_twap_start_time(col: str | None) -> dt_time | None:
    if not col:
        return None
    m = _TWAP_COL_RE.match(str(col).strip())
    if not m:
        return None
    return _parse_hhmm_token(m.group(1))


def _load_model_label_cfg(live_cfg: dict) -> dict:
    model_cfg_ref = live_cfg.get("model_config")
    if not model_cfg_ref:
        return {}
    model_cfg_text = str(model_cfg_ref).strip()
    # Prefer explicit file path (e.g. cbond_daily/config/models/linear_combo_default.json5).
    if model_cfg_text.lower().endswith((".json5", ".json", ".yaml", ".yml")):
        path = Path(model_cfg_text)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            try:
                suffix = path.suffix.lower()
                if suffix == ".json5":
                    import json5

                    model_cfg = json5.loads(path.read_text(encoding="utf-8"))
                elif suffix in (".yaml", ".yml"):
                    import yaml

                    model_cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
                else:
                    model_cfg = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(model_cfg, dict):
                    label_cfg = model_cfg.get("label_cfg")
                    return label_cfg if isinstance(label_cfg, dict) else {}
            except Exception:
                return {}
    try:
        model_cfg = load_config_file(model_cfg_text)
    except Exception:
        return {}
    if not isinstance(model_cfg, dict):
        return {}
    label_cfg = model_cfg.get("label_cfg")
    return label_cfg if isinstance(label_cfg, dict) else {}


def _now_in_live_tz(live_cfg: dict) -> datetime:
    schedule = live_cfg.get("schedule", {}) if isinstance(live_cfg, dict) else {}
    tz_name = "Asia/Shanghai"
    if isinstance(schedule, dict):
        tz_name = str(schedule.get("timezone", "Asia/Shanghai"))
    try:
        return datetime.now(ZoneInfo(tz_name))
    except Exception:
        return datetime.now()


def _should_show_holdings(day_tag: str, live_cfg: dict) -> bool:
    now_tz = _now_in_live_tz(live_cfg)
    today_tag = now_tz.strftime("%Y-%m-%d")
    if _to_iso_day_tag(day_tag) != today_tag:
        return True
    label_cfg = _load_model_label_cfg(live_cfg)
    buy_start = _parse_twap_start_time(label_cfg.get("buy_twap_col"))
    sell_start = _parse_twap_start_time(label_cfg.get("sell_twap_col"))
    now_time = now_tz.time()
    if buy_start is not None and now_time < buy_start:
        return False
    if sell_start is not None and now_time >= sell_start:
        return False
    return True


def _read_holdings(day: str | None = None) -> list[dict]:
    day_tag = _to_iso_day_tag(day)
    live_day_root = _results_live_root() / day_tag
    if not live_day_root.exists():
        return []
    files = sorted(live_day_root.glob("**/trade_list.csv"))
    if not files:
        return []
    latest = files[-1]
    df = pd.read_csv(latest)
    if df.empty:
        return []
    if "code" not in df.columns:
        return []
    if "weight" not in df.columns:
        df["weight"] = pd.NA
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "symbol": str(row.get("code", "")),
                "weight": None if pd.isna(row.get("weight")) else float(row.get("weight")),
            }
        )
    return rows


def _read_trade_list(day: date) -> pd.DataFrame:
    live_day_root = _results_live_root() / f"{day:%Y-%m-%d}"
    if not live_day_root.exists():
        return pd.DataFrame()
    files = sorted(live_day_root.glob("**/trade_list.csv"))
    if not files:
        return pd.DataFrame()
    try:
        df = pd.read_csv(files[-1])
    except Exception:
        return pd.DataFrame()
    if df.empty or "code" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    if "weight" not in work.columns:
        work["weight"] = pd.NA
    return work[["code", "weight"]]


def _parse_day_to_date(day: str | None) -> date:
    if not day:
        return datetime.now().date()
    text = str(day).strip()
    if len(text) == 8 and text.isdigit():
        return datetime.strptime(text, "%Y%m%d").date()
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        return datetime.strptime(text, "%Y-%m-%d").date()
    raise ValueError(f"invalid day: {day}")


def _read_twap_daily(ods_root: str | Path, day: date) -> pd.DataFrame:
    df = read_table_range(ods_root, "market_cbond.daily_twap", day, day)
    if df.empty:
        return df
    if "instrument_code" in df.columns and "exchange_code" in df.columns:
        df = df.copy()
        df["code"] = df["instrument_code"].astype(str) + "." + df["exchange_code"].astype(str)
    return df


def _load_open_days(ods_root: str | Path) -> list[date]:
    cal = read_trading_calendar(ods_root)
    if cal.empty or "calendar_date" not in cal.columns:
        return []
    work = cal.copy()
    if "is_open" in work.columns:
        work = work[work["is_open"].astype(bool)]
    days = pd.to_datetime(work["calendar_date"], errors="coerce").dt.date.dropna().unique().tolist()
    days.sort()
    return days


def _calc_sharpe(ret: pd.Series) -> float:
    s = pd.to_numeric(ret, errors="coerce").dropna()
    if s.empty:
        return 0.0
    vol = float(s.std(ddof=0))
    if vol == 0:
        return 0.0
    return float((float(s.mean()) / vol) * math.sqrt(252.0))


def _calc_vol(ret: pd.Series) -> float:
    s = pd.to_numeric(ret, errors="coerce").dropna()
    if s.empty:
        return 0.0
    return float(float(s.std(ddof=0)) * math.sqrt(252.0))


def _normalize_weights(w: pd.Series) -> pd.Series:
    s = pd.to_numeric(w, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(s.sum())
    if total > 0:
        return s / total
    if len(s) == 0:
        return s
    return pd.Series([1.0 / len(s)] * len(s), index=s.index, dtype=float)


def _build_perf_summary(*, ods_root: str | Path, day: str | None, lookback: int | None) -> dict:
    live_cfg = _load_live_cfg()
    label_cfg = _load_model_label_cfg(live_cfg)
    buy_col = str(
        label_cfg.get("buy_twap_col", live_cfg.get("buy_twap_col", "twap_0945_1000"))
    )
    sell_col = str(
        label_cfg.get("sell_twap_col", live_cfg.get("sell_twap_col", "twap_1430_1442"))
    )
    twap_bps = float(label_cfg.get("twap_bps", live_cfg.get("twap_bps", 1.5)))
    fee_bps = float(label_cfg.get("fee_bps", live_cfg.get("fee_bps", 0.7)))
    cost_bps = twap_bps + fee_bps
    min_amount = float(live_cfg.get("min_amount", 0))
    min_volume = float(live_cfg.get("min_volume", 0))
    default_lb = int(live_cfg.get("perf_lookback_days", 20))
    lookback = max(1, int(lookback if lookback is not None else default_lb))

    asof_day = _parse_day_to_date(day)
    open_days = _load_open_days(ods_root)
    if not open_days:
        return {
            "asof_day": f"{asof_day:%Y-%m-%d}",
            "lookback": lookback,
            "count_days": 0,
            "metrics": {},
            "series": [],
        }
    next_day_map = {open_days[i]: open_days[i + 1] for i in range(len(open_days) - 1)}

    live_root = _results_live_root()
    candidates: list[date] = []
    if live_root.exists():
        for item in live_root.iterdir():
            if not item.is_dir():
                continue
            try:
                d = datetime.strptime(item.name, "%Y-%m-%d").date()
            except ValueError:
                continue
            if d <= asof_day:
                candidates.append(d)
    candidates = sorted(candidates)[-lookback:]

    rows: list[dict] = []
    for trade_day in candidates:
        next_day = next_day_map.get(trade_day)
        if next_day is None:
            continue

        picks = _read_trade_list(trade_day)
        if picks.empty:
            continue

        buy_df = _read_twap_daily(ods_root, trade_day)
        sell_df = _read_twap_daily(ods_root, next_day)
        if buy_df.empty or sell_df.empty:
            continue

        if buy_col not in buy_df.columns or sell_col not in sell_df.columns:
            continue

        merged = picks.merge(buy_df[["code", buy_col]], on="code", how="left").merge(
            sell_df[["code", sell_col]], on="code", how="left"
        )
        merged = merged[
            merged[buy_col].notna()
            & merged[sell_col].notna()
            & (merged[buy_col] > 0)
            & (merged[sell_col] > 0)
        ]
        if merged.empty:
            continue

        buy_px = apply_twap_bps(merged[buy_col], cost_bps, side="buy")
        sell_px = apply_twap_bps(merged[sell_col], cost_bps, side="sell")
        strat_ret = (sell_px - buy_px) / buy_px
        w = _normalize_weights(merged["weight"])
        strategy_return = float((strat_ret * w).sum())

        bench = buy_df[["code", buy_col]].merge(sell_df[["code", sell_col]], on="code", how="inner")
        bench = bench[
            bench[buy_col].notna()
            & bench[sell_col].notna()
            & (bench[buy_col] > 0)
            & (bench[sell_col] > 0)
        ]
        if min_amount > 0 and "amount" in buy_df.columns:
            bench = bench.merge(buy_df[["code", "amount"]], on="code", how="left")
            bench = bench[bench["amount"].fillna(0) >= min_amount]
        if min_volume > 0 and "volume" in buy_df.columns:
            if "volume" not in bench.columns:
                bench = bench.merge(buy_df[["code", "volume"]], on="code", how="left")
            bench = bench[bench["volume"].fillna(0) >= min_volume]

        if bench.empty:
            benchmark_return = float("nan")
        else:
            bench_buy = apply_twap_bps(bench[buy_col], cost_bps, side="buy")
            bench_sell = apply_twap_bps(bench[sell_col], cost_bps, side="sell")
            benchmark_return = float(((bench_sell - bench_buy) / bench_buy).mean())

        rows.append(
            {
                "trade_date": trade_day,
                "next_day": next_day,
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "count": int(len(merged)),
            }
        )

    if not rows:
        return {
            "asof_day": f"{asof_day:%Y-%m-%d}",
            "lookback": lookback,
            "count_days": 0,
            "metrics": {},
            "series": [],
        }

    df = pd.DataFrame(rows).sort_values("trade_date")
    df["strategy_nav"] = (1.0 + df["strategy_return"].fillna(0.0)).cumprod()
    df["benchmark_nav"] = (1.0 + df["benchmark_return"].fillna(0.0)).cumprod()

    metrics = {
        "sharpe": _calc_sharpe(df["strategy_return"]),
        "volatility": _calc_vol(df["strategy_return"]),
        "benchmark_sharpe": _calc_sharpe(df["benchmark_return"]),
        "benchmark_volatility": _calc_vol(df["benchmark_return"]),
    }

    series = [
        {
            "trade_date": f"{row.trade_date:%Y-%m-%d}",
            "next_day": f"{row.next_day:%Y-%m-%d}",
            "strategy_return": float(row.strategy_return),
            "benchmark_return": float(row.benchmark_return)
            if pd.notna(row.benchmark_return)
            else None,
            "strategy_nav": float(row.strategy_nav),
            "benchmark_nav": float(row.benchmark_nav),
            "count": int(row.count),
        }
        for row in df.itertuples()
    ]
    return {
        "asof_day": f"{asof_day:%Y-%m-%d}",
        "lookback": lookback,
        "count_days": int(len(series)),
        "metrics": metrics,
        "series": series,
    }


def _today_day_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _stop_flag_path(day: str | None = None) -> Path:
    tag = _to_iso_day_tag(day)
    return _results_live_root() / tag / "STOP"


def _append_dashboard_log(action: str, message: str) -> None:
    now = datetime.now()
    tag = now.strftime("%Y-%m-%d")
    log_dir = _results_live_root() / tag / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"live_scheduler_{tag}.log"
    line = f"{now:%Y-%m-%d %H:%M:%S} [dashboard] {action} {message}\n"
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(line)


def _live_cfg_path() -> Path:
    return PROJECT_ROOT / "cbond_daily" / "config" / "live_config.json5"


def _load_live_cfg() -> dict:
    return load_config_file("live")


def _save_live_cfg(cfg: dict) -> None:
    _live_cfg_path().write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _config_meta(cfg: dict) -> dict:
    types = {}
    for key, val in cfg.items():
        if isinstance(val, bool):
            types[key] = "bool"
        elif isinstance(val, int):
            types[key] = "int"
        elif isinstance(val, float):
            types[key] = "float"
        elif isinstance(val, dict):
            types[key] = "object"
        elif isinstance(val, list):
            types[key] = "array"
        else:
            types[key] = "str"
    return {"read_only": [], "types": types}


def _set_by_path(target: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    cur = target
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _expand_dotted_payload(payload: dict) -> dict:
    expanded: dict = {}
    for key, val in payload.items():
        if "." in key:
            _set_by_path(expanded, key, val)
        else:
            expanded[key] = val
    return expanded


def _deep_merge_dict(base: dict, updates: dict) -> dict:
    out = dict(base)
    for key, val in updates.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], val)
        else:
            out[key] = val
    return out


def _heartbeat_info(state: dict) -> dict:
    hb_text = state.get("heartbeat")
    hb_age = None
    hb_stale = True
    if hb_text:
        try:
            hb_dt = datetime.fromisoformat(hb_text)
            hb_age = max(0, int((datetime.now() - hb_dt).total_seconds()))
            hb_stale = hb_age > 120
        except Exception:
            hb_age = None
            hb_stale = True
    return {"at": hb_text, "age_seconds": hb_age, "stale": hb_stale}


def _list_daemon_processes() -> list[dict]:
    items: list[dict] = []
    marker = "liveLaunch.scheduler"
    if psutil is not None:
        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                name = str(proc.info.get("name") or "").lower()
                if "python" not in name:
                    continue
                cmd = " ".join(proc.info.get("cmdline") or [])
                if marker not in cmd:
                    continue
                started = ""
                ct = proc.info.get("create_time")
                if ct:
                    started = datetime.fromtimestamp(float(ct)).strftime("%Y-%m-%d %H:%M:%S")
                items.append({"pid": int(proc.info["pid"]), "cmd": cmd, "start": started})
            except Exception:
                continue
        return sorted(items, key=lambda x: x["pid"])

    pid = int(_read_json(_PID_PATH).get("pid", 0) or 0)
    if _is_pid_alive(pid):
        items.append({"pid": pid, "cmd": marker, "start": ""})
    return items


def _kill_pid(pid: int) -> None:
    if not _is_pid_alive(pid):
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            creationflags=WIN_NO_WINDOW,
        )
    else:
        os.kill(pid, signal.SIGTERM)


def create_app() -> Flask:
    paths_cfg = load_config_file("paths")
    results_root = Path(paths_cfg["results"])
    sched_dir = results_root / "live" / "scheduler"
    logs_dir = sched_dir / "logs"
    global _PID_PATH
    _PID_PATH = sched_dir / "pid.json"
    state_path = sched_dir / "state.json"
    ui_log = sched_dir / "scheduler_ui.log"

    template_dir = Path(__file__).resolve().parent / "templates"
    static_dir = Path(__file__).resolve().parent / "static"
    app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.get("/api/log_days")
    def api_log_days():
        live_root = _results_live_root()
        if not live_root.exists():
            return jsonify({"days": [], "current_day": _today_day_tag()})
        days: list[str] = []
        for item in live_root.iterdir():
            if not item.is_dir():
                continue
            if not (item / "logs").exists():
                continue
            try:
                datetime.strptime(item.name, "%Y-%m-%d")
            except ValueError:
                continue
            days.append(item.name)
        days.sort(reverse=True)
        return jsonify({"days": days, "current_day": _today_day_tag()})

    @app.get("/api/logs")
    def api_logs():
        raw_day = request.args.get("day", "").strip() or None
        try:
            day = _to_iso_day_tag(raw_day) if raw_day else None
        except ValueError:
            day = raw_day
            return jsonify({"path": "", "lines": [], "day": day, "error": "invalid day"}), 400
        path, lines = _read_latest_log(day=day)
        return jsonify({"path": path, "lines": lines})

    @app.get("/api/holdings")
    def api_holdings():
        raw_day = request.args.get("day", "").strip() or None
        try:
            day_tag = _to_iso_day_tag(raw_day) if raw_day else _today_day_tag()
        except ValueError:
            day_tag = raw_day
            return jsonify({"rows": [], "day": day_tag, "error": "invalid day"}), 400
        live_cfg = _load_live_cfg()
        if not _should_show_holdings(day_tag, live_cfg):
            return jsonify({"rows": [], "day": day_tag})
        return jsonify({"rows": _read_holdings(day=day_tag), "day": day_tag})

    @app.get("/api/perf_summary")
    def api_perf_summary():
        day = request.args.get("day", "").strip() or None
        lookback_raw = request.args.get("lookback", "").strip()
        lookback = None
        if lookback_raw:
            try:
                lookback = int(lookback_raw)
            except Exception:
                return jsonify({"error": f"invalid lookback: {lookback_raw}"}), 400
        try:
            payload = _build_perf_summary(
                ods_root=paths_cfg["ods_root"],
                day=day,
                lookback=lookback,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(payload)

    @app.get("/api/state")
    def api_state():
        pid_info = _read_json(_PID_PATH)
        state = _read_json(state_path)
        pid = int(pid_info.get("pid", 0) or 0)
        running = _is_pid_alive(pid)
        return jsonify(
            {
                "running": running,
                "pid": pid if running else None,
                "pid_info": pid_info,
                "state": state,
                "now": datetime.now().isoformat(timespec="seconds"),
                "heartbeat": _heartbeat_info(state),
            }
        )

    @app.get("/api/processes")
    def api_processes():
        with _PROCESS_CACHE_LOCK:
            items = list(_PROCESS_CACHE.get("items", []))
        return jsonify({"items": items, "ui_pid": os.getpid()})

    @app.get("/api/config")
    def api_config_get():
        cfg = _load_live_cfg()
        meta = _config_meta(cfg)
        return jsonify({"config": cfg, "meta": meta})

    @app.post("/api/config")
    def api_config_update():
        cfg = _load_live_cfg()
        meta = _config_meta(cfg)
        payload = request.get_json(force=True) or {}
        payload = _expand_dotted_payload(payload)
        updated = {}
        for key, val in payload.items():
            if key not in cfg:
                cfg[key] = val
                updated[key] = val
                continue
            t = meta["types"].get(key, "str")
            try:
                if t == "bool":
                    cfg[key] = bool(val)
                elif t == "int":
                    cfg[key] = int(val)
                elif t == "float":
                    cfg[key] = float(val)
                elif t == "object":
                    if isinstance(val, dict):
                        cur = cfg.get(key, {})
                        if not isinstance(cur, dict):
                            cur = {}
                        cfg[key] = _deep_merge_dict(cur, val)
                    else:
                        cfg[key] = val
                elif t == "array":
                    cfg[key] = val if isinstance(val, list) else cfg.get(key, [])
                else:
                    cfg[key] = str(val)
                updated[key] = cfg[key]
            except Exception:
                cfg[key] = val
                updated[key] = val
        _save_live_cfg(cfg)
        return jsonify({"ok": True, "updated": updated})

    @app.post("/api/start")
    def api_start():
        existing = _list_daemon_processes()
        if existing:
            pid = int(existing[0].get("pid", 0) or 0)
            _write_json(
                _PID_PATH,
                {"pid": pid, "started_at": datetime.now().isoformat(timespec="seconds")},
            )
            return jsonify({"ok": True, "message": "already running", "pid": pid})

        logs_dir.mkdir(parents=True, exist_ok=True)
        fp = ui_log.open("a", encoding="utf-8")
        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
                subprocess, "DETACHED_PROCESS", 0
            )
        proc = subprocess.Popen(
            [sys.executable, "-m", "liveLaunch.scheduler"],
            cwd=str(PROJECT_ROOT),
            stdout=fp,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        _write_json(
            _PID_PATH,
            {"pid": proc.pid, "started_at": datetime.now().isoformat(timespec="seconds")},
        )
        _append_dashboard_log("start", f"scheduler pid={proc.pid}")
        return jsonify({"ok": True, "message": "started", "pid": proc.pid})

    @app.post("/api/stop")
    def api_stop():
        killed = []
        for item in _list_daemon_processes():
            pid = int(item.get("pid", 0) or 0)
            if pid:
                _kill_pid(pid)
                killed.append(pid)

        pid = int(_read_json(_PID_PATH).get("pid", 0) or 0)
        if pid and pid not in killed:
            _kill_pid(pid)
            killed.append(pid)
        _append_dashboard_log("stop", f"killed={killed}")
        return jsonify({"ok": True, "killed": killed})

    @app.post("/api/restart")
    def api_restart():
        _ = api_stop()
        time.sleep(0.5)
        return api_start()

    @app.post("/api/start_open")
    def api_start_open():
        stop_flag = _stop_flag_path()
        if stop_flag.exists():
            stop_flag.unlink()
            _append_dashboard_log("start_open", "removed STOP flag")
        # 贴近 WC：start_open 直接重启调度器，确保当次配置生效。
        _ = api_stop()
        time.sleep(0.5)
        res = api_start()
        _append_dashboard_log("start_open", "scheduler started")
        return res

    @app.post("/api/emergency_stop")
    def api_emergency_stop():
        stop_flag = _stop_flag_path()
        stop_flag.parent.mkdir(parents=True, exist_ok=True)
        stop_flag.write_text("STOP", encoding="utf-8")
        _append_dashboard_log("emergency_stop", f"set STOP at {stop_flag}")
        return jsonify({"ok": True, "stop_flag": str(stop_flag)})

    @app.post("/api/restart_scheduler")
    def api_restart_scheduler():
        stop_flag = _stop_flag_path()
        if stop_flag.exists():
            stop_flag.unlink()
            _append_dashboard_log("restart_scheduler", "removed STOP flag")
        _ = api_stop()
        time.sleep(0.5)
        res = api_start()
        _append_dashboard_log("restart_scheduler", "scheduler restarted")
        return res

    @app.post("/api/sync_holdings")
    def api_sync_holdings():
        day_tag = _today_day_tag()
        live_cfg = _load_live_cfg()
        if _should_show_holdings(day_tag, live_cfg):
            rows = _read_holdings(day=day_tag)
        else:
            rows = []
        _append_dashboard_log("sync_holdings", f"rows={len(rows)}")
        return jsonify({"ok": True, "count": len(rows), "day": day_tag})

    @app.post("/api/shutdown")
    def api_shutdown():
        _append_dashboard_log("shutdown", "ui shutdown requested")
        func = request.environ.get("werkzeug.server.shutdown")
        if func is None:

            def _exit() -> None:
                time.sleep(0.3)
                os._exit(0)

            threading.Thread(target=_exit, daemon=True).start()
            return jsonify({"ok": True, "mode": "force-exit"})
        func()
        return jsonify({"ok": True, "mode": "werkzeug"})

    return app


def main() -> None:
    threading.Thread(target=_process_cache_loop, daemon=True).start()
    app = create_app()
    app.run(host="127.0.0.1", port=5003, debug=False)


if __name__ == "__main__":
    main()
