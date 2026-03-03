from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, render_template, request

from cbond_daily.core.config import load_config_file

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


def _resolve_log_day_root(day: str | None = None) -> Path:
    root = _results_live_root()
    day_tag = day or datetime.now().strftime("%Y%m%d")
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


def _day_tag_to_iso(day: str) -> str:
    return f"{day[:4]}-{day[4:6]}-{day[6:8]}"


def _read_holdings(day: str | None = None) -> list[dict]:
    day_tag = day or datetime.now().strftime("%Y%m%d")
    iso_day = _day_tag_to_iso(day_tag)
    live_day_root = _results_live_root() / iso_day
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


def _today_day_tag() -> str:
    return datetime.now().strftime("%Y%m%d")


def _stop_flag_path(day: str | None = None) -> Path:
    tag = day or _today_day_tag()
    return _results_live_root() / tag / "STOP"


def _append_dashboard_log(action: str, message: str) -> None:
    now = datetime.now()
    tag = now.strftime("%Y%m%d")
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
            return jsonify({"days": [], "current_day": datetime.now().strftime("%Y%m%d")})
        days: list[str] = []
        for item in live_root.iterdir():
            if not item.is_dir():
                continue
            name = item.name
            if len(name) == 8 and name.isdigit() and (item / "logs").exists():
                days.append(name)
        days.sort(reverse=True)
        return jsonify({"days": days, "current_day": datetime.now().strftime("%Y%m%d")})

    @app.get("/api/logs")
    def api_logs():
        day = request.args.get("day", "").strip() or None
        if day and (len(day) != 8 or not day.isdigit()):
            return jsonify({"path": "", "lines": [], "day": day, "error": "invalid day"}), 400
        path, lines = _read_latest_log(day=day)
        return jsonify({"path": path, "lines": lines})

    @app.get("/api/holdings")
    def api_holdings():
        day = request.args.get("day", "").strip() or None
        if day and (len(day) != 8 or not day.isdigit()):
            return jsonify({"rows": [], "day": day, "error": "invalid day"}), 400
        return jsonify({"rows": _read_holdings(day=day)})

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
        rows = _read_holdings(day=_today_day_tag())
        _append_dashboard_log("sync_holdings", f"rows={len(rows)}")
        return jsonify({"ok": True, "count": len(rows)})

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
