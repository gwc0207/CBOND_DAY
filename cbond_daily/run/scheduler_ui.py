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
        else:
            types[key] = "str"
    return {"read_only": [], "types": types}


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
    marker = "cbond_daily.run.schedule_live_daemon"
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

    template_dir = Path(__file__).resolve().parent / "scheduler_web" / "templates"
    static_dir = Path(__file__).resolve().parent / "scheduler_web" / "static"
    app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.get("/api/logs")
    def api_logs():
        state = _read_json(state_path)
        state_log = Path(state.get("log_path", "")) if state.get("log_path") else None
        if state_log is None or not state_log.exists():
            return jsonify({"path": "", "lines": []})
        return jsonify({"path": str(state_log), "lines": _tail(state_log, n=220).splitlines()})

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
        return jsonify({"items": _list_daemon_processes(), "ui_pid": os.getpid()})

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
            [sys.executable, "-m", "cbond_daily.run.schedule_live_daemon"],
            cwd=str(PROJECT_ROOT),
            stdout=fp,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        _write_json(
            _PID_PATH,
            {"pid": proc.pid, "started_at": datetime.now().isoformat(timespec="seconds")},
        )
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
        return jsonify({"ok": True, "killed": killed})

    @app.post("/api/restart")
    def api_restart():
        _ = api_stop()
        time.sleep(0.5)
        return api_start()

    @app.post("/api/shutdown")
    def api_shutdown():
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
    app = create_app()
    app.run(host="127.0.0.1", port=5003, debug=False)


if __name__ == "__main__":
    main()
