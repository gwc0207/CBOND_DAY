from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_daily.core.config import load_config_file

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


def _parse_time(value: str) -> tuple[int, int]:
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError("time must be HH:MM")
    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError("time must be valid HH:MM")
    return hour, minute


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


def main() -> None:
    paths_cfg = load_config_file("paths")
    results_root = Path(paths_cfg["results"])
    sched_dir = results_root / "live" / "scheduler"
    logs_dir = sched_dir / "logs"
    state_path = sched_dir / "state.json"
    pid_path = sched_dir / "pid.json"
    logs_dir.mkdir(parents=True, exist_ok=True)

    old_pid = int(_read_json(pid_path).get("pid", 0) or 0)
    if old_pid and old_pid != os.getpid() and _is_pid_alive(old_pid):
        _write_json(
            state_path,
            {
                "status": "already_running",
                "pid": old_pid,
                "now": datetime.now().isoformat(timespec="seconds"),
                "heartbeat": datetime.now().isoformat(timespec="seconds"),
            },
        )
        return
    _write_json(
        pid_path,
        {
            "pid": os.getpid(),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    last_run_date = None
    while True:
        live_cfg = load_config_file("live")
        schedule = live_cfg.get("schedule", {}) or {}
        target = live_cfg.get("target")
        if not bool(schedule.get("enable", False)):
            _write_json(
                state_path,
                {
                    "status": "disabled",
                    "target": target,
                    "now": datetime.now().isoformat(timespec="seconds"),
                    "heartbeat": datetime.now().isoformat(timespec="seconds"),
                },
            )
            time.sleep(30)
            continue

        time_str = schedule.get("time", "17:00")
        hour, minute = _parse_time(time_str)
        now = datetime.now()
        today = now.date()

        if last_run_date == today:
            prev = _read_json(state_path)
            _write_json(
                state_path,
                {
                    "status": "idle_after_run",
                    "today": str(today),
                    "scheduled_time": time_str,
                    "target": target,
                    "log_path": prev.get("log_path"),
                    "run_started_at": prev.get("run_started_at"),
                    "run_finished_at": prev.get("run_finished_at"),
                    "last_return_code": prev.get("last_return_code"),
                    "heartbeat": now.isoformat(timespec="seconds"),
                },
            )
            time.sleep(30)
            continue

        if now.hour < hour or (now.hour == hour and now.minute < minute):
            _write_json(
                state_path,
                {
                    "status": "waiting_time",
                    "today": str(today),
                    "scheduled_time": time_str,
                    "target": target,
                    "heartbeat": now.isoformat(timespec="seconds"),
                },
            )
            time.sleep(30)
            continue

        log_path = logs_dir / f"live_{today:%Y%m%d}_{now:%H%M%S}.log"
        cmd = [sys.executable, "-m", "cbond_daily.run.live_daily"]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        _write_json(
            state_path,
            {
                "status": "running_live",
                "today": str(today),
                "scheduled_time": time_str,
                "target": target,
                "cmd": cmd,
                "log_path": str(log_path),
                "run_started_at": datetime.now().isoformat(timespec="seconds"),
            },
        )

        with log_path.open("w", encoding="utf-8") as fp:
            flags = WIN_NO_WINDOW if os.name == "nt" else 0
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=fp,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
            while True:
                rc = proc.poll()
                st = _read_json(state_path)
                _write_json(
                    state_path,
                    {
                        **st,
                        "status": "running_live",
                        "today": str(today),
                        "scheduled_time": time_str,
                        "target": target,
                        "cmd": cmd,
                        "log_path": str(log_path),
                        "run_started_at": st.get(
                            "run_started_at",
                            datetime.now().isoformat(timespec="seconds"),
                        ),
                        "heartbeat": datetime.now().isoformat(timespec="seconds"),
                        "worker_pid": int(proc.pid),
                    },
                )
                if rc is not None:
                    break
                time.sleep(5)

        st = _read_json(state_path)
        done = {
            **st,
            "run_finished_at": datetime.now().isoformat(timespec="seconds"),
            "last_return_code": int(rc if rc is not None else -1),
            "heartbeat": datetime.now().isoformat(timespec="seconds"),
        }
        if rc == 0:
            done["status"] = "success"
            last_run_date = today
        else:
            done["status"] = "failed"
        _write_json(state_path, done)
        time.sleep(10)


if __name__ == "__main__":
    main()
