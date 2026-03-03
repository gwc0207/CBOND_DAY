from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_daily.core.config import load_config_file
from cbond_daily.data.io import read_trading_calendar

WIN_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
IDLE_POLL_SECONDS = 300
RUNNING_POLL_SECONDS = 5


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


def _load_open_days(ods_root: str) -> list[date]:
    cal = read_trading_calendar(ods_root)
    if cal.empty or "calendar_date" not in cal.columns:
        return []
    work = cal.copy()
    work["calendar_date"] = pd.to_datetime(work["calendar_date"], errors="coerce").dt.date
    if "is_open" in work.columns:
        work = work[work["is_open"].astype(bool)]
    days = sorted(d for d in work["calendar_date"].dropna().unique().tolist())
    return days


def _next_trading_day(open_days: list[date], run_day: date) -> date:
    for d in open_days:
        if d > run_day:
            return d
    return run_day + timedelta(days=1)


def _day_tag(now_tz: datetime) -> str:
    return now_tz.strftime("%Y%m%d")


def _append_log(results_root: Path, now_tz: datetime, msg: str) -> None:
    day_dir = results_root / "live" / _day_tag(now_tz) / "logs"
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"live_scheduler_{_day_tag(now_tz)}.log"
    line = f"{now_tz:%Y-%m-%d %H:%M:%S %Z} {msg}\n"
    with path.open("a", encoding="utf-8") as fp:
        fp.write(line)


def _stop_flag_path(results_root: Path, now_tz: datetime) -> Path:
    day_dir = results_root / "live" / _day_tag(now_tz)
    return day_dir / "STOP"


def main() -> None:
    paths_cfg = load_config_file("paths")
    ods_root = str(paths_cfg["ods_root"])
    results_root = Path(paths_cfg["results"])
    sched_dir = results_root / "live" / "scheduler"
    state_path = sched_dir / "state.json"
    pid_path = sched_dir / "pid.json"
    sched_dir.mkdir(parents=True, exist_ok=True)

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
    _write_json(state_path, {"status": "booting", "heartbeat": datetime.now().isoformat(timespec="seconds")})

    last_status = ""
    while True:
        live_cfg = load_config_file("live")
        schedule = live_cfg.get("schedule", {}) or {}
        time_str = str(schedule.get("time", "17:00"))
        hour, minute = _parse_time(time_str)
        tz_name = str(schedule.get("timezone", "Asia/Shanghai"))
        tz = pytz.timezone(tz_name)
        now_tz = datetime.now(tz)
        now_local = datetime.now()
        today = now_tz.date()
        open_days = _load_open_days(ods_root)
        target = _next_trading_day(open_days, today)

        st = _read_json(state_path)
        last_target_run = st.get("last_target_run")
        stop_flag = _stop_flag_path(results_root, now_tz)

        def write_status(status: str, extra: dict | None = None, *, force_log: bool = False) -> None:
            payload = {
                **st,
                "status": status,
                "today": str(today),
                "scheduled_time": time_str,
                "timezone": tz_name,
                "target": str(target),
                "heartbeat": now_local.isoformat(timespec="seconds"),
            }
            if extra:
                payload.update(extra)
            _write_json(state_path, payload)
            nonlocal last_status
            if force_log or status != last_status:
                _append_log(
                    results_root,
                    now_tz,
                    f"[heartbeat] status={status} today={today} target={target}",
                )
                last_status = status
            else:
                _append_log(
                    results_root,
                    now_tz,
                    f"[heartbeat] status={status} today={today} target={target}",
                )

        if not bool(schedule.get("enable", False)):
            write_status("disabled")
            time.sleep(IDLE_POLL_SECONDS)
            continue

        if stop_flag.exists():
            write_status("stopped_by_flag")
            time.sleep(IDLE_POLL_SECONDS)
            continue

        if last_target_run == str(target):
            write_status("idle_after_run")
            time.sleep(IDLE_POLL_SECONDS)
            continue

        if now_tz.hour < hour or (now_tz.hour == hour and now_tz.minute < minute):
            write_status("waiting_time")
            time.sleep(IDLE_POLL_SECONDS)
            continue

        cmd = [
            sys.executable,
            "-m",
            "cbond_daily.run.live_daily",
            "--target",
            str(target),
            "--start",
            str(live_cfg.get("start", str(target))),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        run_started_at = now_local.isoformat(timespec="seconds")
        day_logs_dir = results_root / "live" / _day_tag(now_tz) / "logs"
        day_logs_dir.mkdir(parents=True, exist_ok=True)
        day_log_path = day_logs_dir / f"live_scheduler_{_day_tag(now_tz)}.log"

        write_status(
            "running_live",
            {
                "cmd": cmd,
                "run_started_at": run_started_at,
            },
            force_log=True,
        )
        _append_log(results_root, now_tz, f"[run_start] target={target} cmd={' '.join(cmd)}")

        flags = WIN_NO_WINDOW if os.name == "nt" else 0
        with day_log_path.open("a", encoding="utf-8") as log_fp:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log_fp,
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
                        "timezone": tz_name,
                        "target": str(target),
                        "cmd": cmd,
                        "run_started_at": st.get("run_started_at", run_started_at),
                        "heartbeat": datetime.now().isoformat(timespec="seconds"),
                        "worker_pid": int(proc.pid),
                    },
                )
                _append_log(
                    results_root,
                    datetime.now(tz),
                    f"[heartbeat] status=running_live today={today} target={target} worker_pid={proc.pid}",
                )
                if rc is not None:
                    break
                time.sleep(RUNNING_POLL_SECONDS)

        st = _read_json(state_path)
        done = {
            **st,
            "run_finished_at": datetime.now().isoformat(timespec="seconds"),
            "last_return_code": int(rc if rc is not None else -1),
            "heartbeat": datetime.now().isoformat(timespec="seconds"),
        }
        if rc == 0:
            done["status"] = "success"
            done["last_target_run"] = str(target)
        else:
            done["status"] = "failed"
        _write_json(state_path, done)
        _append_log(
            results_root,
            datetime.now(tz),
            f"[run_end] target={target} status={done['status']} return_code={done['last_return_code']}",
        )
        time.sleep(10)


if __name__ == "__main__":
    main()
