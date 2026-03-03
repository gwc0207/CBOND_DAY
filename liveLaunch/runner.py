from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_daily.core.config import load_config_file


def _parse_time(value: str) -> tuple[int, int]:
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError("time must be HH:MM")
    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError("time must be valid HH:MM")
    return hour, minute


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Schedule CBOND_DAY live run at configured time")
    parser.add_argument("--time", type=str, default=None, help="override schedule time (HH:MM)")
    parser.add_argument("--once", action="store_true", help="run immediately once then exit")
    args = parser.parse_args(argv)

    live_cfg = load_config_file("live")
    schedule = live_cfg.get("schedule", {}) or {}
    enabled = bool(schedule.get("enable", False))

    if args.once:
        cmd = [sys.executable, "-m", "cbond_daily.run.live_daily"]
        completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        raise SystemExit(completed.returncode)

    if not enabled:
        print("[scheduler] disabled in live_config.json5 (schedule.enable=false)")
        return

    tz_name = str(schedule.get("timezone", "Asia/Shanghai"))
    tz = pytz.timezone(tz_name)
    time_str = str(args.time or schedule.get("time", "17:00"))
    hour, minute = _parse_time(time_str)

    print(f"[scheduler] enabled, target={time_str}, tz={tz_name}")
    last_run_date = None
    while True:
        now = datetime.now(tz)
        if now.hour == hour and now.minute == minute and last_run_date != now.date():
            print(f"[scheduler] trigger live_daily at {now:%Y-%m-%d %H:%M:%S %Z}")
            cmd = [sys.executable, "-m", "cbond_daily.run.live_daily"]
            rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode
            if rc != 0:
                print(f"[scheduler] live_daily failed: rc={rc}")
            last_run_date = now.date()
        time.sleep(30)


if __name__ == "__main__":
    main()

