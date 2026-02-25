from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pytz

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_daily.core.config import load_config_file
from cbond_daily.run.live_daily import main as run_live_daily


def _parse_time(value: str) -> tuple[int, int]:
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError("time must be HH:MM")
    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError("time must be valid HH:MM")
    return hour, minute


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=str, default=None, help="override schedule time (HH:MM)")
    parser.add_argument("--once", action="store_true", help="run immediately then exit")
    args = parser.parse_args()

    live_cfg = load_config_file("live")
    schedule = live_cfg.get("schedule", {}) or {}
    enabled = bool(schedule.get("enable", False))
    if args.once:
        run_live_daily()
        return
    if not enabled:
        print("[scheduler] disabled in live_config.json5 (schedule.enable=false)")
        return

    tz_name = schedule.get("timezone", "Asia/Shanghai")
    tz = pytz.timezone(tz_name)
    time_str = args.time or schedule.get("time", "17:00")
    hour, minute = _parse_time(time_str)

    print(f"[scheduler] enabled, target={time_str}, tz={tz_name}")
    last_run_date = None
    while True:
        now = datetime.now(tz)
        if now.hour == hour and now.minute == minute:
            if last_run_date != now.date():
                print(f"[scheduler] trigger live_daily at {now:%Y-%m-%d %H:%M:%S %Z}")
                try:
                    run_live_daily()
                except Exception as exc:  # noqa: BLE001
                    print(f"[scheduler] live_daily failed: {exc}")
                last_run_date = now.date()
        time.sleep(30)


if __name__ == "__main__":
    main()
