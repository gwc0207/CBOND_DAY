from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DailySchedule:
    buy_twap_col: str
    sell_twap_col: str
