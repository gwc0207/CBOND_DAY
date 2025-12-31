from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostConfig:
    twap_bps: float = 1.0
