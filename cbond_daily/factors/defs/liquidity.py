from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..base import Factor, FactorRegistry


@dataclass
class Liquidity(Factor):
    name: str = "liquidity"

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data["amount"].fillna(0.0)


@FactorRegistry.register("liquidity")
class LiquidityRegistered(Liquidity):
    pass
