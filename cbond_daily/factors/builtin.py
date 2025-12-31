from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import Factor, FactorRegistry


@dataclass
class IntradayMomentum(Factor):
    name: str = "intraday_momentum"

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data["close_price"] / data["prev_close_price"] - 1.0


@dataclass
class Liquidity(Factor):
    name: str = "liquidity"

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data["amount"].fillna(0.0)


@FactorRegistry.register("intraday_momentum")
class IntradayMomentumRegistered(IntradayMomentum):
    pass


@FactorRegistry.register("liquidity")
class LiquidityRegistered(Liquidity):
    pass
