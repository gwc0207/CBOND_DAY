from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..base import Factor, FactorRegistry
from .. import operators as ops


@dataclass
class StockTrendDelta(Factor):
    """
    CSRank(stock_momentum) * CSRank(delta_proxy)

    stock_momentum = stock_close / stock_close.shift(n) - 1
    delta_proxy = conv_value / pure_redemption_value
    """

    name: str = "stock_trend_delta"

    # columns
    date_col: str = "trade_date"
    stock_close_col: str = "stock_close"
    conv_value_col: str = "base_conv_value"
    pure_redemption_col: str = "base_pure_redemption_value"

    # hyper-parameter
    mom_window: int = 20

    def compute(self, data: pd.DataFrame) -> pd.Series:
        for c in (self.date_col, self.stock_close_col, self.conv_value_col, self.pure_redemption_col):
            if c not in data.columns:
                raise KeyError(f"[{self.name}] missing required column: {c}")

        stock_close = pd.to_numeric(
            data[self.stock_close_col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        # 正股动量
        stock_mom = stock_close / stock_close.shift(self.mom_window) - 1.0

        conv_value = pd.to_numeric(
            data[self.conv_value_col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        pure_redemption = pd.to_numeric(
            data[self.pure_redemption_col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        delta_proxy = conv_value.div(pure_redemption.replace(0, np.nan))

        r_mom = ops.cs_rank(data, stock_mom, self.date_col)
        r_delta = ops.cs_rank(data, delta_proxy, self.date_col)

        factor = r_mom * r_delta
        return factor.astype(float)

    def required_lookback(self) -> int:
        return self.mom_window + 1


@FactorRegistry.register("stock_trend_delta")
class StockTrendDeltaRegistered(StockTrendDelta):
    pass
