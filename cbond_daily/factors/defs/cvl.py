from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..base import Factor, FactorRegistry
from .. import operators as ops


@dataclass
class CVLCarryValueLiquidity(Factor):
    """
    CVL: Carry-Value-Liquidity (cross-sectional rank composite)

    factor = w_prem * rank(-prem) + w_ytm * rank(ytm) + w_liq * rank(log1p(amount))

    - prem: deriv_bond_prem_ratio (越低越好)
    - ytm : deriv_ytm (越高越好)
    - liq : amount (越大越好，log1p 压缩)
    - rank: 按 trade_date 截面 pct rank，抗极值，天然可比
    """

    name: str = "cvl"

    date_col: str = "trade_date"
    prem_col: str = "deriv_bond_prem_ratio"
    ytm_col: str = "deriv_ytm"
    amount_col: str = "amount"

    w_prem: float = 1.0
    w_ytm: float = 1.0
    w_liq: float = 0.6

    def required_lookback(self) -> int:
        return 1

    def compute(self, data: pd.DataFrame) -> pd.Series:
        need = [self.date_col, self.prem_col, self.ytm_col, self.amount_col]
        miss = [c for c in need if c not in data.columns]
        if miss:
            raise KeyError(f"missing columns: {miss}")

        df = data[[self.date_col, self.prem_col, self.ytm_col, self.amount_col]].copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col]).dt.date

        prem = (
            pd.to_numeric(df[self.prem_col], errors="coerce")
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
        )
        ytm = (
            pd.to_numeric(df[self.ytm_col], errors="coerce")
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
        )
        amt = (
            pd.to_numeric(df[self.amount_col], errors="coerce")
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
        )

        prem_rank = ops.cs_rank(df, (-prem), self.date_col)
        ytm_rank = ops.cs_rank(df, ytm, self.date_col)
        liq_rank = ops.cs_rank(df, np.log1p(amt.clip(lower=0.0)), self.date_col)

        wsum = float(abs(self.w_prem) + abs(self.w_ytm) + abs(self.w_liq))
        if wsum <= 0:
            return pd.Series([pd.NA] * len(data), index=data.index)

        out = (self.w_prem * prem_rank + self.w_ytm * ytm_rank + self.w_liq * liq_rank) / wsum
        return out


@FactorRegistry.register("cvl")
class CVLCarryValueLiquidityRegistered(CVLCarryValueLiquidity):
    pass
