from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..base import Factor, FactorRegistry
from .. import operators as ops


@dataclass
class CarryCheapness(Factor):
    """
    CSRank(base_ytm) - CSRank(base_bond_prem_ratio) [- alpha * CSRank(base_duration)]

    直觉：
    - ytm 越高越好（carry）
    - 转股溢价越低越好（便宜）
    - 可选：duration 越低越好（利率敏感度惩罚）
    """

    name: str = "carry_cheapness"

    # column names (match cleaned data)
    date_col: str = "trade_date"
    ytm_col: str = "base_ytm"
    prem_col: str = "base_bond_prem_ratio"

    # optional duration penalty
    use_duration: bool = False
    duration_col: str = "base_duration"
    alpha: float = 0.5

    def compute(self, data: pd.DataFrame) -> pd.Series:
        # basic checks
        for c in (self.date_col, self.ytm_col, self.prem_col):
            if c not in data.columns:
                raise KeyError(f"[{self.name}] missing required column: {c}")

        ytm = pd.to_numeric(data[self.ytm_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        prem = pd.to_numeric(data[self.prem_col], errors="coerce").replace([np.inf, -np.inf], np.nan)

        r_ytm = ops.cs_rank(data, ytm, self.date_col)
        r_prem = ops.cs_rank(data, prem, self.date_col)

        factor = r_ytm - r_prem

        if self.use_duration:
            if self.duration_col not in data.columns:
                raise KeyError(f"[{self.name}] use_duration=True but missing column: {self.duration_col}")
            dur = pd.to_numeric(data[self.duration_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            r_dur = ops.cs_rank(data, dur, self.date_col)
            factor = factor - float(self.alpha) * r_dur

        return factor.astype(float)

    def required_lookback(self) -> int:
        return 1


@FactorRegistry.register("carry_cheapness")
class CarryCheapnessRegistered(CarryCheapness):
    pass
