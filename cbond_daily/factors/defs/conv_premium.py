from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..base import Factor, FactorRegistry


@dataclass
class ConvPremium(Factor):
    name: str = "conv_premium"
    col: str = "deriv_bond_prem_ratio"

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if self.col not in data.columns:
            raise KeyError(f"missing column: {self.col}")
        s = pd.to_numeric(data[self.col], errors="coerce").astype(float)
        s = s.replace([np.inf, -np.inf], np.nan)
        return s


@FactorRegistry.register("conv_premium")
class ConvPremiumRegistered(ConvPremium):
    pass
