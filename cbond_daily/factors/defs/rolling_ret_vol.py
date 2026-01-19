from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..base import Factor, FactorRegistry
from .. import operators as ops


@dataclass
class RollingReturnVol(Factor):
    name: str = "rolling_ret_vol"
    window: int = 14

    date_col: str = "trade_date"
    code_col: str = "code"
    close_col: str = "close_price"
    prev_close_col: str = "prev_close_price"

    def required_lookback(self) -> int:
        return int(self.window)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        for c in [self.code_col, self.close_col, self.prev_close_col]:
            if c not in data.columns:
                raise KeyError(f"missing column: {c}")
        if self.date_col not in data.columns:
            return pd.Series([pd.NA] * len(data), index=data.index)

        df = data[[self.date_col, self.code_col, self.close_col, self.prev_close_col]].copy()
        df[self.close_col] = pd.to_numeric(df[self.close_col], errors="coerce").astype(float)
        df[self.prev_close_col] = pd.to_numeric(df[self.prev_close_col], errors="coerce").astype(float)
        df[self.date_col] = pd.to_datetime(df[self.date_col]).dt.date
        df["_idx"] = df.index
        df = df.sort_values([self.code_col, self.date_col])

        ret = ops.calc_ret(df[self.close_col], df[self.prev_close_col])
        vol = ret.groupby(df[self.code_col]).transform(
            lambda s: s.rolling(self.window, min_periods=self.window).std(ddof=0)
        )
        out = df.set_index("_idx").assign(_out=vol).loc[:, "_out"].reindex(data.index)
        return out


@FactorRegistry.register("rolling_ret_vol")
class RollingReturnVolRegistered(RollingReturnVol):
    pass
