from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..base import Factor, FactorRegistry


@dataclass
class TrendPremLagSimple(Factor):
    """
    极简版：正股动量 × 溢价滞后

    Trend_{i,t} = S_{i,t} / S_{i,t-mom_win} - 1
    Lag_{i,t}   = P_{i,t} - P_{i,t-prem_chg_win}
    Factor_{i,t}= Trend_{i,t} * Lag_{i,t}

    过滤（不满足即NaN）：
    1) Trend > 0
    2) Lag   > 0
    3) 成交额>= 当日截面分位 liq_rank_min（默认0.5，即中位数以上）
    4) 未进入强赎触发流程（若字段存在）
    """

    name: str = "trend_prem_lag_simple"

    # windows
    mom_win: int = 20
    prem_chg_win: int = 5

    # filters
    liq_rank_min: float = 0.50  # 默认：成交额>=当日中位数

    # column names
    date_col: str = "trade_date"
    code_col: str = "code"
    amount_col: str = "amount"

    prem_col_primary: str = "base_bond_prem_ratio"
    prem_col_fallback: str = "deriv_bond_prem_ratio"

    stk_close_col_primary: str = "base_stk_close_price"
    stk_close_col_fallback: str = "deriv_stock_close_price"

    in_trigger_process_col: str = "base_in_trigger_process"

    def required_lookback(self) -> int:
        return int(max(self.mom_win, self.prem_chg_win) + 2)

    @staticmethod
    def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _safe_series(s: pd.Series) -> pd.Series:
        s = s.replace([np.inf, -np.inf], np.nan)
        return s.astype(float)

    @staticmethod
    def _cs_rank(df: pd.DataFrame, values: pd.Series, date_col: str) -> pd.Series:
        # 当日截面分位 rank（0~1）
        return values.groupby(df[date_col]).rank(method="average", pct=True)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        prem_col = self._pick_first_existing(data, [self.prem_col_primary, self.prem_col_fallback])
        stk_close_col = self._pick_first_existing(data, [self.stk_close_col_primary, self.stk_close_col_fallback])

        required = [self.date_col, self.code_col, self.amount_col, prem_col, stk_close_col]
        if any(c is None for c in required):
            return pd.Series(np.nan, index=data.index, name=self.name)

        df = data
        date_s = df[self.date_col]
        code_s = df[self.code_col]

        amount = self._safe_series(df[self.amount_col])
        prem = self._safe_series(df[prem_col])
        stk_close = self._safe_series(df[stk_close_col])

        # Trend: S_t / S_{t-mom} - 1
        trend = stk_close.groupby(code_s).transform(lambda s: s / s.shift(self.mom_win) - 1.0)
        trend = self._safe_series(trend)

        # Lag: P_t - P_{t-chg}
        lag = prem.groupby(code_s).transform(lambda s: s - s.shift(self.prem_chg_win))
        lag = self._safe_series(lag)

        factor = trend * lag
        factor = self._safe_series(factor)

        # filters
        mask_trend = trend > 0.0
        mask_lag = lag > 0.0

        liq_rank = self._cs_rank(df, amount, self.date_col)
        mask_liq = liq_rank >= float(self.liq_rank_min)

        if self.in_trigger_process_col in df.columns:
            in_trigger = df[self.in_trigger_process_col].fillna(0).astype(float)
            mask_trigger = in_trigger <= 0.0
        else:
            mask_trigger = pd.Series(True, index=df.index)

        mask_valid = np.isfinite(trend) & np.isfinite(lag) & np.isfinite(factor)
        mask = mask_trend & mask_lag & mask_liq & mask_trigger & mask_valid

        out = factor.where(mask, np.nan)
        out.name = self.name
        return out


@FactorRegistry.register("trend_prem_lag_simple")
class TrendPremLagSimpleRegistered(TrendPremLagSimple):
    pass
