from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..base import Factor, FactorRegistry


@dataclass
class TrendLaggedPremRepair(Factor):
    name: str = "trend_lagged_prem_repair"

    # windows
    mom_win: int = 20               # 正股趋势窗口
    prem_hist_win: int = 60         # 溢价历史分位窗口
    prem_chg_win: int = 5           # 溢价变化窗口（滞后修复空间）

    # filters
    liq_rank_min: float = 0.30      # 当日成交额截面分位过滤（保留 >= 30%）
    min_year_to_mat: float = 0.30   # 过短久期/到期临近过滤（年）

    # weights
    w_trend: float = 0.50
    w_cheap: float = 0.30
    w_lag: float = 0.20

    # column names (cleaned data wide table)
    date_col: str = "trade_date"
    code_col: str = "code"
    amount_col: str = "amount"

    prem_col_primary: str = "base_bond_prem_ratio"
    prem_col_fallback: str = "deriv_bond_prem_ratio"

    stk_close_col_primary: str = "base_stk_close_price"
    stk_close_col_fallback: str = "deriv_stock_close_price"

    year_to_mat_col_primary: str = "base_year_to_mat"
    year_to_mat_col_fallback: str = "deriv_year_to_mat"

    in_trigger_process_col: str = "base_in_trigger_process"

    def required_lookback(self) -> int:
        # 需要滚动窗口 + shift
        return int(max(self.mom_win, self.prem_hist_win, self.prem_chg_win) + 2)

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
        # pct rank in each trade_date
        return values.groupby(df[date_col]).rank(method="average", pct=True)

    @staticmethod
    def _rolling_last_percentile(values: pd.Series, code_col: pd.Series, window: int) -> pd.Series:
        """
        对每个code 的序列做 rolling，返回“窗口内最后一个值”的分位（0~1）。
        """
        def _pct_last(arr: np.ndarray) -> float:
            x = arr[-1]
            if not np.isfinite(x):
                return np.nan
            valid = arr[np.isfinite(arr)]
            if valid.size == 0:
                return np.nan
            # <= 的经验分位
            return float((valid <= x).mean())

        return values.groupby(code_col).transform(
            lambda s: s.rolling(window, min_periods=max(5, window // 3)).apply(_pct_last, raw=True)
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        # -------- resolve columns --------
        prem_col = self._pick_first_existing(data, [self.prem_col_primary, self.prem_col_fallback])
        stk_close_col = self._pick_first_existing(data, [self.stk_close_col_primary, self.stk_close_col_fallback])
        ytm_col = self._pick_first_existing(data, ["base_ytm", "deriv_ytm"])
        year_to_mat_col = self._pick_first_existing(
            data, [self.year_to_mat_col_primary, self.year_to_mat_col_fallback]
        )

        required = [self.date_col, self.code_col, self.amount_col, prem_col, stk_close_col, year_to_mat_col]
        if any(c is None for c in required):
            # 缺关键字段：直接返回全NaN，避免 pipeline 崩溃
            return pd.Series(np.nan, index=data.index)

        df = data

        date_s = df[self.date_col]
        code_s = df[self.code_col]

        amount = self._safe_series(df[self.amount_col])
        prem = self._safe_series(df[prem_col])          # 转股溢价率（通常是小数，比如 0.20=20%）
        stk_close = self._safe_series(df[stk_close_col])
        year_to_mat = self._safe_series(df[year_to_mat_col])

        # -------- features --------
        # 1) 正股趋势：rolling return over mom_win
        stk_ret_mom = stk_close.groupby(code_s).transform(lambda s: s / s.shift(self.mom_win) - 1.0)
        stk_ret_mom = self._safe_series(stk_ret_mom)

        # 2) 溢价“便宜程度”：自身历史分位（越低越便宜）
        prem_pct = self._rolling_last_percentile(prem, code_s, self.prem_hist_win)
        cheapness = 1.0 - prem_pct  # 越大越便宜（相对自身历史）

        # 3) 溢价“滞后空间”：近期溢价未压缩（prem_chg 越大说明溢价上升/未压缩-> 仍可能有修复空间）
        prem_chg = prem.groupby(code_s).transform(lambda s: s - s.shift(self.prem_chg_win))
        prem_chg = self._safe_series(prem_chg)
        lag_space = prem_chg  # 越大表示近期溢价更“没压下来”，潜在修复空间更大

        # -------- cross-sectional scoring --------
        trend_rank = self._cs_rank(df, stk_ret_mom, self.date_col)          # 越大越好
        cheap_rank = self._cs_rank(df, cheapness, self.date_col)           # 越大越好
        lag_rank = self._cs_rank(df, lag_space, self.date_col)             # 越大越好

        score = self.w_trend * trend_rank + self.w_cheap * cheap_rank + self.w_lag * lag_rank

        # -------- filters --------
        # 流动性：按当日成交额截面分位过滤
        liq_rank = self._cs_rank(df, amount, self.date_col)
        mask_liq = liq_rank >= float(self.liq_rank_min)

        # 到期过滤
        mask_mat = year_to_mat >= float(self.min_year_to_mat)

        # 强赎触发流程过滤（字段存在才过滤）
        if self.in_trigger_process_col in df.columns:
            in_trigger = df[self.in_trigger_process_col].fillna(0).astype(float)
            mask_trigger = in_trigger <= 0.0
        else:
            mask_trigger = pd.Series(True, index=df.index)

        # 基础有效性：避免 NaN 传播导致“假高分”
        mask_valid = (
            np.isfinite(stk_ret_mom)
            & np.isfinite(cheapness)
            & np.isfinite(lag_space)
            & np.isfinite(score)
        )

        mask = mask_liq & mask_mat & mask_trigger & mask_valid

        out = score.where(mask, np.nan)
        out.name = self.name
        return out


@FactorRegistry.register("trend_lagged_prem_repair")
class TrendLaggedPremRepairRegistered(TrendLaggedPremRepair):
    pass
