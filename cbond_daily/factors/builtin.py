from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np
from .base import Factor, FactorRegistry
from . import operators as ops


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


@dataclass
class MeanSigmoidEmaRet(Factor):
    """
    Mean(Sigmoid(EMA(ret, ema_span)), mean_window)

    ret        = close_price / prev_close_price - 1
    ema_span   = 默认 20
    mean_window= 默认 9
    """
    name: str = "mser"
    ema_span: int = 20
    mean_window: int = 9

    def required_lookback(self) -> int:
        return int(max(self.ema_span, self.mean_window))

    def compute(self, data: pd.DataFrame) -> pd.Series:
        ret = data["close_price"] / data["prev_close_price"] - 1.0
        ema = (
            ret
            .groupby(data["code"])
            .transform(lambda x: x.ewm(span=self.ema_span, adjust=False).mean())
        )
        sigmoid = 1.0 / (1.0 + np.exp(-ema))
        factor = (
            sigmoid
            .groupby(data["code"])
            .transform(lambda x: x.rolling(self.mean_window, min_periods=1).mean())
        )
        return factor

@dataclass
class ExprAlpha001(Factor):
    """
    Delta(Div(Max(Pow2(PosMask(MaxDivMin(InvWMA(CSRank(MaxDivMin(CSRank(Max(ret,15)),8)),20),6))),60),close),9)
    """
    name: str = "expr_alpha_001"

    # windows
    max_ret_win: int = 15
    mdm1_win: int = 8
    wma1_win: int = 20
    mdm2_win: int = 6
    max2_win: int = 60
    delta_win: int = 9

    date_col: str = "trade_date"
    code_col: str = "code"
    close_col: str = "close_price"
    prev_close_col: str = "prev_close_price"

    def required_lookback(self) -> int:
        max_win = max(
            self.max_ret_win,
            self.mdm1_win,
            self.wma1_win,
            self.mdm2_win,
            self.max2_win,
        )
        return int(max_win + self.delta_win)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        # 必要字段
        for c in [self.code_col, self.close_col, self.prev_close_col]:
            if c not in data.columns:
                raise KeyError(f"missing column: {c}")
        if self.date_col not in data.columns:
            # 没有时间维度无法做 rolling
            return pd.Series([pd.NA] * len(data), index=data.index)

        df = data[[self.date_col, self.code_col, self.close_col, self.prev_close_col]].copy()
        df[self.close_col] = pd.to_numeric(df[self.close_col], errors="coerce").astype(float)
        df[self.prev_close_col] = pd.to_numeric(df[self.prev_close_col], errors="coerce").astype(float)
        df[self.date_col] = pd.to_datetime(df[self.date_col]).dt.date

        # 为了 rolling 正确：按 code, date 排序；最后再按原 index 还原
        df["_idx"] = df.index
        df = df.sort_values([self.code_col, self.date_col])

        ret = ops.calc_ret(df[self.close_col], df[self.prev_close_col])
        max_ret = ops.rolling_max(ret, df[self.code_col], self.max_ret_win)
        csr1 = ops.cs_rank(df, max_ret, self.date_col)
        mdm1 = ops.max_div_min(csr1, df[self.code_col], self.mdm1_win)
        csr2 = ops.cs_rank(df, mdm1, self.date_col)
        wma1 = ops.inv_wma(csr2, df[self.code_col], self.wma1_win)
        mdm2 = ops.max_div_min(wma1, df[self.code_col], self.mdm2_win)
        pm = ops.pos_mask(mdm2)
        p2 = ops.pow2(pm)
        mx2 = ops.rolling_max(p2, df[self.code_col], self.max2_win)
        close = df[self.close_col].replace([np.inf, -np.inf], np.nan)
        div = mx2 / close
        out_vals = ops.delta(div, df[self.code_col], self.delta_win)

        # 还原顺序
        out = pd.Series(out_vals.values, index=df.index)
        out = df.set_index("_idx").assign(_out=out).loc[:, "_out"].reindex(data.index)
        return out

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


@dataclass
class RollingMaxDrawdown(Factor):
    name: str = "rolling_max_drawdown"
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

        ret = ops.calc_ret(df[self.close_col], df[self.prev_close_col]).fillna(0.0)
        def _mdd(series: pd.Series) -> pd.Series:
            wealth = (1.0 + series).cumprod()
            peak = wealth.cummax()
            dd = wealth / peak - 1.0
            return dd.rolling(self.window, min_periods=self.window).min()

        mdd = ret.groupby(df[self.code_col], sort=False).transform(_mdd)
        out = df.set_index("_idx").assign(_out=mdd).loc[:, "_out"].reindex(data.index)
        return out



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
        对每个 code 的序列做 rolling，返回“窗口内最后一个值”的分位（0~1）。
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
            # 缺关键字段：直接返回全 NaN，避免 pipeline 崩溃
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

        # 3) 溢价“滞后空间”：近期溢价未压缩（prem_chg 越大说明溢价上升/未压缩 -> 仍可能有修复空间）
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


@dataclass
class TrendPremLagSimple(Factor):
    """
    极简版：正股动量 × 溢价滞后

    Trend_{i,t} = S_{i,t} / S_{i,t-mom_win} - 1
    Lag_{i,t}   = P_{i,t} - P_{i,t-prem_chg_win}
    Factor_{i,t}= Trend_{i,t} * Lag_{i,t}

    过滤（不满足则 NaN）：
    1) Trend > 0
    2) Lag   > 0
    3) 成交额 >= 当日截面分位 liq_rank_min（默认 0.5，即中位数以上）
    4) 未进入强赎/触发流程（若字段存在）
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

@dataclass
class OptionCheapness(Factor):
    """
    期权便宜度（Option Cheapness / Vol-Value Mispricing）

    定义：
      O   = CB_close - PureBondValue
      OVR = O / ConvValue
      RV  = AnnualizedStd( log(S_t / S_{t-1}), vol_win )

      Factor = RV / (OVR + eps)

    过滤（不满足则 NaN）：
      1) O > 0
      2) ConvValue > 0
      3) 成交额当日截面分位 >= liq_rank_min
      4) 未进入强赎/触发流程（若字段存在）
      5) year_to_mat >= min_year_to_mat（若字段存在）
    """

    name: str = "option_cheapness"

    # windows
    vol_win: int = 20
    annualize: int = 252
    eps: float = 1e-8

    # filters
    liq_rank_min: float = 0.50
    min_year_to_mat: float = 0.30

    # columns
    date_col: str = "trade_date"
    code_col: str = "code"

    cb_close_col_primary: str = "close_price"
    cb_close_col_fallback: str = "base_cb_close_price"

    amount_col_primary: str = "amount"
    amount_col_fallback: str = "base_cb_amount"

    stock_close_col_primary: str = "base_stk_close_price"
    stock_close_col_fallback: str = "deriv_stock_close_price"

    conv_value_col_primary: str = "deriv_conv_value"
    conv_value_col_fallback: str = "base_conv_value"

    pure_bond_col_primary: str = "deriv_pure_redemption_value"
    pure_bond_col_fallback: str = "base_pure_redemption_value"

    year_to_mat_col_primary: str = "deriv_year_to_mat"
    year_to_mat_col_fallback: str = "base_year_to_mat"

    in_trigger_process_col: str = "base_in_trigger_process"

    def required_lookback(self) -> int:
        return int(self.vol_win + 2)

    @staticmethod
    def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _to_float(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce").astype(float)
        return s.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _cs_rank(df: pd.DataFrame, values: pd.Series, date_col: str) -> pd.Series:
        return values.groupby(df[date_col]).rank(method="average", pct=True)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        cb_close_col = self._pick_first_existing(data, [self.cb_close_col_primary, self.cb_close_col_fallback])
        amount_col = self._pick_first_existing(data, [self.amount_col_primary, self.amount_col_fallback])
        stock_close_col = self._pick_first_existing(data, [self.stock_close_col_primary, self.stock_close_col_fallback])
        conv_value_col = self._pick_first_existing(data, [self.conv_value_col_primary, self.conv_value_col_fallback])
        pure_bond_col = self._pick_first_existing(data, [self.pure_bond_col_primary, self.pure_bond_col_fallback])
        year_to_mat_col = self._pick_first_existing(data, [self.year_to_mat_col_primary, self.year_to_mat_col_fallback])

        required = [self.date_col, self.code_col, cb_close_col, amount_col, stock_close_col, conv_value_col, pure_bond_col]
        if any(c is None for c in required):
            return pd.Series(np.nan, index=data.index, name=self.name)

        # 只取必要列 + 排序保证 rolling 正确，最后再按原 index 还原
        cols = [self.date_col, self.code_col, cb_close_col, amount_col, stock_close_col, conv_value_col, pure_bond_col]
        if year_to_mat_col is not None:
            cols.append(year_to_mat_col)
        if self.in_trigger_process_col in data.columns:
            cols.append(self.in_trigger_process_col)

        df = data[cols].copy()
        df["_idx"] = df.index
        df[self.date_col] = pd.to_datetime(df[self.date_col]).dt.date
        df = df.sort_values([self.code_col, self.date_col])

        cb_close = self._to_float(df[cb_close_col])
        amount = self._to_float(df[amount_col])
        stk_close = self._to_float(df[stock_close_col])
        conv_value = self._to_float(df[conv_value_col])
        pure_bond = self._to_float(df[pure_bond_col])

        # OVR = (CB - PureBond) / ConvValue
        opt_value = cb_close - pure_bond
        ovr = opt_value / conv_value.replace(0.0, np.nan)
        ovr = ovr.replace([np.inf, -np.inf], np.nan)

        # RV = annualized rolling std of log returns of stock close
        log_ret = np.log(stk_close / stk_close.groupby(df[self.code_col]).shift(1))
        log_ret = log_ret.replace([np.inf, -np.inf], np.nan)
        rv = log_ret.groupby(df[self.code_col]).transform(
            lambda s: s.rolling(self.vol_win, min_periods=max(5, self.vol_win // 3)).std()
        )
        rv = rv * np.sqrt(float(self.annualize))
        rv = rv.replace([np.inf, -np.inf], np.nan)

        factor = rv / (ovr + float(self.eps))
        factor = factor.replace([np.inf, -np.inf], np.nan)

        # filters
        mask_opt = opt_value > 0.0
        mask_cv = conv_value > 0.0

        liq_rank = self._cs_rank(df, amount.fillna(0.0), self.date_col)
        mask_liq = liq_rank >= float(self.liq_rank_min)

        if self.in_trigger_process_col in df.columns:
            in_trigger = pd.to_numeric(df[self.in_trigger_process_col], errors="coerce").fillna(0.0).astype(float)
            mask_trigger = in_trigger <= 0.0
        else:
            mask_trigger = pd.Series(True, index=df.index)

        if year_to_mat_col is not None and year_to_mat_col in df.columns:
            ytm = self._to_float(df[year_to_mat_col])
            mask_mat = ytm >= float(self.min_year_to_mat)
        else:
            mask_mat = pd.Series(True, index=df.index)

        mask_valid = np.isfinite(rv) & np.isfinite(ovr) & np.isfinite(factor)
        mask = mask_opt & mask_cv & mask_liq & mask_trigger & mask_mat & mask_valid

        out = factor.where(mask, np.nan)
        out.name = self.name

        # 还原到原 data.index 顺序
        out = df.set_index("_idx").assign(_out=out).loc[:, "_out"].reindex(data.index)
        out.name = self.name
        return out


@FactorRegistry.register("option_cheapness")
class OptionCheapnessRegistered(OptionCheapness):
    pass



@FactorRegistry.register("trend_prem_lag_simple")
class TrendPremLagSimpleRegistered(TrendPremLagSimple):
    pass

@FactorRegistry.register("trend_lagged_prem_repair")
class TrendLaggedPremRepairRegistered(TrendLaggedPremRepair):
    pass



@FactorRegistry.register("conv_premium")
class ConvPremiumRegistered(ConvPremium):
    pass


@FactorRegistry.register("rolling_ret_vol")
class RollingReturnVolRegistered(RollingReturnVol):
    pass


@FactorRegistry.register("rolling_max_drawdown")
class RollingMaxDrawdownRegistered(RollingMaxDrawdown):
    pass


@FactorRegistry.register("intraday_momentum")
class IntradayMomentumRegistered(IntradayMomentum):
    pass


@FactorRegistry.register("liquidity")
class LiquidityRegistered(Liquidity):
    pass


@FactorRegistry.register("mser")
class MeanSigmoidEmaRetRegistered(MeanSigmoidEmaRet):
    pass


@FactorRegistry.register("expr_alpha_001")
class ExprAlpha001Registered(ExprAlpha001):
    pass
