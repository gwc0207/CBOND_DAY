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


@FactorRegistry.register("conv_premium")
class ConvPremiumRegistered(ConvPremium):
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
