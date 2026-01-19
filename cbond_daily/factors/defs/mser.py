from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..base import Factor, FactorRegistry


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


@FactorRegistry.register("mser")
class MeanSigmoidEmaRetRegistered(MeanSigmoidEmaRet):
    pass
