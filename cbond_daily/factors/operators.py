from __future__ import annotations

import numpy as np
import pandas as pd


def calc_ret(close: pd.Series, prev_close: pd.Series) -> pd.Series:
    ret = close / prev_close - 1.0
    return ret.replace([np.inf, -np.inf], np.nan)


def cs_rank(df: pd.DataFrame, values: pd.Series, date_col: str) -> pd.Series:
    return values.groupby(df[date_col]).rank(method="average", pct=True)


def rolling_max(values: pd.Series, code_col: pd.Series, window: int) -> pd.Series:
    return values.groupby(code_col).transform(lambda s: s.rolling(window, min_periods=1).max())


def rolling_min(values: pd.Series, code_col: pd.Series, window: int) -> pd.Series:
    return values.groupby(code_col).transform(lambda s: s.rolling(window, min_periods=1).min())


def max_div_min(values: pd.Series, code_col: pd.Series, window: int) -> pd.Series:
    max_v = rolling_max(values, code_col, window)
    min_v = rolling_min(values, code_col, window).replace(0.0, np.nan)
    return max_v / min_v


def inv_wma(values: pd.Series, code_col: pd.Series, window: int) -> pd.Series:
    weights = np.arange(window, 0, -1, dtype=float)

    def _apply(arr: np.ndarray) -> float:
        ww = weights[-len(arr):]
        denom = ww.sum()
        if denom == 0:
            return np.nan
        return float(np.dot(arr, ww) / denom)

    return values.groupby(code_col).transform(
        lambda s: s.rolling(window, min_periods=1).apply(_apply, raw=True)
    )


def pos_mask(values: pd.Series) -> pd.Series:
    return values.where(values > 0)


def pow2(values: pd.Series) -> pd.Series:
    return values * values


def delta(values: pd.Series, code_col: pd.Series, window: int) -> pd.Series:
    return values.groupby(code_col).transform(lambda s: s - s.shift(window))
