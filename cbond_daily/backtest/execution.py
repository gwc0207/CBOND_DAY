from __future__ import annotations

import pandas as pd


def apply_twap_bps(price: pd.Series, bps: float, *, side: str) -> pd.Series:
    factor = bps / 10000.0
    if side == "buy":
        return price * (1.0 + factor)
    if side == "sell":
        return price * (1.0 - factor)
    raise ValueError(f"unknown side: {side}")
