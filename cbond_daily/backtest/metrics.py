from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(daily_returns: pd.DataFrame, nav_curve: pd.DataFrame) -> dict[str, float]:
    if daily_returns.empty or nav_curve.empty:
        return {
            "days": 0,
            "total_return": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }
    days = int(len(daily_returns))
    total_return = float(nav_curve["nav"].iloc[-1] - 1.0)
    ann_return = float((1.0 + total_return) ** (252.0 / days) - 1.0) if days else 0.0
    daily_vol = float(daily_returns["day_return"].std(ddof=0))
    ann_vol = float(daily_vol * np.sqrt(252.0))
    sharpe = float(ann_return / ann_vol) if ann_vol else 0.0
    nav = nav_curve["nav"].astype(float)
    running_max = nav.cummax()
    drawdown = (nav / running_max) - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    return {
        "days": days,
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }
