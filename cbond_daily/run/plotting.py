from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_nav_plot(
    nav_curve: pd.DataFrame,
    path: str | Path,
    *,
    benchmark: pd.DataFrame | None = None,
) -> None:
    if nav_curve is None or nav_curve.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(nav_curve["trade_date"], nav_curve["nav"], color="#2C6B8F", linewidth=1.6)
    if benchmark is not None and not benchmark.empty:
        ax.plot(
            benchmark["trade_date"],
            benchmark["nav"],
            color="#9C5A3C",
            linewidth=1.4,
            linestyle="--",
        )
    ax.set_title("NAV Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
