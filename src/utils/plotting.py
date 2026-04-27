"""Apply one plotting style across scripts and notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


def set_shared_style() -> None:
    """Set a clean report style once so every figure matches."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    sns.set_palette(
        sns.color_palette(["#00704A", "#27251F", "#D4E9E2", "#CBA258", "#F2F0EB"])
    )


set_shared_style()
