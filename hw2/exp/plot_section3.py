"""Recreate the Section 3 plots from the eight original experiment logs.

Run from hw2: uv run python exp/plot_section3.py
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np


EXP = Path(__file__).resolve().parent
STAMP = "20261013_100836"
CONFIGURATIONS = [
    ("", "Full trajectory", "#0072B2"),
    ("_rtg", "Reward-to-go", "#D55E00"),
    ("_na", "Full trajectory + normalization", "#009E73"),
    ("_rtg_na", "Reward-to-go + normalization", "#CC79A7"),
]

for batch_size, prefix, label, filename in [
    (1000, "cartpole", "Small batch", "cartpole_small_batch"),
    (4000, "cartpole_lb", "Large batch", "cartpole_large_batch"),
]:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    max_steps = 0
    for suffix, legend_label, color in CONFIGURATIONS:
        name = prefix + suffix
        directory = EXP / f"CartPole-v0_{name}_sd1_{STAMP}"
        config = json.loads((directory / "flags.json").read_text())
        assert config["batch_size"] == batch_size and config["exp_name"] == name
        with (directory / "log.csv").open() as file:
            rows = list(csv.DictReader(file))
        assert len(rows) == 100 and int(rows[-1]["step"]) == 99
        steps = np.array([int(row["Train_EnvstepsSoFar"]) for row in rows])
        returns = np.array([float(row["Train_AverageReturn"]) for row in rows])
        assert np.all(np.diff(steps) > 0) and np.all(np.isfinite(returns))
        ax.plot(steps, returns, label=legend_label, color=color, linewidth=1.8)
        max_steps = max(max_steps, steps[-1])

    ax.axhline(200, color="#555555", linestyle="--", linewidth=1, zorder=0)
    ax.set_title(f"CartPole-v0: {label.lower()} ({batch_size:,} steps/update)", fontsize=15)
    ax.set_xlabel("Training environment steps (Train_EnvstepsSoFar)", fontsize=11)
    ax.set_ylabel("Average episode return (Train_AverageReturn)", fontsize=11)
    ax.set_xlim(0, max_steps * 1.02)
    ax.set_ylim(0, 210)
    ax.xaxis.set_major_formatter(EngFormatter(sep=""))
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=10)
    fig.suptitle("100 iterations per run · seed flag 1 · raw curves, no smoothing", fontsize=10)
    for extension in ("png", "pdf"):
        output = EXP / f"{filename}.{extension}"
        fig.savefig(output, dpi=180)
        print(output)
    plt.close(fig)
