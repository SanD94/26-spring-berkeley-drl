"""Plot Chapter 6 runs against actual training environment steps."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Completed run directories")
    parser.add_argument("--output_dir", type=Path, default=Path("exp/chapter6"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    summary = []
    for directory in args.runs:
        flags = json.loads((directory / "flags.json").read_text())
        if flags["env_name"] != "InvertedPendulum-v4":
            parser.error(f"{directory} is not an InvertedPendulum-v4 run")
        with (directory / "log.csv").open() as file:
            rows = list(csv.DictReader(file))
        steps = [int(row["Train_EnvstepsSoFar"]) for row in rows]
        returns = {
            metric: [float(row[metric]) for row in rows]
            for metric in ("Train_AverageReturn", "Eval_AverageReturn")
        }
        label = f"{flags['exp_name']} (seed {flags['seed']})"
        runs.append((label, steps, returns))
        result = {"run": str(directory), "label": label, "training_steps": steps[-1]}
        for metric, values in returns.items():
            solved = [step for step, value in zip(steps, values) if value >= 1000]
            eligible = [value for step, value in zip(steps, values) if step <= 100000]
            result[metric] = {
                "first_1000_steps": solved[0] if solved else None,
                "best_within_100k": max(eligible) if eligible else None,
                "final": values[-1],
            }
        summary.append(result)

    # Keep raw, unsmoothed curves: smoothing can hide the actual success point.
    for name, limit in (("full", None), ("within_100k", 100000)):
        figure, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
        for axis, metric, title in zip(
            axes,
            ("Train_AverageReturn", "Eval_AverageReturn"),
            ("Training returns (before update)", "Evaluation returns (after update)"),
        ):
            for label, steps, returns in runs:
                axis.plot(steps, returns[metric], label=label, linewidth=1.5)
            axis.axhline(1000, color="black", linestyle=":", linewidth=1)
            axis.axvline(100000, color="gray", linestyle="--", linewidth=1)
            axis.set(title=title, xlabel="Training environment steps", ylim=(0, 1050))
            right = limit if limit is not None else max(run[1][-1] for run in runs) * 1.03
            axis.set_xlim(left=0, right=right)
            axis.grid(alpha=0.25)
        axes[0].set_ylabel("Average undiscounted episode return")
        figure.suptitle("Chapter 6: InvertedPendulum-v4 sample efficiency")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
        figure.tight_layout(rect=(0, 0.15, 1, 1))
        for extension in ("png", "pdf"):
            figure.savefig(args.output_dir / f"pendulum_{name}.{extension}", dpi=160)
        plt.close(figure)

    output = json.dumps(summary, indent=2)
    (args.output_dir / "summary.json").write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
