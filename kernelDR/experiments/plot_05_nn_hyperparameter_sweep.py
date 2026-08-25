"""Plot the neural-network hyperparameter sweeps (activation functions, depths)
with the spread over the repeated runs.

Reads one results directory per curve. If a directory contains an aggregated file
(written by aggregate_seed_runs.py), the median over the seeds is drawn as a line
and the min-max range over the seeds as a shaded band; single-run directories are
drawn as a plain line, so old and new results can be compared side by side.
"""

from typing import Annotated

import cyclopts

from kernelDR.experiments.plot_utils import (
    setup_matplotlib, read_nn_results_with_spread, output_path,
)

app = cyclopts.App()

# Distinguishable colors for up to six curves (activations sweep is the largest).
COLORS = ["#440154", "#5EC962", "#3B529B", "#FDE725", "#D01F3C", "#FFC077"]


@app.default
def main(
    results_dirs: Annotated[tuple[str, ...], cyclopts.Parameter(help="One results directory per curve.")],
    labels: Annotated[tuple[str, ...], cyclopts.Parameter(help="Legend entry per directory.")],
    output: Annotated[str, cyclopts.Parameter(help="Output filename.")] = "",
    title_suffix: Annotated[str, cyclopts.Parameter(help="Title suffix.")] = "",
    show_band: Annotated[bool, cyclopts.Parameter(help="Shade the min-max range over the seeds where available.")] = True,
):
    if len(results_dirs) != len(labels):
        raise SystemExit(f"Got {len(results_dirs)} directories but {len(labels)} labels.")
    if not output:
        output = output_path("nn_hyperparameter_sweep.pdf")

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    n_with_spread = 0
    for idx, (rdir, lbl) in enumerate(zip(results_dirs, labels)):
        color = COLORS[idx % len(COLORS)]
        try:
            n_params, l2, h1, spread = read_nn_results_with_spread(rdir)
        except (FileNotFoundError, OSError) as e:
            print(f"Warning: could not read from {rdir}: {e}")
            continue

        if spread is not None:
            n_with_spread += 1
            n_seeds = int(spread["n_seeds"].min())
            lbl = f"{lbl} ({n_seeds} seeds)" if n_seeds == int(spread["n_seeds"].max()) else f"{lbl} (seeds vary)"

        axes[0].semilogy(n_params, l2, 'o-', color=color, markersize=4, label=lbl)
        axes[1].semilogy(n_params, h1, 'o-', color=color, markersize=4, label=lbl)

        if spread is not None and show_band:
            axes[0].fill_between(n_params, spread["l2_min"], spread["l2_max"], color=color, alpha=0.2, linewidth=0)
            axes[1].fill_between(n_params, spread["h1_min"], spread["h1_max"], color=color, alpha=0.2, linewidth=0)

    band_note = " (band: min-max over seeds)" if n_with_spread else ""
    axes[0].set_xlabel("Number of parameters")
    axes[0].set_ylabel("Relative $L^2$-error")
    axes[0].set_title(f"$L^2$-error {title_suffix}".strip() + band_note)
    axes[0].legend()

    axes[1].set_xlabel("Number of parameters")
    axes[1].set_ylabel("Relative $H^1$-error")
    axes[1].set_title(f"$H^1$-error {title_suffix}".strip() + band_note)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved to {output}")
    plt.close()


if __name__ == "__main__":
    app()
