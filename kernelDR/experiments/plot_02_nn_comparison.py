from typing import Annotated

import numpy as np
import cyclopts

from kernelDR.experiments.plot_utils import (
    setup_matplotlib, read_deep_ritz_results, read_nn_results,
    output_path, KERNEL_COLORS, KERNEL_LABELS, NN_COLOR,
)

app = cyclopts.App()


@app.default
def main(
    deep_ritz_dir: Annotated[str, cyclopts.Parameter(help="Directory with deep Ritz results.")] = "reference_results/results_smooth_solution/",
    nn_dir: Annotated[str, cyclopts.Parameter(help="Directory with neural network results.")] = "reference_results/results_neural_network_smooth_solution/",
    k_smoothness: Annotated[int, cyclopts.Parameter(help="Kernel smoothness to compare.")] = 1,
    ep: Annotated[float, cyclopts.Parameter(help="Kernel shape parameter.")] = 1.0,
    output: Annotated[str, cyclopts.Parameter(help="Output filename.")] = "",
    title_suffix: Annotated[str, cyclopts.Parameter(help="Title suffix.")] = "",
):
    if not output:
        output = output_path("nn_comparison_plot.pdf")

    plt = setup_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    color = KERNEL_COLORS[k_smoothness]
    label = KERNEL_LABELS[k_smoothness]

    try:
        h_dr, l2_dr, h1_dr = read_deep_ritz_results(deep_ritz_dir, k_smoothness, ep=ep)
        n_per_dim = np.round(1.0 / h_dr - 1).astype(int)
        n_params_kernel = n_per_dim**2 + 4 * (n_per_dim + 2)
    except (FileNotFoundError, OSError):
        n_params_kernel, l2_dr, h1_dr = None, None, None

    try:
        n_params_nn, l2_nn, h1_nn = read_nn_results(nn_dir)
    except (FileNotFoundError, OSError):
        n_params_nn, l2_nn, h1_nn = None, None, None

    if n_params_kernel is not None:
        axes[0].semilogy(n_params_kernel, l2_dr, 'o-', color=color, markersize=4,
                         label=f"{label}: deep Ritz with kernels")
    if n_params_nn is not None:
        axes[0].semilogy(n_params_nn, l2_nn, '^-', color=NN_COLOR, markersize=4,
                         label="deep Ritz with neural networks")

    if n_params_kernel is not None:
        axes[1].semilogy(n_params_kernel, h1_dr, 'o-', color=color, markersize=4,
                         label=f"{label}: deep Ritz with kernels")
    if n_params_nn is not None:
        axes[1].semilogy(n_params_nn, h1_nn, '^-', color=NN_COLOR, markersize=4,
                         label="deep Ritz with neural networks")

    axes[0].set_xlabel("Number of parameters")
    axes[0].set_ylabel("Relative $L^2$-error")
    axes[0].legend()
    axes[0].set_title(f"$L^2$-error {title_suffix}".strip())

    axes[1].set_xlabel("Number of parameters")
    axes[1].set_ylabel("Relative $H^1$-error")
    axes[1].legend()
    axes[1].set_title(f"$H^1$-error {title_suffix}".strip())

    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved to {output}")
    plt.close()


if __name__ == "__main__":
    app()
