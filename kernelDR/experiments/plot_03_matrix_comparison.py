from typing import Annotated

import cyclopts

from kernelDR.experiments.plot_utils import (
    setup_matplotlib, read_deep_ritz_results, read_matrix_form_results,
    output_path, KERNEL_COLORS, KERNEL_LABELS,
)

app = cyclopts.App()


@app.default
def main(
    deep_ritz_dir: Annotated[str, cyclopts.Parameter(help="Directory with deep Ritz results.")] = "reference_results/results_smooth_solution/",
    matrix_dir: Annotated[str, cyclopts.Parameter(help="Directory with matrix form results.")] = "reference_results/results_matrix_form_smooth_solution/",
    list_kmat: Annotated[tuple[int, ...], cyclopts.Parameter(help="Kernel smoothness values.")] = (0, 1, 2),
    ep: Annotated[float, cyclopts.Parameter(help="Kernel shape parameter.")] = 1.0,
    output: Annotated[str, cyclopts.Parameter(help="Output filename.")] = "",
    title_suffix: Annotated[str, cyclopts.Parameter(help="Title suffix.")] = "",
):
    if not output:
        output = output_path("matrix_comparison_plot.pdf")

    plt = setup_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for kmat in list_kmat:
        color = KERNEL_COLORS[kmat]
        label = KERNEL_LABELS[kmat]

        try:
            h_dr, l2_dr, h1_dr = read_deep_ritz_results(deep_ritz_dir, kmat, ep=ep)
        except (FileNotFoundError, OSError):
            h_dr, l2_dr, h1_dr = None, None, None

        try:
            h_mat, l2_mat, h1_mat = read_matrix_form_results(matrix_dir, kmat)
        except (FileNotFoundError, OSError):
            h_mat, l2_mat, h1_mat = None, None, None

        if h_dr is not None:
            axes[0].loglog(h_dr, l2_dr, 'o-', color=color, markersize=4,
                           label=f"{label}: energy minimization")
        if h_mat is not None:
            axes[0].loglog(h_mat, l2_mat, 's--', color=color, markersize=4,
                           label=f"{label}: linear system")

        if h_dr is not None:
            axes[1].loglog(h_dr, h1_dr, 'o-', color=color, markersize=4,
                           label=f"{label}: energy minimization")
        if h_mat is not None:
            axes[1].loglog(h_mat, h1_mat, 's--', color=color, markersize=4,
                           label=f"{label}: linear system")

    axes[0].set_xlabel("Mesh norm $h$")
    axes[0].set_ylabel("Relative $L^2$-error")
    axes[0].legend()
    axes[0].set_title(f"$L^2$-error {title_suffix}".strip())

    axes[1].set_xlabel("Mesh norm $h$")
    axes[1].set_ylabel("Relative $H^1$-error")
    axes[1].legend()
    axes[1].set_title(f"$H^1$-error {title_suffix}".strip())

    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved to {output}")
    plt.close()


if __name__ == "__main__":
    app()
