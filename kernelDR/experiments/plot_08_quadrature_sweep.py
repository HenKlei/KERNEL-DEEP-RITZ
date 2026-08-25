"""Relative errors of the assembled linear system versus the number of quadrature points.

Companion of the corresponding figure in the paper. The solid lines show the error of
the direct solve for an increasing quadrature, the dashed horizontal lines the error
attained by the deep Ritz method with a new quadrature drawn in every epoch, using the
same centers and kernels. Where the two agree, the accuracy is limited by the
approximation properties of the kernel space rather than by the quadrature.
"""
from typing import Annotated

import cyclopts

from kernelDR.experiments.plot_utils import (
    setup_matplotlib, output_path, KERNEL_COLORS, KERNEL_LABELS,
)

app = cyclopts.App()


def read_sweep(path):
    """Return (n_interior, L2, H1) from a consolidated quadrature sweep table."""
    n_i, l2, h1 = [], [], []
    for line in open(path).readlines()[1:]:
        c = line.split("\t")
        if not c[0].strip():
            continue
        n_i.append(int(c[1]))
        l2.append(float(c[3]))
        h1.append(float(c[4]))
    return n_i, l2, h1


def read_reference(path, n_per_dim):
    """Return the deep Ritz (L2, H1) at ``n_per_dim`` from a consolidated errors table."""
    for line in open(path).readlines()[1:]:
        c = line.split("\t")
        if c and int(c[0]) == n_per_dim:
            return float(c[3]), float(c[4])
    return None, None


@app.default
def main(
    sweep_dir_smooth: Annotated[str, cyclopts.Parameter(help="Directory with the smooth quadrature sweep.")] = "reference_results/results_ni_sweep_smooth/",
    sweep_dir_singular: Annotated[str, cyclopts.Parameter(help="Directory with the singular quadrature sweep.")] = "reference_results/results_ni_sweep_singular/",
    ritz_dir_smooth: Annotated[str, cyclopts.Parameter(help="Deep Ritz results for the smooth example.")] = "reference_results/results_smooth_solution/",
    ritz_dir_singular: Annotated[str, cyclopts.Parameter(help="Deep Ritz results for the singular example.")] = "reference_results/results_singular_solution/",
    n_per_dim: Annotated[int, cyclopts.Parameter(help="Centers per dimension the sweep was run at.")] = 20,
    list_kmat: Annotated[tuple[int, ...], cyclopts.Parameter(help="Kernel smoothness values.")] = (0, 1, 2),
    norm: Annotated[str, cyclopts.Parameter(help="Norm to plot: 'L2' or 'H1'.")] = "L2",
    output: Annotated[str, cyclopts.Parameter(help="Output filename.")] = "",
):
    if not output:
        output = output_path(f"quadrature_sweep_{norm}.pdf")
    idx = 1 if norm.upper() == "L2" else 2

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    panels = ((axes[0], sweep_dir_smooth, ritz_dir_smooth, "Smooth solution"),
              (axes[1], sweep_dir_singular, ritz_dir_singular, "Singular solution"))

    for ax, sweep_dir, ritz_dir, title in panels:
        for kmat in list_kmat:
            color = KERNEL_COLORS[kmat]
            label = KERNEL_LABELS[kmat]
            try:
                data = read_sweep(f"{sweep_dir.rstrip('/')}/errors_k_{kmat}.txt")
            except (FileNotFoundError, OSError):
                continue
            ax.loglog(data[0], data[idx], "o-", color=color, markersize=4, label=label)
            try:
                ref = read_reference(f"{ritz_dir.rstrip('/')}/errors_k_{kmat}.txt", n_per_dim)
            except (FileNotFoundError, OSError):
                ref = (None, None)
            if ref[idx - 1] is not None:
                ax.axhline(ref[idx - 1], color=color, linestyle="--", linewidth=1)
        ax.set_xlabel("Number of quadrature points")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(f"Relative ${{{norm[0]}}}^{{{norm[1:]}}}$-error"
                       if norm.upper() in ("L2", "H1") else f"Relative {norm}-error")
    axes[0].legend()

    fig.suptitle("Dashed: deep Ritz with resampled quadrature", fontsize=9, y=0.02)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved to {output}")


if __name__ == "__main__":
    app()
