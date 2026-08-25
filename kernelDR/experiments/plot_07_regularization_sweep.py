"""Plot the Tikhonov sweep: how well the discrete problem is solved against how
good the resulting approximation is.

Reads the files written by main_08_regularization_sweep.py. The left panel shows
the discrete energy of the unregularised system, which decreases monotonically as
the regularisation is reduced, i.e. the discrete problem is solved ever more
accurately. The right panel shows the relative errors on an independent grid. A
minimum of the errors at a positive regularisation parameter means that the most
accurate solution of the discrete problem is not the best approximation --
a statement about the ill-posedness of the discrete problem rather than about the
quality of the solver.
"""

from typing import Annotated

import cyclopts
import numpy as np
import os

from kernelDR.experiments.plot_utils import setup_matplotlib, output_path

app = cyclopts.App()


@app.default
def main(
    results_file: Annotated[str, cyclopts.Parameter(help="File written by main_08_regularization_sweep.py.")],
    output: Annotated[str, cyclopts.Parameter(help="Output filename.")] = "",
    reference_error: Annotated[float, cyclopts.Parameter(help="Optional horizontal line, e.g. the error reached by Adam with resampled quadrature, to show whether any regularisation reaches it (0 to omit).")] = 0.0,
    reference_label: Annotated[str, cyclopts.Parameter(help="Legend entry for the reference line.")] = "Adam (random quadrature)",
):
    if not os.path.exists(results_file):
        raise SystemExit(f"{results_file} not found.")
    if not output:
        base = os.path.splitext(os.path.basename(results_file))[0]
        output = output_path(f"{base}.pdf")

    data = np.genfromtxt(results_file, skip_header=2)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    regularization, energy = data[:, 0], data[:, 2]
    coeff_norm, error_l2, error_h1 = data[:, 4], data[:, 5], data[:, 6]

    # lambda = 0 cannot be shown on a logarithmic axis; plot it as a horizontal
    # reference instead so the unregularised solve stays visible.
    finite = regularization > 0
    unregularized = None if finite.all() else np.argmin(regularization)

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].semilogx(regularization[finite], energy[finite], 'o-', color="#3B529B", markersize=4)
    if unregularized is not None:
        axes[0].axhline(energy[unregularized], color="grey", linestyle=":", linewidth=1,
                        label=r"$\lambda = 0$")
        axes[0].legend()
    axes[0].set_xlabel(r"Tikhonov parameter $\lambda$")
    axes[0].set_ylabel("Discrete energy")
    axes[0].set_title("Solving the discrete problem")

    axes[1].loglog(regularization[finite], coeff_norm[finite], 'o-', color="#440154", markersize=4)
    axes[1].set_xlabel(r"Tikhonov parameter $\lambda$")
    axes[1].set_ylabel(r"$\|c\|_2$")
    axes[1].set_title("Coefficient magnitude")

    axes[2].loglog(regularization[finite], error_l2[finite], 'o-', color="#D01F3C", markersize=4,
                   label=r"relative $L^2$")
    axes[2].loglog(regularization[finite], error_h1[finite], 's-', color="#5EC962", markersize=4,
                   label=r"relative $H^1$")
    idx_best = int(np.argmin(error_l2))
    axes[2].axvline(max(regularization[idx_best], regularization[finite].min()), color="grey",
                    linestyle="--", linewidth=1,
                    label=rf"best $L^2$ at $\lambda={regularization[idx_best]:.0e}$")
    if reference_error > 0:
        axes[2].axhline(reference_error, color="black", linestyle=":", linewidth=1,
                        label=reference_label)
    axes[2].set_xlabel(r"Tikhonov parameter $\lambda$")
    axes[2].set_ylabel("Relative error")
    axes[2].set_title("Quality of the approximation")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved to {output}")
    print(f"best relative L2-error {error_l2[idx_best]:.4e} at lambda = {regularization[idx_best]:.1e}")
    idx_energy = int(np.argmin(energy))
    print(f"lowest discrete energy {energy[idx_energy]:.6f} at lambda = {regularization[idx_energy]:.1e} "
          f"(relative L2-error {error_l2[idx_energy]:.4e} there)")
    plt.close()


if __name__ == "__main__":
    app()
