"""Plot the CG iteration history of the assembled system: how well the discrete
problem is solved (energy, residual) against how good the approximation is
(relative L2- and H1-errors).

Reads the ``cg_convergence_k_*_n_*.txt`` files written by main_04a/main_04b with
``--cg-error-interval``. A residual that keeps decreasing while the errors grow
again indicates semi-convergence: the truncation of CG regularises an ill-posed
discrete problem, so stopping early gives a better approximation than solving the
system accurately.
"""

from typing import Annotated

import cyclopts
import numpy as np
import os

from kernelDR.experiments.plot_utils import setup_matplotlib, output_path

app = cyclopts.App()


@app.default
def main(
    results_dir: Annotated[str, cyclopts.Parameter(help="Directory holding the cg_convergence_*.txt file.")],
    k_smoothness: Annotated[int, cyclopts.Parameter(help="Kernel smoothness parameter.")] = 2,
    n_per_dim: Annotated[int, cyclopts.Parameter(help="Centers per dimension.")] = 20,
    output: Annotated[str, cyclopts.Parameter(help="Output filename.")] = "",
):
    filepath = os.path.join(results_dir, f"cg_convergence_k_{k_smoothness}_n_{n_per_dim}.txt")
    if not os.path.exists(filepath):
        raise SystemExit(f"{filepath} not found; run main_04a/main_04b with --linear-solver cg "
                         f"and --cg-error-interval > 0.")
    if not output:
        output = output_path(f"cg_semiconvergence_k_{k_smoothness}_n_{n_per_dim}.pdf")

    data = np.genfromtxt(filepath, skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    iterations, energy, residual, error_l2, error_h1 = (data[:, 0], data[:, 1], data[:, 2],
                                                        data[:, 3], data[:, 4])
    # Iteration 0 starts from the zero vector, whose residual is finite but whose
    # energy is exactly 0; drop it from the log-scale plots.
    positive = iterations > 0

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(iterations, energy, color="#3B529B")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("CG iteration")
    axes[0].set_ylabel("Discrete energy")
    axes[0].set_title("Energy of the assembled system")

    axes[1].loglog(iterations[positive], residual[positive], color="#3B529B")
    axes[1].set_xlabel("CG iteration")
    axes[1].set_ylabel(r"$\|Ac-b\|_2$")
    axes[1].set_title("Residual")

    axes[2].loglog(iterations[positive], error_l2[positive], color="#D01F3C", label=r"relative $L^2$")
    axes[2].loglog(iterations[positive], error_h1[positive], color="#5EC962", label=r"relative $H^1$")
    idx_best = int(np.argmin(error_l2))
    axes[2].axvline(iterations[idx_best], color="grey", linestyle="--", linewidth=1,
                    label=f"min. $L^2$ at it. {int(iterations[idx_best])}")
    axes[2].set_xlabel("CG iteration")
    axes[2].set_ylabel("Relative error")
    axes[2].set_title("Error of the approximation")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved to {output}")
    print(f"minimum relative L2-error {error_l2[idx_best]:.4e} at iteration {int(iterations[idx_best])} "
          f"(residual {residual[idx_best]:.4e})")
    print(f"final   relative L2-error {error_l2[-1]:.4e} at iteration {int(iterations[-1])} "
          f"(residual {residual[-1]:.4e})")
    if error_l2[-1] > error_l2[idx_best]:
        print(f"-> semi-convergence: the error grows by a factor "
              f"{error_l2[-1] / error_l2[idx_best]:.2f} after its minimum")
    plt.close()


if __name__ == "__main__":
    app()
