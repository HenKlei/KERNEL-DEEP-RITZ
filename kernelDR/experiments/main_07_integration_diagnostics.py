r"""Quantify the numerical integration errors present in the reported experiments.

Three quantities are computed, all of them for the same problems, domains and
quadrature rules that the other experiment scripts use:

1. **Accuracy of the reported error metrics.** The relative $L^2$- and
   $H^1$-errors are evaluated with the deterministic uniform grid of
   ``uniform_interior_points(n_error)``. Applying the same rule to the reference
   solution, whose norms are known in closed form (see below), shows how accurate
   that rule is. Since numerator and denominator of a relative error use the same
   grid, an error of this size acts as a multiplicative bias that is identical
   for all runs and therefore does not affect the observed convergence rates.

2. **Noise of the training objective.** The energy functional is approximated by
   Monte Carlo quadrature with a new point set in every epoch. Evaluating it
   repeatedly at the exact solution gives the spread of the objective that the
   optimizer sees, which is the relevant scale when loss values of different
   methods are compared.

3. **Convergence of the reported error of a trained model** (optional, requires
   ``--model-path``). The error metrics of an actual kernel approximation are
   recomputed on increasingly fine grids. In contrast to 1., this also covers the
   numerator, whose integrand oscillates on the scale of the centers. Checkpoints
   are written by main_01a/main_01b/main_04a/main_04b with ``--save-models``.

Closed-form reference values used above:

* Unit square, $u = 1 - x_1^2 - x_2^2$:
  $\|u\|_{L^2}^2 = 1 - \tfrac43 + (\tfrac15 + \tfrac29 + \tfrac15)$ and
  $|u|_{H^1}^2 = \int 4x_1^2 + 4x_2^2 = \tfrac83$.
* Circular sector of opening angle $\alpha$ and radius $R$,
  $u = r^s \sin(s\varphi) + 1$ with $s = 1/\alpha$: the gradient has modulus
  $s r^{s-1}$, hence
  $|u|_{H^1}^2 = \int_0^\alpha\!\!\int_0^R s^2 r^{2s-2}\, r \,dr\,d\varphi
  = \alpha s R^{2s} / 2 = R^{2s}/2$, and
  $\|u\|_{L^2}^2 = \big[\tfrac\alpha2 - \tfrac{\sin(2s\alpha)}{4s}\big]
  \tfrac{R^{2s+2}}{2s+2} + \tfrac{2(1-\cos(s\alpha))}{s}\tfrac{R^{s+2}}{s+2}
  + \tfrac{\alpha R^2}{2}$.
"""

from typing import Annotated

import cyclopts
import math
import numpy as np
import os
import torch

from kernelDR.models.model_kernels import load_kernel_model
from kernelDR.problem_definitions.poisson_higher_regularity import PoissonHigherRegularity
from kernelDR.problem_definitions.laplace_pacman import LaplaceOnPacmanDomainSingularSolution
from kernelDR.utils import (compute_L2_norm, compute_H1_semi_norm, compute_relative_L2_error,
                            compute_relative_H1_error, set_seed)
from kernelDR.experiments.plot_utils import save_settings


app = cyclopts.App()


def exact_norms(problem, problem_type):
    """Closed-form (L2 norm, H1 semi-norm) of the reference solution.

    See the module docstring for the derivations.
    """
    if problem_type == "smooth":
        l2_sq = 1. - 4. / 3. + (1. / 5. + 2. / 9. + 1. / 5.)
        h1_semi_sq = 8. / 3.
    else:
        alpha = problem.domain.angle
        radius = problem.domain.radius
        s = 1. / alpha
        h1_semi_sq = radius ** (2 * s) / 2.
        l2_sq = ((alpha / 2. - math.sin(2 * s * alpha) / (4 * s)) * radius ** (2 * s + 2) / (2 * s + 2)
                 + 2 * (1 - math.cos(s * alpha)) / s * radius ** (s + 2) / (s + 2)
                 + alpha * radius ** 2 / 2.)
    return math.sqrt(l2_sq), math.sqrt(h1_semi_sq)


class _ExactSolutionModel(torch.nn.Module):
    """The reference solution wrapped as a model, so that it can be passed to
    ``problem.energy`` (which differentiates its argument via autograd)."""

    def __init__(self, problem):
        super().__init__()
        self.problem = problem
        # problem.energy backpropagates through the model; a dummy parameter keeps
        # the autograd machinery happy without changing the value.
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.problem.reference_solution(x) + 0. * self.dummy


def norm_accuracy_table(problem, problem_type, list_n_error):
    """Grid quadrature of the reference norms vs. their closed-form values."""
    exact_l2, exact_h1 = exact_norms(problem, problem_type)
    rows = []
    for n in list_n_error:
        n_used = problem.domain.uniform_interior_points(n).shape[0]
        l2 = compute_L2_norm(problem.reference_solution, problem, n)
        h1 = compute_H1_semi_norm(problem.gradient_reference_solution, problem, n)
        rows.append((n, n_used, l2, abs(l2 - exact_l2) / exact_l2,
                     h1, abs(h1 - exact_h1) / exact_h1))
    return rows, exact_l2, exact_h1


def drop_singular_points(problem, x):
    """Remove points at which the gradient of the reference solution is unbounded.

    The singular solution on the pacman domain has an unbounded gradient in the
    re-entrant corner, and ``uniform_boundary_points`` contains that corner
    exactly, so the energy of the *exact* solution cannot be evaluated there.
    Random point sets hit the corner with probability zero, and the kernel models
    of the experiments have bounded gradients everywhere, so this only concerns
    the deterministic reference value computed here. Returns (points, n_dropped).
    """
    with torch.no_grad():
        gradients = problem.gradient_reference_solution(x)
        mask = torch.isfinite(gradients).all(dim=1)
        x = x[mask]
    x.requires_grad_()
    return x, int((~mask).sum().item())


def energy_spread_table(problem, list_n_i, n_b, n_repeats):
    """Spread of the Monte Carlo energy at the exact solution, per quadrature size."""
    model = _ExactSolutionModel(problem)
    rows = []
    n_dropped_total = 0
    for n_i in list_n_i:
        # Interior and boundary samples are refined by the same factor, keeping the
        # ratio of the default experiment settings (n_i = 10000, n_b = 1000). Both
        # contributions to the energy then decay at the Monte Carlo rate; scaling
        # only one of them would leave the other dominating the spread.
        n_b_scaled = max(1, int(round(n_b * n_i / list_n_i[0])))
        energies = []
        for _ in range(n_repeats):
            x_i = problem.domain.random_interior_points(n_i)
            x_b = problem.domain.random_boundary_points(n_b_scaled)
            energies.append(problem.energy(model, x_i, x_b).item())
        energies = np.asarray(energies)

        x_i, dropped_i = drop_singular_points(problem, problem.domain.uniform_interior_points(n_i))
        x_b, dropped_b = drop_singular_points(problem, problem.domain.uniform_boundary_points(n_b_scaled))
        n_dropped_total += dropped_i + dropped_b
        uniform_value = problem.energy(model, x_i, x_b).item()

        mean = float(energies.mean())
        std = float(energies.std(ddof=1))
        rows.append((n_i, n_b_scaled, mean, std, std / abs(mean), uniform_value))
    return rows, n_dropped_total


def model_error_table(problem, model, list_n_error):
    """Reported relative errors of a trained model on increasingly fine grids."""
    rows = []
    for n in list_n_error:
        n_used = problem.domain.uniform_interior_points(n).shape[0]
        error_l2 = compute_relative_L2_error(problem, model, n)
        error_h1 = compute_relative_H1_error(problem, model, n)
        rows.append((n, n_used, error_l2, error_h1))
    return rows


@app.default
def main(
    problem_type: Annotated[str, cyclopts.Parameter(help="Problem to analyse: 'smooth' (unit square) or 'singular' (pacman domain).")] = "smooth",
    penalty_parameter: Annotated[float, cyclopts.Parameter(help="Penalty parameter for boundary conditions.")] = 100.0,
    angle: Annotated[float, cyclopts.Parameter(help="Angle of the pacman domain (in radians); only used for 'singular'.")] = 4.71238898038469,
    radius: Annotated[float, cyclopts.Parameter(help="Radius of the pacman domain; only used for 'singular'.")] = 1.5,
    n_b: Annotated[int, cyclopts.Parameter(help="Number of boundary sample points at the smallest interior sample size.")] = 1000,
    n_repeats: Annotated[int, cyclopts.Parameter(help="Number of independent Monte Carlo draws used for the spread of the energy.")] = 200,
    list_n_error: Annotated[tuple[int, ...], cyclopts.Parameter(help="Error evaluation grid sizes to compare. The first value is the setting used in the experiments.")] = (10201, 40000, 160000, 640000),
    list_n_i: Annotated[tuple[int, ...], cyclopts.Parameter(help="Interior quadrature sizes for the energy spread. The first value is the setting used in the experiments.")] = (10000, 40000, 160000),
    model_path: Annotated[str, cyclopts.Parameter(help="Optional kernel model checkpoint (written with --save-models); enables the refinement study of the reported errors.")] = "",
    seed: Annotated[int, cyclopts.Parameter(help="RNG seed for the Monte Carlo draws.")] = 0,
    results_dir: Annotated[str, cyclopts.Parameter(help="Results output directory.")] = "results_integration_diagnostics/",
):
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    set_seed(seed)

    if problem_type == "smooth":
        problem = PoissonHigherRegularity(penalty_parameter=penalty_parameter, device=device)
    elif problem_type == "singular":
        problem = LaplaceOnPacmanDomainSingularSolution(angle=angle, radius=radius,
                                                        penalty_parameter=penalty_parameter, device=device)
    else:
        raise SystemExit(f"Unknown problem type {problem_type!r}; expected 'smooth' or 'singular'.")

    if not results_dir.endswith("/"):
        results_dir += "/"
    os.makedirs(results_dir, exist_ok=True)
    save_settings(results_dir, locals(), filename=f"settings_{problem_type}.txt")

    # 1. Accuracy of the grid quadrature used for the reported error metrics.
    rows, exact_l2, exact_h1 = norm_accuracy_table(problem, problem_type, list_n_error)
    path = results_dir + f"norm_accuracy_{problem_type}.txt"
    with open(path, "w") as f:
        f.write(f"# closed-form norms of the reference solution: "
                f"L2 = {exact_l2:.12e}, H1-semi = {exact_h1:.12e}\n")
        f.write("n_requested\tn_used\tL2_grid\tL2_rel_error\tH1_semi_grid\tH1_rel_error\n")
        for n, n_used, l2, l2_err, h1, h1_err in rows:
            f.write(f"{n}\t{n_used}\t{l2:.12e}\t{l2_err:.6e}\t{h1:.12e}\t{h1_err:.6e}\n")

    print(f"\n=== Accuracy of the error-evaluation grid ({problem_type} problem) ===")
    print(f"closed-form norms: ||u||_L2 = {exact_l2:.8f}, |u|_H1 = {exact_h1:.8f}")
    print(f"{'n_requested':>12} {'n_used':>10} {'L2 rel.err':>12} {'H1 rel.err':>12}")
    for n, n_used, _, l2_err, _, h1_err in rows:
        print(f"{n:>12} {n_used:>10} {l2_err:>12.3e} {h1_err:>12.3e}")
    print(f"-> written to {path}")

    # 2. Noise of the Monte Carlo objective.
    rows, n_dropped = energy_spread_table(problem, list(list_n_i), n_b, n_repeats)
    path = results_dir + f"energy_spread_{problem_type}.txt"
    with open(path, "w") as f:
        f.write(f"# energy of the exact solution, {n_repeats} independent Monte Carlo draws per row\n")
        if n_dropped:
            f.write(f"# {n_dropped} deterministic point(s) with an unbounded reference gradient "
                    f"were excluded from the uniform-grid column\n")
        f.write("n_i\tn_b\tmc_mean\tmc_std\tmc_rel_std\tuniform_grid\n")
        for n_i, n_b_scaled, mean, std, rel_std, uniform_value in rows:
            f.write(f"{n_i}\t{n_b_scaled}\t{mean:.12e}\t{std:.12e}\t{rel_std:.6e}\t{uniform_value:.12e}\n")

    print(f"\n=== Spread of the Monte Carlo energy ({problem_type} problem, {n_repeats} draws) ===")
    if n_dropped:
        print(f"excluded {n_dropped} deterministic point(s) with an unbounded reference gradient "
              f"(the re-entrant corner) from the uniform-grid column")
    print(f"{'n_i':>10} {'n_b':>8} {'mean':>14} {'std':>12} {'rel. std':>11} {'uniform grid':>14}")
    for n_i, n_b_scaled, mean, std, rel_std, uniform_value in rows:
        print(f"{n_i:>10} {n_b_scaled:>8} {mean:>14.6f} {std:>12.4f} {rel_std:>11.2e} {uniform_value:>14.6f}")
    print(f"-> written to {path}")

    # 3. Refinement study for a trained model.
    if model_path:
        model = load_kernel_model(model_path, device=device)
        rows = model_error_table(problem, model, list_n_error)
        path = results_dir + f"error_refinement_{problem_type}.txt"
        finest_l2, finest_h1 = rows[-1][2], rows[-1][3]
        with open(path, "w") as f:
            f.write(f"# relative errors of {model_path} on increasingly fine evaluation grids\n")
            f.write("n_requested\tn_used\trel_L2\trel_H1\tL2_dev_from_finest\tH1_dev_from_finest\n")
            for n, n_used, error_l2, error_h1 in rows:
                f.write(f"{n}\t{n_used}\t{error_l2:.12e}\t{error_h1:.12e}\t"
                        f"{abs(error_l2 - finest_l2) / finest_l2:.6e}\t"
                        f"{abs(error_h1 - finest_h1) / finest_h1:.6e}\n")

        print(f"\n=== Reported errors of {os.path.basename(model_path)} under grid refinement ===")
        print(f"{'n_requested':>12} {'n_used':>10} {'rel. L2':>13} {'rel. H1':>13} "
              f"{'L2 dev.':>10} {'H1 dev.':>10}")
        for n, n_used, error_l2, error_h1 in rows:
            print(f"{n:>12} {n_used:>10} {error_l2:>13.6e} {error_h1:>13.6e} "
                  f"{abs(error_l2 - finest_l2) / finest_l2:>10.2e} "
                  f"{abs(error_h1 - finest_h1) / finest_h1:>10.2e}")
        print(f"-> written to {path}")


if __name__ == "__main__":
    app()
