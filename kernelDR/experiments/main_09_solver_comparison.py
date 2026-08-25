r"""Compare the solvers on one and the same discrete problem.

The comparison in the paper contrasts a direct solve, the conjugate gradient
method and the Adam optimizer, but those runs differ in more than the solver: the
optimizer runs use the Lagrange basis, drop quadrature points that coincide with a
center, and add no Tikhonov term, while the assembled system does none of the
first two and adds one. This script removes those differences. All methods

* use the same centers and the same basis,
* see the same fixed quadrature points, and
* are measured with the same two quantities:

  - how accurately the discrete problem is solved: the discrete energy
    $\tfrac12 c^\top A c - b^\top c$ and the norm of its gradient $\|Ac-b\|_2$,
    both with respect to the system assembled from exactly those points;
  - how good the resulting approximation is: the relative $L^2$- and $H^1$-errors
    on an independent uniform grid.

The only method that does not minimise this functional is Adam with resampled
quadrature, which draws new points every epoch. Its row is therefore the
interesting one: it is expected to reach a *higher* discrete energy while giving a
*smaller* error, which is precisely the distinction between optimization
performance and generalization.

Since the basis is shared, the coefficient vectors of all methods are directly
comparable, and $\|c\|_2$ is reported as a measure of how strongly a method
exploits the near-null directions of the discrete problem.
"""

from typing import Annotated

import cyclopts
import numpy as np
import os
import scipy as sp
import scipy.sparse.linalg as spla
import time
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from kernelDR.matrix_form import assemble_system_vectorized
from kernelDR.models.model_kernels import FlatKernelModel
from kernelDR.problem_definitions.poisson_higher_regularity import PoissonHigherRegularity
from kernelDR.problem_definitions.laplace_pacman import LaplaceOnPacmanDomainSingularSolution
from kernelDR.training import train_model
from kernelDR.utils import (compute_relative_L2_error, compute_relative_H1_error,
                            drop_points_near_centers, set_seed)
from kernelDR.experiments.plot_utils import save_settings


app = cyclopts.App()


@app.default
def main(
    problem_type: Annotated[str, cyclopts.Parameter(help="Problem to analyse: 'smooth' (unit square) or 'singular' (pacman domain).")] = "smooth",
    kernel: Annotated[str, cyclopts.Parameter(help="Kernel type.")] = "matern",
    k_smoothness: Annotated[int, cyclopts.Parameter(help="Kernel smoothness parameter.")] = 2,
    ep: Annotated[float, cyclopts.Parameter(help="Kernel shape parameter.")] = 1.0,
    n_per_dim: Annotated[int, cyclopts.Parameter(help="Centers per dimension.")] = 20,
    flag_lagrange: Annotated[bool, cyclopts.Parameter(help="Use the Lagrange basis for all methods (as the optimizer runs of main_01 do) instead of the plain kernel basis. The spanned space is the same either way.")] = False,
    penalty_parameter: Annotated[float, cyclopts.Parameter(help="Penalty parameter for boundary conditions.")] = 100.0,
    angle: Annotated[float, cyclopts.Parameter(help="Angle of the pacman domain (in radians); only used for 'singular'.")] = 4.71238898038469,
    radius: Annotated[float, cyclopts.Parameter(help="Radius of the pacman domain; only used for 'singular'.")] = 1.5,
    n_i: Annotated[int, cyclopts.Parameter(help="Number of interior quadrature points.")] = 10000,
    n_b: Annotated[int, cyclopts.Parameter(help="Number of boundary quadrature points.")] = 1000,
    n_error: Annotated[int, cyclopts.Parameter(help="Number of error evaluation points.")] = 640000,
    use_uniform_quadrature: Annotated[bool, cyclopts.Parameter(help="Use a uniform tensor-product grid as the common fixed quadrature instead of a fixed random sample.")] = True,
    list_regularization: Annotated[tuple[float, ...], cyclopts.Parameter(help="Tikhonov parameters for the direct solves. The default covers the value used in the paper and the one that minimises the error according to main_08.")] = (1e-10, 1e-5),
    solver: Annotated[str, cyclopts.Parameter(help="Solver type for scipy.linalg.solve; falls back to 'sym' if the factorisation fails.")] = "pos",
    cg_maxiter: Annotated[int, cyclopts.Parameter(help="Maximum CG iterations.")] = 10000,
    cg_rtol: Annotated[float, cyclopts.Parameter(help="CG relative tolerance.")] = 1e-12,
    cg_error_interval: Annotated[int, cyclopts.Parameter(help="Evaluate the errors of the CG iterate every this many iterations, to also report the best iterate.")] = 10,
    n_epochs: Annotated[int, cyclopts.Parameter(help="Number of Adam epochs.")] = 10000,
    lr: Annotated[float, cyclopts.Parameter(help="Initial learning rate for Adam.")] = 5e-2,
    gamma: Annotated[float, cyclopts.Parameter(help="Learning rate decay factor for Adam.")] = 0.5,
    num_logs: Annotated[int, cyclopts.Parameter(help="Number of log points during the Adam runs.")] = 40,
    skip_adam: Annotated[bool, cyclopts.Parameter(help="Only compare the matrix-based solvers (the Adam runs dominate the runtime).")] = False,
    chunk_size: Annotated[int, cyclopts.Parameter(help="Chunk size for the vectorized assembly (0 processes all points at once).")] = 0,
    seed: Annotated[int, cyclopts.Parameter(help="RNG seed for the quadrature points.")] = 0,
    results_dir: Annotated[str, cyclopts.Parameter(help="Results output directory.")] = "results_solver_comparison/",
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
    save_settings(results_dir, locals(), filename=f"settings_{problem_type}_k{k_smoothness}_n{n_per_dim}.txt")

    # Centers, as in all other experiments.
    centers_inner = problem.domain.uniform_interior_points(n_per_dim ** 2)
    centers_boundary = problem.domain.uniform_boundary_points(4 * (n_per_dim + 2))
    centers = torch.vstack([centers_inner, centers_boundary]).detach()
    centers.requires_grad_()

    model_params = {"str_kernel": kernel, "k_smoothness": k_smoothness, "ctrs": centers,
                    "ep": ep, "flag_lagrange": flag_lagrange}
    num_coeffs = len(FlatKernelModel(problem.domain.dim, problem.output_dim, **model_params).coeffs)

    # The one quadrature rule that every method sees. Points coinciding with a
    # center are dropped once, here, so that the assembled system and the optimizer
    # runs discretise exactly the same functional -- the energy has no second
    # derivative with respect to the coefficients at such points, see
    # drop_points_near_centers.
    if use_uniform_quadrature:
        x_i = problem.domain.uniform_interior_points(n_i)
        x_b = problem.domain.uniform_boundary_points(n_b)
    else:
        x_i = problem.domain.random_interior_points(n_i)
        x_b = problem.domain.random_boundary_points(n_b)

    n_i_raw, n_b_raw = x_i.shape[0], x_b.shape[0]
    x_i = drop_points_near_centers(x_i, centers)
    x_b = drop_points_near_centers(x_b, centers)
    dropped = (n_i_raw - x_i.shape[0]) + (n_b_raw - x_b.shape[0])

    print(f"{num_coeffs} unknowns, common quadrature: {x_i.shape[0]} interior + {x_b.shape[0]} boundary "
          f"points ({'uniform grid' if use_uniform_quadrature else 'fixed random sample'}), "
          f"{'Lagrange' if flag_lagrange else 'plain kernel'} basis")
    if dropped:
        print(f"  dropped {dropped} quadrature point(s) coinciding with a center")

    model = FlatKernelModel(problem.domain.dim, problem.output_dim, **model_params)
    A, b = assemble_system_vectorized(problem, model, x_i, x_b, num_coeffs, chunk_size=chunk_size)
    print(f"assembled: ||A||_2 = {np.linalg.norm(A, 2):.4e}, "
          f"condition number = {np.linalg.cond(A):.4e}")

    def model_from_coeffs(coeffs):
        new_model = FlatKernelModel(problem.domain.dim, problem.output_dim, **model_params)
        new_model.coeffs = nn.Parameter(torch.tensor(np.asarray(coeffs)).to(centers.device))
        return new_model

    def evaluate(label, setting, model_to_check, elapsed):
        """The two sides of the comparison for one method."""
        coeffs = model_to_check.coeffs.detach().cpu().numpy().reshape(-1)
        # Discrete energy of the common functional, evaluated directly from the
        # energy functional rather than from A and b (the two agree, see below).
        energy = problem.energy(model_to_check, x_i, x_b).item()
        return {
            "label": label,
            "setting": setting,
            "energy": energy,
            "energy_from_system": float(0.5 * coeffs @ (A @ coeffs) - b @ coeffs),
            "residual": float(np.linalg.norm(A @ coeffs - b)),
            "coeff_norm": float(np.linalg.norm(coeffs)),
            "l2": compute_relative_L2_error(problem, model_to_check, n_error),
            "h1": compute_relative_H1_error(problem, model_to_check, n_error),
            "time_s": elapsed,
        }

    rows = []

    # 1. Direct solve (Cholesky, or an indefinite factorisation if A is not
    #    numerically positive definite) for each Tikhonov parameter.
    for regularization in list_regularization:
        matrix = A + regularization * np.eye(num_coeffs)
        t_start = time.time()
        try:
            coeffs = sp.linalg.solve(matrix, b, assume_a=solver)
            used = solver
        except (np.linalg.LinAlgError, sp.linalg.LinAlgError, ValueError):
            coeffs = sp.linalg.solve(matrix, b, assume_a="sym")
            used = "sym (fallback)"
        elapsed = time.time() - t_start
        rows.append(evaluate("direct solve", f"lambda={regularization:.0e}, {used}",
                             model_from_coeffs(coeffs), elapsed))

    # 2. Conjugate gradient on the same system, without any regularisation.
    #    Both the final and the best iterate are reported, since the truncation of
    #    CG acts as a regularisation of its own.
    best = {"l2": np.inf, "coeffs": None, "iteration": 0}
    state = {"iteration": 0}

    def cg_callback(xk):
        state["iteration"] += 1
        if cg_error_interval > 0 and state["iteration"] % cg_error_interval == 0:
            error_l2 = compute_relative_L2_error(problem, model_from_coeffs(xk), n_error)
            if error_l2 < best["l2"]:
                best.update(l2=error_l2, coeffs=np.array(xk), iteration=state["iteration"])

    t_start = time.time()
    coeffs_cg, info = spla.cg(A, b, x0=np.zeros_like(b), rtol=cg_rtol, maxiter=cg_maxiter,
                              callback=cg_callback)
    elapsed = time.time() - t_start
    rows.append(evaluate("CG", f"{state['iteration']} iterations, info={info}",
                         model_from_coeffs(coeffs_cg), elapsed))
    if best["coeffs"] is not None and best["iteration"] != state["iteration"]:
        rows.append(evaluate("CG (best iterate)", f"stopped after {best['iteration']} iterations",
                             model_from_coeffs(best["coeffs"]), float("nan")))

    # 3./4. Adam on exactly the same quadrature points, and Adam with a new point
    #       set in every epoch. The fixed-quadrature run keeps the common point set
    #       unchanged, so that it minimises exactly the assembled functional; those
    #       points were already filtered once above. The resampled run draws new
    #       points in every epoch, which have to be filtered again each time -- a
    #       single draw landing on a center makes the energy undefined there and
    #       takes the whole run to NaN.
    if not skip_adam:
        for label, fixed in (("Adam (fixed quadrature)", True), ("Adam (random quadrature)", False)):
            adam_model = FlatKernelModel(problem.domain.dim, problem.output_dim, **model_params)
            optimizer = optim.Adam(adam_model.parameters(), lr=lr)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=200, min_lr=1e-6)
            set_seed(seed)
            t_start = time.time()
            train_model(problem, adam_model, n_i, n_b, n_epochs, optimizer, centers,
                        scheduler=scheduler, fixed_integration_points=fixed,
                        flag_best_model=False, num_logs=num_logs, n_error=n_error,
                        remove_center_points=not fixed,
                        fixed_x_i=x_i if fixed else None,
                        fixed_x_b=x_b if fixed else None)
            elapsed = time.time() - t_start
            rows.append(evaluate(label, f"{n_epochs} epochs, lr={lr}", adam_model, elapsed))

    # Consistency check: the energy functional evaluated on the common quadrature
    # must agree with the quadratic form of the assembled system.
    deviations = [abs(row["energy"] - row["energy_from_system"]) / max(abs(row["energy"]), 1e-30)
                  for row in rows]
    print(f"\nenergy functional vs. assembled quadratic form: "
          f"largest relative deviation {max(deviations):.2e}")

    path = results_dir + f"solver_comparison_{problem_type}_k{k_smoothness}_n{n_per_dim}.txt"
    with open(path, "w") as f:
        f.write(f"# {num_coeffs} unknowns, {x_i.shape[0]} interior + {x_b.shape[0]} boundary "
                f"quadrature points, {'uniform grid' if use_uniform_quadrature else 'fixed random sample'}, "
                f"{'Lagrange' if flag_lagrange else 'plain kernel'} basis\n")
        f.write("method\tsetting\tenergy\tresidual\tcoeff_norm\tL2-error\tH1-error\ttime_s\n")
        for row in rows:
            f.write(f"{row['label']}\t{row['setting']}\t{row['energy']:.6f}\t{row['residual']:.6e}\t"
                    f"{row['coeff_norm']:.6e}\t{row['l2']:.6e}\t{row['h1']:.6e}\t{row['time_s']:.2f}\n")

    print(f"\n{'method':<26}{'energy':>12}{'||Ac-b||':>12}{'||c||':>12}{'rel. L2':>12}{'rel. H1':>12}")
    for row in rows:
        print(f"{row['label']:<26}{row['energy']:>12.4f}{row['residual']:>12.2e}"
              f"{row['coeff_norm']:>12.2e}{row['l2']:>12.3e}{row['h1']:>12.3e}")
    print(f"-> written to {path}")

    minimizes = [row for row in rows if not row["label"].startswith("Adam (random")]
    if minimizes:
        lowest = min(minimizes, key=lambda row: row["energy"])
        best_error = min(rows, key=lambda row: row["l2"])
        print(f"\nlowest energy on the common functional: {lowest['label']} "
              f"({lowest['energy']:.4f}, rel. L2 {lowest['l2']:.3e})")
        print(f"smallest error:                         {best_error['label']} "
              f"({best_error['energy']:.4f}, rel. L2 {best_error['l2']:.3e})")


if __name__ == "__main__":
    app()
