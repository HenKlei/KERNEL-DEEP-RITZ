r"""Separate optimization performance from generalization for the assembled system.

The assembled system is solved for a range of Tikhonov parameters $\lambda$, and
for each solution both

* how well the discrete problem is solved -- the discrete energy
  $\tfrac12 c^\top A c - b^\top c$ of the *unregularised* system and the residual
  $\|Ac-b\|_2$ -- and
* how good the resulting approximation is -- the relative $L^2$- and $H^1$-errors
  on an independent uniform grid

are recorded. Decreasing $\lambda$ necessarily decreases the discrete energy, since
the unregularised problem is what is being minimised more and more accurately. If
the errors have their minimum at some $\lambda^\ast > 0$ instead, then the discrete
problem is ill-posed and a more accurate solve gives a worse approximation, which
is a statement about regularisation rather than about solver quality.

The coefficient norm $\|c\|_2$ is recorded as well: overfitting the quadrature
points shows up as coefficients of very large magnitude that cancel almost
exactly at those points while the approximation oscillates elsewhere.

The sweep either assembles the system itself or reloads one written earlier by
main_04a/main_04b via ``--system-dir`` (the ``original_A.npy`` / ``b.npy`` files in
the ``system_k_<k>_n_<n>/`` subdirectory), which keeps repeated sweeps cheap.
"""

from typing import Annotated

import cyclopts
import numpy as np
import os
import scipy as sp
import torch
import torch.nn as nn

from kernelDR.matrix_form import assemble_system_vectorized
from kernelDR.models.model_kernels import FlatKernelModel
from kernelDR.problem_definitions.poisson_higher_regularity import PoissonHigherRegularity
from kernelDR.problem_definitions.laplace_pacman import LaplaceOnPacmanDomainSingularSolution
from kernelDR.utils import compute_relative_L2_error, compute_relative_H1_error, set_seed
from kernelDR.experiments.plot_utils import save_settings


app = cyclopts.App()


def build_centers(problem, n_per_dim):
    """Interior and boundary centers, as used by all other experiments."""
    centers_inner = problem.domain.uniform_interior_points(n_per_dim ** 2)
    # (n+2) due to having the boundary corner points always
    centers_boundary = problem.domain.uniform_boundary_points(4 * (n_per_dim + 2))
    centers = torch.vstack([centers_inner, centers_boundary]).detach()
    centers.requires_grad_()
    return centers


@app.default
def main(
    problem_type: Annotated[str, cyclopts.Parameter(help="Problem to analyse: 'smooth' (unit square) or 'singular' (pacman domain).")] = "smooth",
    kernel: Annotated[str, cyclopts.Parameter(help="Kernel type.")] = "matern",
    k_smoothness: Annotated[int, cyclopts.Parameter(help="Kernel smoothness parameter.")] = 2,
    ep: Annotated[float, cyclopts.Parameter(help="Kernel shape parameter.")] = 1.0,
    n_per_dim: Annotated[int, cyclopts.Parameter(help="Centers per dimension.")] = 20,
    penalty_parameter: Annotated[float, cyclopts.Parameter(help="Penalty parameter for boundary conditions.")] = 100.0,
    angle: Annotated[float, cyclopts.Parameter(help="Angle of the pacman domain (in radians); only used for 'singular'.")] = 4.71238898038469,
    radius: Annotated[float, cyclopts.Parameter(help="Radius of the pacman domain; only used for 'singular'.")] = 1.5,
    n_i: Annotated[int, cyclopts.Parameter(help="Number of interior quadrature points.")] = 10000,
    n_b: Annotated[int, cyclopts.Parameter(help="Number of boundary quadrature points.")] = 1000,
    n_error: Annotated[int, cyclopts.Parameter(help="Number of error evaluation points.")] = 640000,
    use_uniform_quadrature: Annotated[bool, cyclopts.Parameter(help="Assemble from a uniform tensor-product grid instead of random samples (matches the CG and Adam-fixed runs).")] = False,
    solver: Annotated[str, cyclopts.Parameter(help="Solver type for scipy.linalg.solve; falls back to 'sym' if the factorisation fails.")] = "pos",
    system_dir: Annotated[str, cyclopts.Parameter(help="Reuse a system assembled earlier: directory containing original_A.npy and b.npy. Assembles a new system if empty.")] = "",
    list_regularization: Annotated[tuple[float, ...], cyclopts.Parameter(help="Tikhonov parameters to sweep. The default spans no regularisation up to heavy regularisation.")] = (
        0.0, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7,
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0),
    chunk_size: Annotated[int, cyclopts.Parameter(help="Chunk size for the vectorized assembly (0 processes all points at once).")] = 0,
    seed: Annotated[int, cyclopts.Parameter(help="RNG seed for the quadrature points.")] = 0,
    results_dir: Annotated[str, cyclopts.Parameter(help="Results output directory.")] = "results_regularization_sweep/",
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

    centers = build_centers(problem, n_per_dim)
    model_params = {"str_kernel": kernel, "k_smoothness": k_smoothness, "ctrs": centers,
                    "ep": ep, "flag_lagrange": False}
    num_coeffs = len(FlatKernelModel(problem.domain.dim, problem.output_dim, **model_params).coeffs)

    if system_dir:
        if not system_dir.endswith("/"):
            system_dir += "/"
        A = np.load(system_dir + "original_A.npy")
        b = np.load(system_dir + "b.npy")
        print(f"Loaded the assembled system from {system_dir} ({A.shape[0]} unknowns)")
        if A.shape[0] != num_coeffs:
            raise SystemExit(f"The loaded system has {A.shape[0]} unknowns but the given centers give "
                             f"{num_coeffs}; check --n-per-dim.")
    else:
        if use_uniform_quadrature:
            x_i = problem.domain.uniform_interior_points(n_i)
            x_b = problem.domain.uniform_boundary_points(n_b)
        else:
            x_i = problem.domain.random_interior_points(n_i)
            x_b = problem.domain.random_boundary_points(n_b)
        model = FlatKernelModel(problem.domain.dim, problem.output_dim, **model_params)
        A, b = assemble_system_vectorized(problem, model, x_i, x_b, num_coeffs, chunk_size=chunk_size)
        print(f"Assembled the system ({num_coeffs} unknowns, "
              f"{'uniform' if use_uniform_quadrature else 'random'} quadrature)")

    norm_A = np.linalg.norm(A, 2)
    condition_number = np.linalg.cond(A)
    print(f"||A||_2 = {norm_A:.4e}, condition number = {condition_number:.4e}")

    rows = []
    for regularization in list_regularization:
        matrix = A + regularization * np.eye(num_coeffs)
        try:
            coeffs = sp.linalg.solve(matrix, b, assume_a=solver)
            used_solver = solver
        except (np.linalg.LinAlgError, sp.linalg.LinAlgError, ValueError):
            coeffs = sp.linalg.solve(matrix, b, assume_a="sym")
            used_solver = "sym"

        model = FlatKernelModel(problem.domain.dim, problem.output_dim, **model_params)
        model.coeffs = nn.Parameter(torch.tensor(coeffs).to(centers.device))

        # Energy and residual of the *unregularised* system: the objective that is
        # being minimised, so that lowering lambda can only improve them.
        energy = float(0.5 * coeffs @ (A @ coeffs) - b @ coeffs)
        residual = float(np.linalg.norm(A @ coeffs - b))
        error_l2 = compute_relative_L2_error(problem, model, n_error)
        error_h1 = compute_relative_H1_error(problem, model, n_error)

        rows.append((regularization, regularization / norm_A, energy, residual,
                     float(np.linalg.norm(coeffs)), error_l2, error_h1))
        print(f"  lambda {regularization:9.1e}  energy {energy:14.6f}  residual {residual:10.3e}  "
              f"||c|| {np.linalg.norm(coeffs):10.3e}  L2 {error_l2:10.4e}  H1 {error_h1:10.4e}"
              f"{'' if used_solver == solver else '  (fallback solver)'}")

    # The quadrature size is part of the name, so that a sweep over n_i does not
    # overwrite its own results.
    quadrature_tag = "uniform" if use_uniform_quadrature else "random"
    path = (results_dir + f"regularization_sweep_{problem_type}_k{k_smoothness}_n{n_per_dim}"
                          f"_{quadrature_tag}_ni{n_i}.txt")
    with open(path, "w") as f:
        f.write(f"# ||A||_2 = {norm_A:.12e}, condition number = {condition_number:.12e}, "
                f"{num_coeffs} unknowns, "
                f"{'uniform' if use_uniform_quadrature else 'random'} quadrature\n")
        f.write("lambda\tlambda_rel\tenergy\tresidual\tcoeff_norm\tL2-error\tH1-error\n")
        for regularization, rel, energy, residual, coeff_norm, error_l2, error_h1 in rows:
            f.write(f"{regularization:.6e}\t{rel:.6e}\t{energy:.12e}\t{residual:.6e}\t"
                    f"{coeff_norm:.6e}\t{error_l2:.6e}\t{error_h1:.6e}\n")
    print(f"-> written to {path}")

    best_l2 = min(rows, key=lambda row: row[5])
    lowest_energy = min(rows, key=lambda row: row[2])
    print(f"\nsmallest L2-error  {best_l2[5]:.4e} at lambda = {best_l2[0]:.1e} "
          f"(energy {best_l2[2]:.6f})")
    print(f"lowest energy      {lowest_energy[2]:.6f} at lambda = {lowest_energy[0]:.1e} "
          f"(L2-error {lowest_energy[5]:.4e})")
    if best_l2[0] != lowest_energy[0]:
        print(f"-> the most accurate solution of the discrete problem is not the best approximation: "
              f"the L2-error is a factor {lowest_energy[5] / best_l2[5]:.2f} larger there")


if __name__ == "__main__":
    app()
