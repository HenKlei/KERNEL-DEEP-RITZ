from datetime import datetime
from typing import Annotated

import cyclopts
import numpy as np
import scipy.sparse.linalg as spla
import time
import torch
import torch.nn as nn
import os

from kernelDR.matrix_form import assemble_and_solve_system
from kernelDR.models.model_kernels import FlatKernelModel, save_kernel_model
from kernelDR.problem_definitions.laplace_pacman import LaplaceOnPacmanDomainSingularSolution
from kernelDR.utils import (compute_relative_L2_error, compute_relative_H1_error, peak_memory_mb,
                            reset_peak_memory, set_seed)
from kernelDR.experiments.plot_utils import save_settings


app = cyclopts.App()


@app.default
def main(
    kernel: Annotated[str, cyclopts.Parameter(help="Kernel type.")] = "matern",
    ep: Annotated[float, cyclopts.Parameter(help="Kernel shape parameter.")] = 1.0,
    penalty_parameter: Annotated[float, cyclopts.Parameter(help="Penalty parameter for boundary conditions.")] = 100.0,
    n_i: Annotated[int, cyclopts.Parameter(help="Number of interior sample points.")] = 10000,
    n_b: Annotated[int, cyclopts.Parameter(help="Number of boundary sample points.")] = 1000,
    n_error: Annotated[int, cyclopts.Parameter(help="Number of error evaluation points. The default resolves the approximation error, which oscillates on the scale of the centers; a coarser grid understates the H1-errors and biases fitted convergence rates (see the README).")] = 640000,
    regularization: Annotated[float, cyclopts.Parameter(help="Tikhonov regularization added to the diagonal of the kernel matrix; needed to stabilise Cholesky/CG for higher kernel smoothness.")] = 1e-10,
    solver: Annotated[str, cyclopts.Parameter(help="Solver type for scipy.linalg.solve (e.g. 'pos', 'gen', 'sym').")] = "pos",
    linear_solver: Annotated[str, cyclopts.Parameter(help="Linear solver to use after assembly: 'direct' (Cholesky/LU via scipy.linalg.solve) or 'cg' (conjugate gradient, with per-iteration energy logging).")] = "direct",
    use_uniform_quadrature: Annotated[bool, cyclopts.Parameter(help="If True, assemble the linear system from a uniform tensor-product grid (matching --fixed-integration-points in main_01); default uses random samples.")] = False,
    flag_lagrange: Annotated[bool, cyclopts.Parameter(help="Use the Lagrange basis of the kernel space instead of the plain kernel basis. The spanned space is the same, but the resulting system is far better conditioned; the optimizer runs of main_01 use it, the assembled system does not by default.")] = False,
    assembly: Annotated[str, cyclopts.Parameter(help="Assembly strategy: 'vectorized' (exploits the linearity in the coefficients) or 'loop' (original entry-by-entry version, orders of magnitude slower, kept for validation).")] = "vectorized",
    chunk_size: Annotated[int, cyclopts.Parameter(help="Process the quadrature points in chunks of this size during the vectorized assembly, to bound the memory of the (n_points, n_centers, dim) gradient tensor. 0 processes all points at once.")] = 0,
    cg_maxiter: Annotated[int, cyclopts.Parameter(help="Maximum CG iterations.")] = 10000,
    cg_rtol: Annotated[float, cyclopts.Parameter(help="CG relative tolerance.")] = 1e-12,
    cg_error_interval: Annotated[int, cyclopts.Parameter(help="Evaluate the relative L2-/H1-errors every this many CG iterations (0 to disable). Written to cg_convergence_k_*_n_*.txt, which shows whether the errors grow again while the residual keeps decreasing.")] = 10,
    angle: Annotated[float, cyclopts.Parameter(help="Angle of the pacman domain (in radians).")] = 4.71238898038469,
    radius: Annotated[float, cyclopts.Parameter(help="Radius of the pacman domain.")] = 1.5,
    seed: Annotated[int, cyclopts.Parameter(help="RNG seed for the quadrature points used to assemble the system (no effect with --use-uniform-quadrature). Use a separate --results-dir per seed when repeating a run.")] = 0,
    save_models: Annotated[bool, cyclopts.Parameter(help="Save the solution of every configuration, so that it can be re-evaluated later (see main_07_integration_diagnostics.py).")] = False,
    list_kmat: Annotated[tuple[int, ...], cyclopts.Parameter(help="Kernel smoothness values.")] = (0, 1, 2),
    list_n_per_dim: Annotated[tuple[int, ...], cyclopts.Parameter(help="Centers per dimension.")] = (1, 2, 4, 8, 12, 16, 20),
    results_dir: Annotated[str, cyclopts.Parameter(help="Results output directory.")] = "results_matrix_form_singular_solution/",
):
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    set_seed(seed)

    problem = LaplaceOnPacmanDomainSingularSolution(angle=angle, radius=radius,
                                                    penalty_parameter=penalty_parameter, device=device)

    if not results_dir.endswith("/"):
        results_dir += "/"
    os.makedirs(results_dir, exist_ok=True)
    save_settings(results_dir, locals())

    h_values = 1. / (np.array(list_n_per_dim) + 1)

    for idx_kmat, kmat in enumerate(list_kmat):
        with open(results_dir + "errors_k_" + str(kmat) + ".txt", "w") as f:
            f.write("n\th ~ 1 / sqrt(n)\tn_centers\tL2-error\tH1-error\ttime_solve_s\tpeak_mem_mb\n")
        with open(results_dir + "condition_numbers_k_" + str(kmat) + ".txt", "w") as f:
            f.write("n\th ~ 1 / sqrt(n)\tCondition number\n")
        for idx_n, n_per_dim in enumerate(list_n_per_dim):
            print(datetime.now().strftime("%H:%M:%S"), kmat, n_per_dim)
            # Use centers both in the interior and the boundary --> this seems to improve accuracy
            centers_inner = problem.domain.uniform_interior_points(n_per_dim**2)
            # (n+2) due to having the boundary corner points always
            centers_boundary = problem.domain.uniform_boundary_points(4 * (n_per_dim + 2))

            # Remove interior centers too close to the boundary (avoids near-duplicate
            # centers that make the kernel Gram matrix singular).
            array_dist = torch.cdist(centers_inner, centers_boundary)
            mask = torch.min(array_dist, dim=1).values > 1/10 * (2 * problem.domain.radius / n_per_dim)
            centers_inner = centers_inner[mask, :]

            centers = torch.vstack([centers_inner, centers_boundary]).detach()
            centers.requires_grad_()

            save_mat_path = results_dir + f"system_k_{kmat}_n_{n_per_dim}/"
            os.makedirs(save_mat_path, exist_ok=True)

            model_params = {"str_kernel": kernel, "k_smoothness": kmat, "ctrs": centers, "ep": ep, "flag_lagrange": flag_lagrange}
            reset_peak_memory()
            t_start = time.time()
            model, A, b = assemble_and_solve_system(problem, FlatKernelModel, n_i, n_b, model_params=model_params,
                                                    return_linear_system=True, regularization=regularization,
                                                    save_mat_path=save_mat_path, solver=solver,
                                                    use_uniform_quadrature=use_uniform_quadrature,
                                                    assembly=assembly, chunk_size=chunk_size)
            elapsed_time = time.time() - t_start
            run_peak_mem_mb = peak_memory_mb()

            if linear_solver == "cg":
                # Re-solve via conjugate gradient on the same (regularised) system,
                # logging the quadratic Dirichlet energy 0.5 c^T A c - b^T c and the
                # residual norm ||A c - b||_2 at each iteration.
                #
                # CG operates on the assembled system, so its quadrature is fixed by
                # construction. Tracking the relative errors alongside the energy
                # separates how well the discrete problem is solved (energy, residual)
                # from how good the resulting approximation is (L2-, H1-error): if the
                # errors start to grow again while the residual keeps decreasing, the
                # truncation of CG acts as a regularisation of an ill-posed discrete
                # problem, rather than CG being an inferior solver.
                energy_log = []
                residual_log = []
                error_log = []
                c_A = A
                c_b = b
                x_curr = np.zeros_like(b)
                energy_log.append(0.5 * x_curr @ (c_A @ x_curr) - c_b @ x_curr)
                residual_log.append(float(np.linalg.norm(c_A @ x_curr - c_b)))

                def evaluate_errors(iteration, coeffs):
                    model.coeffs = nn.Parameter(torch.tensor(coeffs).to(model.ctrs.device))
                    error_log.append((iteration,
                                      float(0.5 * coeffs @ (c_A @ coeffs) - c_b @ coeffs),
                                      float(np.linalg.norm(c_A @ coeffs - c_b)),
                                      compute_relative_L2_error(problem, model, n_error),
                                      compute_relative_H1_error(problem, model, n_error)))

                if cg_error_interval > 0:
                    evaluate_errors(0, x_curr)

                def cg_callback(xk):
                    energy_log.append(float(0.5 * xk @ (c_A @ xk) - c_b @ xk))
                    residual_log.append(float(np.linalg.norm(c_A @ xk - c_b)))
                    iteration = len(energy_log) - 1
                    if cg_error_interval > 0 and iteration % cg_error_interval == 0:
                        evaluate_errors(iteration, xk)

                t_cg_start = time.time()
                c_cg, info = spla.cg(c_A, c_b, x0=np.zeros_like(b),
                                     rtol=cg_rtol, maxiter=cg_maxiter,
                                     callback=cg_callback)
                cg_time = time.time() - t_cg_start
                model.coeffs = nn.Parameter(torch.tensor(c_cg).to(model.ctrs.device))
                elapsed_time = cg_time
                cg_log_path = results_dir + f"cg_energy_k_{kmat}_n_{n_per_dim}.txt"
                with open(cg_log_path, "w") as f:
                    f.write("iteration\tenergy\tresidual_norm\n")
                    for i, (e, r) in enumerate(zip(energy_log, residual_log)):
                        f.write(f"{i}\t{e:.12e}\t{r:.6e}\n")
                print(f"  CG: {len(energy_log) - 1} iterations, info={info}, "
                      f"final energy={energy_log[-1]:.4e}, "
                      f"final residual={residual_log[-1]:.4e}, log saved to {cg_log_path}")

                if error_log:
                    # Final iterate, in case it is not a multiple of the interval.
                    if error_log[-1][0] != len(energy_log) - 1:
                        evaluate_errors(len(energy_log) - 1, c_cg)
                        model.coeffs = nn.Parameter(torch.tensor(c_cg).to(model.ctrs.device))
                    cg_error_path = results_dir + f"cg_convergence_k_{kmat}_n_{n_per_dim}.txt"
                    with open(cg_error_path, "w") as f:
                        f.write("iteration\tenergy\tresidual_norm\tL2-error\tH1-error\n")
                        for it, e, r, error_l2, error_h1 in error_log:
                            f.write(f"{it}\t{e:.12e}\t{r:.6e}\t{error_l2:.6e}\t{error_h1:.6e}\n")
                    best = min(error_log, key=lambda row: row[3])
                    print(f"  CG errors logged every {cg_error_interval} iteration(s) to {cg_error_path}")
                    print(f"    smallest L2-error {best[3]:.4e} at iteration {best[0]} "
                          f"(residual {best[2]:.4e}); final L2-error {error_log[-1][3]:.4e} "
                          f"at iteration {error_log[-1][0]}")

            if save_models:
                save_kernel_model(model, save_mat_path + "model.pt")
                print(f"  Saved solution to {save_mat_path}model.pt")

            condition_num = np.linalg.cond(A)

            error_L2 = compute_relative_L2_error(problem, model, n_error)
            error_H1 = compute_relative_H1_error(problem, model, n_error)

            with open(results_dir + "errors_k_" + str(kmat) + ".txt", "a") as f:
                f.write(f"{n_per_dim}\t{h_values[idx_n]}\t{centers.shape[0]}\t"
                        f"{error_L2}\t{error_H1}\t{elapsed_time:.4f}\t{run_peak_mem_mb:.2f}\n")
            with open(results_dir + "condition_numbers_k_" + str(kmat) + ".txt", "a") as f:
                f.write(f"{n_per_dim}\t{h_values[idx_n]}\t{condition_num}\n")


if __name__ == "__main__":
    app()
