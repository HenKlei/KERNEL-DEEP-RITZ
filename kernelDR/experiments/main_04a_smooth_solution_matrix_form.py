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
from kernelDR.models.model_kernels import FlatKernelModel
from kernelDR.problem_definitions.poisson_higher_regularity import PoissonHigherRegularity
from kernelDR.utils import compute_relative_L2_error, compute_relative_H1_error, peak_memory_mb, reset_peak_memory
from kernelDR.experiments.plot_utils import save_settings


app = cyclopts.App()


@app.default
def main(
    kernel: Annotated[str, cyclopts.Parameter(help="Kernel type.")] = "matern",
    ep: Annotated[float, cyclopts.Parameter(help="Kernel shape parameter.")] = 1.0,
    penalty_parameter: Annotated[float, cyclopts.Parameter(help="Penalty parameter for boundary conditions.")] = 100.0,
    n_i: Annotated[int, cyclopts.Parameter(help="Number of interior sample points.")] = 10000,
    n_b: Annotated[int, cyclopts.Parameter(help="Number of boundary sample points.")] = 1000,
    n_error: Annotated[int, cyclopts.Parameter(help="Number of error evaluation points.")] = 10201,
    regularization: Annotated[float, cyclopts.Parameter(help="Tikhonov regularization added to the diagonal of the kernel matrix; needed to stabilise Cholesky/CG for higher kernel smoothness.")] = 1e-10,
    solver: Annotated[str, cyclopts.Parameter(help="Solver type for scipy.linalg.solve (e.g. 'pos', 'gen', 'sym').")] = "pos",
    linear_solver: Annotated[str, cyclopts.Parameter(help="Linear solver to use after assembly: 'direct' (Cholesky/LU via scipy.linalg.solve) or 'cg' (conjugate gradient, with per-iteration energy logging).")] = "direct",
    use_uniform_quadrature: Annotated[bool, cyclopts.Parameter(help="If True, assemble the linear system from a uniform tensor-product grid (matching --fixed-integration-points in main_01); default uses random samples.")] = False,
    cg_maxiter: Annotated[int, cyclopts.Parameter(help="Maximum CG iterations.")] = 10000,
    cg_rtol: Annotated[float, cyclopts.Parameter(help="CG relative tolerance.")] = 1e-12,
    list_kmat: Annotated[tuple[int, ...], cyclopts.Parameter(help="Kernel smoothness values.")] = (0, 1, 2),
    list_n_per_dim: Annotated[tuple[int, ...], cyclopts.Parameter(help="Centers per dimension.")] = (1, 2, 4, 8, 12, 16, 20),
    results_dir: Annotated[str, cyclopts.Parameter(help="Results output directory.")] = "results_matrix_form_smooth_solution/",
):
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    problem = PoissonHigherRegularity(penalty_parameter=penalty_parameter, device=device)

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

            centers = torch.vstack([centers_inner, centers_boundary]).detach()
            centers.requires_grad_()

            save_mat_path = results_dir + f"system_k_{kmat}_n_{n_per_dim}/"
            os.makedirs(save_mat_path, exist_ok=True)

            model_params = {"str_kernel": kernel, "k_smoothness": kmat, "ctrs": centers, "ep": ep, "flag_lagrange": False}
            reset_peak_memory()
            t_start = time.time()
            model, A, b = assemble_and_solve_system(problem, FlatKernelModel, n_i, n_b, model_params=model_params,
                                                    return_linear_system=True, regularization=regularization,
                                                    save_mat_path=save_mat_path, solver=solver,
                                                    use_uniform_quadrature=use_uniform_quadrature)
            elapsed_time = time.time() - t_start
            run_peak_mem_mb = peak_memory_mb()

            if linear_solver == "cg":
                # Re-solve via conjugate gradient on the same (regularised) system,
                # logging the quadratic Dirichlet energy 0.5 c^T A c - b^T c and the
                # residual norm ||A c - b||_2 at each iteration.
                energy_log = []
                residual_log = []
                c_A = A
                c_b = b
                x_curr = np.zeros_like(b)
                energy_log.append(0.5 * x_curr @ (c_A @ x_curr) - c_b @ x_curr)
                residual_log.append(float(np.linalg.norm(c_A @ x_curr - c_b)))

                def cg_callback(xk):
                    energy_log.append(float(0.5 * xk @ (c_A @ xk) - c_b @ xk))
                    residual_log.append(float(np.linalg.norm(c_A @ xk - c_b)))

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
                        f.write(f"{i}\t{e:.6e}\t{r:.6e}\n")
                print(f"  CG: {len(energy_log) - 1} iterations, info={info}, "
                      f"final energy={energy_log[-1]:.4e}, "
                      f"final residual={residual_log[-1]:.4e}, log saved to {cg_log_path}")

            cond = np.linalg.cond(A)
            error_L2 = compute_relative_L2_error(problem, model, n_error)
            error_H1 = compute_relative_H1_error(problem, model, n_error)

            with open(results_dir + "errors_k_" + str(kmat) + ".txt", "a") as f:
                f.write(f"{n_per_dim}\t{h_values[idx_n]}\t{centers.shape[0]}\t"
                        f"{error_L2}\t{error_H1}\t{elapsed_time:.4f}\t{run_peak_mem_mb:.2f}\n")
            with open(results_dir + "condition_numbers_k_" + str(kmat) + ".txt", "a") as f:
                f.write(f"{n_per_dim}\t{h_values[idx_n]}\t{cond}\n")


if __name__ == "__main__":
    app()
