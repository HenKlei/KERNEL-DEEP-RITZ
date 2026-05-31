from datetime import datetime
from typing import Annotated

import cyclopts
import numpy as np
import os
import scipy as sp
import time
import torch
import torch.nn as nn

from kernelDR.models.model_kernels import FlatKernelModel
from kernelDR.problem_definitions.laplace_pacman import LaplaceOnPacmanDomainSingularSolution
from kernelDR.utils import compute_relative_L2_error, compute_relative_H1_error, peak_memory_mb, reset_peak_memory
from kernelDR.experiments.plot_utils import save_settings


app = cyclopts.App()


@app.default
def main(
    kernel: Annotated[str, cyclopts.Parameter(help="Kernel type.")] = "matern",
    ep: Annotated[float, cyclopts.Parameter(help="Kernel shape parameter.")] = 1.0,
    penalty_parameter: Annotated[float, cyclopts.Parameter(help="Penalty parameter for boundary conditions.")] = 100.0,
    n_error: Annotated[int, cyclopts.Parameter(help="Number of error evaluation points.")] = 10201,
    angle: Annotated[float, cyclopts.Parameter(help="Angle of the pacman domain (in radians).")] = 4.71238898038469,
    radius: Annotated[float, cyclopts.Parameter(help="Radius of the pacman domain.")] = 1.5,
    list_kmat: Annotated[tuple[int, ...], cyclopts.Parameter(help="Kernel smoothness values.")] = (0, 1, 2),
    list_n_per_dim: Annotated[tuple[int, ...], cyclopts.Parameter(help="Centers per dimension.")] = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20),
    regularization: Annotated[float, cyclopts.Parameter(help="Tikhonov regularization added to the kernel diagonal.")] = 0.0,
    results_dir: Annotated[str, cyclopts.Parameter(help="Results output directory.")] = "results_interpolation_singular_solution/",
):
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    problem = LaplaceOnPacmanDomainSingularSolution(angle=angle, radius=radius,
                                                    penalty_parameter=penalty_parameter, device=device)

    if not results_dir.endswith("/"):
        results_dir += "/"
    os.makedirs(results_dir, exist_ok=True)
    save_settings(results_dir, locals())

    array_errors_L2 = np.zeros((len(list_kmat), len(list_n_per_dim)))
    array_errors_H1 = np.zeros((len(list_kmat), len(list_n_per_dim)))
    array_times = np.zeros((len(list_kmat), len(list_n_per_dim)))
    array_peak_mem = np.zeros((len(list_kmat), len(list_n_per_dim)))
    array_n_centers = np.zeros((len(list_kmat), len(list_n_per_dim)), dtype=int)

    for idx_kmat, kmat in enumerate(list_kmat):
        # Run the computation
        for idx_n, n_per_dim in enumerate(list_n_per_dim):
            print(datetime.now().strftime("%H:%M:%S"), kmat, n_per_dim)
            # Use centers both in the interior and the boundary --> this seems to improve accuracy
            centers_inner = problem.domain.uniform_interior_points(n_per_dim**2)
            # (n+2) due to having the boundary corner points always
            centers_boundary = problem.domain.uniform_boundary_points(4 * (n_per_dim + 2))

            # Remove interior centers that are too close to the boundary (avoids
            # near-duplicate centers that make the kernel Gram matrix singular).
            array_dist = torch.cdist(centers_inner, centers_boundary)
            mask = torch.min(array_dist, dim=1).values > 1/10 * (2 * problem.domain.radius / n_per_dim)
            centers_inner = centers_inner[mask, :]

            centers = torch.vstack([centers_inner, centers_boundary]).detach()
            centers.requires_grad_()

            model = FlatKernelModel(problem.domain.dim, problem.output_dim,
                                    str_kernel=kernel, k_smoothness=kmat, ctrs=centers, ep=ep, flag_lagrange=False)

            # Compute interpolant (optional Tikhonov regularization for ill-conditioned kernel Gram matrices)
            reset_peak_memory()
            t_start = time.time()
            K_mat = model.kernel.eval(centers, centers).detach().cpu().numpy()
            if regularization > 0.0:
                K_mat = K_mat + regularization * np.eye(K_mat.shape[0])
            coeffs = sp.linalg.solve(K_mat,
                                     problem.reference_solution(centers).detach().cpu().numpy(), assume_a='pos')
            elapsed_time = time.time() - t_start
            run_peak_mem_mb = peak_memory_mb()

            model.coeffs = nn.Parameter(torch.from_numpy(coeffs).to(model.ctrs.device))

            error_L2 = compute_relative_L2_error(problem, model, n_error)
            array_errors_L2[idx_kmat, idx_n] = error_L2

            error_H1 = compute_relative_H1_error(problem, model, n_error)
            array_errors_H1[idx_kmat, idx_n] = error_H1
            array_times[idx_kmat, idx_n] = elapsed_time
            array_peak_mem[idx_kmat, idx_n] = run_peak_mem_mb
            array_n_centers[idx_kmat, idx_n] = centers.shape[0]

    h_values = 1. / (np.array(list_n_per_dim) + 1)

    for k_idx, k in enumerate(list_kmat):
        with open(results_dir + "errors_k_" + str(k) + ".txt", "w") as f:
            f.write("n\th ~ 1 / sqrt(n)\tn_centers\tL2-error\tH1-error\ttime_solve_s\tpeak_mem_mb\n")
            for n, h, n_c, l2_err, h1_err, t, mem in zip(list_n_per_dim, h_values, array_n_centers[k_idx],
                                                          array_errors_L2[k_idx], array_errors_H1[k_idx],
                                                          array_times[k_idx], array_peak_mem[k_idx]):
                f.write(f"{n}\t{h}\t{int(n_c)}\t{l2_err}\t{h1_err}\t{t:.4f}\t{mem:.2f}\n")


if __name__ == "__main__":
    app()
