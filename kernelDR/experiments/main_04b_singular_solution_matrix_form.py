from datetime import datetime
import numpy as np
import torch
import os

from kernelDR.matrix_form import assemble_and_solve_system
from kernelDR.models.model_kernels import FlatKernelModel
from kernelDR.problem_definitions.laplace_pacman import LaplaceOnPacmanDomainSingularSolution
from kernelDR.utils import compute_relative_L2_error, compute_relative_H1_error


torch.set_default_dtype(torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

angle = 3.*torch.pi/2.
radius = 1.5
problem = LaplaceOnPacmanDomainSingularSolution(angle=angle, radius=radius, penalty_parameter=100., device=device)

filepath_prefix = "results_matrix_form_singular_solution/"
os.makedirs(filepath_prefix, exist_ok=True)

n_i = 10000
n_b = 1000
n_error = 10201

regularization = 0

kernel = "matern"
ep = 1
list_kmat = [0, 1, 2]

list_n_per_dim = [1, 2, 4, 8, 12, 16, 20]

h_values = 1. / (np.array(list_n_per_dim) + 1)

for idx_kmat, kmat in enumerate(list_kmat):
    with open(filepath_prefix + "errors_k_" + str(kmat) + ".txt", "w") as f:
        f.write("n\th ~ 1 / sqrt(n)\tL2-error\tH1-error\n")
    with open(filepath_prefix + "condition_numbers_k_" + str(kmat) + ".txt", "w") as f:
        f.write("n\th ~ 1 / sqrt(n)\tCondition number\n")
    for idx_n, n_per_dim in enumerate(list_n_per_dim):
        print(datetime.now().strftime("%H:%M:%S"), kmat, n_per_dim)
        # Use centers both in the interior and the boundary --> this seems to improve accuracy
        centers_inner = problem.domain.uniform_interior_points(n_per_dim**2)
        n_inner_centers = len(centers_inner)
        # (n+2) due to having the boundary corner points always
        centers_boundary = problem.domain.uniform_boundary_points(4 * (n_per_dim + 2))
        n_boundary_centers = len(centers_boundary)

        centers = torch.vstack([centers_inner, centers_boundary]).detach()
        centers.requires_grad_()

        save_mat_path = filepath_prefix + f"system_k_{kmat}_n_{n_per_dim}/"
        os.makedirs(save_mat_path, exist_ok=True)

        model_params = {"str_kernel": kernel, "k_smoothness": kmat, "ctrs": centers, "ep": ep, "flag_lagrange": False}
        model, A, b = assemble_and_solve_system(problem, FlatKernelModel, n_i, n_b, model_params=model_params,
                                                return_linear_system=True, regularization=regularization,
                                                save_mat_path=save_mat_path)

        condition_num = np.linalg.cond(A)

        error_L2 = compute_relative_L2_error(problem, model, n_error)
        error_H1 = compute_relative_H1_error(problem, model, n_error)

        with open(filepath_prefix + "errors_k_" + str(kmat) + ".txt", "a") as f:
            f.write(f"{n_per_dim}\t{h_values[idx_n]}\t{error_L2}\t{error_H1}\n")
        with open(filepath_prefix + "condition_numbers_k_" + str(kmat) + ".txt", "a") as f:
            f.write(f"{n_per_dim}\t{h_values[idx_n]}\t{condition_num}\n")
