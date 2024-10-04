from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import os

from kernelDR.models.model_kernels import FlatKernelModel
from kernelDR.problem_definitions.poisson_higher_regularity import PoissonHigherRegularityRefSol
from kernelDR.utils import compute_relative_L2_error, compute_relative_H1_error


torch.set_default_dtype(torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
problem = PoissonHigherRegularityRefSol(penalty_parameter=100., device=device)

filepath_prefix = "results_matrix_form_smooth_solution/"
os.makedirs(filepath_prefix, exist_ok=True)

n_i = 10000
n_b = 1000
n_error = 10201

regularization = 0

kernel = "matern"
ep = 1
list_kmat = [0, 1, 2]

list_n_per_dim = [1, 2, 4, 8, 12, 16, 20]

num_reg_params = 50
regularization_parameters = np.geomspace(1e-14, 1e0, num=num_reg_params)

h_values = 1. / (np.array(list_n_per_dim) + 1)

for idx_kmat, kmat in enumerate(list_kmat):
    for idx_n, n_per_dim in enumerate(list_n_per_dim):
        with open(filepath_prefix + "errors_condition_numbers_k_" + str(kmat) + "_n_per_dim_" + str(n_per_dim) + ".txt", "w") as f:
            f.write("Regularization parameter\tL2-error\tH1-error\tCondition number\n")
        print(datetime.now().strftime("%H:%M:%S"), kmat, n_per_dim)
        centers_inner = problem.domain.uniform_interior_points(n_per_dim**2)
        n_inner_centers = len(centers_inner)
        # (n+2) due to having the boundary corner points always
        centers_boundary = problem.domain.uniform_boundary_points(4 * (n_per_dim + 2))
        n_boundary_centers = len(centers_boundary)

        centers = torch.vstack([centers_inner, centers_boundary]).detach()
        centers.requires_grad_()

        save_mat_path = filepath_prefix + f"system_k_{kmat}_n_{n_per_dim}/"
        A = np.load(save_mat_path + "original_A.npy")
        b = np.load(save_mat_path + "b.npy")

        model_params = {"str_kernel": kernel, "k_smoothness": kmat, "ctrs": centers, "ep": ep, "flag_lagrange": False}
        model = FlatKernelModel(problem.domain.dim, problem.output_dim, **model_params)
        num_coeffs = len(model.coeffs)

        for idx_reg, reg_param in enumerate(regularization_parameters):
            A_reg = A + reg_param * np.eye(num_coeffs)
            coeffs = np.linalg.solve(A_reg, b)

            model = FlatKernelModel(problem.domain.dim, problem.output_dim, **model_params)
            model.coeffs = nn.Parameter(torch.tensor(coeffs))

            condition_num = np.linalg.cond(A_reg)
            error_L2 = compute_relative_L2_error(problem, model, n_error)
            error_H1 = compute_relative_H1_error(problem, model, n_error)

            with open(filepath_prefix + "errors_condition_numbers_k_" + str(kmat) + "_n_per_dim_" + str(n_per_dim) + ".txt", "a") as f:
                f.write(f"{reg_param}\t{error_L2}\t{error_H1}\t{condition_num}\n")
