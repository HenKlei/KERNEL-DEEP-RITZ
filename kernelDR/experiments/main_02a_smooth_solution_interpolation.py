import numpy as np
import os
from datetime import datetime
import torch
import torch.nn as nn

from kernelDR.models.model_kernels import FlatKernelModel
from kernelDR.problem_definitions.poisson_higher_regularity import PoissonHigherRegularityRefSol
from kernelDR.utils import compute_relative_L2_error, compute_relative_H1_error


torch.set_default_dtype(torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
problem = PoissonHigherRegularityRefSol(penalty_parameter=100., device=device)

n_error = 10201

filepath_prefix = "results_interpolation_smooth_solution/"
os.makedirs(filepath_prefix, exist_ok=True)

kernel = "matern"
ep = 1
list_kmat = [0, 1, 2]

list_n_per_dim = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

array_errors_L2 = np.zeros((len(list_kmat), len(list_n_per_dim)))
array_errors_H1 = np.zeros((len(list_kmat), len(list_n_per_dim)))

for idx_kmat, kmat in enumerate(list_kmat):
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

        model = FlatKernelModel(problem.domain.dim, problem.output_dim,
                                str_kernel=kernel, k_smoothness=kmat, ctrs=centers, ep=ep, flag_lagrange=False)

        # Compute interpolant
        coeffs = np.linalg.solve(model.kernel.eval(centers, centers).detach().numpy(),
                                 problem.reference_solution(centers).detach().numpy())

        model.coeffs = nn.Parameter(torch.from_numpy(coeffs))

        error_L2 = compute_relative_L2_error(problem, model, n_error)
        array_errors_L2[idx_kmat, idx_n] = error_L2

        error_H1 = compute_relative_H1_error(problem, model, n_error)
        array_errors_H1[idx_kmat, idx_n] = error_H1

h_values = 1. / (np.array(list_n_per_dim) + 1)

for k in list_kmat:
    with open(filepath_prefix + "errors_k_" + str(k) + ".txt", "w") as f:
        f.write("n\th ~ 1 / sqrt(n)\tL2-error\tH1-error\n")
        for n, h, l2_err, h1_err in zip(list_n_per_dim, h_values, array_errors_L2[k], array_errors_H1[k]):
            f.write(f"{n}\t{h}\t{l2_err}\t{h1_err}\n")
