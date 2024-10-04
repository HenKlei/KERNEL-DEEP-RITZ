from datetime import datetime
import numpy as np
import time
import torch
from torch import optim
from torch.optim import lr_scheduler
import os

from kernelDR.models.model_kernels import FlatKernelModel
from kernelDR.problem_definitions.poisson_higher_regularity import PoissonHigherRegularityRefSol
from kernelDR.training import train_model
from kernelDR.utils import compute_L2_norm


torch.set_default_dtype(torch.float64)

# Define filepaths, device and setting
timestr = time.strftime("%Y%m%d-%H%M%S")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

problem = PoissonHigherRegularityRefSol(penalty_parameter=100., device=device)

# Define kernel
kernel = "matern"
k_smoothness = 1
ep = 1

filepath_prefix = f"results_smooth_solution_k{k_smoothness}/"

os.makedirs(filepath_prefix, exist_ok=True)

# Define method
method = "flat_kernel"
print(f"Cuda available: {torch.cuda.is_available()}")

# Define several quantities
n_i = 10000
n_b = 1000
n_error = 10201
n_plot = 1000

sol_norm = compute_L2_norm(problem.reference_solution, problem, n=10201)

# Run the computation
list_n_per_dim = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
list_n_epochs = [100000 for _ in list_n_per_dim]

list_error_L2 = []
list_error_H1 = []

list_final_losses = []
list_true_n_inner = []

for idx_n, (n_per_dim, n_epochs) in enumerate(zip(list_n_per_dim, list_n_epochs)):
    # Use centers both in the interior and the boundary --> this seems to improve accuracy
    centers_inner = problem.domain.uniform_interior_points(n_per_dim**2)
    n_inner_centers = len(centers_inner)
    # (n+2) due to having the boundary corner points always
    centers_boundary = problem.domain.uniform_boundary_points(4 * (n_per_dim + 2))
    n_boundary_centers = len(centers_boundary)
    centers = torch.vstack([centers_inner, centers_boundary]).detach()
    centers.requires_grad_()

    list_true_n_inner.append(len(centers_inner))
    print(datetime.now().strftime("%H:%M:%S"), f"Training with n={len(centers)} centers ({n_inner_centers} inner "
                                               f"centers, {n_boundary_centers} boundary centers) ...")

    # Define model, optimizer and scheduler
    model = FlatKernelModel(problem.domain.dim, problem.output_dim,
                            str_kernel=kernel, k_smoothness=k_smoothness, ctrs=centers, ep=ep, flag_lagrange=True)

    optimizer = optim.Adam(model.parameters(), lr=5e-2)

    list_milestones = [150, 300, 500, 750, 1300, 2000, 3000, 5000, 8000, 16000, 25000, 350000, 50000, 65000, 80000]

    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=list_milestones, gamma=0.5)

    num_logs = 400
    final_loss, list_loss, list_L2, list_H1 = train_model(problem, model, n_i, n_b, n_epochs, optimizer, centers,
                                                          scheduler=scheduler, fixed_integration_points=False,
                                                          flag_best_model=False, num_logs=num_logs)
    list_epochs_log = list(np.unique(np.geomspace(1, n_epochs, num=num_logs, dtype=int, endpoint=True)))
    list_loss_subset = [list_loss[idx] for idx in list_epochs_log]

    list_final_losses.append(final_loss)

    print(datetime.now().strftime("%H:%M:%S"), 'Computation {}/{} finished.'.format(idx_n + 1, len(list_n_per_dim)))

    list_epochs = list(np.unique(np.geomspace(1, n_epochs, num_logs, dtype=int, endpoint=True)))

    # Save results
    with open(filepath_prefix + "conv_results_{}_{}_{}_{}.txt".format(k_smoothness, ep, n_per_dim, n_epochs), "w") as f:
        f.write('i, epoch, loss, l2_err, h1_err\n')
        for i, (epoch, loss, l2_err, h1_err) in enumerate(zip(list_epochs, list_loss_subset, list_L2, list_H1)):
            f.write('{:3d} {:5d} {:.5e} {:.5e} {:.5e}\n'.format(i, epoch, loss, l2_err, h1_err))
