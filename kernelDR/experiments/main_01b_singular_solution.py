from datetime import datetime
from typing import Annotated

import cyclopts
import numpy as np
import torch
from torch import optim
from torch.optim import lr_scheduler
import os

from kernelDR.models.model_kernels import FlatKernelModel, TwoLayerKernelModel, save_kernel_model
from kernelDR.problem_definitions.laplace_pacman import LaplaceOnPacmanDomainSingularSolution
from kernelDR.training import train_model
from kernelDR.utils import count_parameters, peak_memory_mb, reset_peak_memory, set_seed
from kernelDR.experiments.plot_utils import save_settings


app = cyclopts.App()


@app.default
def main(
    model_type: Annotated[str, cyclopts.Parameter(help="Model type: 'flat' or '2l'.")] = "flat",
    kernel: Annotated[str, cyclopts.Parameter(help="Kernel type.")] = "matern",
    k_smoothness: Annotated[int, cyclopts.Parameter(help="Kernel smoothness parameter.")] = 1,
    ep: Annotated[float, cyclopts.Parameter(help="Kernel shape parameter.")] = 1.0,
    penalty_parameter: Annotated[float, cyclopts.Parameter(help="Penalty parameter for boundary conditions.")] = 100.0,
    n_i: Annotated[int, cyclopts.Parameter(help="Number of interior sample points.")] = 10000,
    n_b: Annotated[int, cyclopts.Parameter(help="Number of boundary sample points.")] = 1000,
    n_error: Annotated[int, cyclopts.Parameter(help="Number of error evaluation points. The default resolves the approximation error, which oscillates on the scale of the centers; a coarser grid understates the H1-errors and biases fitted convergence rates (see the README).")] = 640000,
    optimizer_type: Annotated[str, cyclopts.Parameter(help="Optimizer: 'adam' or 'lbfgs'.")] = "adam",
    n_epochs: Annotated[int, cyclopts.Parameter(help="Number of training epochs.")] = 10000,
    lr: Annotated[float, cyclopts.Parameter(help="Initial learning rate.")] = 5e-2,
    gamma: Annotated[float, cyclopts.Parameter(help="Learning rate decay factor (Adam only).")] = 0.5,
    num_logs: Annotated[int, cyclopts.Parameter(help="Number of log points.")] = 50,
    early_stopping_patience: Annotated[int, cyclopts.Parameter(help="Stop training if no improvement for this many epochs (0 to disable).")] = 0,
    angle: Annotated[float, cyclopts.Parameter(help="Angle of the pacman domain (in radians).")] = 4.71238898038469,
    radius: Annotated[float, cyclopts.Parameter(help="Radius of the pacman domain.")] = 1.5,
    fixed_integration_points: Annotated[bool, cyclopts.Parameter(help="Use fixed integration points instead of random.")] = False,
    seed: Annotated[int, cyclopts.Parameter(help="RNG seed for the quadrature point sampling. Use a separate --results-dir per seed when repeating a run.")] = 0,
    save_models: Annotated[bool, cyclopts.Parameter(help="Save the trained model of every configuration, so that it can be re-evaluated later (see main_07_integration_diagnostics.py).")] = False,
    list_n_per_dim: Annotated[tuple[int, ...], cyclopts.Parameter(help="Centers per dimension.")] = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20),
    results_dir: Annotated[str, cyclopts.Parameter(help="Results output directory.")] = "",
):
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    set_seed(seed)

    problem = LaplaceOnPacmanDomainSingularSolution(angle=angle, radius=radius,
                                                    penalty_parameter=penalty_parameter, device=device)

    if not results_dir:
        results_dir = f"results_singular_solution_{kernel}_k{k_smoothness}/"
    if not results_dir.endswith("/"):
        results_dir += "/"
    os.makedirs(results_dir, exist_ok=True)
    save_settings(results_dir, locals())

    print(f"Cuda available: {torch.cuda.is_available()}")

    list_n_epochs = [n_epochs for _ in list_n_per_dim]

    with open(results_dir + "timings.txt", "w") as f:
        f.write("n_per_dim\tn_centers\tnum_params\tepochs\ttime_train_s\tpeak_mem_mb\n")

    list_final_losses = []
    list_true_n_inner = []

    for idx_n, (n_per_dim, n_ep) in enumerate(zip(list_n_per_dim, list_n_epochs)):
        # Use centers both in the interior and the boundary
        centers_inner = problem.domain.uniform_interior_points(n_per_dim**2)
        n_inner_centers = len(centers_inner)
        # (n+2) due to having the boundary corner points always
        centers_boundary = problem.domain.uniform_boundary_points(4 * (n_per_dim + 2))
        n_boundary_centers = len(centers_boundary)

        # Remove interior-centers that are too close to the boundary (due to Pacman domain)
        array_dist = torch.cdist(centers_inner, centers_boundary)
        mask = torch.min(array_dist, dim=1).values > 1/10 * (2 * problem.domain.radius / n_per_dim)
        centers_inner = centers_inner[mask, :]

        centers = torch.vstack([centers_inner, centers_boundary]).detach()
        centers.requires_grad_()

        list_true_n_inner.append(len(centers_inner))
        print(datetime.now().strftime("%H:%M:%S"), f"Training with n={len(centers)} centers ({n_inner_centers} inner "
                                                   f"centers, {n_boundary_centers} boundary centers) ...")

        # Define model, optimizer and scheduler
        if model_type == "2l":
            model = TwoLayerKernelModel(problem.domain.dim, problem.output_dim,
                                        str_kernel=kernel, k_smoothness=k_smoothness, ctrs=centers, ep=ep)
        else:
            model = FlatKernelModel(problem.domain.dim, problem.output_dim,
                                    str_kernel=kernel, k_smoothness=k_smoothness, ctrs=centers, ep=ep, flag_lagrange=True)

        if optimizer_type == "lbfgs":
            optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=20, line_search_fn="strong_wolfe")
            scheduler = None
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=200, min_lr=1e-6)

        num_params = count_parameters(model)
        reset_peak_memory()
        final_loss, list_loss, list_L2, list_H1, actual_epochs, elapsed_time = train_model(problem, model, n_i, n_b, n_ep, optimizer, centers,
                                                              scheduler=scheduler, fixed_integration_points=fixed_integration_points,
                                                              flag_best_model=False, num_logs=num_logs, early_stopping_patience=early_stopping_patience,
                                                              n_error=n_error)
        run_peak_mem_mb = peak_memory_mb()
        list_epochs_log = list(np.unique(np.geomspace(1, actual_epochs, num=num_logs, dtype=int, endpoint=True)))
        list_loss_subset = [list_loss[idx] for idx in list_epochs_log]

        list_final_losses.append(final_loss)

        print(datetime.now().strftime("%H:%M:%S"), 'Computation {}/{} finished.'.format(idx_n + 1, len(list_n_per_dim)))

        list_epochs = list(np.unique(np.geomspace(1, actual_epochs, num_logs, dtype=int, endpoint=True)))

        # Save results
        with open(results_dir + "conv_results_{}_{}_{}_{}.txt".format(k_smoothness, ep, n_per_dim, actual_epochs), "w") as f:
            f.write('i, epoch, loss, l2_err, h1_err\n')
            for i, (epoch, loss, l2_err, h1_err) in enumerate(zip(list_epochs, list_loss_subset, list_L2, list_H1)):
                f.write('{:3d} {:5d} {:.5e} {:.5e} {:.5e}\n'.format(i, epoch, loss, l2_err, h1_err))

        # Full per-epoch loss history (for plotting energy vs iterations against CG).
        with open(results_dir + "loss_history_{}_{}_{}_{}.txt".format(k_smoothness, ep, n_per_dim, actual_epochs), "w") as f:
            f.write("iteration\tloss\n")
            for i, ls in enumerate(list_loss):
                f.write(f"{i}\t{ls:.6e}\n")

        with open(results_dir + "timings.txt", "a") as f:
            f.write(f"{n_per_dim}\t{len(centers)}\t{num_params}\t{actual_epochs}\t"
                    f"{elapsed_time:.2f}\t{run_peak_mem_mb:.2f}\n")

        if save_models:
            model_path = results_dir + f"model_k_{k_smoothness}_ep_{ep}_n_{n_per_dim}.pt"
            save_kernel_model(model, model_path)
            print(f"Saved trained model to {model_path}")

        # Free GPU memory from this run
        del model, optimizer, scheduler, centers
        del final_loss, list_loss, list_L2, list_H1, list_loss_subset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    app()
