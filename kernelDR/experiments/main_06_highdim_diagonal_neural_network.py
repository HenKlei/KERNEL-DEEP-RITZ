"""Neural-network baseline for the high-dimensional sinus-along-diagonal
Poisson problem, to be compared against the kernel runs in
``main_05_highdim_diagonal.py``.

Same problem setup, same Deep Ritz energy, same trainer — only the
ansatz changes from a (flat / two-layer) kernel model to a fully
connected tanh-MLP.
"""
from datetime import datetime
from typing import Annotated

import cyclopts
import math
import os
import torch
from torch import optim
from torch.optim import lr_scheduler

from kernelDR.models.model_nn import NeuralNetworkModel
from kernelDR.problem_definitions.highdim_diagonal import HighDimDiagonalExample
from kernelDR.training import train_model
from kernelDR.utils import count_parameters, peak_memory_mb, reset_peak_memory
from kernelDR.experiments.plot_utils import save_settings


app = cyclopts.App()


@app.default
def main(
    d_x: Annotated[int, cyclopts.Parameter(help="Spatial dimension.")] = 10,
    frequency: Annotated[float, cyclopts.Parameter(help="Frequency of the diagonal sinus (μ=0 ⇒ π).")] = math.pi,
    penalty_parameter: Annotated[float, cyclopts.Parameter(help="Boundary penalty.")] = 100.0,
    n_i: Annotated[int, cyclopts.Parameter(help="Interior sample points per Ritz batch.")] = 10000,
    n_b: Annotated[int, cyclopts.Parameter(help="Boundary sample points per Ritz batch.")] = 1000,
    n_epochs: Annotated[int, cyclopts.Parameter(help="Training epochs.")] = 10000,
    lr: Annotated[float, cyclopts.Parameter(help="Initial learning rate.")] = 5e-2,
    gamma: Annotated[float, cyclopts.Parameter(help="LR decay factor.")] = 0.5,
    num_layers: Annotated[int, cyclopts.Parameter(help="Number of hidden layers.")] = 2,
    activation: Annotated[str, cyclopts.Parameter(help="Activation function: tanh, relu, gelu, silu, softplus, sin.")] = "tanh",
    list_num_neurons_per_layer: Annotated[tuple[int, ...], cyclopts.Parameter(help="Hidden width sweep.")] = (8, 16, 32, 64, 128),
    n_error: Annotated[int, cyclopts.Parameter(help="Spatial error eval points.")] = 10000,
    fixed_integration_points: Annotated[bool, cyclopts.Parameter(help="Use deterministic samples.")] = False,
    num_logs: Annotated[int, cyclopts.Parameter(help="Convergence log frequency.")] = 100,
    early_stopping_patience: Annotated[int, cyclopts.Parameter(help="Early-stop patience (0 to disable).")] = 0,
    seed: Annotated[int, cyclopts.Parameter(help="RNG seed for reproducibility.")] = 0,
    results_dir: Annotated[str, cyclopts.Parameter(help="Output dir.")] = "",
):
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    torch.manual_seed(seed)

    if not results_dir:
        results_dir = f"results_highdim_diagonal_d{d_x}_nn/"
    if not results_dir.endswith("/"):
        results_dir += "/"
    os.makedirs(results_dir, exist_ok=True)
    save_settings(results_dir, locals())

    print(f"Cuda available: {torch.cuda.is_available()}")

    problem = HighDimDiagonalExample(d_x=d_x, frequency=frequency,
                                     penalty_parameter=penalty_parameter, device=device)

    with open(results_dir + "summary.txt", "w") as f:
        f.write("width\tnum_params\tfinal_loss\tL2_err\tH1_err\ttime_train_s\tpeak_mem_mb\n")

    for idx_n, width in enumerate(list_num_neurons_per_layer):
        model = NeuralNetworkModel(problem.domain.dim, problem.output_dim, width, num_layers, activation=activation).to(device)
        num_params = count_parameters(model)
        print(datetime.now().strftime("%H:%M:%S"),
              f"Training NN: width={width}, depth={num_layers}, num_params={num_params} (d_x={d_x}) ...")

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=200, min_lr=1e-6)

        reset_peak_memory()
        final_loss, list_loss, list_L2, list_H1, actual_epochs, elapsed_time = train_model(
            problem, model, n_i, n_b, n_epochs, optimizer,
            scheduler=scheduler, fixed_integration_points=fixed_integration_points,
            flag_best_model=False, num_logs=num_logs,
            early_stopping_patience=early_stopping_patience, n_error=n_error)
        run_peak_mem_mb = peak_memory_mb()

        l2_final = list_L2[-1] if list_L2 else float('nan')
        h1_final = list_H1[-1] if list_H1 else float('nan')

        with open(results_dir + f"conv_results_w{width}.txt", "w") as f:
            f.write("i\tepoch\tloss\tl2_err\th1_err\n")
            import numpy as np
            list_epochs_log = list(np.unique(np.geomspace(1, actual_epochs, num=num_logs,
                                                         dtype=int, endpoint=True)))
            list_loss_subset = [list_loss[idx] for idx in list_epochs_log if idx < len(list_loss)]
            for i, (ep_, ls, l2, h1) in enumerate(zip(list_epochs_log, list_loss_subset,
                                                      list_L2, list_H1)):
                f.write(f"{i}\t{ep_}\t{ls:.5e}\t{l2:.5e}\t{h1:.5e}\n")

        with open(results_dir + "summary.txt", "a") as f:
            f.write(f"{width}\t{num_params}\t{final_loss:.5e}\t"
                    f"{l2_final:.5e}\t{h1_final:.5e}\t{elapsed_time:.2f}\t{run_peak_mem_mb:.2f}\n")

        # Save trained model state for post-hoc analysis / error recomputation.
        torch.save({
            "state_dict": model.state_dict(),
            "width": width,
            "num_layers": num_layers,
            "input_dim": problem.domain.dim,
            "output_dim": problem.output_dim,
            "d_x": d_x,
            "num_params": num_params,
        }, results_dir + f"model_state_w{width}.pt")

        del model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    app()
