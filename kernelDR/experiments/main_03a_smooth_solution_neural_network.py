from datetime import datetime
from typing import Annotated

import cyclopts
import torch
from torch import optim
from torch.optim import lr_scheduler
import os

from kernelDR.models.model_nn import NeuralNetworkModel
from kernelDR.problem_definitions.poisson_higher_regularity import PoissonHigherRegularity
from kernelDR.training import train_model
from kernelDR.utils import compute_relative_L2_error, compute_relative_H1_error, count_parameters, peak_memory_mb, reset_peak_memory
from kernelDR.experiments.plot_utils import save_settings


app = cyclopts.App()


@app.default
def main(
    penalty_parameter: Annotated[float, cyclopts.Parameter(help="Penalty parameter for boundary conditions.")] = 100.0,
    n_i: Annotated[int, cyclopts.Parameter(help="Number of interior sample points.")] = 10000,
    n_b: Annotated[int, cyclopts.Parameter(help="Number of boundary sample points.")] = 1000,
    n_error: Annotated[int, cyclopts.Parameter(help="Number of error evaluation points.")] = 10201,
    n_epochs: Annotated[int, cyclopts.Parameter(help="Number of training epochs.")] = 100000,
    lr: Annotated[float, cyclopts.Parameter(help="Initial learning rate.")] = 5e-2,
    gamma: Annotated[float, cyclopts.Parameter(help="Learning rate decay factor.")] = 0.5,
    num_logs: Annotated[int, cyclopts.Parameter(help="Number of log points.")] = 400,
    early_stopping_patience: Annotated[int, cyclopts.Parameter(help="Stop training if no improvement for this many epochs (0 to disable).")] = 0,
    num_layers: Annotated[int, cyclopts.Parameter(help="Number of hidden layers.")] = 2,
    activation: Annotated[str, cyclopts.Parameter(help="Activation function: tanh, relu, gelu, silu, softplus, sin.")] = "tanh",
    scheduler_type: Annotated[str, cyclopts.Parameter(help="LR scheduler: 'multistep' (MultiStepLR schedule) or 'plateau' (ReduceLROnPlateau).")] = "multistep",
    list_num_neurons_per_layer: Annotated[tuple[int, ...], cyclopts.Parameter(help="Neurons per layer.")] = (1, 2, 3, 6, 8, 12, 14),
    results_dir: Annotated[str, cyclopts.Parameter(help="Results output directory.")] = "results_neural_network_smooth_solution/",
):
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    problem = PoissonHigherRegularity(penalty_parameter=penalty_parameter, device=device)

    if not results_dir.endswith("/"):
        results_dir += "/"
    os.makedirs(results_dir, exist_ok=True)
    save_settings(results_dir, locals())

    with open(results_dir + "convergence_results.txt", "w") as f:
        f.write("Neurons per layer\tTotal number of parameters\tL2-error\tH1-error\tFinal loss\ttime_train_s\tpeak_mem_mb\n")

    print(f"Cuda available: {torch.cuda.is_available()}")

    list_n_epochs = [n_epochs for _ in list_num_neurons_per_layer]

    for idx_n, (n, n_ep) in enumerate(zip(list_num_neurons_per_layer, list_n_epochs)):
        model = NeuralNetworkModel(problem.domain.dim, problem.output_dim, n, num_layers, activation=activation).to(device)

        num_parameters = count_parameters(model)
        print(f"Number of parameters: {num_parameters}")

        optimizer = optim.Adam(model.parameters(), lr=lr)

        if scheduler_type == "multistep":
            # Original schedule used for the first-submission results.
            milestones = [150, 300, 500, 750, 1300, 2000, 3000, 5000, 8000,
                          16000, 25000, 350000, 50000, 65000, 80000]
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        else:
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=200, min_lr=1e-6)

        reset_peak_memory()
        final_loss, list_loss, list_L2, list_H1, actual_epochs, elapsed_time = train_model(problem, model, n_i, n_b, n_ep, optimizer,
                                                              scheduler=scheduler, fixed_integration_points=False,
                                                              flag_best_model=False, num_logs=num_logs, early_stopping_patience=early_stopping_patience,
                                                              n_error=n_error)
        run_peak_mem_mb = peak_memory_mb()

        print(datetime.now().strftime("%H:%M:%S"),
              'Computation {}/{} finished.'.format(idx_n + 1, len(list_num_neurons_per_layer)))

        error_L2 = compute_relative_L2_error(problem, model, n_error)

        error_H1 = compute_relative_H1_error(problem, model, n_error)

        with open(results_dir + "convergence_results.txt", "a") as f:
            f.write(f"{n}\t{num_parameters}\t{error_L2}\t{error_H1}\t{final_loss}\t{elapsed_time:.2f}\t{run_peak_mem_mb:.2f}\n")

        # Free GPU memory from this run
        del model, optimizer, scheduler
        del final_loss, list_loss, list_L2, list_H1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    app()
