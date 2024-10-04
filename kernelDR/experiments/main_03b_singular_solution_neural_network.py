from datetime import datetime
import time
import torch
from torch import optim
from torch.optim import lr_scheduler
import os

from kernelDR.models.model_nn import NeuralNetworkModel
from kernelDR.problem_definitions.laplace_pacman import LaplaceOnPacmanDomainSingularSolution
from kernelDR.training import train_model
from kernelDR.utils import compute_relative_L2_error, compute_relative_H1_error, count_parameters


torch.set_default_dtype(torch.float64)

# Define filepaths, device and setting
timestr = time.strftime("%Y%m%d-%H%M%S")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

angle = 3.*torch.pi/2.
radius = 1.5
problem = LaplaceOnPacmanDomainSingularSolution(angle=angle, radius=radius, penalty_parameter=100., device=device)

filepath_prefix = "results_neural_network_singular_solution/"
os.makedirs(filepath_prefix, exist_ok=True)

with open(filepath_prefix + "convergence_results.txt", "w") as f:
    f.write("Neurons per layer\tTotal number of parameters\tL2-error\tH1-error\tFinal loss\n")

print(f"Cuda available: {torch.cuda.is_available()}")

# Define several quantities
n_i = 10000
n_b = 1000
n_error = 10201
num_logs = 400

# Run the computation
list_num_neurons_per_layer = [1, 2, 3, 6, 8, 12, 14]
list_n_epochs = [100000 for _ in list_num_neurons_per_layer]


for idx_n, (n, n_epochs) in enumerate(zip(list_num_neurons_per_layer, list_n_epochs)):
    model = NeuralNetworkModel(problem.domain.dim, problem.output_dim, n, 2)

    num_parameters = count_parameters(model)
    print(f"Number of parameters: {num_parameters}")

    optimizer = optim.Adam(model.parameters(), lr=5e-2)

    # Manual testing of differnt milestones (learning rate scheduling)
    list_milestones = [150, 300, 500, 750, 1300, 2000, 3000, 5000, 8000, 16000, 25000, 350000, 50000, 65000, 80000]

    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=list_milestones, gamma=0.5)

    final_loss, list_loss, list_L2, list_H1 = train_model(problem, model, n_i, n_b, n_epochs, optimizer,
                                                          scheduler=scheduler, fixed_integration_points=False,
                                                          flag_best_model=False, num_logs=num_logs)

    print(datetime.now().strftime("%H:%M:%S"),
          'Computation {}/{} finished.'.format(idx_n + 1, len(list_num_neurons_per_layer)))

    error_L2 = compute_relative_L2_error(problem, model, n_error)

    error_H1 = compute_relative_H1_error(problem, model, n_error)

    with open(filepath_prefix + "convergence_results.txt", "a") as f:
        f.write(f"{n}\t{num_parameters}\t{error_L2}\t{error_H1}\t{final_loss}\n")
