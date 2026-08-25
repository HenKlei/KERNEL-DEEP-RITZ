r"""Recompute the reported errors of saved models on a finer evaluation grid.

The relative errors reported by the experiment scripts are evaluated on the
uniform grid of ``uniform_interior_points(n_error)``. That grid has to resolve the
approximation error, which oscillates on the scale of the centers, so the accuracy
of the metric degrades as the centers refine. For the~$H^1$-norm the resulting
bias is not constant but grows with the number of centers, which tilts fitted
convergence rates rather than cancelling out of them.

Evaluating a fine grid during training would be wasteful, since the errors are
logged a few hundred times per run only to draw the error-versus-epoch curves.
This script instead recomputes the *final* errors from the checkpoints written by
``--save-models``, which costs minutes rather than a repetition of the training.

Handles the checkpoint layouts of all experiment families:

* ``model_k_<k>_ep_<ep>_n_<n>.pt``      -- main_01a / main_01b (kernel, Adam)
* ``system_k_<k>_n_<n>/model.pt``       -- main_04a / main_04b (kernel, linear system)
* ``model_seed<s>_width<w>.pt``         -- main_03a / main_03b (neural network)
"""

from typing import Annotated

import cyclopts
import glob
import os
import re
import torch

from kernelDR.models.model_kernels import load_kernel_model
from kernelDR.models.model_nn import load_nn_model
from kernelDR.problem_definitions.poisson_higher_regularity import PoissonHigherRegularity
from kernelDR.problem_definitions.laplace_pacman import LaplaceOnPacmanDomainSingularSolution
from kernelDR.utils import compute_relative_L2_error, compute_relative_H1_error, count_parameters


app = cyclopts.App()


def discover_checkpoints(results_dir):
    """Find checkpoints and the parameters identifying them.

    Returns a list of dicts with keys ``path``, ``kind`` and the identifying
    values parsed from the file name.
    """
    found = []

    for path in sorted(glob.glob(os.path.join(results_dir, "model_k_*_n_*.pt"))):
        match = re.search(r"model_k_(-?\d+)_ep_([0-9.eE+-]+)_n_(\d+)\.pt$", os.path.basename(path))
        if match:
            found.append({"path": path, "kind": "kernel", "k": int(match.group(1)),
                          "ep": float(match.group(2)), "n_per_dim": int(match.group(3)), "seed": None})

    for path in sorted(glob.glob(os.path.join(results_dir, "system_k_*_n_*", "model.pt"))):
        match = re.search(r"system_k_(-?\d+)_n_(\d+)$", os.path.basename(os.path.dirname(path)))
        if match:
            found.append({"path": path, "kind": "kernel", "k": int(match.group(1)),
                          "ep": None, "n_per_dim": int(match.group(2)), "seed": None})

    for path in sorted(glob.glob(os.path.join(results_dir, "model_seed*_width*.pt"))):
        match = re.search(r"model_seed(\d+)_width(\d+)\.pt$", os.path.basename(path))
        if match:
            found.append({"path": path, "kind": "nn", "k": None, "ep": None,
                          "seed": int(match.group(1)), "n_per_dim": int(match.group(2))})

    return found


@app.default
def main(
    results_dir: Annotated[str, cyclopts.Parameter(help="Directory holding the checkpoints written with --save-models.")],
    problem_type: Annotated[str, cyclopts.Parameter(help="Problem the checkpoints belong to: 'smooth' or 'singular'.")] = "smooth",
    n_error: Annotated[int, cyclopts.Parameter(help="Number of error evaluation points. About 640000 is needed for a relative H1-error accurate to well below one percent.")] = 640000,
    n_error_reference: Annotated[int, cyclopts.Parameter(help="Grid the original values were computed on, reported alongside for comparison (0 to skip).")] = 10201,
    penalty_parameter: Annotated[float, cyclopts.Parameter(help="Penalty parameter for boundary conditions.")] = 100.0,
    angle: Annotated[float, cyclopts.Parameter(help="Angle of the pacman domain (in radians); only used for 'singular'.")] = 4.71238898038469,
    radius: Annotated[float, cyclopts.Parameter(help="Radius of the pacman domain; only used for 'singular'.")] = 1.5,
    chunk_size: Annotated[int, cyclopts.Parameter(help="Evaluation points processed at once; bounds the memory of the fine grid.")] = 50000,
    output: Annotated[str, cyclopts.Parameter(help="Output filename (relative to the results directory unless absolute).")] = "",
):
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    if problem_type == "smooth":
        problem = PoissonHigherRegularity(penalty_parameter=penalty_parameter, device=device)
    elif problem_type == "singular":
        problem = LaplaceOnPacmanDomainSingularSolution(angle=angle, radius=radius,
                                                        penalty_parameter=penalty_parameter, device=device)
    else:
        raise SystemExit(f"Unknown problem type {problem_type!r}; expected 'smooth' or 'singular'.")

    if not results_dir.endswith("/"):
        results_dir += "/"

    checkpoints = discover_checkpoints(results_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found in {results_dir}. The experiment scripts write them "
                         f"only with --save-models.")

    if not output:
        output = f"recomputed_errors_nerror{n_error}.txt"
    output_path = output if os.path.isabs(output) else os.path.join(results_dir, output)

    print(f"Recomputing the errors of {len(checkpoints)} checkpoint(s) on {n_error} points "
          f"({problem_type} problem)")
    header = (f"{'checkpoint':<34}{'n':>5}{'params':>8}{'rel. L2':>13}{'rel. H1':>13}")
    if n_error_reference:
        header += f"{'L2 was':>13}{'H1 was':>13}{'H1 low by':>11}"
    print(header)

    rows = []
    for entry in checkpoints:
        if entry["kind"] == "kernel":
            model = load_kernel_model(entry["path"], device=device)
        else:
            model = load_nn_model(entry["path"], device=device)

        error_l2 = compute_relative_L2_error(problem, model, n_error, chunk_size=chunk_size)
        error_h1 = compute_relative_H1_error(problem, model, n_error, chunk_size=chunk_size)

        reference = None
        if n_error_reference:
            reference = (compute_relative_L2_error(problem, model, n_error_reference, chunk_size=chunk_size),
                         compute_relative_H1_error(problem, model, n_error_reference, chunk_size=chunk_size))

        num_params = count_parameters(model)
        name = os.path.relpath(entry["path"], results_dir)
        rows.append({**entry, "name": name, "num_params": num_params,
                     "l2": error_l2, "h1": error_h1, "reference": reference})

        line = f"{name:<34}{entry['n_per_dim']:>5}{num_params:>8}{error_l2:>13.5e}{error_h1:>13.5e}"
        if reference:
            line += (f"{reference[0]:>13.5e}{reference[1]:>13.5e}"
                     f"{(1 - reference[1] / error_h1) * 100:>10.1f}%")
        print(line)

    with open(output_path, "w") as f:
        f.write(f"# relative errors recomputed on {n_error} evaluation points, {problem_type} problem\n")
        f.write("checkpoint\tk\tep\tseed\tn\tnum_params\tL2-error\tH1-error")
        if n_error_reference:
            f.write(f"\tL2-error_n{n_error_reference}\tH1-error_n{n_error_reference}")
        f.write("\n")
        for row in rows:
            f.write(f"{row['name']}\t{row['k']}\t{row['ep']}\t{row['seed']}\t{row['n_per_dim']}\t"
                    f"{row['num_params']}\t{row['l2']:.12e}\t{row['h1']:.12e}")
            if row["reference"]:
                f.write(f"\t{row['reference'][0]:.12e}\t{row['reference'][1]:.12e}")
            f.write("\n")
    print(f"-> written to {output_path}")


if __name__ == "__main__":
    app()
