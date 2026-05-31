"""Deep Ritz on a high-dimensional sinus-along-diagonal Poisson problem.

This is the non-parametric variant of the example in Sec. 4.3 of
arXiv:2507.06731 (parameter fixed to μ = 0). Centers are pre-selected
by a P-greedy procedure on a candidate point pool, then a flat or
two-layer kernel model is trained by minimising the Deep Ritz energy.
The motivation is that the solution varies only along the diagonal of
(0, 1)^{d_x}: an isotropic kernel suffers from the curse of
dimensionality, while a two-layer kernel can learn the dominant
direction during training.
"""
from datetime import datetime
from typing import Annotated

import cyclopts
import math
import numpy as np
import os
import time
import torch
from torch import optim
from torch.optim import lr_scheduler

from kernelDR.models.model_kernels import FlatKernelModel, TwoLayerKernelModel
from kernelDR.problem_definitions.highdim_diagonal import HighDimDiagonalExample
from kernelDR.training import train_model
from kernelDR.utils import count_parameters, peak_memory_mb, reset_peak_memory
from kernelDR.experiments.plot_utils import save_settings


app = cyclopts.App()


@torch.no_grad()
def p_greedy(kernel, candidates, n_max):
    """Pick ``n_max`` indices from ``candidates`` by P-greedy.

    Iteratively selects the candidate that maximises the current power
    function in the kernel's native space. Implementation uses the
    Newton-basis update P_n^2(x) = P_{n-1}^2(x) - v_n(x)^2 with
    v_n(x) = (k(x, x_n) - sum_{i<n} v_i(x_n) v_i(x)) / P_{n-1}(x_n).
    """
    n_cand = candidates.shape[0]
    assert n_max <= n_cand, "n_max exceeds number of candidates"

    diag = float(kernel.eval(candidates[:1], candidates[:1]).item())
    P2 = torch.full((n_cand,), diag, device=candidates.device, dtype=candidates.dtype)
    V = torch.zeros((n_cand, n_max), device=candidates.device, dtype=candidates.dtype)
    selected = []
    chosen_mask = torch.zeros(n_cand, dtype=torch.bool, device=candidates.device)

    for n in range(n_max):
        masked = P2.clone()
        masked[chosen_mask] = -float('inf')
        idx = int(torch.argmax(masked).item())
        if P2[idx].item() <= 0.:
            break
        selected.append(idx)
        chosen_mask[idx] = True

        k_col = kernel.eval(candidates, candidates[idx:idx + 1]).squeeze(-1)
        if n == 0:
            v_n = k_col / torch.sqrt(P2[idx])
        else:
            v_n = (k_col - V[:, :n] @ V[idx, :n]) / torch.sqrt(P2[idx])
        V[:, n] = v_n
        P2 = torch.clamp(P2 - v_n ** 2, min=0.)

    return selected


@torch.no_grad()
def print_trainable_parameters(model, stage, n_centers):
    print(datetime.now().strftime("%H:%M:%S"),
          f"Trainable parameters ({stage}) for n_centers={n_centers}:")
    with np.printoptions(precision=6, suppress=True, linewidth=1000):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            print(datetime.now().strftime("%H:%M:%S"),
                  f"  {name} shape={tuple(param.shape)}")
            print(param.detach().cpu().numpy())


@app.default
def main(
    d_x: Annotated[int, cyclopts.Parameter(help="Spatial dimension.")] = 10,
    model_type: Annotated[str, cyclopts.Parameter(help="Model type: 'flat' or '2l'.")] = "2l",
    kernel: Annotated[str, cyclopts.Parameter(help="Kernel type.")] = "matern",
    k_smoothness: Annotated[int, cyclopts.Parameter(help="Kernel smoothness (matern/wendland).")] = 2,
    ep: Annotated[float, cyclopts.Parameter(help="Kernel shape parameter.")] = 1.0,
    frequency: Annotated[float, cyclopts.Parameter(help="Frequency of the diagonal sinus (μ=0 ⇒ π).")] = math.pi,
    penalty_parameter: Annotated[float, cyclopts.Parameter(help="Boundary penalty.")] = 100.0,
    n_i: Annotated[int, cyclopts.Parameter(help="Interior sample points per Ritz batch.")] = 10000,
    n_b: Annotated[int, cyclopts.Parameter(help="Boundary sample points per Ritz batch.")] = 1000,
    optimizer_type: Annotated[str, cyclopts.Parameter(help="Optimizer: 'adam' or 'lbfgs'.")] = "adam",
    n_epochs: Annotated[int, cyclopts.Parameter(help="Training epochs.")] = 1000,
    lr: Annotated[float, cyclopts.Parameter(help="Initial learning rate (for coeffs).")] = 5e-2,
    matrix_lr_mult: Annotated[float, cyclopts.Parameter(help="LR multiplier for the 2L linear layer (Adam only). Effective matrix LR = lr * matrix_lr_mult.")] = 1.0,
    gamma: Annotated[float, cyclopts.Parameter(help="LR decay factor (Adam only).")] = 0.5,
    n_candidates_interior: Annotated[int, cyclopts.Parameter(help="Interior candidate pool for P-greedy.")] = 5000,
    n_candidates_boundary: Annotated[int, cyclopts.Parameter(help="Boundary candidate pool for P-greedy.")] = 5000,
    list_n_centers: Annotated[tuple[int, ...], cyclopts.Parameter(help="Center counts to sweep.")] = (10, 20, 30),
    n_error: Annotated[int, cyclopts.Parameter(help="Spatial error eval points.")] = 10000,
    fixed_integration_points: Annotated[bool, cyclopts.Parameter(help="Use deterministic samples.")] = False,
    num_logs: Annotated[int, cyclopts.Parameter(help="Convergence log frequency.")] = 50,
    early_stopping_patience: Annotated[int, cyclopts.Parameter(help="Early-stop patience (0 to disable).")] = 0,
    seed: Annotated[int, cyclopts.Parameter(help="RNG seed for reproducibility.")] = 0,
    matrix_init: Annotated[str, cyclopts.Parameter(help="Path to a .npy file or .npz with key 'matrix' to initialise the 2L linear layer (ignored for flat).")] = "",
    freeze_matrix: Annotated[bool, cyclopts.Parameter(help="Freeze the 2L linear layer (only train coeffs). Only meaningful together with --matrix-init.")] = False,
    results_dir: Annotated[str, cyclopts.Parameter(help="Output dir.")] = "",
):
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    torch.manual_seed(seed)

    if not results_dir:
        results_dir = f"results_highdim_diagonal_d{d_x}_{model_type}/"
    if not results_dir.endswith("/"):
        results_dir += "/"
    os.makedirs(results_dir, exist_ok=True)
    save_settings(results_dir, locals())

    print(f"Cuda available: {torch.cuda.is_available()}")

    problem = HighDimDiagonalExample(d_x=d_x, frequency=frequency,
                                     penalty_parameter=penalty_parameter, device=device)

    # P-greedy candidate pool: random interior + random boundary points.
    cand_int = problem.domain.random_interior_points(n_candidates_interior).detach()
    cand_bnd = problem.domain.random_boundary_points(n_candidates_boundary).detach()
    candidates = torch.vstack([cand_int, cand_bnd]).detach()

    # Use the same kernel family for P-greedy as for the trained model.
    ref_model = FlatKernelModel(d_x, problem.output_dim, str_kernel=kernel,
                                k_smoothness=k_smoothness, ctrs=candidates[:1].clone(),
                                ep=ep, flag_lagrange=False)
    p_kernel = ref_model.kernel

    n_max = max(list_n_centers)
    print(datetime.now().strftime("%H:%M:%S"),
          f"P-greedy: {candidates.shape[0]} candidates -> {n_max} centers ...")
    reset_peak_memory()
    t_pgreedy_start = time.time()
    sel_indices = p_greedy(p_kernel, candidates, n_max)
    pgreedy_time_s = time.time() - t_pgreedy_start
    pgreedy_peak_mem_mb = peak_memory_mb()
    selected_centers_full = candidates[sel_indices].detach()
    print(datetime.now().strftime("%H:%M:%S"),
          f"  selected {len(sel_indices)} centers in {pgreedy_time_s:.2f}s "
          f"(peak GPU mem {pgreedy_peak_mem_mb:.1f} MB; "
          f"interior pool: {n_candidates_interior}, boundary pool: {n_candidates_boundary})")

    with open(results_dir + "pgreedy_cost.txt", "w") as f:
        f.write("n_candidates\tn_selected\ttime_setup_s\tpeak_mem_mb\n")
        f.write(f"{candidates.shape[0]}\t{len(sel_indices)}\t{pgreedy_time_s:.4f}\t{pgreedy_peak_mem_mb:.2f}\n")

    with open(results_dir + "summary.txt", "w") as f:
        f.write("n_centers\tnum_params\tfinal_loss\tL2_err\tH1_err\ttime_train_s\tpeak_mem_mb\n")

    if model_type == "2l":
        with open(results_dir + "anisotropy_alignment.txt", "w") as f:
            f.write("n_centers\tv1_alignment_ratio\ttrace_BtB\n")

    for n_centers in list_n_centers:
        if n_centers > selected_centers_full.shape[0]:
            print(f"Skipping n_centers={n_centers}: only {selected_centers_full.shape[0]} selected")
            continue

        centers = selected_centers_full[:n_centers].clone().detach()
        centers.requires_grad_()

        if model_type == "2l":
            model = TwoLayerKernelModel(d_x, problem.output_dim, str_kernel=kernel,
                                        k_smoothness=k_smoothness, ctrs=centers, ep=ep)
            if matrix_init:
                if matrix_init.endswith(".npz"):
                    B_init = np.load(matrix_init)["matrix"]
                else:
                    B_init = np.load(matrix_init)
                B_init = np.asarray(B_init)
                if B_init.shape != (d_x, d_x):
                    raise ValueError(f"matrix_init has shape {B_init.shape}, expected ({d_x}, {d_x})")
                model.matrix.data = torch.from_numpy(B_init).to(model.matrix.device).to(model.matrix.dtype)
                if freeze_matrix:
                    model.matrix.requires_grad_(False)
                print(datetime.now().strftime("%H:%M:%S"),
                      f"  warm-started matrix from {matrix_init}"
                      f"{' (frozen)' if freeze_matrix else ''}")
        else:
            model = FlatKernelModel(d_x, problem.output_dim, str_kernel=kernel,
                                    k_smoothness=k_smoothness, ctrs=centers, ep=ep,
                                    flag_lagrange=False)

        if optimizer_type == "lbfgs":
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.LBFGS(trainable, lr=lr, max_iter=20,
                                    line_search_fn="strong_wolfe")
            scheduler = None
        else:
            if model_type == "2l" and model.matrix.requires_grad:
                # Higher LR on the linear layer so its updates aren't drowned
                # out by the n_centers-many coefficient gradients.
                matrix_lr = lr * matrix_lr_mult
                optimizer = optim.Adam([
                    {"params": [model.coeffs], "lr": lr},
                    {"params": [model.matrix], "lr": matrix_lr},
                ])
                print(datetime.now().strftime("%H:%M:%S"),
                      f"  param-group LRs: coeffs={lr}, matrix={matrix_lr}")
            else:
                trainable = [p for p in model.parameters() if p.requires_grad]
                optimizer = optim.Adam(trainable, lr=lr)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma,
                                                      patience=200, min_lr=1e-6)

        num_params = count_parameters(model)
        print(datetime.now().strftime("%H:%M:%S"),
              f"Training {model_type} model with {n_centers} centers, "
              f"{num_params} trainable params (d_x={d_x}) ...")

        reset_peak_memory()
        matrix_trajectory_path = ""
        training_metrics_path = results_dir + f"training_metrics_n{n_centers}.npz"
        if model_type == "2l":
            matrix_trajectory_path = results_dir + f"matrix_trajectory_n{n_centers}.npy"
        final_loss, list_loss, list_L2, list_H1, actual_epochs, elapsed_time = train_model(
            problem, model, n_i, n_b, n_epochs, optimizer, centers,
            scheduler=scheduler, fixed_integration_points=fixed_integration_points,
            flag_best_model=False, num_logs=num_logs,
            early_stopping_patience=early_stopping_patience, n_error=n_error,
            matrix_snapshot_interval=10, matrix_trajectory_path=matrix_trajectory_path,
            training_metrics_path=training_metrics_path)
        run_peak_mem_mb = peak_memory_mb()

        if model_type == "2l":
            B_print = model.matrix.detach().cpu().numpy()
            _, singular_values, _ = np.linalg.svd(B_print, full_matrices=False)
            with np.printoptions(precision=3, suppress=True, linewidth=1000):
                print(datetime.now().strftime("%H:%M:%S"),
                      f"Optimized matrix for n_centers={n_centers}:\n{B_print}")
                print(datetime.now().strftime("%H:%M:%S"),
                      f"Singular values for n_centers={n_centers}:\n{singular_values}")

        list_epochs_log = list(np.unique(np.geomspace(1, actual_epochs, num=num_logs,
                                                     dtype=int, endpoint=True)))
        list_loss_subset = [list_loss[idx] for idx in list_epochs_log if idx < len(list_loss)]

        with open(results_dir + f"conv_results_n{n_centers}.txt", "w") as f:
            f.write("i\tepoch\tloss\tl2_err\th1_err\n")
            for i, (ep_, ls, l2, h1) in enumerate(zip(list_epochs_log, list_loss_subset,
                                                      list_L2, list_H1)):
                f.write(f"{i}\t{ep_}\t{ls:.5e}\t{l2:.5e}\t{h1:.5e}\n")

        l2_final = list_L2[-1] if list_L2 else float('nan')
        h1_final = list_H1[-1] if list_H1 else float('nan')
        with open(results_dir + "summary.txt", "a") as f:
            f.write(f"{n_centers}\t{num_params}\t{final_loss:.5e}\t"
                    f"{l2_final:.5e}\t{h1_final:.5e}\t{elapsed_time:.2f}\t{run_peak_mem_mb:.2f}\n")

        # Save trained model state (centers + coeffs, plus the learned linear
        # layer for 2L models) so the anisotropy can be inspected post-hoc.
        state = {
            "centers": model.ctrs.detach().cpu().numpy(),
            "coeffs": model.coeffs.detach().cpu().numpy(),
            "model_type": model_type,
            "d_x": d_x,
            "n_centers": n_centers,
        }
        if model_type == "2l":
            B = model.matrix.detach().cpu().numpy()
            state["matrix"] = B
            np.save(results_dir + f"matrix_n{n_centers}.npy", B)
            # Quick alignment diagnostic: how much of B^T B's energy is
            # captured by the diagonal direction v_1 = (1,...,1)/sqrt(d).
            BtB = B.T @ B
            v1 = np.ones(d_x) / np.sqrt(d_x)
            energy_v1 = float(v1 @ BtB @ v1)
            energy_total = float(np.trace(BtB))
            ratio = energy_v1 / energy_total if energy_total > 0 else float("nan")
            print(datetime.now().strftime("%H:%M:%S"),
                  f"  v_1^T B^T B v_1 / tr(B^T B) = {ratio:.4f} "
                  f"(1.0 = pure diagonal direction; 1/d = {1.0 / d_x:.4f} = isotropic)")
            with open(results_dir + "anisotropy_alignment.txt", "a") as f:
                f.write(f"{n_centers}\t{ratio:.6f}\t{energy_total:.6e}\n")
        np.savez(results_dir + f"model_state_n{n_centers}.npz", **state)

        del model, optimizer, scheduler, centers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    app()
