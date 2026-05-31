import numpy as np
import os
import time
import torch

from kernelDR.utils import compute_relative_L2_error, compute_relative_H1_error
from datetime import datetime


def train_model(problem, model, n_i, n_b, epochs, optimizer, centers=None, scheduler=None,
                fixed_integration_points=False, flag_best_model=True, list_epochs_log=None, num_logs=40,
                n_error=10201, early_stopping_patience=0, matrix_snapshot_interval=10,
                matrix_trajectory_path=None, training_metrics_path=None):
    if list_epochs_log is None:
        list_epochs_log = list(np.unique(np.geomspace(1, epochs, num=num_logs, dtype=int, endpoint=True)))

    best_loss = None
    best_epoch = 0
    early_stopping_counter = 0

    best_model_filename = "best_deep_ritz.mdl"

    if centers is None:
        # check whether model has attribute ctrs
        if hasattr(model, "ctrs"):
            centers = model.ctrs
        else:
            centers = None

    if os.path.exists(best_model_filename):
        os.remove(best_model_filename)

    def remove_centers(x):
        if centers is None:
            return x
        else:
            spatial_dim = x.shape[1]
            tol = 1e-3
            with torch.no_grad():
                for c in centers:
                    sel = torch.linalg.norm(x - c[:spatial_dim], dim=-1) < tol
                    mask = ~sel
                    x = x[mask]
            x.requires_grad_()
            return x

    # generate the data set
    if fixed_integration_points:
        x_i = problem.domain.random_interior_points(n_i)
        x_i = remove_centers(x_i)
        x_b = problem.domain.random_boundary_points(n_b)
        x_b = remove_centers(x_b)

    list_loss = []
    list_L2 = []
    list_H1 = []
    matrix_snapshots = []
    log_epochs = []
    log_losses = []
    log_lrs = []

    def closure():
        optimizer.zero_grad()

        if not fixed_integration_points:
            x_i = problem.domain.random_interior_points(n_i)
            x_i = remove_centers(x_i)
            x_b = problem.domain.random_boundary_points(n_b)
            x_b = remove_centers(x_b)
        else:
            x_i = problem.domain.uniform_interior_points(n_i)
            x_i = remove_centers(x_i)
            x_b = problem.domain.uniform_boundary_points(n_b)
            x_b = remove_centers(x_b)

        loss = problem.energy(model, x_i, x_b)
        loss.backward()
        return loss

    t_start = time.time()

    for epoch in range(epochs + 1):
        loss = closure()

        # Check for NaN values. If detected, continue with next epoch.
        flag_nan = False
        total_norm = 0

        list_params = list(model.parameters())
        for idx_p, p in enumerate(list_params):
            if p.grad is None:
                continue
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

            if torch.isnan(param_norm):
                print(f"nan detected in epoch {epoch}, continuing ...")
                flag_nan = True
                break

        if flag_nan:
            continue

        optimizer.step(closure)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss.item())
            else:
                scheduler.step()

        current_loss = loss.item()

        if (hasattr(model, "matrix") and matrix_snapshot_interval > 0 and epoch > 0
                and epoch % matrix_snapshot_interval == 0):
            matrix_snapshots.append(model.matrix.detach().cpu().numpy()[None, ...])

        if flag_best_model and epoch > .8 * epochs:
            if best_loss is None or current_loss < best_loss:
                best_loss = current_loss
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_filename)

        list_loss.append(current_loss)

        # Early stopping
        if early_stopping_patience > 0:
            if best_loss is None or current_loss < best_loss - 1e-10:
                if not flag_best_model:
                    best_loss = current_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
                break

        if epoch in list_epochs_log:
            if hasattr(problem, 'compute_relative_L2_error'):
                L2_error = problem.compute_relative_L2_error(model, n=n_error)
            else:
                L2_error = compute_relative_L2_error(problem, model, n=n_error)
            if hasattr(problem, 'compute_relative_H1_error'):
                H1_error = problem.compute_relative_H1_error(model, n=n_error)
            else:
                H1_error = compute_relative_H1_error(problem, model, n=n_error)

            list_L2.append(L2_error)
            list_H1.append(H1_error)
            log_epochs.append(epoch)
            log_losses.append(current_loss)
            log_lrs.append(optimizer.param_groups[0]['lr'])

            print(datetime.now().strftime("%H:%M:%S"), 
                  "epoch: {}\tloss: {:.3e}\tlearning rate {:.3e}\tL2 error: {:.3e}\tH1 error: {:.3e}".format(
                      epoch, loss.item(), optimizer.param_groups[0]['lr'], L2_error, H1_error))

    elapsed_time = time.time() - t_start

    if matrix_trajectory_path and matrix_snapshots:
        matrix_trajectory = np.concatenate(matrix_snapshots, axis=0)
        trajectory_dir = os.path.dirname(matrix_trajectory_path)
        if trajectory_dir:
            os.makedirs(trajectory_dir, exist_ok=True)
        np.save(matrix_trajectory_path, matrix_trajectory)
        print(f"Saved matrix trajectory to {matrix_trajectory_path} with shape {matrix_trajectory.shape}")

    if training_metrics_path and log_epochs:
        metrics_dir = os.path.dirname(training_metrics_path)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        np.savez(
            training_metrics_path,
            epoch=np.asarray(log_epochs, dtype=int),
            loss=np.asarray(log_losses, dtype=float),
            learning_rate=np.asarray(log_lrs, dtype=float),
            l2_error=np.asarray(list_L2, dtype=float),
            h1_error=np.asarray(list_H1, dtype=float),
        )
        print(f"Saved training metrics to {training_metrics_path} with {len(log_epochs)} checkpoints")

    if flag_best_model:
        print(f"Best epoch: {best_epoch}\tBest loss: {best_loss}")
        model.load_state_dict(torch.load(best_model_filename))
    else:
        best_loss = loss.item()

    print(f"Training time: {elapsed_time:.2f}s ({epoch} epochs)")

    return best_loss, list_loss, list_L2, list_H1, epoch, elapsed_time
