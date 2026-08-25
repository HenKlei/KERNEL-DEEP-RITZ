import numpy as np
import os
import time
import torch

from kernelDR.utils import compute_relative_L2_error, compute_relative_H1_error, drop_points_near_centers
from datetime import datetime


def train_model(problem, model, n_i, n_b, epochs, optimizer, centers=None, scheduler=None,
                fixed_integration_points=False, flag_best_model=True, list_epochs_log=None, num_logs=40,
                n_error=10201, early_stopping_patience=0, matrix_snapshot_interval=10,
                matrix_trajectory_path=None, training_metrics_path=None, best_model_path=None,
                remove_center_points=True, fixed_x_i=None, fixed_x_b=None):
    """Minimise the energy functional of ``problem`` over the parameters of ``model``.

    With ``fixed_integration_points`` the same quadrature points are used in every
    epoch, otherwise a new set is drawn per epoch. By default the fixed set is the
    uniform tensor-product grid; passing ``fixed_x_i`` / ``fixed_x_b`` uses a given
    point set instead, which allows a fixed *random* rule and lets an optimizer run
    be compared against a linear system assembled from exactly the same points.

    ``remove_center_points`` drops quadrature points that coincide with a center.
    Set it to False to minimise exactly the functional that the matrix-form
    assembly discretises, which does not remove any points.
    """
    if list_epochs_log is None:
        list_epochs_log = list(np.unique(np.geomspace(1, epochs, num=num_logs, dtype=int, endpoint=True)))

    best_loss = None
    best_epoch = 0
    early_stopping_counter = 0

    # Unique per process, so that several runs launched concurrently in the same
    # working directory (e.g. a seed sweep) do not overwrite or delete each
    # other's checkpoint.
    keep_checkpoint = best_model_path is not None
    if best_model_path is None:
        best_model_path = f"best_deep_ritz_{os.getpid()}.mdl"
    best_model_filename = best_model_path

    if centers is None:
        # check whether model has attribute ctrs
        if hasattr(model, "ctrs"):
            centers = model.ctrs
        else:
            centers = None

    if flag_best_model and os.path.exists(best_model_filename):
        os.remove(best_model_filename)

    def remove_centers(x):
        if centers is None or not remove_center_points:
            return x
        return drop_points_near_centers(x, centers)

    # Generate the fixed quadrature once, outside the optimization loop: either the
    # given point set, or by default the uniform tensor-product grid.
    if fixed_integration_points:
        x_i_fixed = fixed_x_i if fixed_x_i is not None else problem.domain.uniform_interior_points(n_i)
        x_b_fixed = fixed_x_b if fixed_x_b is not None else problem.domain.uniform_boundary_points(n_b)
        x_i_fixed = remove_centers(x_i_fixed).detach().clone().requires_grad_()
        x_b_fixed = remove_centers(x_b_fixed).detach().clone().requires_grad_()

    list_loss = []
    list_L2 = []
    list_H1 = []
    matrix_snapshots = []
    log_epochs = []
    log_losses = []
    log_lrs = []

    def closure():
        optimizer.zero_grad()

        if fixed_integration_points:
            x_i, x_b = x_i_fixed, x_b_fixed
        else:
            x_i = remove_centers(problem.domain.random_interior_points(n_i))
            x_b = remove_centers(problem.domain.random_boundary_points(n_b))

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
        if not keep_checkpoint:
            os.remove(best_model_filename)
    else:
        best_loss = loss.item()

    print(f"Training time: {elapsed_time:.2f}s ({epoch} epochs)")

    return best_loss, list_loss, list_L2, list_H1, epoch, elapsed_time
