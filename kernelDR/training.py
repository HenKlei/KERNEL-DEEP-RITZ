import numpy as np
import os
import torch

from kernelDR.utils import compute_relative_L2_error, compute_relative_H1_error
from datetime import datetime


def train_model(problem, model, n_i, n_b, epochs, optimizer, centers=None, scheduler=None,
                fixed_integration_points=False, flag_best_model=True, list_epochs_log=None, num_logs=40):
    if list_epochs_log is None:
        list_epochs_log = list(np.unique(np.geomspace(1, epochs, num=num_logs, dtype=int, endpoint=True)))

    best_loss = None
    best_epoch = 0

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
            tol = 1e-3
            with torch.no_grad():
                for c in centers:
                    sel = torch.linalg.norm(x - c, dim=-1) < tol
                    mask = np.ones(x.shape[0], dtype=bool)
                    mask[sel] = False
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

    for epoch in range(epochs + 1):
        loss = closure()

        # Check for NaN values. If detected, continue with next epoch.
        flag_nan = False
        total_norm = 0

        list_params = list(model.parameters())
        for idx_p, p in enumerate(list_params):
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

            if np.isnan(param_norm):
                print(f"nan detected in epoch {epoch}, continuing ...")
                flag_nan = True
                break

        if flag_nan:
            continue

        optimizer.step(closure)
        if scheduler is not None:
            scheduler.step()

        if epoch > .8 * epochs and flag_best_model:
            if not best_loss or loss < best_loss:
                best_loss = loss.item()
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_filename)

        list_loss.append(loss.item())

        if epoch in list_epochs_log:
            L2_error = compute_relative_L2_error(problem, model, n=10201)
            H1_error = compute_relative_H1_error(problem, model, n=10201)

            list_L2.append(L2_error)
            list_H1.append(H1_error)

            print(datetime.now().strftime("%H:%M:%S"), 
                  "epoch: {}\tloss: {:.3e}\tlearning rate {:.3e}\tL2 error: {:.3e}\tH1 error: {:.3e}".format(
                      epoch, loss.item(), optimizer.param_groups[0]['lr'], L2_error, H1_error))

    if flag_best_model:
        print(f"Best epoch: {best_epoch}\tBest loss: {best_loss}")
        model.load_state_dict(torch.load(best_model_filename))
    else:
        best_loss = loss.item()

    return best_loss, list_loss, list_L2, list_H1
