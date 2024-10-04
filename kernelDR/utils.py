import numpy as np
import math
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def my_arctan(x1, x2):
    # Returns angle in the interval [0, 2pi]
    phi = np.arctan2(x2, x1)
    phi += (phi < 0) * 2 * math.pi
    return phi


def my_arctan_torch(x1, x2):
    # Returns angle in the interval [0, 2pi]
    phi = torch.arctan2(x2, x1)
    phi += (phi < 0) * 2 * math.pi
    return phi


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = my_arctan(x, y)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def gradient(f, x, y=None):
    if y is None:
        y = f(x)
    return torch.autograd.grad(y, x, torch.ones_like(y), retain_graph=True, create_graph=True)[0]


def compute_L2_norm(func, problem, n):
    x_vals = problem.domain.uniform_interior_points(n)
    y_vals = func(x_vals)
    return np.sqrt(problem.domain.volume_domain * torch.mean(y_vals ** 2).item())


def compute_relative_L2_error(problem, model, n):
    err_norm = compute_L2_norm(lambda x: problem.reference_solution(x) - model(x), problem, n)
    sol_norm = compute_L2_norm(problem.reference_solution, problem, n)
    return err_norm / sol_norm


def compute_L2_norm_bdry(func, problem, n):
    x_vals = problem.domain.uniform_boundary_points(n)
    y_vals = func(x_vals)
    return np.sqrt(torch.mean(y_vals ** 2).item())


def compute_relative_L2_error_bdry(problem, model, n):
    err_norm = compute_L2_norm_bdry(lambda x: problem.reference_solution(x) - model(x), problem, n)
    sol_norm = compute_L2_norm_bdry(problem.reference_solution, problem, n)
    return err_norm / sol_norm


def compute_H1_semi_norm(grad, problem, n):
    x_vals = problem.domain.uniform_interior_points(n)
    y_grad = grad(x_vals)
    return np.sqrt(problem.domain.volume_domain * torch.mean(torch.sum(y_grad**2, dim=1)).item())


def compute_H1_norm(func, grad, problem, n):
    return np.sqrt(compute_L2_norm(func, problem, n)**2 + compute_H1_semi_norm(grad, problem, n)**2)


def compute_relative_H1_error(problem, model, n):
    err_norm = compute_H1_norm(lambda x: problem.reference_solution(x) - model(x),
                               lambda x: problem.gradient_reference_solution(x) - gradient(model, x),
                               problem, n)
    sol_norm = compute_H1_norm(problem.reference_solution, problem.gradient_reference_solution, problem, n)
    return err_norm / sol_norm
