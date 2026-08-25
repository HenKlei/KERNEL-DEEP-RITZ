import numpy as np
import math
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed):
    """Seed the RNGs used by the experiments (torch on all devices, NumPy).

    Repeated runs of the same configuration with different seeds differ in the
    quadrature point sets drawn during training and, for neural networks, in the
    initialisation of the weights. Note that seeding fixes the random streams but
    does not guarantee bitwise reproducibility across different GPUs or library
    versions; it makes the repetitions of a hyperparameter study well-defined and
    independent of each other.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def reset_peak_memory():
    """Reset CUDA peak-memory tracking; no-op on CPU."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb():
    """Return CUDA peak allocated memory since last reset, in MB. Returns 0 on CPU."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    return 0.0


# Number of points whose distances to all centers are computed at once. With the
# largest quadrature sizes used here the full distance matrix would be several GB.
DISTANCE_CHUNK_SIZE = 50000


def drop_points_near_centers(x, centers, tol=1e-3):
    """Remove points of ``x`` that lie within ``tol`` of one of the ``centers``.

    Quadrature points coinciding with a center have to be excluded whenever second
    derivatives of the energy are taken: the kernel is evaluated at zero distance
    there, where its second derivative is not defined, and the resulting gradient
    of the energy with respect to the coefficients is NaN. Assembling the linear
    system only needs first derivatives and is therefore unaffected, which is why
    the optimizer and the matrix form discretise slightly different functionals
    unless the same filtered point set is used for both.

    The distances are computed in chunks: the full ``(len(x), len(centers))`` matrix
    is never materialised, which for the largest quadrature sizes used here would be
    several gigabytes on its own and dominate the peak memory of the whole run.
    """
    if centers is None:
        return x
    spatial_dim = x.shape[1]
    ctrs = centers[:, :spatial_dim]
    with torch.no_grad():
        masks = []
        for start in range(0, x.shape[0], DISTANCE_CHUNK_SIZE):
            chunk = x[start:start + DISTANCE_CHUNK_SIZE]
            masks.append(~(torch.cdist(chunk, ctrs) < tol).any(dim=1))
        x = x[torch.cat(masks) if len(masks) > 1 else masks[0]]
    x.requires_grad_()
    return x


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


# Number of evaluation points processed at once when computing error norms. The
# quadrature is a sum over the points, so chunking changes only the summation
# order; it keeps the memory of the (n_points, n_centers) kernel evaluation and of
# the autograd graph bounded, which is what makes fine error grids affordable.
ERROR_CHUNK_SIZE = 50000


def _evaluation_chunks(x, chunk_size):
    if not chunk_size or chunk_size <= 0 or x.shape[0] <= chunk_size:
        yield x
    else:
        for start in range(0, x.shape[0], chunk_size):
            yield x[start:start + chunk_size]


def compute_L2_norm(func, problem, n, chunk_size=ERROR_CHUNK_SIZE):
    x_vals = problem.domain.uniform_interior_points(n)
    total = 0.
    for chunk in _evaluation_chunks(x_vals, chunk_size):
        total += torch.sum(func(chunk) ** 2).item()
    return np.sqrt(problem.domain.volume_domain * total / x_vals.shape[0])


def compute_relative_L2_error(problem, model, n, chunk_size=ERROR_CHUNK_SIZE):
    err_norm = compute_L2_norm(lambda x: problem.reference_solution(x) - model(x), problem, n,
                               chunk_size=chunk_size)
    sol_norm = compute_L2_norm(problem.reference_solution, problem, n, chunk_size=chunk_size)
    return err_norm / sol_norm


def compute_L2_norm_bdry(func, problem, n):
    x_vals = problem.domain.uniform_boundary_points(n)
    y_vals = func(x_vals)
    return np.sqrt(torch.mean(y_vals ** 2).item())


def compute_relative_L2_error_bdry(problem, model, n):
    err_norm = compute_L2_norm_bdry(lambda x: problem.reference_solution(x) - model(x), problem, n)
    sol_norm = compute_L2_norm_bdry(problem.reference_solution, problem, n)
    return err_norm / sol_norm


def compute_H1_semi_norm(grad, problem, n, chunk_size=ERROR_CHUNK_SIZE):
    x_vals = problem.domain.uniform_interior_points(n)
    total = 0.
    for chunk in _evaluation_chunks(x_vals, chunk_size):
        total += torch.sum(torch.sum(grad(chunk) ** 2, dim=1)).item()
    return np.sqrt(problem.domain.volume_domain * total / x_vals.shape[0])


def compute_H1_norm(func, grad, problem, n, chunk_size=ERROR_CHUNK_SIZE):
    return np.sqrt(compute_L2_norm(func, problem, n, chunk_size=chunk_size)**2
                   + compute_H1_semi_norm(grad, problem, n, chunk_size=chunk_size)**2)


def compute_relative_H1_error(problem, model, n, chunk_size=ERROR_CHUNK_SIZE):
    err_norm = compute_H1_norm(lambda x: problem.reference_solution(x) - model(x),
                               lambda x: problem.gradient_reference_solution(x) - gradient(model, x),
                               problem, n, chunk_size=chunk_size)
    sol_norm = compute_H1_norm(problem.reference_solution, problem.gradient_reference_solution, problem, n,
                               chunk_size=chunk_size)
    return err_norm / sol_norm
