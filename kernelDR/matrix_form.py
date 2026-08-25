import numpy as np
import scipy as sp
import torch
import torch.nn as nn

from kernelDR.utils import drop_points_near_centers, gradient


def basis_values_and_gradients(model, x, num_coeffs):
    """Values and gradients of all basis functions of a coefficient-linear model.

    The models used here depend linearly on their coefficients, so setting the
    coefficient matrix to the identity evaluates all basis functions in a single
    forward pass. The gradients then require one backward pass per basis function
    instead of one model evaluation per matrix entry.

    Returns (values, gradients, x) with shapes (N, M), (N, M, d) and (N, d); the
    returned points are the ones the derivatives were taken with respect to.
    """
    x = x.detach().clone().requires_grad_(True)
    model.coeffs = nn.Parameter(torch.eye(num_coeffs, dtype=x.dtype, device=x.device))

    values = model(x).reshape(x.shape[0], num_coeffs)
    ones = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
    gradients = torch.empty(x.shape[0], num_coeffs, x.shape[1], dtype=x.dtype, device=x.device)
    for m in range(num_coeffs):
        gradients[:, m, :] = torch.autograd.grad(values[:, m], x, ones, retain_graph=True)[0]

    return values.detach(), gradients.detach(), x.detach()


def _chunks(x, chunk_size):
    if chunk_size and chunk_size > 0:
        for start in range(0, x.shape[0], chunk_size):
            yield x[start:start + chunk_size]
    else:
        yield x


def assemble_system_vectorized(problem, model, x_i, x_b, num_coeffs, chunk_size=0):
    """Assemble the linear system by exploiting the linearity in the coefficients.

    Mathematically identical to assemble_system_loop, but the quadrature sums are
    evaluated as matrix products over all basis functions at once, which removes
    the double loop over the matrix entries. Note that the summation order differs,
    so the entries agree only up to floating point rounding.

    ``chunk_size`` bounds the memory of the (N, M, d) gradient tensor by splitting
    the quadrature points; the result is unaffected since the quadrature is a sum
    over the points.
    """
    dtype, device = x_i.dtype, x_i.device
    penalty = problem.penalty_parameter

    A = torch.zeros(num_coeffs, num_coeffs, dtype=dtype, device=device)
    b = torch.zeros(num_coeffs, dtype=dtype, device=device)

    # Interior contribution: diffusion * <grad u, grad v> + reaction * u * v, and
    # the source term of the right hand side.
    for chunk in _chunks(x_i, chunk_size):
        values, gradients, points = basis_values_and_gradients(model, chunk, num_coeffs)
        diffusion = problem.diffusion(points)
        reaction = problem.reaction(points)
        source = problem.source(points)

        A += torch.einsum('qmk,qnk->mn', gradients, diffusion[:, None, None] * gradients)
        A += values.T @ (reaction[:, None] * values)
        b += values.T @ source

    A *= problem.domain.volume_domain / x_i.shape[0]
    b *= problem.domain.volume_domain / x_i.shape[0]

    # Boundary contribution of the Nitsche-type penalty formulation.
    A_boundary = torch.zeros_like(A)
    b_boundary = torch.zeros_like(b)
    for chunk in _chunks(x_b, chunk_size):
        values, gradients, points = basis_values_and_gradients(model, chunk, num_coeffs)
        normals = problem.domain.outer_unit_normal(points)
        boundary_values = problem.dirichlet_boundary_values(points)

        normal_derivatives = torch.einsum('qmk,qk->qm', gradients, normals)

        A_boundary += normal_derivatives.T @ values
        A_boundary += values.T @ normal_derivatives
        A_boundary -= penalty * (values.T @ values)

        b_boundary += normal_derivatives.T @ boundary_values
        b_boundary -= penalty * (values.T @ boundary_values)

    A -= A_boundary * problem.domain.volume_boundary / x_b.shape[0]
    b -= b_boundary * problem.domain.volume_boundary / x_b.shape[0]

    return A.cpu().numpy(), b.cpu().numpy()


def assemble_system_loop(problem, model_ansatz, model_test, x_i, x_b, num_coeffs):
    """Assemble the linear system entry by entry.

    This is the original implementation, kept as a reference to validate the
    vectorized assembly against. It evaluates the full model once per matrix
    entry and is therefore orders of magnitude slower.
    """
    A = np.zeros((num_coeffs, num_coeffs))
    b = np.zeros(num_coeffs)

    for i in range(num_coeffs):
        e_i = torch.zeros(num_coeffs)
        e_i[i] = 1.
        model_test.coeffs = nn.Parameter(e_i)
        v_i = model_test(x_i)
        v_b = model_test(x_b)
        grads_v_i = gradient(model_test, x_i, y=v_i)
        grads_v_b = gradient(model_test, x_b, y=v_b)
        for j in range(num_coeffs):
            e_j = torch.zeros(num_coeffs)
            e_j[j] = 1.
            model_ansatz.coeffs = nn.Parameter(e_j)
            u_i = model_ansatz(x_i)
            u_b = model_ansatz(x_b)
            grads_u_i = gradient(model_ansatz, x_i, y=u_i)
            grads_u_b = gradient(model_ansatz, x_b, y=u_b)
            A[i, j] = problem.bilinear_form(x_i, u_i, v_i, x_b, u_b, v_b, grads_u_i, grads_v_i, grads_u_b, grads_v_b)
        b[i] = problem.right_hand_side(x_i, v_i, x_b, v_b, grads_v_i, grads_v_b)

    return A, b


def assemble_and_solve_system(problem, model_class, n_i, n_b, model_params={}, return_linear_system=False,
                              regularization=0., save_mat_path=None, solver='pos',
                              use_uniform_quadrature=False, assembly='vectorized', chunk_size=0):
    if use_uniform_quadrature:
        x_i = problem.domain.uniform_interior_points(n_i)
        x_b = problem.domain.uniform_boundary_points(n_b)
    else:
        x_i = problem.domain.random_interior_points(n_i)
        x_b = problem.domain.random_boundary_points(n_b)

    # Quadrature points coinciding with a center are dropped, as in the optimizer runs
    # (see drop_points_near_centers) and in the solver comparison. For a radial kernel
    # the gradient vanishes at zero distance, so such a point contributes a degenerate
    # row to the boundary terms. With a uniform grid commensurate with the centers this
    # happens for many points at once and spoils the assembled system.
    centers = model_params.get('ctrs')
    if centers is not None:
        n_i_raw, n_b_raw = x_i.shape[0], x_b.shape[0]
        x_i = drop_points_near_centers(x_i, centers)
        x_b = drop_points_near_centers(x_b, centers)
        dropped = (n_i_raw - x_i.shape[0]) + (n_b_raw - x_b.shape[0])
        if dropped:
            print(f"  dropped {dropped} quadrature point(s) coinciding with a center")

    model_ansatz = model_class(problem.domain.dim, problem.output_dim, **model_params)

    num_coeffs = len(model_ansatz.coeffs)

    if assembly == 'vectorized':
        A, b = assemble_system_vectorized(problem, model_ansatz, x_i, x_b, num_coeffs,
                                          chunk_size=chunk_size)
    elif assembly == 'loop':
        model_test = model_class(problem.domain.dim, problem.output_dim, **model_params)
        A, b = assemble_system_loop(problem, model_ansatz, model_test, x_i, x_b, num_coeffs)
    else:
        raise ValueError(f"Unknown assembly {assembly!r}; expected 'vectorized' or 'loop'.")

    if save_mat_path is not None:
        np.save(save_mat_path + "original_A", A)
        np.save(save_mat_path + "b", b)

    A = A + regularization * np.eye(num_coeffs)

    if save_mat_path is not None:
        np.save(save_mat_path + "regularized_A", A)

    coeffs = sp.linalg.solve(A, b, assume_a=solver)

    model = model_class(problem.domain.dim, problem.output_dim, **model_params)
    model.coeffs = nn.Parameter(torch.tensor(coeffs))

    if return_linear_system:
        return model, A, b
    return model
