import numpy as np
import torch
import torch.nn as nn

from kernelDR.utils import gradient


def assemble_and_solve_system(problem, model_class, n_i, n_b, model_params={}, return_linear_system=False,
                              regularization=0., save_mat_path=None):
    x_i = problem.domain.random_interior_points(n_i)
    x_b = problem.domain.random_boundary_points(n_b)

    model_ansatz = model_class(problem.domain.dim, problem.output_dim, **model_params)
    model_test = model_class(problem.domain.dim, problem.output_dim, **model_params)

    num_coeffs = len(model_ansatz.coeffs)

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

    if save_mat_path is not None:
        np.save(save_mat_path + "original_A", A)
        np.save(save_mat_path + "b", b)

    A = A + regularization * np.eye(num_coeffs)

    if save_mat_path is not None:
        np.save(save_mat_path + "regularized_A", A)

    coeffs = np.linalg.solve(A, b)
    model = model_class(problem.domain.dim, problem.output_dim, **model_params)
    model.coeffs = nn.Parameter(torch.tensor(coeffs))

    if return_linear_system:
        return model, A, b
    return model
