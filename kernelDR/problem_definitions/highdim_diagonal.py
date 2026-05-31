"""High-dimensional sinus-along-diagonal example (non-parametric).

Reproduces — at fixed parameter μ = 0 — the high-dimensional setup of
Section 4.3 of Haasdonk, Wenzel, Santin "Kernel-based Greedy
Approximation of Parametric Elliptic BVPs" (arXiv:2507.06731): a
Poisson problem on the unit hypercube whose exact solution is a
sinus wave depending only on the diagonal direction.

The diagonal-dominant structure makes this a target case for
problem-adapted (e.g. anisotropic or two-layer) kernels.
"""
import math

import numpy as np
import torch

from kernelDR.problem_definitions.base import DeepRitzExample
from kernelDR.problem_definitions.domains import UnitHypercubeDomain
from kernelDR.utils import gradient


class HighDimDiagonalExample(DeepRitzExample):
    """Poisson problem -Δu = f on Ω = (0,1)^{d_x} with u = g on ∂Ω,
    where the exact solution is

        u(x) = sin(<x, κ>),    κ = (κ_scalar, ..., κ_scalar)^T

    with κ_scalar := frequency / d_x. The default frequency = π
    corresponds to μ = 0 in the parametric paper (half period along the
    diagonal of the cube). The solution varies only along the diagonal
    direction v_1 = (1, ..., 1)^T / sqrt(d_x).
    """

    def __init__(self, d_x, frequency=math.pi, penalty_parameter=1.,
                 device=torch.device('cpu'), output_dim=1):
        super().__init__(penalty_parameter=penalty_parameter, device=device, output_dim=output_dim)
        self.d_x = d_x
        self.frequency = frequency
        self._kappa_scalar = frequency / d_x
        self.domain = UnitHypercubeDomain(dim=d_x, device=device)

    def _phase(self, x):
        return self._kappa_scalar * x.sum(dim=-1)

    def diffusion(self, x):
        return torch.ones(x.shape[0], device=x.device)

    def reaction(self, x):
        return torch.zeros(x.shape[0], device=x.device)

    def source(self, x):
        # -Δu = |κ|^2 · u with |κ|^2 = d_x · κ_scalar^2 = frequency^2 / d_x
        return self.d_x * self._kappa_scalar ** 2 * torch.sin(self._phase(x))

    def dirichlet_boundary_values(self, x):
        return torch.sin(self._phase(x))

    def reference_solution(self, x):
        return torch.sin(self._phase(x))

    def gradient_reference_solution(self, x):
        cos_phase = torch.cos(self._phase(x))
        return self._kappa_scalar * cos_phase[..., None] * torch.ones_like(x)

    def compute_relative_L2_error(self, model, n, n_repeats=5, chunk_size=10000):
        """Monte-Carlo relative L^2 error.

        Overrides the default uniform-grid quadrature because in d_x dimensions
        a tensor-product grid with O(n) total points has only O(n^{1/d_x})
        points per axis, which is unusable for d_x >= 5 or so. Random sampling
        gives unbiased estimates with standard error O(1/sqrt(n * n_repeats)).
        Sampling is split into chunks of ``chunk_size`` to keep the
        ``(chunk_size, n_centers)`` kernel matrix small enough to fit in
        memory for large kernel models.
        """
        sq_err, sq_ref = [], []
        for _ in range(n_repeats):
            tot_err, tot_ref, tot_n = 0.0, 0.0, 0
            for start in range(0, n, chunk_size):
                cur = min(chunk_size, n - start)
                x = self.domain.random_interior_points(cur).detach()
                with torch.no_grad():
                    y_model = model(x)
                    y_ref = self.reference_solution(x)
                tot_err += torch.sum((y_ref - y_model) ** 2).item()
                tot_ref += torch.sum(y_ref ** 2).item()
                tot_n += cur
                del x, y_model, y_ref
            sq_err.append(tot_err / tot_n)
            sq_ref.append(tot_ref / tot_n)
        return float(np.sqrt(np.mean(sq_err)) / np.sqrt(np.mean(sq_ref)))

    def compute_relative_H1_error(self, model, n, n_repeats=5, chunk_size=10000):
        """Monte-Carlo relative H^1 error (same rationale as L^2; chunked
        to bound the autograd graph size at large n_centers)."""
        sq_L2, sq_H1_semi, sq_L2_ref, sq_H1_semi_ref = [], [], [], []
        for _ in range(n_repeats):
            t_L2, t_H1, t_L2_ref, t_H1_ref, tot_n = 0.0, 0.0, 0.0, 0.0, 0
            for start in range(0, n, chunk_size):
                cur = min(chunk_size, n - start)
                x = self.domain.random_interior_points(cur)
                y_model = model(x)
                grad_model = gradient(model, x, y=y_model)
                with torch.no_grad():
                    y_ref = self.reference_solution(x)
                    grad_ref = self.gradient_reference_solution(x)
                    t_L2 += torch.sum((y_ref - y_model.detach()) ** 2).item()
                    t_H1 += torch.sum(torch.sum((grad_ref - grad_model.detach()) ** 2, dim=1)).item()
                    t_L2_ref += torch.sum(y_ref ** 2).item()
                    t_H1_ref += torch.sum(torch.sum(grad_ref ** 2, dim=1)).item()
                tot_n += cur
                del x, y_model, grad_model, y_ref, grad_ref
            sq_L2.append(t_L2 / tot_n)
            sq_H1_semi.append(t_H1 / tot_n)
            sq_L2_ref.append(t_L2_ref / tot_n)
            sq_H1_semi_ref.append(t_H1_ref / tot_n)
        err = np.sqrt(np.mean(sq_L2) + np.mean(sq_H1_semi))
        ref = np.sqrt(np.mean(sq_L2_ref) + np.mean(sq_H1_semi_ref))
        return float(err / ref)
