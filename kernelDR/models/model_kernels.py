import math
import numpy as np
import torch
import torch.nn as nn

# Shim for NumPy >= 2.0: vkoga_2L's Wendland kernel calls ``np.math.factorial``,
# which was removed in NumPy 2.0. Restoring the alias keeps the library
# working without forking it.
if not hasattr(np, "math"):
    np.math = math

from vkoga_2L import tkernels  # noqa: E402


class FlatKernelModel(nn.Module):
    def __init__(self, in_N, out_N, str_kernel='matern', k_smoothness=2, ctrs=None, ep=1, flag_lagrange=False):
        super().__init__()
        # set parameters
        self.in_N = in_N
        self.out_N = out_N
        self.str_kernel = str_kernel
        self.k_smoothness = k_smoothness
        self.ep = ep
        self.flag_lagrange = flag_lagrange

        assert self.str_kernel in ('matern', 'wendland'), 'Pick other kernel!'
        if self.str_kernel == 'matern':
            self.kernel = tkernels.Matern(k=self.k_smoothness, ep=self.ep, flag_normalize_x=True, flag_normalize_y=True)
        elif self.str_kernel == 'wendland':
            self.kernel = tkernels.Wendland(k=self.k_smoothness, ep=self.ep, d=self.in_N, flag_normalize_y=True)

        if ctrs is None:
            ctrs = torch.rand(100, self.in_N) * 2 - 1

        self.register_buffer('ctrs', ctrs)
        self.coeffs = nn.Parameter(torch.zeros(self.ctrs.shape[0], self.out_N))

        if self.flag_lagrange:
            A = self.kernel.eval(self.ctrs, self.ctrs)
            self.register_buffer('invA', torch.inverse(A).detach().clone())

    def forward(self, x):
        if self.flag_lagrange:
            return (self.kernel.eval(x, self.ctrs) @ self.invA @ self.coeffs).squeeze()
        else:
            return (self.kernel.eval(x, self.ctrs) @ self.coeffs).squeeze()


class TwoLayerKernelModel(nn.Module):
    def __init__(self, in_N, out_N, str_kernel='matern', k_smoothness=2, ctrs=None, ep=1):
        super().__init__()
        # set parameters
        self.in_N = in_N
        self.out_N = out_N
        self.str_kernel = str_kernel
        self.k_smoothness = k_smoothness
        self.ep = ep

        assert self.str_kernel in ('matern', 'wendland'), 'Pick other kernel!'
        if self.str_kernel == 'matern':
            self.kernel = tkernels.Matern(k=self.k_smoothness, ep=self.ep, flag_normalize_x=True, flag_normalize_y=True)
        elif self.str_kernel == 'wendland':
            self.kernel = tkernels.Wendland(k=self.k_smoothness, ep=self.ep, d=self.in_N, flag_normalize_y=True)

        if ctrs is None:
            ctrs = torch.rand(100, self.in_N) * 2 - 1

        self.register_buffer('ctrs', ctrs)
        self.coeffs = nn.Parameter(torch.zeros(self.ctrs.shape[0], self.out_N))
        self.matrix = nn.Parameter(torch.eye(self.in_N, dtype=self.ctrs.dtype, device=self.ctrs.device))

    def forward(self, x):
        return (self.kernel.eval(x @ self.matrix.T, self.ctrs @ self.matrix.T) @ self.coeffs).squeeze()
