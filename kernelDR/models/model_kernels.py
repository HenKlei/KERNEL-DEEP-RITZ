import torch
import torch.nn as nn
from vkoga_2L import tkernels
import math


class FlatKernelModel(nn.Module):
    def __init__(self, in_N, out_N, str_kernel='matern', k_smoothness=2, ctrs=None, ep=1, flag_lagrange=False):
        super(FlatKernelModel, self).__init__()
        # set parameters
        self.in_N = in_N
        self.out_N = out_N
        self.str_kernel = str_kernel
        self.k_smoothness = k_smoothness
        self.ctrs = ctrs
        self.ep = ep
        self.flag_lagrange = flag_lagrange

        assert self.str_kernel in ('matern', 'wendland', 'gaussian'), 'Pick other kernel!'
        if self.str_kernel == 'matern':
            self.kernel = tkernels.Matern(k=self.k_smoothness, ep=self.ep, flag_normalize_x=True, flag_normalize_y=True)
        elif self.str_kernel == 'wendland':
            self.kernel = tkernels.Wendland(k=self.k_smoothness, ep=self.ep, d=self.in_N, flag_normalize_y=True)
        elif self.str_kernel == 'gaussian':
            self.kernel = tkernels.Gaussian(ep=self.ep)

        if self.ctrs is None:   # if no centers are provided, generate random centers
            self.ctrs = torch.rand(100, self.in_N) * 2 - 1

        self.coeffs = nn.Parameter(1e-1 * torch.randn(self.ctrs.shape[0], self.out_N))

        if self.flag_lagrange:
            A = self.kernel.eval(self.ctrs, self.ctrs)
            self.invA = torch.inverse(A).detach().clone()

    def forward(self, x):

        if self.flag_lagrange:
            return (self.kernel.eval(x, self.ctrs) @ self.invA @ self.coeffs).squeeze()
        else:
            return (self.kernel.eval(x, self.ctrs) @ self.coeffs).squeeze()
