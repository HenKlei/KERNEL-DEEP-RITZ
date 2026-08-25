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


def save_kernel_model(model, path):
    """Persist a kernel model together with everything needed to rebuild it.

    Storing the centers and the kernel parameters alongside the coefficients
    makes the checkpoint self-contained, so that a trained model can be
    re-evaluated later (e.g. by the integration diagnostics) without knowing the
    command line that produced it.
    """
    torch.save({
        "model_class": type(model).__name__,
        "in_N": model.in_N,
        "out_N": model.out_N,
        "str_kernel": model.str_kernel,
        "k_smoothness": model.k_smoothness,
        "ep": model.ep,
        "flag_lagrange": bool(getattr(model, "flag_lagrange", False)),
        "ctrs": model.ctrs.detach().cpu(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }, path)


def load_kernel_model(path, device=None):
    """Rebuild a kernel model saved by save_kernel_model."""
    checkpoint = torch.load(path, map_location="cpu")

    classes = {"FlatKernelModel": FlatKernelModel, "TwoLayerKernelModel": TwoLayerKernelModel}
    if checkpoint["model_class"] not in classes:
        raise ValueError(f"Unknown model class {checkpoint['model_class']!r} in {path}")
    model_class = classes[checkpoint["model_class"]]

    ctrs = checkpoint["ctrs"]
    if device is not None:
        ctrs = ctrs.to(device)

    kwargs = {"str_kernel": checkpoint["str_kernel"], "k_smoothness": checkpoint["k_smoothness"],
              "ctrs": ctrs, "ep": checkpoint["ep"]}
    if model_class is FlatKernelModel:
        kwargs["flag_lagrange"] = checkpoint["flag_lagrange"]

    model = model_class(checkpoint["in_N"], checkpoint["out_N"], **kwargs)

    # The matrix-form scripts assign the solution of the linear system as a flat
    # coefficient vector, whereas the constructor allocates shape (n_ctrs, out_N).
    # Both give the same forward pass, so reshape rather than reject the load.
    state_dict = dict(checkpoint["state_dict"])
    own_state = model.state_dict()
    for key, value in state_dict.items():
        if (key in own_state and value.shape != own_state[key].shape
                and value.numel() == own_state[key].numel()):
            state_dict[key] = value.reshape(own_state[key].shape)

    model.load_state_dict(state_dict)
    return model


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
