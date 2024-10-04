import torch

from kernelDR.problem_definitions.base import DeepRitzExample
from kernelDR.problem_definitions.domains import RectDomain


class PoissonHigherRegularity(DeepRitzExample):
    """
    -\Delta u(x) = 1, x\in \Omega,
    u(x) = 0, x\in \partial \Omega
    \Omega = (0,1) \times (0,1)  # noqa
    """
    def __init__(self, penalty_parameter=1., device=torch.device('cpu')):
        super().__init__(penalty_parameter=penalty_parameter, device=device, output_dim=1)

        self.domain = RectDomain(xmin=0., xmax=1., ymin=0., ymax=1., device=device)

    def diffusion(self, x):
        return torch.ones(x.shape[0])

    def reaction(self, x):
        return torch.zeros(x.shape[0])

    def source(self, x):
        return torch.ones(x.shape[0])

    def dirichlet_boundary_values(self, x):
        return torch.zeros(x.shape[0])

    def reference_solution(self, x):
        raise NotImplementedError


class PoissonHigherRegularityRefSol(DeepRitzExample):
    """
    -\Delta u(x) = 4, x\in \Omega,
    u(x) = 1 - x1**2 - x2**2, x=[x1, x2]\in \partial \Omega
    \Omega = (0,1) \times (0,1)  # noqa
    Solution given by u(x) = 1 - x1**2 - x2**2          <-- I checked this via PDE-VKOGA: The solution is correct
    """
    def __init__(self, penalty_parameter=1., device=torch.device('cpu')):
        super().__init__(penalty_parameter=penalty_parameter, device=device, output_dim=1)

        self.domain = RectDomain(xmin=0., xmax=1., ymin=0., ymax=1., device=device)

    def diffusion(self, x):
        return torch.ones(x.shape[0])

    def reaction(self, x):
        return torch.zeros(x.shape[0])

    def source(self, x):
        return 4 * torch.ones(x.shape[0])

    def dirichlet_boundary_values(self, x):
        return 1 - x[..., 0]**2 - x[..., 1]**2

    def reference_solution(self, x):
        return 1 - x[..., 0]**2 - x[..., 1]**2

    def gradient_reference_solution(self, x):
        return -2. * x
