import torch

from kernelDR.problem_definitions.base import DeepRitzExample
from kernelDR.problem_definitions.domains.circular_sector import CircularSectorDomain
from kernelDR.problem_definitions.domains.circular_sector import my_arctan_torch


class LaplaceOnPacmanDomain(DeepRitzExample):
    """
    -\Delta u(x) = 0, x\in \Omega,
    u(x) = x1 * x2 + 1, x=[x1, x2]\in \partial \Omega
    \Omega = CircularSectorDomain with opening angle alpha  # noqa
    Solution given by u(x) = x1 * x2 + 1
    """
    def __init__(self, angle=3.*torch.pi/2., radius=1., penalty_parameter=1., device=torch.device('cpu')):
        self.domain = CircularSectorDomain(angle=angle, radius=radius, device=device)

        super().__init__(penalty_parameter=penalty_parameter, device=device, output_dim=1)

    def diffusion(self, x):
        return torch.ones(x.shape[0])

    def reaction(self, x):
        return torch.zeros(x.shape[0])

    def source(self, x):
        return torch.zeros(x.shape[0])

    def dirichlet_boundary_values(self, x):
        return x[..., 0] * x[..., 1] + 1.

    def reference_solution(self, x):
        return x[..., 0] * x[..., 1] + 1.

    def gradient_reference_solution(self, x):
        return torch.stack([x[..., 1], x[..., 0]], axis=-1)


class LaplaceOnPacmanDomainSingularSolution(DeepRitzExample):
    """
    -\Delta u(x) = 0, x\in \Omega,
    u(x) = np.linalg.norm(x, axis=1, keepdims=True) ** (1 / alpha) * np.sin(my_arctan(x[:, [0]], x[:, [1]]) / alpha) + 1, x=[x1, x2]\in \partial \Omega
    \Omega = CircularSectorDomain with opening angle alpha  # noqa
    Solution given by u(x) = np.linalg.norm(x, axis=1, keepdims=True) ** (1 / alpha) * np.sin(my_arctan(x[:, [0]], x[:, [1]]) / alpha) + 1
    """
    def __init__(self, angle=3.*torch.pi/2., radius=1., penalty_parameter=1., device=torch.device('cpu')):
        self.domain = CircularSectorDomain(angle=angle, radius=radius, device=device)

        super().__init__(penalty_parameter=penalty_parameter, device=device, output_dim=1)

    def diffusion(self, x):
        return torch.ones(x.shape[0])

    def reaction(self, x):
        return torch.zeros(x.shape[0])

    def source(self, x):
        return torch.zeros(x.shape[0])

    def dirichlet_boundary_values(self, x):
        return (torch.norm(x, dim=1)**(1 / self.domain.angle)
                * torch.sin(my_arctan_torch(x[:, 0], x[:, 1]) / self.domain.angle) + 1)

    def reference_solution(self, x):
        return (torch.norm(x, dim=1)**(1 / self.domain.angle)
                * torch.sin(my_arctan_torch(x[:, 0], x[:, 1]) / self.domain.angle) + 1)

    def gradient_reference_solution(self, x):
        phi = my_arctan_torch(x[:, [0]], x[:, [1]])
        r = torch.norm(x, dim=1, keepdim=True)

        grad_x1 = (torch.cos(phi) * torch.sin(phi / self.domain.angle)
                   - torch.sin(phi) * torch.cos(phi / self.domain.angle)) / self.domain.angle * r ** (1/self.domain.angle - 1)
        grad_x2 = (torch.sin(phi) * torch.sin(phi / self.domain.angle)
                   + torch.cos(phi) * torch.cos(phi / self.domain.angle)) / self.domain.angle * r ** (1/self.domain.angle - 1)
        return torch.concatenate([grad_x1, grad_x2], axis=1)
