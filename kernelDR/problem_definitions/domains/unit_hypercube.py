import numpy as np
import torch

from kernelDR.problem_definitions.domains.base import BaseDomain


class UnitHypercubeDomain(BaseDomain):
    def __init__(self, dim, device=torch.device('cpu')):
        super().__init__(device=device)

        self.dim = dim

        self.volume_domain = 1.
        self.volume_boundary = 2. * self.dim

    def random_interior_points(self, n):
        points = torch.rand(n, self.dim).to(self.device)
        points[..., 0] = points[..., 0]
        points[..., 1] = points[..., 1]
        points.requires_grad_()
        return points

    def uniform_interior_points(self, n):
        n_per_dim = int(np.power(n, 1. / self.dim))
        delta_x = 1. / (n_per_dim + 1)
        x = torch.linspace(delta_x, 1. - delta_x, n_per_dim)
        points = torch.cartesian_prod(*([x] * self.dim))
        points.requires_grad_()
        points = points.to(self.device)
        return points

    def random_boundary_points(self, n):
        n_per_face = int(n / self.volume_boundary)
        points = None
        for d in range(self.dim):
            xb = torch.rand((n_per_face, self.dim))
            xb[..., d] = 0.
            if points is not None:
                points = torch.vstack([points, xb])
            else:
                points = xb
            xb = torch.rand((n_per_face, self.dim))
            xb[..., d] = 1.
            points = torch.vstack([points, xb])
        points = points.to(self.device)
        points.requires_grad_()
        return points

    def uniform_boundary_points(self, n):
        n_per_face = int(n / self.volume_boundary)
        points = None
        x = torch.linspace(0., 1., int(np.power(n_per_face, 1. / (self.dim - 1))))
        for d in range(self.dim):
            xb = torch.cartesian_prod(*([x] * self.dim))
            xb[..., d] = 0.
            if points is not None:
                points = torch.vstack([points, xb])
            else:
                points = xb
            xb = torch.cartesian_prod(*([x] * self.dim))
            xb[..., d] = 1.
            points = torch.vstack([points, xb])
        points = points.to(self.device)

        # dismiss non-unique points!!!
        if points.shape[0] > 1:
            points = torch.unique(points, dim=0)

        points.requires_grad_()
        return points

    def outer_unit_normal(self, x):
        res = torch.zeros_like(x)
        for d in range(self.dim):
            res[torch.isclose(x[..., d], torch.zeros(x.shape[0])), d] = -1.
            res[torch.isclose(x[..., d], torch.ones(x.shape[0])), d] = 1.
        return res
