import numpy as np
import torch

from kernelDR.problem_definitions.domains.base import BaseDomain


class RectDomain(BaseDomain):
    def __init__(self, xmin, xmax, ymin, ymax, device=torch.device('cpu')):
        super().__init__(device=device)

        self.dim = 2

        assert xmin < xmax and ymin < ymax
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.volume_domain = (self.xmax - self.xmin) * (self.ymax - self.ymin)
        self.volume_boundary = 2. * ((self.xmax - self.xmin) + (self.ymax - self.ymin))

    def random_interior_points(self, n):
        points = torch.rand(n, self.dim).to(self.device)
        points[..., 0] = points[..., 0] * (self.xmax - self.xmin) + self.xmin
        points[..., 1] = points[..., 1] * (self.ymax - self.ymin) + self.ymin
        points.requires_grad_()
        return points

    def uniform_interior_points(self, n):
        n_per_dim = int(np.sqrt(n))
        delta_x = (self.xmax - self.xmin) / (n_per_dim + 1)
        delta_y = (self.ymax - self.ymin) / (n_per_dim + 1)
        x1 = torch.linspace(self.xmin + delta_x, self.xmax - delta_x, n_per_dim)
        x2 = torch.linspace(self.ymin + delta_y, self.ymax - delta_y, n_per_dim)
        points = torch.cartesian_prod(x1, x2)
        points.requires_grad_()
        points = points.to(self.device)
        return points

    def random_boundary_points(self, n):
        n_per_segment_x = int(n * (self.xmax - self.xmin) / self.volume_boundary)
        n_per_segment_y = int(n * (self.ymax - self.ymin) / self.volume_boundary)
        xb1 = torch.stack([self.xmin + (self.xmax - self.xmin) * torch.rand(n_per_segment_x),
                           self.ymin * torch.ones(n_per_segment_x)], axis=-1)
        xb2 = torch.stack([self.xmin + (self.xmax - self.xmin) * torch.rand(n_per_segment_x),
                           self.ymax * torch.ones(n_per_segment_x)], axis=-1)
        xb3 = torch.stack([self.xmin * torch.ones(n_per_segment_y),
                           self.ymin + (self.ymax - self.ymin) * torch.rand(n_per_segment_y)], axis=-1)
        n_remaining = n - 2 * n_per_segment_x - n_per_segment_y
        xb4 = torch.stack([self.xmax * torch.ones(n_remaining),
                           self.ymin + (self.ymax - self.ymin) * torch.rand(n_remaining)], axis=-1)
        points = torch.vstack([xb1, xb2, xb3, xb4]).to(self.device)
        assert points.shape[0] == n
        points.requires_grad_()
        return points

    def uniform_boundary_points(self, n):
        n_per_segment_x = int(n * (self.xmax - self.xmin) / self.volume_boundary)
        n_per_segment_y = int(n * (self.ymax - self.ymin) / self.volume_boundary)
        xb1 = torch.stack([torch.linspace(self.xmin, self.xmax, n_per_segment_x),
                           self.ymin * torch.ones(n_per_segment_x)], axis=-1)
        xb2 = torch.stack([torch.linspace(self.xmin, self.xmax, n_per_segment_x),
                           self.ymax * torch.ones(n_per_segment_x)], axis=-1)
        xb3 = torch.stack([self.xmin * torch.ones(n_per_segment_y),
                           torch.linspace(self.ymin, self.ymax, n_per_segment_y)], axis=-1)
        n_remaining = n - 2 * n_per_segment_x - n_per_segment_y
        xb4 = torch.stack([self.xmax * torch.ones(n_remaining),
                           torch.linspace(self.ymin, self.ymax, n_remaining)], axis=-1)
        points = torch.vstack([xb1, xb2, xb3, xb4]).to(self.device)
        assert points.shape[0] == n

        # dismiss non-unique points!!!
        if points.shape[0] > 1:
            points = torch.unique(points, dim=0)

        points.requires_grad_()
        return points

    def outer_unit_normal(self, x):
        res = torch.zeros_like(x)
        res[torch.isclose(x[..., 0], self.xmin * torch.ones(x.shape[0])), 0] = -1.
        res[torch.isclose(x[..., 0], self.xmax * torch.ones(x.shape[0])), 0] = 1.
        res[torch.isclose(x[..., 1], self.ymin * torch.ones(x.shape[0])), 1] = -1.
        res[torch.isclose(x[..., 1], self.ymax * torch.ones(x.shape[0])), 1] = 1.
        return res
