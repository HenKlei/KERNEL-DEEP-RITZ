import torch
import numpy as np
import math

from kernelDR.problem_definitions.domains.base import BaseDomain


class CircularSectorDomain(BaseDomain):

    def __init__(self, angle=2.*torch.pi, radius=1., device=torch.device('cpu')):
        super().__init__(device=device)
        assert 0. < angle <= 2.*torch.pi

        self.dim = 2

        self.angle = angle
        self.radius = radius

        self.volume_domain = self.angle / 2 * self.radius**2
        self.volume_boundary = 2. * self.radius + self.angle * self.radius

    def random_interior_points(self, n):
        lengths = self.radius * torch.sqrt(torch.rand(n))
        angles = self.angle * torch.rand(n)

        x = lengths * torch.cos(angles)
        y = lengths * torch.sin(angles)
        points = torch.stack([x, y], axis=-1)
        points = points.to(self.device)
        points.requires_grad_()
        return points

    def uniform_interior_points(self, n):
        factor_area = (4 * self.radius**2) / self.volume_domain
        n_per_dim = int(np.sqrt(factor_area * n))     # we want to have roughly n centers within the pacman domain
        x1 = torch.linspace(-self.radius, self.radius, n_per_dim)
        x2 = torch.linspace(-self.radius, self.radius, n_per_dim)

        points = torch.cartesian_prod(x1, x2)
        points = points[torch.linalg.norm(points, axis=-1) < self.radius, :]        # remove points outside the circle
        points = points[my_arctan(points[:, 0], points[:, 1]) < self.angle, :]      # remove points outside the sector
        points = points[torch.linalg.norm(points, axis=-1) > 1e-10, :]        # remove points too close to origin

        points = points.to(self.device)
        points.requires_grad_()
        return points

    def random_boundary_points(self, n):
        if self.angle < 2.*torch.pi:
            n_circlular_sector = int(n * (self.angle * self.radius / self.volume_boundary))
            n_line_y_0 = (n - n_circlular_sector) // 2
            n_other_line = n - n_circlular_sector - n_line_y_0

            angles = self.angle * torch.rand(n_circlular_sector)
            x = self.radius * torch.cos(angles)
            y = self.radius * torch.sin(angles)
            points = torch.stack([x, y], axis=-1)

            lengths = self.radius * torch.rand(n_line_y_0)
            points = torch.vstack([points, torch.stack([lengths, torch.zeros_like(lengths)], axis=-1)])

            lengths_2 = self.radius * torch.rand(n_other_line)
            x_2 = lengths_2 * torch.cos(torch.tensor(self.angle))
            y_2 = lengths_2 * torch.sin(torch.tensor(self.angle))
            points = torch.vstack([points, torch.stack([x_2, y_2], axis=-1)])
            assert points.shape[0] == n
        else:
            angles = self.angle * torch.rand(n)
            x = self.radius * torch.cos(angles)
            y = self.radius * torch.sin(angles)
            points = torch.stack([x, y], axis=-1)
        points = points.to(self.device)
        points.requires_grad_()
        return points

    def uniform_boundary_points(self, n):
        if self.angle < 2.*torch.pi:
            length = 2 * self.radius + self.angle * self.radius

            length_straight_line = self.radius

            n_straight_line = int(n * length_straight_line / length)
            n_circular_sector = n - 2*n_straight_line

            angles = self.angle * torch.linspace(0, 1, n_circular_sector+1)[:-1]        # remove endpoint (will be used in straight_line)
            x = self.radius * torch.cos(angles)
            y = self.radius * torch.sin(angles)
            points = torch.stack([x, y], axis=-1)

            lengths = self.radius * torch.linspace(0, 1, n_straight_line+1)[:-1]        # remove endpoint (is used in circular sector)
            points = torch.vstack([points, torch.stack([lengths, torch.zeros_like(lengths)], axis=-1)])

            lengths_2 = self.radius * torch.linspace(0, 1, n_straight_line+1)[1:]        # remove startpoint (is used in other straight line)
            x_2 = lengths_2 * torch.cos(torch.tensor(self.angle))
            y_2 = lengths_2 * torch.sin(torch.tensor(self.angle))
            points = torch.vstack([points, torch.stack([x_2, y_2], axis=-1)])
            assert points.shape[0] == n

        else:
            angles = self.angle * torch.linspace(0, 1, n+1)[:-1]            # remove endpoint (will coincide with first point)
            x = self.radius * torch.cos(angles)
            y = self.radius * torch.sin(angles)
            points = torch.stack([x, y], axis=-1)
        points = points.to(self.device)
        points.requires_grad_()
        return points

    def outer_unit_normal(self, x):
        res = torch.zeros_like(x)
        if self.angle < 2.*torch.pi:
            res[..., 0] = torch.cos(torch.tensor(self.angle + torch.pi / 2.))
            res[..., 1] = torch.sin(torch.tensor(self.angle + torch.pi / 2.))
            mask = torch.isclose(x[..., 1], torch.zeros(x.shape[0]))
            res[mask, 0] = 0.
            res[mask, 1] = -1.
            mask = torch.isclose(torch.linalg.norm(x, axis=-1), self.radius * torch.ones(x.shape[0]))
            res[mask] = x[mask] / torch.linalg.norm(x, axis=-1)[mask, None]
        else:
            res = x / torch.linalg.norm(x, axis=-1)
        return res


def my_arctan(x1, x2):
    # Returns angle in the interval [0, 2pi]

    phi = np.arctan2(x2, x1)
    phi += (phi < 0) * 2 * math.pi

    return phi


def my_arctan_torch(x1, x2):
    # Returns angle in the interval [0, 2pi]

    phi = torch.arctan2(x2, x1)
    phi += (phi < 0) * 2 * math.pi

    return phi
