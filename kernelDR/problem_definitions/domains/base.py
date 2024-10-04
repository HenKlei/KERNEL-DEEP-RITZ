import torch


class BaseDomain:
    def __init__(self, device=torch.device('cpu')):
        self.device = device

    def random_interior_points(self, n):
        raise NotImplementedError

    def uniform_interior_points(self, n):
        raise NotImplementedError

    def random_boundary_points(self, n):
        raise NotImplementedError

    def uniform_boundary_points(self, n):
        raise NotImplementedError

    def outer_unit_normal(self, x):
        raise NotImplementedError
