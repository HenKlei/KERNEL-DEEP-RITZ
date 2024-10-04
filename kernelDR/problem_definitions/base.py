import torch

from kernelDR.utils import gradient


class DeepRitzExample:
    def __init__(self, penalty_parameter=1., device=torch.device('cpu'), output_dim=1):
        self.penalty_parameter = penalty_parameter
        self.device = device
        self.output_dim = output_dim

    def diffusion(self, x):
        raise NotImplementedError

    def reaction(self, x):
        raise NotImplementedError

    def source(self, x):
        raise NotImplementedError

    def dirichlet_boundary_values(self, x):
        raise NotImplementedError

    def reference_solution(self, x):
        raise NotImplementedError

    def energy(self, model, x_i, x_b):
        y_i = model(x_i)
        y_b = model(x_b)
        grads_i = gradient(model, x_i, y=y_i)
        grads_b = gradient(model, x_b, y=y_b)
        return self._energy(x_i, y_i, x_b, y_b, grads_i, grads_b)

    def _energy(self, x_i, y_i, x_b, y_b, grads_i, grads_b):
        return (0.5 * self.bilinear_form(x_i, y_i, y_i, x_b, y_b, y_b, grads_i, grads_i, grads_b, grads_b)
                - self.right_hand_side(x_i, y_i, x_b, y_b, grads_i, grads_b))

    def bilinear_form(self, x_i, u_i, v_i, x_b, u_b, v_b, grads_u_i, grads_v_i, grads_u_b, grads_v_b):
        diffusion_i = self.diffusion(x_i)
        reaction_i = self.reaction(x_i)
        outer_normals = self.domain.outer_unit_normal(x_b)
        return (torch.mean(torch.sum(diffusion_i[..., None] * grads_u_i * grads_v_i, dim=1)
                           + reaction_i * u_i * v_i) * self.domain.volume_domain
                - torch.mean(u_b * torch.sum(grads_v_b * outer_normals, dim=1)
                             + v_b * torch.sum(grads_u_b * outer_normals, dim=1)
                             - self.penalty_parameter * u_b * v_b) * self.domain.volume_boundary)

    def right_hand_side(self, x_i, v_i, x_b, v_b, grads_v_i, grads_v_b):
        source_i = self.source(x_i)
        boundary_vals = self.dirichlet_boundary_values(x_b)
        outer_normals = self.domain.outer_unit_normal(x_b)
        return (torch.mean(v_i * source_i) * self.domain.volume_domain
                - torch.mean(boundary_vals * torch.sum(grads_v_b * outer_normals, dim=1)
                             - self.penalty_parameter * boundary_vals * v_b) * self.domain.volume_boundary)
