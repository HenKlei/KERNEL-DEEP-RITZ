import numpy as np
import torch

from kernelDR.problem_definitions.base import DeepRitzExample
from kernelDR.problem_definitions.domains import RectDomain
from kernelDR.utils import gradient


class FixedParameterExample(DeepRitzExample):
    def __init__(self, penalty_parameter=1., device=torch.device('cpu'), output_dim=1, parameter_value=1.):
        super().__init__(penalty_parameter=penalty_parameter, device=device, output_dim=output_dim)
        self.parameter_value = parameter_value

        self.domain = RectDomain(xmin=0., xmax=1., ymin=0., ymax=1., device=device)

    def diffusion(self, x):
        return self.parameter_value * torch.ones(x.shape[0])

    def reaction(self, x):
        return torch.zeros(x.shape[0])

    def source(self, x):
        return 4 * torch.ones(x.shape[0])

    def dirichlet_boundary_values(self, x):
        return (1 - x[..., 0]**2 - x[..., 1]**2) / self.parameter_value

    def reference_solution(self, x):
        return (1 - x[..., 0]**2 - x[..., 1]**2) / self.parameter_value

    def gradient_reference_solution(self, x):
        return (-2. * x) / self.parameter_value


class ParametricExample:
    def __init__(self, parameters=torch.linspace(0.5, 5., 10), penalty_parameter=1.,
                 device=torch.device('cpu'), output_dim=1, n_error_params=100):
        self.parameters = parameters
        self.output_dim = output_dim
        self.device = device
        self.penalty_parameter = penalty_parameter
        self.n_error_params = n_error_params
        self.param_min = parameters.min().item()
        self.param_max = parameters.max().item()
        self.fixed_parameter_examples = [FixedParameterExample(penalty_parameter=penalty_parameter,
                                                               device=device, output_dim=output_dim,
                                                               parameter_value=p.item()) for p in parameters]
        self.domain = self.fixed_parameter_examples[0].domain

    def _rescale_param(self, mu):
        """Rescale parameter from [param_min, param_max] to [0, 1]."""
        return (mu - self.param_min) / (self.param_max - self.param_min)

    def energy(self, model, x_i, x_b):
        spatial_dim = x_i.shape[1]
        total = torch.tensor(0., device=x_i.device)
        for ex in self.fixed_parameter_examples:
            mu = ex.parameter_value
            mu_scaled = self._rescale_param(mu)
            # Augment spatial points with rescaled parameter value
            x_i_aug = torch.hstack([x_i, mu_scaled * torch.ones(x_i.shape[0], 1, device=x_i.device)])
            x_b_aug = torch.hstack([x_b, mu_scaled * torch.ones(x_b.shape[0], 1, device=x_b.device)])
            x_i_aug.requires_grad_()
            x_b_aug.requires_grad_()

            # Forward pass on augmented inputs
            y_i = model(x_i_aug)
            y_b = model(x_b_aug)

            # Gradients w.r.t. all inputs, sliced to spatial dimensions only
            grads_i = gradient(model, x_i_aug, y=y_i)[:, :spatial_dim]
            grads_b = gradient(model, x_b_aug, y=y_b)[:, :spatial_dim]

            # Pass spatial coordinates to _energy (for normals, diffusion, etc.)
            total = total + ex._energy(x_i, y_i, x_b, y_b, grads_i, grads_b)
        return total / len(self.fixed_parameter_examples)

    def compute_relative_L2_error(self, model, n):
        """Compute relative L2 error integrated over spatial domain and parameter space."""
        error_params = torch.linspace(self.param_min, self.param_max, self.n_error_params)

        x = self.domain.uniform_interior_points(n)
        squared_errors = []
        squared_refs = []
        for mu in error_params:
            ex = FixedParameterExample(penalty_parameter=self.penalty_parameter,
                                       device=self.device, parameter_value=mu.item())
            mu_scaled = self._rescale_param(mu.item())
            x_aug = torch.hstack([x, mu_scaled * torch.ones(x.shape[0], 1, device=x.device)])
            with torch.no_grad():
                y_model = model(x_aug)
            y_ref = ex.reference_solution(x)
            squared_errors.append(torch.mean((y_ref - y_model) ** 2).item())
            squared_refs.append(torch.mean(y_ref ** 2).item())

        err_norm = np.sqrt(np.mean(squared_errors))
        sol_norm = np.sqrt(np.mean(squared_refs))
        return err_norm / sol_norm

    def compute_relative_H1_error(self, model, n):
        """Compute relative H1 error integrated over spatial domain and parameter space."""
        error_params = torch.linspace(self.param_min, self.param_max, self.n_error_params)

        x = self.domain.uniform_interior_points(n)
        spatial_dim = x.shape[1]

        squared_L2_errors = []
        squared_H1_semi_errors = []
        squared_L2_refs = []
        squared_H1_semi_refs = []

        for mu in error_params:
            ex = FixedParameterExample(penalty_parameter=self.penalty_parameter,
                                       device=self.device, parameter_value=mu.item())
            mu_scaled = self._rescale_param(mu.item())
            x_aug = torch.hstack([x, mu_scaled * torch.ones(x.shape[0], 1, device=x.device)])
            x_aug.requires_grad_()

            y_model = model(x_aug)
            grad_model = gradient(model, x_aug, y=y_model)[:, :spatial_dim]

            with torch.no_grad():
                y_ref = ex.reference_solution(x)
                grad_ref = ex.gradient_reference_solution(x)

                squared_L2_errors.append(torch.mean((y_ref - y_model.detach()) ** 2).item())
                squared_H1_semi_errors.append(
                    torch.mean(torch.sum((grad_ref - grad_model.detach()) ** 2, dim=1)).item())
                squared_L2_refs.append(torch.mean(y_ref ** 2).item())
                squared_H1_semi_refs.append(torch.mean(torch.sum(grad_ref ** 2, dim=1)).item())

        err_norm = np.sqrt(np.mean(squared_L2_errors) + np.mean(squared_H1_semi_errors))
        sol_norm = np.sqrt(np.mean(squared_L2_refs) + np.mean(squared_H1_semi_refs))
        return err_norm / sol_norm
