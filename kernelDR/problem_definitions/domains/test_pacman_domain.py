import torch
import matplotlib.pyplot as plt

from kernelDR.problem_definitions.domains.circular_sector import CircularSectorDomain


angle = 3.5 * torch.pi / 2.
radius = 1.5
dom = CircularSectorDomain(angle=angle, radius=radius)
n_inner = 1000
# points_int = dom.random_interior_points(n_inner).detach().numpy()
points_int = dom.uniform_interior_points(n_inner).detach().numpy()

plt.scatter(points_int[..., 0], points_int[..., 1])

n_boundary = 100
points_bd = dom.random_boundary_points(n_boundary).detach().numpy()
normals = dom.outer_unit_normal(torch.from_numpy(points_bd)).detach().numpy()

plt.scatter(points_bd[..., 0], points_bd[..., 1])
plt.quiver(points_bd[..., 0], points_bd[..., 1], normals[..., 0], normals[..., 1], angles='xy', scale_units='xy', scale=1.)
plt.xlim(-radius-1.5, radius+1.5)
plt.ylim(-radius-1.5, radius+1.5)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show(block=False)
