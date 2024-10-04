import matplotlib.pyplot as plt

from kernelDR.problem_definitions.domains.unit_hypercube import UnitHypercubeDomain


dim = 2
dom = UnitHypercubeDomain(dim)

n_inner = 1000
n_boundary = 100

points = dom.random_interior_points(n_inner).detach().numpy()
plt.scatter(points[..., 0], points[..., 1])

points = dom.random_boundary_points(n_boundary)

normals = dom.outer_unit_normal(points).detach().numpy()
points = points.detach().numpy()

plt.scatter(points[..., 0], points[..., 1])
plt.quiver(points[..., 0], points[..., 1], normals[..., 0], normals[..., 1], angles='xy', scale_units='xy', scale=1.)
plt.xlim(-1.5, 2.5)
plt.ylim(-1.5, 2.5)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()


points = dom.uniform_interior_points(n_inner).detach().numpy()
plt.scatter(points[..., 0], points[..., 1])

points = dom.uniform_boundary_points(n_boundary)

normals = dom.outer_unit_normal(points).detach().numpy()
points = points.detach().numpy()

plt.scatter(points[..., 0], points[..., 1])
plt.quiver(points[..., 0], points[..., 1], normals[..., 0], normals[..., 1], angles='xy', scale_units='xy', scale=1.)
plt.xlim(-1.5, 2.5)
plt.ylim(-1.5, 2.5)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()
