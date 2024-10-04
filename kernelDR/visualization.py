from matplotlib import pyplot as plt
from matplotlib import cm
import pathlib
import torch


def _plot(points, values, title="", save_at=None, block=False):
    plt.figure()
    vals = plt.scatter(points.detach().numpy()[..., 0], points.detach().numpy()[..., 1], c=values)
    plt.colorbar(vals)
    plt.title(title)
    if save_at is not None:
        pathlib.Path(save_at).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_at + f"scatter_{title.replace(' ', '_')}")
        plt.close()
    else:
        plt.show(block=False)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vals = ax.plot_trisurf(points.detach().numpy()[..., 0], points.detach().numpy()[..., 1],
                           values.reshape(-1), cmap=cm.viridis)
    plt.colorbar(vals)
    plt.title(title)
    if save_at is not None:
        plt.savefig(save_at + f"trisurf_{title.replace(' ', '_')}")
        plt.close()
    else:
        plt.show(block=block)


def plot_predicted_solution(problem, model, n, title="", save_at=None, block=False):
    with torch.no_grad():
        interior_points = problem.domain.uniform_interior_points(n)
        predictions = model(interior_points).cpu().detach().numpy()

    _plot(interior_points, predictions, title=title, save_at=save_at, block=block)


def plot_reference_solution(problem, n, title="", save_at=None, block=False):
    with torch.no_grad():
        interior_points = problem.domain.uniform_interior_points(n)
        values = problem.reference_solution(interior_points).cpu().detach().numpy()

    _plot(interior_points, values, title=title, save_at=save_at, block=block)
