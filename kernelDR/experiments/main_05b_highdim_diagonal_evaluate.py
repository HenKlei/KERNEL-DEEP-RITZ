"""Interactive utilities to inspect stored high-dimensional diagonal results.

Typical interactive workflow:

	from kernelDR.experiments import main_05b_highdim_diagonal_evaluate as ev
	best = ev.collect_best_checkpoint_metrics("kernelDR/experiments/tiz_results")
	fig = ev.plot_best_errors_vs_centers(best)
"""

from pathlib import Path
from typing import Optional

import numpy as np

from kernelDR.experiments.plot_utils import setup_matplotlib


def _extract_n_centers(path: Path, prefix: str, suffix: str) -> Optional[int]:
	name = path.name
	if not (name.startswith(prefix) and name.endswith(suffix)):
		return None
	raw = name[len(prefix):-len(suffix)]
	return int(raw) if raw.isdigit() else None


def discover_n_centers(results_dir="kernelDR/experiments/results_highdim_diagonal_d5_2l/"):
	"""Return sorted n_centers values found in saved file names."""
	results_path = Path(results_dir)
	if not results_path.exists():
		raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

	metrics_files = list(results_path.glob("training_metrics_n*.npz"))
	trajectory_files = list(results_path.glob("matrix_trajectory_n*.npy"))
	matrix_files = list(results_path.glob("matrix_n*.npy"))

	n_metrics = {
		n for n in (
			_extract_n_centers(p, "training_metrics_n", ".npz") for p in metrics_files
		) if n is not None
	}
	n_trajectory = {
		n for n in (
			_extract_n_centers(p, "matrix_trajectory_n", ".npy") for p in trajectory_files
		) if n is not None
	}
	n_matrix = {
		n for n in (
			_extract_n_centers(p, "matrix_n", ".npy") for p in matrix_files
		) if n is not None
	}

	return sorted(n_metrics | n_trajectory | n_matrix)


def _load_training_metrics(metrics_path: Path):
	if not metrics_path.exists():
		return None
	data = np.load(metrics_path)
	required = ["epoch", "loss", "learning_rate", "l2_error", "h1_error"]
	if not all(k in data for k in required):
		return None
	return {
		"epoch": data["epoch"],
		"loss": data["loss"],
		"learning_rate": data["learning_rate"],
		"l2_error": data["l2_error"],
		"h1_error": data["h1_error"],
	}


def _load_matrix_trajectory(trajectory_path: Path):
	if not trajectory_path.exists():
		return None
	trajectory = np.load(trajectory_path)
	if trajectory.ndim != 3:
		return None
	return trajectory


def load_run(results_dir, n_centers, matrix_snapshot_interval=10):
	"""Load one run and return raw + derived quantities for inspection."""
	results_path = Path(results_dir)
	if not results_path.exists():
		raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

	metrics_path = results_path / f"training_metrics_n{n_centers}.npz"
	trajectory_path = results_path / f"matrix_trajectory_n{n_centers}.npy"
	matrix_path = results_path / f"matrix_n{n_centers}.npy"

	metrics = _load_training_metrics(metrics_path)
	trajectory = _load_matrix_trajectory(trajectory_path)
	matrix = np.load(matrix_path) if matrix_path.exists() else None

	run = {
		"results_dir": str(results_path),
		"n_centers": int(n_centers),
		"matrix_snapshot_interval": int(matrix_snapshot_interval),
		"metrics": metrics,
		"trajectory": trajectory,
		"matrix": matrix,
		"paths": {
			"training_metrics": str(metrics_path),
			"matrix_trajectory": str(trajectory_path),
			"matrix": str(matrix_path),
		},
	}

	if metrics is not None:
		run["epochs"] = metrics["epoch"]
	else:
		run["epochs"] = None

	if trajectory is not None:
		if run["epochs"] is not None and len(run["epochs"]) == trajectory.shape[0]:
			snapshot_epochs = run["epochs"]
		else:
			snapshot_epochs = matrix_snapshot_interval * np.arange(1, trajectory.shape[0] + 1)

		singular_values = np.linalg.svd(trajectory, compute_uv=False)
		delta = trajectory - trajectory[0][None, :, :]
		fro_delta = np.linalg.norm(delta, axis=(1, 2))
		fro_ref = np.linalg.norm(trajectory[0])
		rel_fro_delta = fro_delta / max(fro_ref, 1e-14)

		run["snapshot_epochs"] = snapshot_epochs
		run["singular_values"] = singular_values
		run["rel_fro_delta"] = rel_fro_delta
	else:
		run["snapshot_epochs"] = None
		run["singular_values"] = None
		run["rel_fro_delta"] = None

	return run


def collect_best_checkpoint_metrics(results_dir, matrix_snapshot_interval=10):
	"""Collect best-checkpoint L2/H1 errors over all available n_centers.

	For each run, the checkpoint with the smallest logged loss in training_metrics_n*.npz is selected.
	Returns a dictionary with arrays sorted by n_centers.
	"""
	n_values = discover_n_centers(results_dir)
	if not n_values:
		return {
			"n_centers": np.array([], dtype=int),
			"best_epoch": np.array([], dtype=int),
			"best_loss": np.array([], dtype=float),
			"best_l2_error": np.array([], dtype=float),
			"best_h1_error": np.array([], dtype=float),
		}

	rows = []
	for n_centers in n_values:
		run = load_run(results_dir, n_centers, matrix_snapshot_interval=matrix_snapshot_interval)
		metrics = run["metrics"]
		if metrics is None or metrics["loss"].size == 0:
			continue
		best_idx = int(np.argmin(metrics["loss"]))
		rows.append(
			(
				int(n_centers),
				int(metrics["epoch"][best_idx]),
				float(metrics["loss"][best_idx]),
				float(metrics["l2_error"][best_idx]),
				float(metrics["h1_error"][best_idx]),
			)
		)

	if not rows:
		return {
			"n_centers": np.array([], dtype=int),
			"best_epoch": np.array([], dtype=int),
			"best_loss": np.array([], dtype=float),
			"best_l2_error": np.array([], dtype=float),
			"best_h1_error": np.array([], dtype=float),
		}

	rows.sort(key=lambda row: row[0])
	data = np.array(rows, dtype=float)
	return {
		"n_centers": data[:, 0].astype(int),
		"best_epoch": data[:, 1].astype(int),
		"best_loss": data[:, 2],
		"best_l2_error": data[:, 3],
		"best_h1_error": data[:, 4],
	}


def plot_best_errors_vs_centers(best_metrics):
	"""Plot best-model L2 and H1 errors against the number of centers."""
	plt = setup_matplotlib()
	fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))

	n_centers = best_metrics["n_centers"]
	best_l2 = best_metrics["best_l2_error"]
	best_h1 = best_metrics["best_h1_error"]

	if n_centers.size == 0:
		ax.text(0.5, 0.5, "No metrics found", ha="center", va="center")
		ax.set_axis_off()
		return fig

	ax.semilogx(n_centers, best_l2, "o-", markersize=4, label="Best logged $L^2$ error")
	ax.semilogx(n_centers, best_h1, "s-", markersize=4, label="Best logged $H^1$ error")
	ax.set_xlabel("Number of centers")
	ax.set_ylabel("Relative error")
	ax.set_title("Best-model errors vs centers")
	ax.set_yscale("log")
	ax.grid(True, which="both", alpha=0.3)
	ax.legend()

	fig.tight_layout()
	return fig


def save_best_errors_plot(results_dir, output_dir="", matrix_snapshot_interval=10):
	"""Save the aggregate best-model error plot and return the file path."""
	best_metrics = collect_best_checkpoint_metrics(
		results_dir,
		matrix_snapshot_interval=matrix_snapshot_interval,
	)
	fig = plot_best_errors_vs_centers(best_metrics)
	out_dir = Path(output_dir) if output_dir else Path(results_dir) / "figures"
	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / "best_errors_vs_centers.png"
	fig.savefig(out_path)
	plt = setup_matplotlib()
	plt.close(fig)
	print(f"Saved {out_path}")
	return str(out_path)


def plot_run(run_data, max_singular_values=3):
	"""Create and return diagnostic figures for one loaded run.

	Returns (fig_diagnostics, fig_heatmap_or_none).
	"""
	plt = setup_matplotlib()

	n_centers = run_data["n_centers"]
	metrics = run_data["metrics"]
	trajectory = run_data["trajectory"]
	matrix = run_data["matrix"]

	fig, axes = plt.subplots(2, 2, figsize=(11, 8))

	if metrics is None:
		axes[0, 0].text(0.5, 0.5, "No training_metrics file", ha="center", va="center")
		axes[0, 1].text(0.5, 0.5, "No training_metrics file", ha="center", va="center")
		axes[1, 0].text(0.5, 0.5, "No training_metrics file", ha="center", va="center")
	else:
		epochs = metrics["epoch"]
		axes[0, 0].semilogy(epochs, metrics["loss"], "o-", markersize=3)
		axes[0, 0].set_title("Loss")
		axes[0, 0].set_xlabel("Epoch")
		axes[0, 0].set_ylabel("Loss")

		axes[0, 1].plot(epochs, metrics["learning_rate"], "o-", markersize=3)
		axes[0, 1].set_title("Learning Rate")
		axes[0, 1].set_xlabel("Epoch")
		axes[0, 1].set_ylabel("LR")

		axes[1, 0].semilogy(epochs, metrics["l2_error"], "o-", markersize=3, label="L2")
		axes[1, 0].semilogy(epochs, metrics["h1_error"], "s-", markersize=3, label="H1")
		axes[1, 0].set_title("Errors")
		axes[1, 0].set_xlabel("Epoch")
		axes[1, 0].set_ylabel("Relative error")
		axes[1, 0].legend()

	if trajectory is None:
		axes[1, 1].text(0.5, 0.5, "No matrix_trajectory file", ha="center", va="center")
	else:
		snapshot_epochs = run_data["snapshot_epochs"]
		rel_fro_delta = run_data["rel_fro_delta"]
		singular_values = run_data["singular_values"]

		axes[1, 1].plot(snapshot_epochs, rel_fro_delta, "o-", markersize=3,
						label="relative ||B_t - B_0||_F")

		max_lines = min(max_singular_values, singular_values.shape[1])
		for i in range(max_lines):
			axes[1, 1].plot(snapshot_epochs, singular_values[:, i], "--", linewidth=1.0,
							label=f"sigma_{i + 1}")

		axes[1, 1].set_title("Matrix evolution")
		axes[1, 1].set_xlabel("Epoch")
		axes[1, 1].set_ylabel("Value")
		axes[1, 1].legend()

	fig.suptitle(f"High-dimensional diagonal diagnostics (n_centers={n_centers})")
	fig.tight_layout()

	fig_m = None
	if matrix is not None:
		fig_m, ax_m = plt.subplots(1, 1, figsize=(5.5, 4.8))
		im = ax_m.imshow(matrix, cmap="RdBu_r")
		ax_m.set_title(f"Optimized matrix B (n_centers={n_centers})")
		ax_m.set_xlabel("Column")
		ax_m.set_ylabel("Row")
		fig_m.colorbar(im, ax=ax_m, shrink=0.85)
		fig_m.tight_layout()

	return fig, fig_m


def save_run_plots(run_data, output_dir="", max_singular_values=3):
	"""Save figures for one run and return output file paths."""
	out_dir = Path(output_dir) if output_dir else Path(run_data["results_dir"]) / "figures"
	out_dir.mkdir(parents=True, exist_ok=True)

	fig, fig_m = plot_run(run_data, max_singular_values=max_singular_values)

	n_centers = run_data["n_centers"]
	diag_path = out_dir / f"highdim_diagnostics_n{n_centers}.png"
	fig.savefig(diag_path)

	heatmap_path = None
	if fig_m is not None:
		heatmap_path = out_dir / f"matrix_heatmap_n{n_centers}.png"
		fig_m.savefig(heatmap_path)

	plt = setup_matplotlib()
	plt.close(fig)
	if fig_m is not None:
		plt.close(fig_m)

	return {
		"diagnostics": str(diag_path),
		"heatmap": str(heatmap_path) if heatmap_path is not None else None,
	}


def run_all(
	results_dir="kernelDR/experiments/results_highdim_diagonal_d5_2l/",
	output_dir="",
	matrix_snapshot_interval=10,
	max_singular_values=3,
):
	"""Load all available runs in a directory and save their plots.

	Returns a dict: n_centers -> {diagnostics, heatmap}.
	"""
	n_values = discover_n_centers(results_dir)
	if not n_values:
		print(f"No matching result files found in {results_dir}")
		return {}

	saved = {}
	for n_centers in n_values:
		run_data = load_run(
			results_dir=results_dir,
			n_centers=n_centers,
			matrix_snapshot_interval=matrix_snapshot_interval,
		)
		out = save_run_plots(
			run_data,
			output_dir=output_dir,
			max_singular_values=max_singular_values,
		)
		saved[n_centers] = out
		print(f"Saved plots for n_centers={n_centers}: {out}")

	return saved


if __name__ == "__main__":
	# Simple executable entry point: generates only the aggregate best-errors plot.
	results_dir = "kernelDR/experiments/tiz_results/results_highdim_diagonal_d10_2l"
	output_dir = "kernelDR/experiments/tiz_results/results_highdim_diagonal_d10_2l/figures"
	save_best_errors_plot(
		results_dir=results_dir,
		output_dir=output_dir,
		matrix_snapshot_interval=10,
	)

