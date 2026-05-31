"""Shared utilities for experiment results and plotting."""

import inspect
import numpy as np
import os
from datetime import datetime

OUTPUT_DIR = "figures/"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def output_path(filename):
    ensure_output_dir()
    return os.path.join(OUTPUT_DIR, filename)


def save_settings(results_dir, local_vars):
    """Save experiment settings to a text file.

    Call as: save_settings(results_dir, locals())
    """
    with open(os.path.join(results_dir, "settings.txt"), "w") as f:
        f.write(f"date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        for key, val in local_vars.items():
            if key.startswith("_") or callable(val) or inspect.ismodule(val):
                continue
            f.write(f"{key}: {val}\n")


# Consistent colors for kernel smoothness values
KERNEL_COLORS = {0: "#3B529B", 1: "#FDE725", 2: "#5EC962"}  # viridis blue, yellow, green
KERNEL_LABELS = {0: r"$\nu=1/2$", 1: r"$\nu=3/2$", 2: r"$\nu=5/2$"}
NN_COLOR = "#440154"  # viridis violet


def setup_matplotlib():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.figsize": (5, 4),
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    })
    return plt


def _load_txt(filepath):
    """Load a text data file, auto-detecting whether it has a header."""
    with open(filepath) as f:
        first_line = f.readline().strip()
    try:
        [float(x) for x in first_line.replace(",", " ").split()]
        skip = 0  # first line is data
    except ValueError:
        skip = 1  # first line is a header
    data = np.genfromtxt(filepath, skip_header=skip)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def _load_txt_with_header(filepath):
    """Load a text data file together with its header column names.

    Returns (data, header_cols). If the file has no header, header_cols is None.
    Used by readers that need to look up columns by name in order to remain
    robust against schema changes (e.g. added n_centers / peak_mem_mb cols).
    """
    with open(filepath) as f:
        first_line = f.readline().strip()
    try:
        [float(x) for x in first_line.replace(",", " ").split()]
        header_cols = None
        skip = 0
    except ValueError:
        # Tab- or whitespace-separated header. Comma-separated headers
        # (e.g. "i, epoch, loss, ...") also need handling.
        header_cols = [c.strip() for c in first_line.replace(",", "\t").split("\t") if c.strip()]
        skip = 1
    data = np.genfromtxt(filepath, skip_header=skip)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data, header_cols


def _column_index(header_cols, *candidates):
    """Find the first matching column name in ``header_cols``. Case-insensitive
    substring match, so "L2-error" / "L2_err" / "l2 error" all match candidate "l2"."""
    if header_cols is None:
        return None
    lowered = [c.lower() for c in header_cols]
    for cand in candidates:
        c = cand.lower()
        for i, name in enumerate(lowered):
            if c == name or c in name:
                return i
    return None


def _read_h_l2_h1(filepath, h_default_col=1, l2_default_col=2, h1_default_col=3):
    """Load (h, L2, H1) columns from a results file by header name when
    available, with positional fallback for older schema-less files."""
    data, header_cols = _load_txt_with_header(filepath)
    h_idx = _column_index(header_cols, "h ~", "h_", "h ") or h_default_col
    l2_idx = _column_index(header_cols, "L2-error", "L2_err", "L2") or l2_default_col
    h1_idx = _column_index(header_cols, "H1-error", "H1_err", "H1") or h1_default_col
    return data[:, h_idx], data[:, l2_idx], data[:, h1_idx]


def read_deep_ritz_results(results_dir, k_smoothness, ep=1.0, list_n_per_dim=None):
    """Read deep Ritz results. Supports both consolidated errors_k_*.txt files
    and per-center-count conv_results_* files (extracts final epoch)."""
    # Try consolidated format first (errors_k_*.txt)
    consolidated = os.path.join(results_dir, f"errors_k_{k_smoothness}.txt")
    if os.path.exists(consolidated):
        return _read_h_l2_h1(consolidated)

    # Fall back to per-center-count conv_results files (kept positional —
    # conv_results_*.txt schema is "i epoch loss l2_err h1_err", unchanged).
    if list_n_per_dim is None:
        list_n_per_dim = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    h_values = []
    l2_errors = []
    h1_errors = []

    for n in list_n_per_dim:
        pattern = f"conv_results_{k_smoothness}_{ep}_{n}_"
        matching = [f for f in os.listdir(results_dir) if f.startswith(pattern) and f.endswith(".txt")]
        if not matching:
            continue
        filepath = os.path.join(results_dir, matching[0])
        data = _load_txt(filepath)
        h = 1.0 / (n + 1)
        h_values.append(h)
        l2_errors.append(data[-1, 3])
        h1_errors.append(data[-1, 4])

    return np.array(h_values), np.array(l2_errors), np.array(h1_errors)


def read_interpolation_results(results_dir, k_smoothness):
    filepath = os.path.join(results_dir, f"errors_k_{k_smoothness}.txt")
    return _read_h_l2_h1(filepath)


def read_nn_results(results_dir):
    """Returns (num_params, L2, H1) for the NN sweep file."""
    filepath = os.path.join(results_dir, "convergence_results.txt")
    data, header_cols = _load_txt_with_header(filepath)
    np_idx = _column_index(header_cols, "Total number of parameters", "num_params") or 1
    l2_idx = _column_index(header_cols, "L2-error", "L2_err", "L2") or 2
    h1_idx = _column_index(header_cols, "H1-error", "H1_err", "H1") or 3
    return data[:, np_idx], data[:, l2_idx], data[:, h1_idx]


def read_matrix_form_results(results_dir, k_smoothness):
    filepath = os.path.join(results_dir, f"errors_k_{k_smoothness}.txt")
    return _read_h_l2_h1(filepath)


def estimate_convergence_rate(h_values, errors, skip_first=5):
    """Least-squares slope of log(error) vs. log(h) over the asymptotic regime.

    By default the first ``skip_first=5`` (pre-asymptotic) rows are dropped,
    matching ``pgfplotstableread[skip first n=5]`` used in the paper.
    """
    h_values = np.asarray(h_values)[skip_first:]
    errors = np.asarray(errors)[skip_first:]
    mask = (errors > 0) & (h_values > 0)
    if np.sum(mask) < 2:
        return np.nan
    log_h = np.log(h_values[mask])
    log_err = np.log(errors[mask])
    coeffs = np.polyfit(log_h, log_err, 1)
    return coeffs[0]
