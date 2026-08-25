"""Aggregate repeated runs of an experiment (one file per RNG seed) into a single
file with summary statistics, so that figures can show the spread over seeds
instead of a single run.

The default settings aggregate the neural-network hyperparameter sweeps written by
main_03a/main_03b, i.e. the files ``convergence_results_seed*.txt`` inside one
results directory. The script is generic, though: any set of files sharing a
schema can be aggregated by grouping on a key column (``--key-column``).

The column order of the output is chosen so that existing pgfplots ``table[x
index=1, y index=2]`` statements keep working: index 0 is the key, index 1 the
parameter count, index 2/3/4 the medians over the seeds of the statistics
columns. The per-statistic detail blocks follow afterwards, so error bars can be
added via ``y error plus index=`` / ``y error minus index=``, and min-max bands
via the ``*_min`` / ``*_max`` columns.
"""

from typing import Annotated

import cyclopts
import glob
import numpy as np
import os

from kernelDR.experiments.plot_utils import _load_txt_with_header, _column_index


app = cyclopts.App()


# Short, space-free names used in the header of the aggregated file.
_SHORT_LABELS = {
    "neurons per layer": "neurons",
    "total number of parameters": "n_params",
    "l2-error": "L2",
    "h1-error": "H1",
    "l2_err": "L2",
    "h1_err": "H1",
    "final loss": "loss",
    "final_loss": "loss",
    "time_train_s": "time",
    "peak_mem_mb": "mem",
    "n_centers": "n_centers",
    "num_params": "n_params",
}

# Statistics computed per column, in the order they are written out.
_STATS = ("min", "max", "err_minus", "err_plus", "mean", "std", "geomean")


def _short_label(name):
    key = name.strip().lower()
    if key in _SHORT_LABELS:
        return _SHORT_LABELS[key]
    return "_".join(name.strip().split())


def _column_stats(values):
    """Summary statistics of the repetitions of one quantity.

    ``err_minus`` / ``err_plus`` are the distances from the median to the min and
    max, i.e. the asymmetric error bars belonging to the median. The geometric
    mean is reported as well because the errors are plotted on logarithmic axes;
    it is NaN whenever a value is non-positive (the loss, for instance).
    """
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    vmin, vmax = float(np.min(values)), float(np.max(values))
    geomean = float(np.exp(np.mean(np.log(values)))) if np.all(values > 0) else float("nan")
    return {
        "median": median,
        "min": vmin,
        "max": vmax,
        "err_minus": median - vmin,
        "err_plus": vmax - median,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "geomean": geomean,
    }


def collect_runs(files, key_column, passthrough_columns, stat_columns):
    """Read the per-seed files and group their rows by the key column.

    Returns (grouped, resolved_names) where ``grouped`` maps the key value to a
    dict with the passthrough values and the lists of repeated statistics.
    """
    grouped = {}
    resolved = None

    for path in files:
        data, header_cols = _load_txt_with_header(path)
        if header_cols is None:
            raise ValueError(f"{path} has no header row; cannot resolve columns by name.")

        key_idx = _column_index(header_cols, key_column)
        if key_idx is None:
            raise ValueError(f"Key column {key_column!r} not found in {path} (columns: {header_cols}).")
        pass_idx = []
        for name in passthrough_columns:
            idx = _column_index(header_cols, name)
            if idx is None:
                raise ValueError(f"Column {name!r} not found in {path} (columns: {header_cols}).")
            pass_idx.append(idx)
        stat_idx = []
        for name in stat_columns:
            idx = _column_index(header_cols, name)
            if idx is None:
                raise ValueError(f"Column {name!r} not found in {path} (columns: {header_cols}).")
            stat_idx.append(idx)

        names = ([header_cols[key_idx]]
                 + [header_cols[i] for i in pass_idx]
                 + [header_cols[i] for i in stat_idx])
        if resolved is None:
            resolved = names
        elif resolved != names:
            raise ValueError(f"{path} has a different schema than the previous files: "
                             f"{names} vs {resolved}.")

        for row in data:
            key = row[key_idx]
            entry = grouped.setdefault(key, {"passthrough": [row[i] for i in pass_idx],
                                             "stats": [[] for _ in stat_idx]})
            if entry["passthrough"] != [row[i] for i in pass_idx]:
                raise ValueError(f"Inconsistent values in the passthrough columns for key {key} "
                                 f"in {path}: {[row[i] for i in pass_idx]} vs {entry['passthrough']}.")
            for slot, i in enumerate(stat_idx):
                entry["stats"][slot].append(row[i])

    return grouped, resolved


def write_aggregate(output_path, grouped, resolved_names, n_passthrough):
    key_label = _short_label(resolved_names[0])
    pass_labels = [_short_label(n) for n in resolved_names[1:1 + n_passthrough]]
    stat_labels = [_short_label(n) for n in resolved_names[1 + n_passthrough:]]

    header = ([key_label] + pass_labels
              + [f"{lbl}_median" for lbl in stat_labels]
              + ["n_seeds"])
    for lbl in stat_labels:
        header += [f"{lbl}_{stat}" for stat in _STATS]

    with open(output_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for key in sorted(grouped):
            entry = grouped[key]
            stats = [_column_stats(vals) for vals in entry["stats"]]
            n_seeds = len(entry["stats"][0])

            row = [key] + entry["passthrough"] + [s["median"] for s in stats] + [n_seeds]
            for s in stats:
                row += [s[stat] for stat in _STATS]
            f.write("\t".join(_format(v) for v in row) + "\n")

    return header


def _format(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if float(value).is_integer() and abs(float(value)) < 1e15:
        return str(int(round(float(value))))
    return f"{float(value):.12e}"


@app.default
def main(
    results_dir: Annotated[str, cyclopts.Parameter(help="Directory holding the per-seed result files.")],
    pattern: Annotated[str, cyclopts.Parameter(help="Glob pattern of the per-seed files inside the results directory.")] = "convergence_results_seed*.txt",
    key_column: Annotated[str, cyclopts.Parameter(help="Column identifying a configuration; rows sharing this value are treated as repetitions.")] = "Neurons per layer",
    passthrough_columns: Annotated[tuple[str, ...], cyclopts.Parameter(help="Columns that are constant across repetitions and copied verbatim.")] = ("Total number of parameters",),
    stat_columns: Annotated[tuple[str, ...], cyclopts.Parameter(help="Columns to summarise over the repetitions.")] = ("L2-error", "H1-error", "Final loss"),
    output: Annotated[str, cyclopts.Parameter(help="Output filename (relative to the results directory unless absolute).")] = "convergence_results_aggregated.txt",
    min_seeds: Annotated[int, cyclopts.Parameter(help="Warn about configurations aggregated from fewer than this many repetitions.")] = 2,
):
    if not results_dir.endswith("/"):
        results_dir += "/"

    files = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not files:
        raise SystemExit(f"No files matching {pattern!r} in {results_dir}")

    grouped, resolved = collect_runs(files, key_column, list(passthrough_columns), list(stat_columns))

    output_path = output if os.path.isabs(output) else os.path.join(results_dir, output)
    header = write_aggregate(output_path, grouped, resolved, len(passthrough_columns))

    print(f"Aggregated {len(files)} file(s) from {results_dir}:")
    for path in files:
        print(f"  {os.path.relpath(path, results_dir)}")
    print(f"Wrote {output_path}")
    print("Columns (index: name):")
    print("  " + ", ".join(f"{i}: {name}" for i, name in enumerate(header)))

    # Spread report: the factor between the best and the worst repetition tells
    # at a glance whether the single-run conclusions are robust.
    first_stat = stat_columns[0]
    print(f"\nSpread over the repetitions ({first_stat}):")
    print(f"  {_short_label(resolved[0]):>12} {'n_seeds':>8} {'median':>12} {'min':>12} {'max':>12} {'max/min':>9}")
    for key in sorted(grouped):
        vals = np.asarray(grouped[key]["stats"][0], dtype=float)
        s = _column_stats(vals)
        ratio = s["max"] / s["min"] if s["min"] > 0 else float("nan")
        flag = "  <-- few repetitions" if vals.size < min_seeds else ""
        print(f"  {key:>12g} {vals.size:>8d} {s['median']:>12.3e} {s['min']:>12.3e} "
              f"{s['max']:>12.3e} {ratio:>9.2f}{flag}")


if __name__ == "__main__":
    app()
