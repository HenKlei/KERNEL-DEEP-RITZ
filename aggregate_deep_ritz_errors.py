"""Aggregate per-``n_per_dim`` Deep Ritz convergence files into the single
``errors_k_*.txt`` files that the paper plots expect.

The Deep Ritz experiment scripts (``main_01a``, ``main_01b``) write one
``conv_results_{k}_{ep}_{n_per_dim}_{n_epochs}.txt`` per spatial resolution
into a result directory like ``results_smooth_solution_matern_k0/``. The
paper template, in contrast, used to read a single aggregate
``results_smooth_solution_errors_k_0.txt`` per kernel-smoothness with
columns

    n   h ~ 1 / sqrt(n)   L2-error   H1-error   time_s

This script scans the working directory for the per-experiment Deep Ritz
result folders and writes the corresponding aggregate next to them. It
uses only the standard library — drop it in any directory that contains
the ``results_*_solution_*_k*/`` subfolders and run

    python aggregate_deep_ritz_errors.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from glob import glob
from statistics import median


def conv_n_seed(filename: str) -> tuple[int, int] | None:
    """Extract ``(n_per_dim, seed)`` from a conv_results file name.

    Matches both ``conv_results_{k}_{ep}_{n}_{epochs}.txt`` (the run with the
    default seed) and ``conv_results_{k}_{ep}_{n}_{epochs}_seed{s}.txt`` (the
    repetitions), so that a directory holding several seeds is aggregated into
    one row per ``n_per_dim`` with the spread over the repetitions.
    """
    m = re.match(r"conv_results_\d+_[\d\.]+_(\d+)_\d+(?:_seed(\d+))?\.txt$",
                 os.path.basename(filename))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)) if m.group(2) is not None else 0


def conv_n(filename: str) -> int | None:
    """Extract ``n_per_dim`` from a conv_results file name."""
    parsed = conv_n_seed(filename)
    return parsed[0] if parsed else None


def last_l2_h1(path: str) -> tuple[float, float] | None:
    """Return ``(L2, H1)`` from the final data row of a conv_results file."""
    last = None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith(("i,", "i\t")):
                last = s
    if last is None:
        return None
    toks = last.replace(",", " ").split()
    try:
        return float(toks[3]), float(toks[4])
    except (IndexError, ValueError):
        return None


def recomputed_n_error(path: str) -> int:
    """Extract the evaluation grid size from a recomputed-errors filename."""
    m = re.search(r"recomputed_errors_nerror(\d+)\.txt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_recomputed(results_dir: str, k: int | None = None):
    """Load errors recomputed on a finer grid by ``main_10_recompute_errors``.

    The errors logged during training are evaluated on the comparatively coarse
    default grid, which does not resolve the approximation error at the finer
    center distributions. When a recomputed file is present its values are
    preferred; the finest available grid wins if there are several.

    Returns ``({n_per_dim: (L2, H1)}, n_error, path)``, or ``({}, None, None)``.
    """
    candidates = glob(os.path.join(results_dir, "recomputed_errors_nerror*.txt"))
    candidates = [c for c in candidates if recomputed_n_error(c) > 0]
    if not candidates:
        return {}, None, None

    path = max(candidates, key=recomputed_n_error)
    out: dict[int, tuple[float, float]] = {}
    header: list[str] | None = None
    with open(path) as f:
        for line in f:
            s = line.rstrip("\n")
            if not s.strip() or s.lstrip().startswith("#"):
                continue
            toks = s.split("\t")
            if header is None:
                header = toks
                continue
            row = dict(zip(header, toks))
            if k is not None and row.get("k") not in (None, "None", ""):
                try:
                    if int(row["k"]) != k:
                        continue
                except ValueError:
                    pass
            try:
                out[int(row["n"])] = (float(row["L2-error"]), float(row["H1-error"]))
            except (KeyError, ValueError):
                continue
    return out, recomputed_n_error(path), path


def load_timings(results_dir: str) -> dict[int, dict[str, float]]:
    """Parse ``timings.txt`` (if any) into ``{n_per_dim: {col: value}}``.

    The Deep Ritz scripts write a ``timings.txt`` with header
    ``n_per_dim  n_centers  num_params  epochs  time_train_s  peak_mem_mb``.
    Any subset of those columns is picked up.
    """
    p = os.path.join(results_dir, "timings.txt")
    if not os.path.exists(p):
        return {}
    out: dict[int, dict[str, float]] = {}
    with open(p) as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            n_idx = header.index("n_per_dim")
        except ValueError:
            return {}
        for line in f:
            toks = line.rstrip("\n").split("\t")
            if len(toks) <= n_idx:
                continue
            try:
                key = int(toks[n_idx])
            except ValueError:
                continue
            row: dict[str, float] = {}
            for i, name in enumerate(header):
                if i == n_idx or i >= len(toks):
                    continue
                try:
                    row[name] = float(toks[i])
                except ValueError:
                    continue
            out[key] = row
    return out


def aggregate(results_dir: str, out_path: str, k: int | None = None,
              use_recomputed: bool = True) -> bool:
    """Build a single ``errors_k_*.txt`` from per-``n_per_dim`` conv files.

    Schema written (one row per ``n_per_dim``, indices in brackets):

        n[0]  h ~ 1 / sqrt(n)[1]  n_centers[2]
            L2-error[3]  H1-error[4]  time_train_s[5]  peak_mem_mb[6]

    If several seeds are present (files ``..._seed{s}.txt`` next to the run with
    the default seed), the error columns hold the median over the repetitions and
    the spread is appended:

        n_seeds[7]  L2_min[8]  L2_max[9]  L2_err_minus[10]  L2_err_plus[11]
            H1_min[12]  H1_max[13]  H1_err_minus[14]  H1_err_plus[15]

    so that indices 1, 3 and 4 keep their meaning for existing plot statements.

    Matches the schema of the interpolation and matrix-form
    ``errors_k_*.txt`` files. For flat-kernel models (the default for
    ``main_01a``/``main_01b``) ``n_centers`` equals the number of trainable
    parameters, which is the natural x-axis for the convergence plots.
    """
    timings = load_timings(results_dir)
    recomputed, n_error, recomputed_path = ({}, None, None)
    if use_recomputed:
        recomputed, n_error, recomputed_path = load_recomputed(results_dir, k)

    # Collect every seed of every resolution: {n_per_dim: {seed: (L2, H1)}}
    per_n: dict[int, dict[int, tuple[float, float]]] = {}
    n_from_recomputed = 0
    for conv in glob(os.path.join(results_dir, "conv_results_*.txt")):
        parsed = conv_n_seed(conv)
        if parsed is None:
            continue
        n, seed = parsed
        if n in recomputed:
            # A recomputed value is a single number per resolution, so it replaces
            # the whole set of repetitions and the row carries no spread.
            per_n[n] = {0: recomputed[n]}
            n_from_recomputed += 1
            continue
        l2h1 = last_l2_h1(conv)
        if l2h1 is None:
            continue
        per_n.setdefault(n, {})[seed] = l2h1
    if not per_n:
        return False

    n_seeds_max = max(len(v) for v in per_n.values())
    rows: list[tuple] = []
    for n in sorted(per_n):
        seeds = per_n[n]
        l2s = [v[0] for v in seeds.values()]
        h1s = [v[1] for v in seeds.values()]
        med_l2, med_h1 = median(l2s), median(h1s)
        meta = timings.get(n, {})
        rows.append((
            n, 1.0 / (n + 1), int(meta.get("n_centers", 0)), med_l2, med_h1,
            meta.get("time_train_s", 0.0), meta.get("peak_mem_mb", 0.0),
            len(seeds),
            min(l2s), max(l2s), med_l2 - min(l2s), max(l2s) - med_l2,
            min(h1s), max(h1s), med_h1 - min(h1s), max(h1s) - med_h1,
        ))

    with open(out_path, "w") as f:
        if n_seeds_max > 1:
            # Indices 1, 3 and 4 keep their meaning, so existing plot statements
            # continue to work; the spread is appended for the error bars.
            f.write(
                "n\th ~ 1 / sqrt(n)\tn_centers\tL2-error\tH1-error\t"
                "time_train_s\tpeak_mem_mb\tn_seeds\t"
                "L2_min\tL2_max\tL2_err_minus\tL2_err_plus\t"
                "H1_min\tH1_max\tH1_err_minus\tH1_err_plus\n"
            )
            for r in rows:
                f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\t{r[5]:.4f}\t{r[6]:.2f}\t"
                        f"{r[7]}\t{r[8]:.12g}\t{r[9]:.12g}\t{r[10]:.12g}\t{r[11]:.12g}\t"
                        f"{r[12]:.12g}\t{r[13]:.12g}\t{r[14]:.12g}\t{r[15]:.12g}\n")
        else:
            # Only one run per resolution: keep the classic schema rather than
            # writing zero-length error bars, which would suggest that repetitions
            # were performed and showed no variability.
            f.write(
                "n\th ~ 1 / sqrt(n)\tn_centers\t"
                "L2-error\tH1-error\ttime_train_s\tpeak_mem_mb\n"
            )
            for r in rows:
                f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\t{r[5]:.4f}\t{r[6]:.2f}\n")
    if n_seeds_max > 1:
        counts = sorted({len(v) for v in per_n.values()})
        print(f"    aggregated over {counts if len(counts) > 1 else counts[0]} seed(s) per "
              f"resolution, median with min/max appended")

    # Reported on stdout rather than as a comment line in the file: the plotting
    # code reads the first line as the column header, so the schema has to stay
    # exactly as the other errors_k_*.txt files.
    if n_from_recomputed:
        print(f"    using errors recomputed on {n_error} points for {n_from_recomputed}/{len(rows)} "
              f"row(s) from {os.path.basename(recomputed_path)}")
        missing = sorted(set(recomputed) - {row[0] for row in rows})
        if missing:
            print(f"    note: {os.path.basename(recomputed_path)} also holds n = {missing}, "
                  f"which have no conv_results file and were skipped")
    elif recomputed_path:
        print(f"    note: {os.path.basename(recomputed_path)} matched no n_per_dim; "
              f"used the errors logged during training")
    return True


# Directory naming convention produced by ``main_01a`` / ``main_01b``:
#
#     results_{problem}_solution_{kernel}_k{k}/
#
# The paper expects aggregates named ``results_{problem}_solution_errors_k_{k}.txt``
# for the main-body matern runs, and equivalently with a kernel infix
# (e.g. ``results_smooth_solution_wendland_errors_k_0.txt``) for the appendix
# runs. The default output naming follows that convention.
DIR_PATTERN = re.compile(
    r"^results_(?P<problem>smooth|singular)_solution_(?P<kernel>matern|wendland)_k(?P<k>\d+)$"
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "root", nargs="?", default=".",
        help="Directory holding the Deep Ritz result folders. Defaults to the cwd.",
    )
    p.add_argument(
        "--kernel", choices=("matern", "wendland", "all"), default="all",
        help="Only aggregate result folders for this kernel family.",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Where to write the aggregates. Defaults to <root>.",
    )
    p.add_argument(
        "--no-recomputed", action="store_true",
        help=(
            "Ignore recomputed_errors_nerror*.txt and use the errors logged during "
            "training, as written by main_01a/main_01b. By default the recomputed "
            "values are preferred, since the grid used during training does not "
            "resolve the approximation error at the finer center distributions."
        ),
    )
    p.add_argument(
        "--omit-kernel-in-name", action="store_true",
        help=(
            "Omit the kernel family from the output filename "
            "(so matern aggregates become ``results_<problem>_solution_errors_k_<k>.txt``, "
            "the convention used in the paper main body). "
            "Cannot be used with --kernel all unless only one kernel family is present."
        ),
    )
    args = p.parse_args(argv)

    out_dir = args.out_dir or args.root
    os.makedirs(out_dir, exist_ok=True)

    matches: list[tuple[str, str, str, int]] = []  # (path, problem, kernel, k)
    for entry in sorted(os.listdir(args.root)):
        m = DIR_PATTERN.match(entry)
        if not m:
            continue
        if args.kernel != "all" and m.group("kernel") != args.kernel:
            continue
        matches.append((
            os.path.join(args.root, entry),
            m.group("problem"),
            m.group("kernel"),
            int(m.group("k")),
        ))

    if not matches:
        print(f"No Deep Ritz result folders found in {args.root!r}", file=sys.stderr)
        return 1

    if args.omit_kernel_in_name:
        kernels = {kernel for _, _, kernel, _ in matches}
        if len(kernels) > 1:
            print(
                "--omit-kernel-in-name requires a single kernel family; "
                f"found {sorted(kernels)}. Re-run with --kernel <name>.",
                file=sys.stderr,
            )
            return 2

    n_ok = 0
    for path, problem, kernel, k in matches:
        if args.omit_kernel_in_name:
            out_name = f"results_{problem}_solution_errors_k_{k}.txt"
        else:
            out_name = f"results_{problem}_solution_{kernel}_errors_k_{k}.txt"
        out_path = os.path.join(out_dir, out_name)
        if aggregate(path, out_path, k=k, use_recomputed=not args.no_recomputed):
            print(f"  {path}  ->  {out_path}")
            n_ok += 1
        else:
            print(f"  {path}  (no conv_results_*.txt; skipped)", file=sys.stderr)
    print(f"\nAggregated {n_ok} result folder(s).")
    return 0 if n_ok else 3


if __name__ == "__main__":
    sys.exit(main())
