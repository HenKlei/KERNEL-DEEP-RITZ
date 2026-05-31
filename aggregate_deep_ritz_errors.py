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


def conv_n(filename: str) -> int | None:
    """Extract ``n_per_dim`` from ``conv_results_{k}_{ep}_{n}_{epochs}.txt``."""
    m = re.match(r"conv_results_\d+_[\d\.]+_(\d+)_\d+\.txt$", os.path.basename(filename))
    return int(m.group(1)) if m else None


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


def aggregate(results_dir: str, out_path: str) -> bool:
    """Build a single ``errors_k_*.txt`` from per-``n_per_dim`` conv files.

    Schema written (one row per ``n_per_dim``, indices in brackets):

        n[0]  h ~ 1 / sqrt(n)[1]  n_centers[2]
            L2-error[3]  H1-error[4]  time_train_s[5]  peak_mem_mb[6]

    Matches the schema of the interpolation and matrix-form
    ``errors_k_*.txt`` files. For flat-kernel models (the default for
    ``main_01a``/``main_01b``) ``n_centers`` equals the number of trainable
    parameters, which is the natural x-axis for the convergence plots.
    """
    timings = load_timings(results_dir)
    rows: list[tuple] = []
    for conv in glob(os.path.join(results_dir, "conv_results_*.txt")):
        n = conv_n(conv)
        if n is None:
            continue
        l2h1 = last_l2_h1(conv)
        if l2h1 is None:
            continue
        l2, h1 = l2h1
        h = 1.0 / (n + 1)
        meta = timings.get(n, {})
        n_centers = int(meta.get("n_centers", 0))
        t = meta.get("time_train_s", 0.0)
        peak_mem = meta.get("peak_mem_mb", 0.0)
        rows.append((n, h, n_centers, l2, h1, t, peak_mem))
    if not rows:
        return False
    rows.sort()
    with open(out_path, "w") as f:
        f.write(
            "n\th ~ 1 / sqrt(n)\tn_centers\t"
            "L2-error\tH1-error\ttime_train_s\tpeak_mem_mb\n"
        )
        for n, h, nc, l2, h1, t, pm in rows:
            f.write(f"{n}\t{h}\t{nc}\t{l2}\t{h1}\t{t:.4f}\t{pm:.2f}\n")
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
        if aggregate(path, out_path):
            print(f"  {path}  ->  {out_path}")
            n_ok += 1
        else:
            print(f"  {path}  (no conv_results_*.txt; skipped)", file=sys.stderr)
    print(f"\nAggregated {n_ok} result folder(s).")
    return 0 if n_ok else 3


if __name__ == "__main__":
    sys.exit(main())
