"""Build the high-dimensional results table reported in the paper.

For each method group (flat kernel, two-layered kernel, neural network) and
each kernel-smoothness value ν the row reports the best run (smallest final
loss) across the swept expansion sizes / widths. The output mirrors the
``\\Cref{tab:errors-high-dimensional}`` table in the paper.
"""
from pathlib import Path
from typing import Annotated

import cyclopts
import numpy as np


app = cyclopts.App()


# Map main_05 k_smoothness to the Matérn ν value reported in the paper.
NU_BY_K = {0: "1/2", 1: "3/2", 2: "5/2"}

GROUP_FLAT = "Flat kernel"
GROUP_2L = "Two-layered kernel"
GROUP_NN = "Neural networks"


def _read_settings(settings_path: Path):
    settings = {}
    if not settings_path.exists():
        return settings
    with open(settings_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            settings[key.strip()] = value.strip()
    return settings


def _read_summary(summary_path: Path):
    if not summary_path.exists():
        return None, None
    with open(summary_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
    if not header:
        return None, None
    columns = header.split("\t")
    data = np.genfromtxt(summary_path, delimiter="\t", skip_header=1)
    if data.size == 0:
        return columns, None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return columns, data


def _find_col(columns, name_candidates):
    lowered = [c.lower() for c in columns]
    for cand in name_candidates:
        cand_l = cand.lower()
        for i, col in enumerate(lowered):
            if cand_l == col or cand_l in col:
                return i
    return None


def _group_and_nu(folder_name, settings):
    """Return (group_label, nu_label) for one result folder."""
    if folder_name.startswith("main05"):
        model_type = settings.get("model_type", "")
        try:
            k = int(settings.get("k_smoothness", ""))
        except ValueError:
            return None, None
        nu = NU_BY_K.get(k)
        if nu is None:
            return None, None
        if model_type == "flat":
            return GROUP_FLAT, nu
        if model_type == "2l":
            return GROUP_2L, nu
        return None, None
    if folder_name.startswith("main06"):
        return GROUP_NN, "--"
    return None, None


def collect_best_rows(results_dir: Path):
    """Pick the row with smallest final loss per (group, ν)."""
    best: dict[tuple[str, str], dict] = {}

    for folder in sorted(results_dir.iterdir()):
        if not folder.is_dir():
            continue
        settings = _read_settings(folder / "settings.txt")
        group, nu = _group_and_nu(folder.name, settings)
        if group is None:
            continue

        columns, data = _read_summary(folder / "summary.txt")
        if columns is None or data is None:
            continue

        idx_loss = _find_col(columns, ["final_loss", "loss"])
        idx_l2 = _find_col(columns, ["l2_err", "l2"])
        idx_h1 = _find_col(columns, ["h1_err", "h1"])
        if None in (idx_loss, idx_l2, idx_h1):
            continue

        for row in data:
            loss = float(row[idx_loss])
            entry = {
                "group": group,
                "nu": nu,
                "loss": loss,
                "l2": float(row[idx_l2]),
                "h1": float(row[idx_h1]),
                "folder": folder.name,
            }
            key = (group, nu)
            if key not in best or loss < best[key]["loss"]:
                best[key] = entry

    return best


def _format_sci(value):
    """Format like '5.569 \\cdot 10^{-2}'."""
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10 ** exponent
    return f"{mantissa:.3f} \\cdot 10^{{{exponent}}}"


def to_latex_table(best, caption, label):
    group_order = [GROUP_FLAT, GROUP_2L, GROUP_NN]
    nu_order = ["1/2", "3/2", "5/2", "--"]

    lines = [
        "\\begin{table}[htbp]%",
        "\t\\centering",
        "\t\\begin{tabular}{l|ccc}",
        "\t\t\\toprule",
        "\t\tMethod & $\\nu$ & $L^2$-error & $H^1$-error \\\\",
        "\t\t\\midrule\\midrule",
    ]

    for group in group_order:
        group_rows = []
        for nu in nu_order:
            entry = best.get((group, nu))
            if entry is None:
                continue
            group_rows.append(
                f"\t\t{group} & ${entry['nu']}$ "
                f"& ${_format_sci(entry['l2'])}$ "
                f"& ${_format_sci(entry['h1'])}$ \\\\"
            )
        if not group_rows:
            continue
        lines.extend(group_rows)
        if group != group_order[-1]:
            lines.append("\t\t\\midrule")

    lines.extend([
        "\t\t\\bottomrule",
        "\t\\end{tabular}",
        f"\t\\caption{{{caption}}}",
        f"\t\\label{{{label}}}",
        "\\end{table}",
    ])
    return "\n".join(lines) + "\n"


@app.default
def main(
    results_dir: Annotated[str, cyclopts.Parameter(help="Directory containing high-dimensional result folders.")] = "results_highdim/",
    output_tex: Annotated[str, cyclopts.Parameter(help="Output .tex file path.")] = "results_highdim/main_05_06_best_table.tex",
    caption: Annotated[str, cyclopts.Parameter(help="LaTeX table caption.")] = "Best result per expansion size (selected by lowest loss across all expansion sizes) for the high-dimensional example.",
    label: Annotated[str, cyclopts.Parameter(help="LaTeX table label.")] = "tab:errors-high-dimensional",
):
    batch_dir = Path(results_dir)
    if not batch_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    best = collect_best_rows(batch_dir)
    if not best:
        raise RuntimeError(f"No valid summary rows found in: {results_dir}")

    table_tex = to_latex_table(best, caption=caption, label=label)

    out_path = Path(output_tex)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table_tex, encoding="utf-8")

    print(f"Found best rows for {len(best)} (method, ν) cells.")
    print(f"Saved LaTeX table to {out_path}")
    print("Preview:")
    print(table_tex)


if __name__ == "__main__":
    app()
