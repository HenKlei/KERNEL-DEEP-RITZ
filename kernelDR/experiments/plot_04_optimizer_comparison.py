from typing import Annotated

import numpy as np
import cyclopts

from kernelDR.experiments.plot_utils import setup_matplotlib, output_path

app = cyclopts.App()


def read_training_curve(results_dir, k_smoothness, ep, n_per_dim, n_epochs):
    import os
    pattern = f"conv_results_{k_smoothness}_{ep}_{n_per_dim}_"
    matching = [f for f in os.listdir(results_dir) if f.startswith(pattern) and f.endswith(".txt")]
    if not matching:
        filename = f"conv_results_{k_smoothness}_{ep}_{n_per_dim}_{n_epochs}.txt"
        filepath = os.path.join(results_dir, filename)
    else:
        filepath = os.path.join(results_dir, matching[0])
    data = np.genfromtxt(filepath, skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 1], data[:, 2], data[:, 3], data[:, 4]


@app.default
def main(
    results_dirs: Annotated[tuple[str, ...], cyclopts.Parameter(help="Result directories to compare.")],
    labels: Annotated[tuple[str, ...], cyclopts.Parameter(help="Labels for each directory.")],
    k_smoothness: Annotated[int, cyclopts.Parameter(help="Kernel smoothness parameter.")] = 1,
    ep: Annotated[float, cyclopts.Parameter(help="Kernel shape parameter.")] = 1.0,
    n_per_dim: Annotated[int, cyclopts.Parameter(help="Centers per dimension to plot.")] = 4,
    n_epochs: Annotated[int, cyclopts.Parameter(help="Number of epochs (for filename matching).")] = 1000,
    output: Annotated[str, cyclopts.Parameter(help="Output filename.")] = "",
):
    if not output:
        output = output_path("optimizer_comparison.pdf")

    plt = setup_matplotlib()

    colors =["#3B529B", "#FDE725", "#5EC962", "#440154", "#D01F3C", "#FFC077"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for idx, (rdir, lbl) in enumerate(zip(results_dirs, labels)):
        color = colors[idx % len(colors)]
        try:
            epochs, losses, l2_errs, h1_errs = read_training_curve(rdir, k_smoothness, ep, n_per_dim, n_epochs)
        except (FileNotFoundError, OSError) as e:
            print(f"Warning: could not read from {rdir}: {e}")
            continue

        axes[0].plot(epochs, losses, color=color, label=lbl)
        axes[1].semilogy(epochs, l2_errs, color=color, label=lbl)
        axes[2].semilogy(epochs, h1_errs, color=color, label=lbl)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss")
    axes[0].legend()

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Relative $L^2$-error")
    axes[1].set_title("$L^2$-error")
    axes[1].legend()

    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Relative $H^1$-error")
    axes[2].set_title("$H^1$-error")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved to {output}")
    plt.close()


if __name__ == "__main__":
    app()
