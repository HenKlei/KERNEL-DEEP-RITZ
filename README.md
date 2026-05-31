[![DOI](https://zenodo.org/badge/867681687.svg)](https://doi.org/10.5281/zenodo.13890995)

```
# ~~~
# This file is part of the paper:
#
#           " Kernel Methods in the Deep Ritz framework:
#                       Theory and practice "
#
#   https://github.com/HenKlei/KERNEL-DEEP-RITZ.git
#
# Copyright 2026 all developers. All rights reserved.
# License: Licensed as BSD-2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Hendrik Kleikamp, Tizian Wenzel
# ~~~
```

# Solution of elliptic PDEs via kernel methods
In this repository, we provide the code used for the numerical experiments in the paper "Kernel Methods in the Deep Ritz framework: Theory and practice" by Hendrik Kleikamp and Tizian Wenzel.

You find the preprint [here](https://arxiv.org/abs/2410.03503).

## Installation

### Without Docker
On a system with Python 3.12 and `git`, run:
```bash
git clone https://github.com/HenKlei/KERNEL-DEEP-RITZ.git
cd KERNEL-DEEP-RITZ
python -m venv venv
source venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```
Replace the PyTorch index URL with `https://download.pytorch.org/whl/cu126` (or the appropriate
version for your driver) if you want GPU support.

### Using Docker
On a system with `docker` (see [here](https://www.docker.com/get-started/) for details),
the following commands should be sufficient to install the `kernelDR` package with all required dependencies
in a new docker container:
```bash
git clone https://github.com/HenKlei/KERNEL-DEEP-RITZ.git
cd KERNEL-DEEP-RITZ
docker build -t kernel-deep-ritz .
```
To run the code in an interactive bash-session, use the command
```bash
docker run -it kernel-deep-ritz bash
```

## Running the experiments
The scripts that produce the results in the paper are in
[`kernelDR/experiments/`](kernelDR/experiments/). All wall-clock estimates
quoted below were measured on a single **NVIDIA H200 GPU** (Adam runs are
GPU-resident; the matrix-form linear-solver and the kernel-interpolation
runs assemble on GPU but the solve itself uses CPU SciPy/LAPACK).

### Convergence study: Deep Ritz with kernel ansatz
Smooth solution (Poisson on `(0, 1)^2`) for Matérn kernels with smoothness
parameters `k = 0, 1, 2` (≈ 2.25 h per command):
```bash
python kernelDR/experiments/main_01a_smooth_solution.py --k-smoothness 0 --results-dir results_smooth_solution_matern_k0/
python kernelDR/experiments/main_01a_smooth_solution.py --k-smoothness 1 --results-dir results_smooth_solution_matern_k1/
python kernelDR/experiments/main_01a_smooth_solution.py --k-smoothness 2 --results-dir results_smooth_solution_matern_k2/
```
Singular solution (Laplace on the pacman domain) for the same three smoothness
values (≈ 2.25 h per command):
```bash
python kernelDR/experiments/main_01b_singular_solution.py --k-smoothness 0 --results-dir results_singular_solution_matern_k0/
python kernelDR/experiments/main_01b_singular_solution.py --k-smoothness 1 --results-dir results_singular_solution_matern_k1/
python kernelDR/experiments/main_01b_singular_solution.py --k-smoothness 2 --results-dir results_singular_solution_matern_k2/
```

### Convergence study: kernel interpolation reference
Reference solutions interpolated directly with the same kernels used as baseline
(≈ 5–10 min per command):
```bash
python kernelDR/experiments/main_02a_smooth_solution_interpolation.py --results-dir results_interpolation_smooth_solution/
python kernelDR/experiments/main_02b_singular_solution_interpolation.py --results-dir results_interpolation_singular_solution/
```

### Comparison against neural networks
The kernel ansatz is replaced by a fully connected neural network on the same
smooth/singular problems. All networks are trained for 100,000 epochs with a
`MultiStepLR` learning-rate schedule (the script defaults). Network widths are
chosen so the number of trainable parameters roughly matches the kernel sweep
(`n_per_dim = 1, 2, …, 20` ≈ 13 … 488 centers). Each command below runs the
full width sweep for one (activation, depth) configuration and takes
≈ 2.5–4.5 h depending on depth. Result directory names encode the activation
function and the network depth explicitly, i.e.
`results_neural_network_{smooth,singular}_solution_{activation}_depth{num_layers}/`.

The main-body figure compares the kernel ansatz against a GELU network of depth
two (the configuration selected from the activation and depth sweeps in the
appendix):
```bash
python kernelDR/experiments/main_03a_smooth_solution_neural_network.py --activation gelu --num-layers 2 \
    --results-dir results_neural_network_smooth_solution_gelu_depth2/
python kernelDR/experiments/main_03b_singular_solution_neural_network.py --activation gelu --num-layers 2 \
    --results-dir results_neural_network_singular_solution_gelu_depth2/
```

### Comparison against the direct linear system (matrix form)
Assemble and solve the kernel-collocation linear system directly. A tiny
Tikhonov term (`--regularization 1e-10`, the script default) is added to the
diagonal to stabilise the Cholesky solve at large `n_per_dim` and high kernel
smoothness (Matérn `k=2`); the bias is several orders of magnitude below the
achievable errors, so the convergence rates are unaffected. Each command sweeps
`k = 0, 1, 2` internally and takes ≈ 1–1.5 h on the H200.
```bash
python kernelDR/experiments/main_04a_smooth_solution_matrix_form.py --results-dir results_matrix_form_smooth_solution/
python kernelDR/experiments/main_04b_singular_solution_matrix_form.py --results-dir results_matrix_form_singular_solution/
```

### Iterative-solver comparison (CG vs Adam on fixed vs stochastic quadrature)
To compare the iterative behaviour of conjugate gradient on the assembled
linear system against the Adam-based Deep Ritz minimisation, three runs are
performed for a single configuration (Matérn `k=2`, `n_per_dim = 20`,
`n_centers = 484`). The CG run is sub-minute; each Adam run takes ≈ 27 min,
so the whole section takes ≈ 55 min.

```bash
# 1. CG on the fixed uniform-grid linear system (per-iteration energy and
#    residual norm are written to cg_energy_k_2_n_20.txt).
python kernelDR/experiments/main_04a_smooth_solution_matrix_form.py \
    --linear-solver cg --use-uniform-quadrature \
    --list-kmat 2 --list-n-per-dim 20 \
    --results-dir results_matrix_form_smooth_solution_cg_uniform/

# 2. Adam with fixed uniform-grid quadrature (deterministic energy per epoch).
python kernelDR/experiments/main_01a_smooth_solution.py \
    --k-smoothness 2 --list-n-per-dim 20 \
    --fixed-integration-points \
    --results-dir results_smooth_solution_matern_k2_fixed/

# 3. Adam with stochastic (resampled) quadrature -- the standard Deep Ritz
#    setup, run once at the same n_per_dim to obtain a matching loss history.
python kernelDR/experiments/main_01a_smooth_solution.py \
    --k-smoothness 2 --list-n-per-dim 20 \
    --results-dir results_smooth_solution_matern_k2_singlerun/
```

### High-dimensional example
Section 4.3 of the paper applies the method to a `d = 10` Poisson problem with
a sine-along-the-diagonal solution and compares flat kernels, two-layered (2L)
kernels and a neural network. All runs use Matérn kernels with shape parameter
`ep = 0.1`, sweep over expansion sizes `n_centers ∈ {10, 20, 30}` (kernels) and
widths `{8, 16, 32}` (NN), and train for 10,000 epochs with early stopping
(patience 500). 
The kernel runs:
```bash
for model_type in flat 2l; do
    for k in 0 1 2; do
        python -m kernelDR.experiments.main_05_highdim_diagonal \
            --d-x 10 --model-type $model_type --kernel matern --k-smoothness $k --ep 0.1 \
            --list-n-centers 10 --list-n-centers 20 --list-n-centers 30 \
            --n-epochs 10000 --early-stopping-patience 500 \
            --results-dir results_highdim/main05_d10_${model_type}_matern_s${k}_ep01/
    done
done
```
The neural-network baseline (depth-2 GELU network, sweep over widths):
```bash
python -m kernelDR.experiments.main_06_highdim_diagonal_neural_network \
    --d-x 10 --num-layers 2 --activation gelu \
    --list-num-neurons-per-layer 8 --list-num-neurons-per-layer 16 --list-num-neurons-per-layer 32 \
    --n-epochs 10000 --early-stopping-patience 500 \
    --results-dir results_highdim/main06_d10_l2_gelu/
```

For a deeper per-run inspection (convergence curves, matrix-trajectory plots,
etc.), use
```bash
python kernelDR/experiments/main_05b_highdim_diagonal_evaluate.py --help
```

### Plotting
The figures in the paper itself are typeset with `pgfplots` inside the LaTeX
sources and read the result files directly. To inspect the same data outside
LaTeX (matplotlib PDFs), the `plot_*.py` scripts in
[`kernelDR/experiments/`](kernelDR/experiments/) reproduce most of the
convergence and comparison plots.

A frozen copy of the data shipped with the paper lives in `reference_results/`,
and the default arguments of `plot_01_convergence.py`, `plot_02_nn_comparison.py`,
and `plot_03_matrix_comparison.py` point at the corresponding subdirectories
there. The result-generation commands above intentionally write into the cwd
(not into `reference_results/`) so that re-running never overwrites the shipped
reference. To plot your own re-run, pass `--deep-ritz-dir`, `--interpolation-dir`,
etc. explicitly to the directories you produced.

Convergence (`L^2` and `H^1` errors versus the mesh norm `h`, kernel Deep Ritz
vs. kernel interpolation; one plot per problem). `plot_01_convergence.py` reads
consolidated `errors_k_{0,1,2}.txt` files from a single directory. The smooth
and singular subdirectories under `reference_results/` already contain those
aggregated files, so:
```bash
python kernelDR/experiments/plot_01_convergence.py \
    --deep-ritz-dir reference_results/results_smooth_solution/ \
    --interpolation-dir reference_results/results_interpolation_smooth_solution/ \
    --output convergence_smooth.pdf
python kernelDR/experiments/plot_01_convergence.py \
    --deep-ritz-dir reference_results/results_singular_solution/ \
    --interpolation-dir reference_results/results_interpolation_singular_solution/ \
    --output convergence_singular.pdf
```
If you re-run the Deep Ritz experiments yourself, the per-`k` folders
(`results_{smooth,singular}_solution_matern_k{0,1,2}/`) contain only
`conv_results_*.txt` files, not the consolidated `errors_k_*.txt` that
`plot_01` expects. Aggregate them first with
```bash
python aggregate_deep_ritz_errors.py --kernel matern --out-dir <target_dir>/
```
and then rename / symlink the three resulting
`results_{smooth,singular}_solution_matern_errors_k_{0,1,2}.txt` files to
`errors_k_{0,1,2}.txt` inside per-problem directories that you can pass to
`--deep-ritz-dir`.

Kernel-vs.-network comparison at matched parameter counts (one activation /
depth combination per call):
```bash
python kernelDR/experiments/plot_02_nn_comparison.py \
    --deep-ritz-dir reference_results/results_smooth_solution/ \
    --nn-dir reference_results/results_neural_network_smooth_solution_gelu_depth2/ \
    --k-smoothness 1 --output nn_smooth_gelu_depth2.pdf
```
Energy minimisation (Adam) vs. direct linear-system solve, side-by-side for
`k = 0, 1, 2`:
```bash
python kernelDR/experiments/plot_03_matrix_comparison.py \
    --deep-ritz-dir reference_results/results_smooth_solution/ \
    --matrix-dir reference_results/results_matrix_form_smooth_solution/ \
    --output matrix_comparison_smooth.pdf
```
Iterative-solver comparison (loss / energy / `L^2` / `H^1` over iterations for
CG vs. Adam-fixed vs. Adam-stochastic). The three input directories are the
ones produced in the *Iterative-solver comparison* section above (or the
corresponding copies under `reference_results/`):
```bash
python kernelDR/experiments/plot_04_optimizer_comparison.py \
    --results-dirs reference_results/results_matrix_form_smooth_solution_cg_uniform/ \
    --results-dirs reference_results/results_smooth_solution_matern_k2_fixed/ \
    --results-dirs reference_results/results_smooth_solution_matern_k2_singlerun/ \
    --labels "CG (uniform)" --labels "Adam (uniform)" --labels "Adam (stochastic)" \
    --k-smoothness 2 --n-per-dim 20 --output optimizer_comparison.pdf
```
Pass `--help` to any script to see the remaining options (kernel shape
parameter `ep`, title suffixes, etc.).

## Appendix
The appendix of the paper contains (i) the activation-function and
network-depth sweeps used to motivate the GELU / depth-2 baseline of the
main-body neural-network comparison and (ii) the repeat of the convergence
study with the compactly supported Wendland kernels.

### Neural network activation and depth sweeps
Each command runs the full width sweep for one configuration (≈ 2.5–4.5 h per
command depending on depth).

**Activation functions (depth 2).** Sweep all six activations at fixed depth 2:
```bash
for act in tanh relu gelu silu softplus sin; do
    python kernelDR/experiments/main_03a_smooth_solution_neural_network.py --activation $act --num-layers 2 \
        --results-dir results_neural_network_smooth_solution_${act}_depth2/
    python kernelDR/experiments/main_03b_singular_solution_neural_network.py --activation $act --num-layers 2 \
        --results-dir results_neural_network_singular_solution_${act}_depth2/
done
```

**Network depth.** The depth sweep fixes the activation to the best performer
from the activation sweep above (GELU in the commands below; adjust if another
activation is selected) and varies the number of hidden layers. Widths are
reduced as depth grows to keep parameter counts comparable. The depth-2 point
is the GELU / depth-2 run from the main-body NN comparison (or, equivalently,
the corresponding run from the activation sweep above).
```bash
# Depth 1 (widths chosen so n_params ≈ 15 … 500)
python kernelDR/experiments/main_03a_smooth_solution_neural_network.py --activation gelu --num-layers 1 \
    --list-num-neurons-per-layer 2 --list-num-neurons-per-layer 3 --list-num-neurons-per-layer 5 \
    --list-num-neurons-per-layer 8 --list-num-neurons-per-layer 11 --list-num-neurons-per-layer 16 \
    --list-num-neurons-per-layer 20 --results-dir results_neural_network_smooth_solution_gelu_depth1/
python kernelDR/experiments/main_03b_singular_solution_neural_network.py --activation gelu --num-layers 1 \
    --list-num-neurons-per-layer 2 --list-num-neurons-per-layer 3 --list-num-neurons-per-layer 5 \
    --list-num-neurons-per-layer 8 --list-num-neurons-per-layer 11 --list-num-neurons-per-layer 16 \
    --list-num-neurons-per-layer 20 --results-dir results_neural_network_singular_solution_gelu_depth1/

# Depth 4 (widths ≈ 13 … 480 params)
python kernelDR/experiments/main_03a_smooth_solution_neural_network.py --activation gelu --num-layers 4 \
    --list-num-neurons-per-layer 1 --list-num-neurons-per-layer 2 --list-num-neurons-per-layer 3 \
    --list-num-neurons-per-layer 4 --list-num-neurons-per-layer 6 --list-num-neurons-per-layer 8 \
    --list-num-neurons-per-layer 10 --results-dir results_neural_network_smooth_solution_gelu_depth4/
python kernelDR/experiments/main_03b_singular_solution_neural_network.py --activation gelu --num-layers 4 \
    --list-num-neurons-per-layer 1 --list-num-neurons-per-layer 2 --list-num-neurons-per-layer 3 \
    --list-num-neurons-per-layer 4 --list-num-neurons-per-layer 6 --list-num-neurons-per-layer 8 \
    --list-num-neurons-per-layer 10 --results-dir results_neural_network_singular_solution_gelu_depth4/

# Depth 8 (widths ≈ 20 … 480 params)
python kernelDR/experiments/main_03a_smooth_solution_neural_network.py --activation gelu --num-layers 8 \
    --list-num-neurons-per-layer 1 --list-num-neurons-per-layer 2 --list-num-neurons-per-layer 3 \
    --list-num-neurons-per-layer 4 --list-num-neurons-per-layer 5 --list-num-neurons-per-layer 6 \
    --list-num-neurons-per-layer 7 --results-dir results_neural_network_smooth_solution_gelu_depth8/
python kernelDR/experiments/main_03b_singular_solution_neural_network.py --activation gelu --num-layers 8 \
    --list-num-neurons-per-layer 1 --list-num-neurons-per-layer 2 --list-num-neurons-per-layer 3 \
    --list-num-neurons-per-layer 4 --list-num-neurons-per-layer 5 --list-num-neurons-per-layer 6 \
    --list-num-neurons-per-layer 7 --results-dir results_neural_network_singular_solution_gelu_depth8/
```

### Wendland kernel
All scripts in [`kernelDR/experiments/`](kernelDR/experiments/) support the
`--kernel {matern,wendland}` flag (Matérn is the default); `--k-smoothness`
selects the smoothness as for Matérn.

Deep Ritz convergence (≈ 2.25 h per command):
```bash
python kernelDR/experiments/main_01a_smooth_solution.py --kernel wendland --k-smoothness 0 --results-dir results_smooth_solution_wendland_k0/
python kernelDR/experiments/main_01a_smooth_solution.py --kernel wendland --k-smoothness 1 --results-dir results_smooth_solution_wendland_k1/
python kernelDR/experiments/main_01a_smooth_solution.py --kernel wendland --k-smoothness 2 --results-dir results_smooth_solution_wendland_k2/

python kernelDR/experiments/main_01b_singular_solution.py --kernel wendland --k-smoothness 0 --results-dir results_singular_solution_wendland_k0/
python kernelDR/experiments/main_01b_singular_solution.py --kernel wendland --k-smoothness 1 --results-dir results_singular_solution_wendland_k1/
python kernelDR/experiments/main_01b_singular_solution.py --kernel wendland --k-smoothness 2 --results-dir results_singular_solution_wendland_k2/
```
Interpolation reference (sweeps `k = 0, 1, 2` internally; ≈ 5–10 min per command):
```bash
python kernelDR/experiments/main_02a_smooth_solution_interpolation.py --kernel wendland --results-dir results_interpolation_smooth_solution_wendland/
python kernelDR/experiments/main_02b_singular_solution_interpolation.py --kernel wendland --results-dir results_interpolation_singular_solution_wendland/
```
Direct linear system (≈ 1–1.5 h per command):
```bash
python kernelDR/experiments/main_04a_smooth_solution_matrix_form.py --kernel wendland --results-dir results_matrix_form_smooth_solution_wendland/
python kernelDR/experiments/main_04b_singular_solution_matrix_form.py --kernel wendland --results-dir results_matrix_form_singular_solution_wendland/
```

## Questions
If you have any questions, feel free to contact us via email at <hendrik.kleikamp@uni-graz.at>.

## License (BSD-2-Clause)
The code is licensed under BSD-2-clause, see [LICENSE.txt](LICENSE.txt).
