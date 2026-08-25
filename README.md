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
quoted below were measured on a single **NVIDIA A100-SXM4-80GB GPU**, on a
node with an **Intel Xeon Gold 6542Y CPU** (Adam runs are GPU-resident; the
matrix-form linear-solver and the kernel-interpolation runs assemble on GPU
but the solve itself uses CPU SciPy/LAPACK).

Every script that involves randomness accepts `--seed` (default `0`), which
fixes the quadrature point sets drawn during training and, for the neural
networks, the initialisation of the weights. The interpolation scripts
(`main_02a`/`main_02b`) are deterministic and therefore take no seed. Seeding
makes repeated runs well-defined and independent of each other; it does not
guarantee bitwise reproducibility across different GPUs or library versions.
To repeat a run, pass a different `--seed` together with a different
`--results-dir` — except for the neural-network sweeps, which take a list of
seeds and write one result file per seed, see
[Repeated runs and error bars](#repeated-runs-and-error-bars).

The reported relative errors are evaluated on a fixed grid of `--n-error` points
that is independent of the quadrature points used during training. The defaults
are chosen such that the reported values are converged: a uniform grid of
`640000` points in two dimensions, and Monte Carlo integration in the
high-dimensional example, where a tensor-product grid would degenerate.
`--num-logs` controls how often the errors are evaluated during training.

### Convergence study: Deep Ritz with kernel ansatz
Smooth solution (Poisson on `(0, 1)^2`) for Matérn kernels with smoothness
parameters `k = 0, 1, 2` (≈ 2.5 h per command):
```bash
python kernelDR/experiments/main_01a_smooth_solution.py --k-smoothness 0 --results-dir results_smooth_solution_matern_k0/
python kernelDR/experiments/main_01a_smooth_solution.py --k-smoothness 1 --results-dir results_smooth_solution_matern_k1/
python kernelDR/experiments/main_01a_smooth_solution.py --k-smoothness 2 --results-dir results_smooth_solution_matern_k2/
```
Singular solution (Laplace on the pacman domain) for the same three smoothness
values (≈ 2.5 h per command):
```bash
python kernelDR/experiments/main_01b_singular_solution.py --k-smoothness 0 --results-dir results_singular_solution_matern_k0/
python kernelDR/experiments/main_01b_singular_solution.py --k-smoothness 1 --results-dir results_singular_solution_matern_k1/
python kernelDR/experiments/main_01b_singular_solution.py --k-smoothness 2 --results-dir results_singular_solution_matern_k2/
```
Adding `--save-models` stores the trained model of every configuration, which
allows the errors to be recomputed afterwards with
`main_10_recompute_errors.py`, for instance on a different evaluation grid.

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
Assemble and solve the kernel-collocation linear system directly. The system is
assembled in the **Lagrange basis** of the kernel space (`--flag-lagrange`), the
same basis the optimiser runs use. The spanned space, and hence the discrete
problem, is identical to the one in the basis of kernel translates, but the
condition number of the stiffness matrix differs by up to twelve orders of
magnitude — for the smooth example it ranges from `4.0e3` to `1.2e15` for the
translates and stays between `1.3e1` and `2.6e2` in the Lagrange basis. No
regularisation is therefore needed (`--regularization 0`), and the computed
errors agree with those obtained in the plain basis to three digits. Each
command sweeps `k = 0, 1, 2` internally.

Quadrature points that coincide with a centre are dropped before assembly, as in
the optimiser runs. This matters for `--use-uniform-quadrature`, where a grid
commensurate with the centres makes many points coincide at once: without the
filter the assembled system is spoiled (relative `L2`-error `1.03` instead of
`1.7e-1` at `n_per_dim = 20`, `k = 2`).

Since the ansatz depends linearly on its coefficients, the system is assembled by
evaluating all basis functions and their gradients once and forming the quadrature
sums as matrix products (`--assembly vectorized`, the default), which reduces the
cost from `O(n_points * n_centers^3)` to `O(n_points * n_centers^2)`. The original
entry-by-entry version is kept as `--assembly loop`; both agree to machine
precision.
```bash
python kernelDR/experiments/main_04a_smooth_solution_matrix_form.py --flag-lagrange --regularization 0 --results-dir results_matrix_form_smooth_solution/
python kernelDR/experiments/main_04b_singular_solution_matrix_form.py --flag-lagrange --regularization 0 --results-dir results_matrix_form_singular_solution/
```
The Wendland counterparts in the appendix use the same commands with
`--kernel wendland`. Add `--save-models` to keep the solutions for a later
re-evaluation on a different grid, see `main_10_recompute_errors.py`. Each
command takes ≈ 20 min on a GPU.

### Iterative-solver comparison (CG vs Adam on fixed vs stochastic quadrature)
To compare the iterative behaviour of conjugate gradient on the assembled
linear system against the Adam-based Deep Ritz minimisation, three runs are
performed for a single configuration (Matérn `k=2`, `n_per_dim = 20`,
`n_centers = 484`). In the Lagrange basis CG converges in 87 iterations, so the
solve itself is sub-minute — the cost of the first command is dominated by the
error evaluation at every `--cg-error-interval`-th iteration (≈ 26 min on a CPU
with interval 2, ≈ 3 min with `--cg-error-interval 0`, which records only the
energy and the residual). Each Adam run takes ≈ 7 min on a GPU.

```bash
# 1. CG on the fixed uniform-grid linear system (per-iteration energy and
#    residual norm are written to cg_energy_k_2_n_20.txt; the relative errors
#    every --cg-error-interval iterations to cg_convergence_k_2_n_20.txt).
python kernelDR/experiments/main_04a_smooth_solution_matrix_form.py \
    --flag-lagrange --regularization 0 \
    --linear-solver cg --use-uniform-quadrature \
    --list-kmat 2 --list-n-per-dim 20 --cg-error-interval 2 \
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

CG operates on the assembled system, so its quadrature is fixed by construction;
`--use-uniform-quadrature` selects the same uniform grid that
`--fixed-integration-points` uses for Adam. With `--cg-error-interval` the
relative errors of the current iterate are recorded alongside the energy and the
residual in `cg_convergence_k_<k>_n_<n>.txt`, which separates how accurately the
discrete system is solved from how good the resulting approximation is:
```bash
python kernelDR/experiments/plot_06_cg_semiconvergence.py \
    --results-dir results_matrix_form_smooth_solution_cg_uniform/ --k-smoothness 2 --n-per-dim 20
```

### Optimization versus generalization
Two scripts report, side by side, how accurately the discrete problem is solved
(discrete energy, residual, coefficient norm) and how good the resulting
approximation is (relative errors on the independent grid).

`main_08_regularization_sweep.py` does so over a range of Tikhonov parameters, and
`plot_07_regularization_sweep.py` plots the result (minutes on a CPU):
```bash
python -m kernelDR.experiments.main_08_regularization_sweep \
    --problem-type smooth --k-smoothness 2 --n-per-dim 20 --use-uniform-quadrature \
    --results-dir results_regularization_sweep/
python kernelDR/experiments/plot_07_regularization_sweep.py \
    --results-file results_regularization_sweep/regularization_sweep_smooth_k2_n20_uniform_ni10000.txt \
    --reference-error 3.90e-4
```
`--n-i` varies the number of fixed quadrature points the system is built from, and
`--system-dir` reuses a system assembled earlier by `main_04a`/`main_04b`.

`main_09_solver_comparison.py` does so for the direct solve, CG and both Adam
variants on one and the same discrete problem — same centers, same basis, same
quadrature points (`--skip-adam` restricts it to the matrix-based solvers):
```bash
python -m kernelDR.experiments.main_09_solver_comparison \
    --problem-type smooth --k-smoothness 2 --n-per-dim 20 \
    --results-dir results_solver_comparison/
```

### Influence of the number of quadrature points
Answers whether a fixed quadrature reaches the accuracy of the resampled
optimisation by simply using more points. The centers are held fixed while the
quadrature is refined, so the approximation error stays constant and only the
quadrature error changes. `--skip-adam` restricts the run to the matrix-based
solvers, which makes each configuration cheap.

```bash
for problem in smooth singular; do
  for k in 0 1 2; do
    for ni in 10000 40000 160000 640000; do
      python -m kernelDR.experiments.main_09_solver_comparison \
          --problem-type $problem --k-smoothness $k --n-per-dim 20 \
          --flag-lagrange --skip-adam --use-uniform-quadrature \
          --list-regularization 0 --n-i $ni --n-b $((ni/10)) \
          --chunk-size 20000 \
          --results-dir results_ni_sweep_${problem}_k${k}/ni${ni}/
    done
  done
done
```
24 runs, a few minutes each on a GPU; the `n_i = 640000` ones dominate. Use
`--chunk-size` to bound the memory of the `(n_points, n_centers, dim)` gradient
tensor during assembly.

The per-run files are consolidated into one table per problem and smoothness
(one row per quadrature size) for plotting, see
`reference_results/results_ni_sweep_{smooth,singular}/errors_k_*.txt`.

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
The errors of the high-dimensional example are evaluated by Monte Carlo
integration rather than on a uniform grid, which in `d = 10` would place only
`floor(n^(1/d))` points per axis. `HighDimDiagonalExample` overrides the error
routines accordingly, and `--n-error` sets the number of samples per repetition,
of which five are averaged.

For a deeper per-run inspection (convergence curves, matrix-trajectory plots,
etc.), use
```bash
python kernelDR/experiments/main_05b_highdim_diagonal_evaluate.py --help
```

### Integration diagnostics
Quantifies how accurate the quadrature actually is, which is what justifies the
`--n-error` default. Three things are measured: the spread of the energy over
independent Monte Carlo draws, how well a grid reproduces the closed-form norms
of the reference solution, and how much the reported errors of a trained model
change when the evaluation grid is refined.

```bash
python -m kernelDR.experiments.main_07_integration_diagnostics \
    --problem-type smooth --results-dir results_integration_diagnostics/
python -m kernelDR.experiments.main_07_integration_diagnostics \
    --problem-type singular --results-dir results_integration_diagnostics/
```
Pass `--model-path` (a checkpoint written with `--save-models`) to include the
error-refinement table for that model. On the grid used throughout, going from
`640000` to `1440000` evaluation points changes the reported errors by less than
`0.2 %` in the `L2`-norm and `0.7 %` in the `H1`-norm.

Note that on the pacman domain a uniform grid resolves the `H1` semi-norm of the
reference solution only to a few percent, and not monotonically, because the grid
cannot follow the sector boundary exactly. This affects the absolute values of the
singular `H1`-errors but not the convergence rates, since it is a constant factor.

### Which command produces which figure
The figures in the paper are typeset from the result files listed below. All of
them are shipped in `reference_results/`, so the paper can be rebuilt without
re-running anything.

| Paper | Content | Result directory | Section above |
|---|---|---|---|
| Fig. 1, 7 | kernel deep Ritz vs. interpolation | `results_{smooth,singular}_solution/`, `results_interpolation_{smooth,singular}_solution/` | Convergence study, interpolation reference |
| Fig. 2, 8 | kernel vs. neural network | as above plus `results_neural_network_*_gelu_depth2/` | Comparison against neural networks |
| Fig. 3, 9 | kernel deep Ritz vs. assembled system | as above plus `results_matrix_form_{smooth,singular}_solution/` | Direct linear system |
| Fig. 4, 5 | energy over the iterations, CG semi-convergence | `results_matrix_form_smooth_solution_cg_uniform/`, `results_smooth_solution_matern_k2_{fixed,singlerun}/` | Iterative-solver comparison |
| Fig. 6 | the singular reference solution | none, drawn analytically | — |
| Fig. 10 | error vs. number of quadrature points | `results_ni_sweep_{smooth,singular}/` | Influence of the number of quadrature points |
| Fig. 11–14 | Wendland counterparts of Fig. 1, 3, 7, 9 | the `*_wendland` directories | Wendland kernel |
| Fig. 15–18 | network activation and depth sweeps | `results_neural_network_*_depth*/` | Neural network activation and depth sweeps |
| Tab. 2 | solvers on one discrete problem | `results_solver_comparison_lagrange/` | Optimization versus generalization |
| Tab. 3 | high-dimensional example | `results_highdim/` | High-dimensional example |

The `_plain_basis` directories hold the earlier runs in the basis of kernel
translates. They are kept because the condition numbers quoted in the text come
from them, but no figure reads them.

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
Error versus the number of quadrature points, for both examples side by side,
with the resampled deep Ritz errors as dashed reference lines:
```bash
python kernelDR/experiments/plot_08_quadrature_sweep.py --norm L2 --output quadrature_sweep_L2.pdf
python kernelDR/experiments/plot_08_quadrature_sweep.py --norm H1 --output quadrature_sweep_H1.pdf
```

Pass `--help` to any script to see the remaining options (kernel shape
parameter `ep`, title suffixes, etc.).

## Appendix
The appendix of the paper contains (i) the activation-function and
network-depth sweeps used to motivate the GELU / depth-2 baseline of the
main-body neural-network comparison and (ii) the repeat of the convergence
study with the compactly supported Wendland kernels.

### Neural network activation and depth sweeps
Each command runs the full width sweep for one configuration and one seed
(≈ 2.5–4.5 h per command depending on depth). The sweeps are repeated for
several seeds, see [Repeated runs and error bars](#repeated-runs-and-error-bars)
below; `--list-seeds` can be given repeatedly to run several seeds in one
process, and every seed writes its own `convergence_results_seed<SEED>.txt`.

**Activation functions (depth 2).** Sweep all six activations at fixed depth 2:
```bash
for act in tanh relu gelu silu softplus sin; do
    python kernelDR/experiments/main_03a_smooth_solution_neural_network.py --activation $act --num-layers 2 \
        --list-seeds 0 --results-dir results_neural_network_smooth_solution_${act}_depth2/
    python kernelDR/experiments/main_03b_singular_solution_neural_network.py --activation $act --num-layers 2 \
        --list-seeds 0 --results-dir results_neural_network_singular_solution_${act}_depth2/
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

### Repeated runs and error bars
Both the deep Ritz runs with the kernel ansatz and the neural-network sweeps are
repeated for three RNG seeds, so that the figures show the median with error bars
spanning the minimum and the maximum instead of a single run.

For the kernel runs the variability is small — the largest ratio between the
maximum and the minimum error over three seeds is about `1.5` at the finest mesh
norm for the smooth example and about `1.02` for the singular one, in which case
the error bars are thinner than the line width. The network runs vary far more,
up to a factor of `345` for the deeper architectures.

The kernel sweep is one process per (problem, smoothness, seed, mesh size). One
`main_01` call sweeps all mesh sizes sequentially, so splitting on
`--list-n-per-dim` as well is what makes the sweep parallel:
```bash
# seeds 1 and 2 (seed 0 is the original run); 132 jobs, ~14 GPU-h in total
for seed in 1 2; do
  for prob in smooth singular; do
    [ $prob = smooth ] && script=main_01a_smooth_solution || script=main_01b_singular_solution
    for k in 0 1 2; do
      for n in 1 2 4 6 8 10 12 14 16 18 20; do
        python -m kernelDR.experiments.$script \
            --k-smoothness $k --seed $seed --list-n-per-dim $n \
            --results-dir results_kernel_seeds/${prob}_k${k}_seed${seed}/n${n}/
      done
    done
  done
done
```
Give each process its own `--results-dir`: `main_01` truncates `timings.txt` at
startup and writes `settings.txt`, so concurrent processes sharing a directory
would clobber both. The `conv_results_*` file names are unique per mesh size, so
the split runs merge by copying them together.

> **Note.** `aggregate_deep_ritz_errors.py` rebuilds `errors_k_*.txt` from the
> `conv_results_*` files of a single run. Its file-name pattern does not match the
> `_seed{N}` files, so re-running it on a directory that already holds a
> multi-seed aggregate would silently drop the median and the error-bar columns.

The neural-network sweeps are repeated for several RNG seeds, so
that the figures can show the spread over the repetitions instead of a single
run. One process handles one (configuration, seed) pair; the seeds are
independent of each other, while within one seed the random streams are paired
across the widths (the RNG is re-seeded per width). A single (width, seed) run
therefore reproduces exactly the corresponding row of the full sweep, which
makes it possible to re-run individual configurations without repeating the
whole sweep.

The runs are small — at most a few hundred parameters, peaking below 100 MB of
device memory — and leave the GPU mostly idle, so several of them should be
executed concurrently. The launcher script does this for the complete appendix
sweep (6 activations at depth 2, plus depths 1, 4 and 8, on both problems):
```bash
# Distribute the sweep over four GPUs, six concurrent runs on each:
GPUS="0 1 2 3" JOBS_PER_GPU=6 SEEDS="0 1 2" ./run_appendix_seed_sweeps.sh

# Single device (or CPU): one queue, NPAR runs at a time
NPAR=8 SEEDS="0 1 2" ./run_appendix_seed_sweeps.sh
```
This prepares 54 runs (9 configurations × 2 problems × 3 seeds). With `GPUS` set,
the runs are distributed over the given devices round-robin, each device gets its
own queue of `JOBS_PER_GPU` concurrent runs pinned via `CUDA_VISIBLE_DEVICES`, and
each process is limited to `OMP_THREADS` (default 2) CPU threads so that many
processes can share the host. Set `DRY_RUN=1` to print the job list and the
per-device distribution without running anything, `SAVE_MODELS=1` to keep the
trained networks, and `OUT_ROOT` to write elsewhere than the current directory.
Per-run logs land in `logs_appendix_sweeps/`. Sequentially the whole sweep is
roughly 135–160 GPU-h, so the concurrency is what makes it practical.

Afterwards, each results directory is condensed into one file holding the
summary statistics over its seeds:
```bash
for d in results_neural_network_*_depth*/; do
    python -m kernelDR.experiments.aggregate_seed_runs --results-dir "$d"
done
```
This writes `convergence_results_aggregated.txt` and prints the spread per
configuration. The column order is chosen so that the `\addplot table[x index=1,
y index=2]` statements in the LaTeX sources keep working unchanged:

| index | column | meaning |
| --- | --- | --- |
| 0 | `neurons` | neurons per layer (the key the repetitions are grouped by) |
| 1 | `n_params` | total number of trainable parameters |
| 2, 3, 4 | `L2_median`, `H1_median`, `loss_median` | median over the seeds |
| 5 | `n_seeds` | number of repetitions this row was aggregated from |
| 6 … 12 | `L2_min`, `L2_max`, `L2_err_minus`, `L2_err_plus`, `L2_mean`, `L2_std`, `L2_geomean` | spread of the relative $L^2$-error |
| 13 … 19 | `H1_…` | same for the relative $H^1$-error |
| 20 … 26 | `loss_…` | same for the final loss (`loss_geomean` is `nan`, the loss is negative) |

`*_err_minus` / `*_err_plus` are the distances from the median to the smallest
and largest repetition, i.e. the asymmetric error bars belonging to the median:
```latex
\addplot[..., error bars/.cd, y dir=both, y explicit]
    table[x index=1, y index=2, y error minus index=8, y error plus index=9] {..._aggregated.txt};
```
For a shaded min-max band instead, use the `*_min` / `*_max` columns with
`\addplot[name path=…]` and `\addplot fill between`.

The script is generic: `--pattern`, `--key-column`, `--passthrough-columns` and
`--stat-columns` make it applicable to any set of result files sharing a schema,
for instance the `summary.txt` files of the high-dimensional runs.

To inspect the sweeps outside LaTeX, `plot_05_nn_hyperparameter_sweep.py` draws
the median as a line and the min-max range over the seeds as a band. Directories
without an aggregated file (e.g. the single-run reference data of the first
submission) are drawn as a plain line, so old and new results can be compared
directly:
```bash
python kernelDR/experiments/plot_05_nn_hyperparameter_sweep.py \
    --results-dirs results_neural_network_smooth_solution_gelu_depth2/ \
    --results-dirs results_neural_network_smooth_solution_tanh_depth2/ \
    --labels "GELU" --labels "tanh" --output activations_smooth.pdf
```

### Wendland kernel
All scripts in [`kernelDR/experiments/`](kernelDR/experiments/) support the
`--kernel {matern,wendland}` flag (Matérn is the default); `--k-smoothness`
selects the smoothness as for Matérn.

Deep Ritz convergence (≈ 2.25 h per command):
```bash
for k in 0 1 2; do
    python kernelDR/experiments/main_01a_smooth_solution.py --kernel wendland --k-smoothness $k \
        --n-error 640000 --num-logs 50 --save-models --results-dir results_smooth_solution_wendland_k$k/
    python kernelDR/experiments/main_01b_singular_solution.py --kernel wendland --k-smoothness $k \
        --n-error 640000 --num-logs 50 --save-models --results-dir results_singular_solution_wendland_k$k/
done
```
Interpolation reference (sweeps `k = 0, 1, 2` internally; ≈ 5–10 min per command):
```bash
python kernelDR/experiments/main_02a_smooth_solution_interpolation.py --kernel wendland --results-dir results_interpolation_smooth_solution_wendland/
python kernelDR/experiments/main_02b_singular_solution_interpolation.py --kernel wendland --results-dir results_interpolation_singular_solution_wendland/
```
Direct linear system (≈ 1–1.5 h per command):
```bash
python kernelDR/experiments/main_04a_smooth_solution_matrix_form.py --kernel wendland --n-error 640000 --save-models --results-dir results_matrix_form_smooth_solution_wendland/
python kernelDR/experiments/main_04b_singular_solution_matrix_form.py --kernel wendland --n-error 640000 --save-models --results-dir results_matrix_form_singular_solution_wendland/
```

## Questions
If you have any questions, feel free to contact us via email at <hendrik.kleikamp@uni-graz.at>.

## License (BSD-2-Clause)
The code is licensed under BSD-2-clause, see [LICENSE.txt](LICENSE.txt).
