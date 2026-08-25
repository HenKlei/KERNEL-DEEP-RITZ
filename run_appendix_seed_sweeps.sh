#!/usr/bin/env bash
#
# Run the neural-network hyperparameter sweeps of the appendix (activation
# functions at depth 2, and the depth sweep) repeatedly for several RNG seeds,
# so that the figures can report the spread over the repetitions.
#
# The individual runs are small (at most a few hundred parameters) and leave the
# GPU mostly idle, so several of them are executed concurrently; NPAR controls
# how many. One process handles one (configuration, seed) pair and writes
# convergence_results_seed<SEED>.txt into the configuration's results directory.
#
# Usage:
#   ./run_appendix_seed_sweeps.sh                   # 4 jobs at a time, seeds 0 1 2
#   NPAR=8 SEEDS="0 1 2" ./run_appendix_seed_sweeps.sh
#   DRY_RUN=1 ./run_appendix_seed_sweeps.sh         # only print what would run
#
# Environment:
#   NPAR        number of concurrent runs (default 4)
#   SEEDS       space-separated list of seeds (default "0 1 2")
#   SAVE_MODELS if set, pass --save-models so that the error metrics of the
#               trained networks can be recomputed later on a different
#               evaluation grid without repeating the training
#   OUT_ROOT   directory the results directories are created in (default ".")
#   PYTHON     interpreter to use (default "python")
#   DRY_RUN    if set, print the job list instead of running it
#
# After the runs have finished, aggregate each configuration with
#   python -m kernelDR.experiments.aggregate_seed_runs --results-dir <dir>/

set -euo pipefail

NPAR="${NPAR:-4}"
SEEDS="${SEEDS:-0 1 2}"
OUT_ROOT="${OUT_ROOT:-.}"
PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-}"
JOBS_PER_GPU="${JOBS_PER_GPU:-6}"
OMP_THREADS="${OMP_THREADS:-2}"
SAVE_MODELS="${SAVE_MODELS:-}"

SMOOTH_SCRIPT="kernelDR/experiments/main_03a_smooth_solution_neural_network.py"
SINGULAR_SCRIPT="kernelDR/experiments/main_03b_singular_solution_neural_network.py"

LOG_DIR="${OUT_ROOT}/logs_appendix_sweeps"
mkdir -p "$LOG_DIR"

job_dir="$(mktemp -d)"
trap 'rm -rf "$job_dir"' EXIT

# With GPUS set, the runs are distributed over the given devices round-robin and
# each device gets its own queue of JOBS_PER_GPU concurrent runs. The models here
# have at most a few hundred parameters and peak below 100 MB of device memory, so
# the limit is how many processes can keep a device busy, not memory.
if [ -n "$GPUS" ]; then
    read -r -a gpu_array <<< "$GPUS"
    n_gpus=${#gpu_array[@]}
else
    n_gpus=0
fi
job_index=0

# Expand "2 3 5" into "--list-num-neurons-per-layer 2 --list-num-neurons-per-layer 3 ..."
widths_flags() {
    local flags=""
    for w in $1; do
        flags="$flags --list-num-neurons-per-layer $w"
    done
    echo "$flags"
}

# add_job <problem: smooth|singular> <activation> <num_layers> <widths> <seed>
add_job() {
    local problem="$1" activation="$2" depth="$3" widths="$4" seed="$5"
    local script results_dir flags

    if [ "$problem" = "smooth" ]; then
        script="$SMOOTH_SCRIPT"
    else
        script="$SINGULAR_SCRIPT"
    fi
    results_dir="${OUT_ROOT}/results_neural_network_${problem}_solution_${activation}_depth${depth}/"
    flags="$(widths_flags "$widths")"
    [ -n "$SAVE_MODELS" ] && flags="$flags --save-models"

    local command="$PYTHON $script --activation $activation --num-layers $depth${flags} \
--list-seeds $seed --results-dir $results_dir \
> ${LOG_DIR}/${problem}_${activation}_depth${depth}_seed${seed}.log 2>&1"

    if [ "$n_gpus" -gt 0 ]; then
        local gpu=${gpu_array[$((job_index % n_gpus))]}
        printf '%s\0' "CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$OMP_THREADS $command" \
            >> "${job_dir}/gpu_${gpu}"
    else
        printf '%s\0' "$command" >> "${job_dir}/all"
    fi
    job_index=$((job_index + 1))
}

# Widths per depth, chosen so that the parameter counts stay comparable
# (identical to the commands documented in the README).
WIDTHS_DEPTH1="2 3 5 8 11 16 20"
WIDTHS_DEPTH2="1 2 3 6 8 12 14"
WIDTHS_DEPTH4="1 2 3 4 6 8 10"
WIDTHS_DEPTH8="1 2 3 4 5 6 7"

for seed in $SEEDS; do
    for problem in smooth singular; do
        # Activation sweep at depth 2. The GELU/depth-2 run is shared with the
        # depth sweep below and therefore only launched here.
        for activation in tanh relu gelu silu softplus sin; do
            add_job "$problem" "$activation" 2 "$WIDTHS_DEPTH2" "$seed"
        done
        # Depth sweep at the activation selected from the sweep above.
        add_job "$problem" gelu 1 "$WIDTHS_DEPTH1" "$seed"
        add_job "$problem" gelu 4 "$WIDTHS_DEPTH4" "$seed"
        add_job "$problem" gelu 8 "$WIDTHS_DEPTH8" "$seed"
    done
done

echo "Prepared ${job_index} runs (seeds: ${SEEDS})."
if [ "$n_gpus" -gt 0 ]; then
    echo "Devices: ${GPUS} (${JOBS_PER_GPU} concurrent run(s) per device, "\
"${OMP_THREADS} CPU thread(s) each) -> $((n_gpus * JOBS_PER_GPU)) runs at a time."
    for gpu in "${gpu_array[@]}"; do
        count=$(tr -cd '\0' < "${job_dir}/gpu_${gpu}" | wc -c)
        echo "  GPU ${gpu}: ${count} run(s)"
    done
else
    echo "No devices given (set GPUS to distribute over several GPUs); "\
"running ${NPAR} at a time."
fi
echo "Logs: ${LOG_DIR}/"

if [ -n "${DRY_RUN:-}" ]; then
    if [ "$n_gpus" -gt 0 ]; then
        for gpu in "${gpu_array[@]}"; do
            tr '\0' '\n' < "${job_dir}/gpu_${gpu}"
        done
    else
        tr '\0' '\n' < "${job_dir}/all"
    fi
    exit 0
fi

if [ "$n_gpus" -gt 0 ]; then
    # One independent queue per device, so that the per-device concurrency is
    # exactly JOBS_PER_GPU regardless of how long individual runs take.
    for gpu in "${gpu_array[@]}"; do
        xargs -0 -P "$JOBS_PER_GPU" -I {} bash -c '{}' < "${job_dir}/gpu_${gpu}" &
    done
    wait
else
    xargs -0 -P "$NPAR" -I {} bash -c '{}' < "${job_dir}/all"
fi

echo "All runs finished. Aggregate them with, for example:"
echo "  for d in ${OUT_ROOT}/results_neural_network_*_depth*/; do \\"
echo "      $PYTHON -m kernelDR.experiments.aggregate_seed_runs --results-dir \"\$d\"; \\"
echo "  done"
