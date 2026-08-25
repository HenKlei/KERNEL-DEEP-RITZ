for d in results_neural_network_*_depth*/; do
    	python -m kernelDR.experiments.aggregate_seed_runs --results-dir "$d"
done
