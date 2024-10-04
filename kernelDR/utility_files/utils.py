import numpy as np


def compute_conv_rates(array_errors, array_input, list_idx_start, list_idx_stop):
    # array_errors: Contains the errors
    # array_input: Contains the number of centers
    # list_idx_start, list_idx_stop: Used to compute an average convergence rate (using different intervals)

    array_input = np.array(array_input)

    list_arrays_rates = []
    list_rates = []

    for idx_run in range(array_errors.shape[0]):
        # Loop over different run

        indices = np.isinf(array_errors[idx_run, :])        # exclude inf values (inf values caused me some problems)
        vector_errors = array_errors[idx_run, ~indices]
        array_input_ = array_input[~indices]

        # Compute several possible decay rates (using different intervals) and finally take the mean value
        array_rates = np.zeros((len(list_idx_start), len(list_idx_stop)))

        for idx1, idx_start in enumerate(list_idx_start):
            for idx2, idx_stop in enumerate(list_idx_stop):
                array_rates[idx1, idx2] = (np.log(vector_errors[idx_stop]) - np.log(vector_errors[idx_start])) \
                                          / (np.log(array_input_[idx_stop]) - np.log(array_input_[idx_start]))

        list_arrays_rates.append(array_rates)
        list_rates.append(np.mean(array_rates))



    return list_rates, list_arrays_rates
