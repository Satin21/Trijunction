import numpy as np
from codes.trijunction_matrices import *


def base_ham(parameters):
    parameters.update(potential=zero_potential)
    return trijunction.hamiltonian_submatrix(
        sparse=True, params=f_params(**parameters)
    )


def adaptive_two_parameters(
    xy,
    voltages,
    params,
    gates
):

    num_ham = base_ham(params)

    for i, gate in enumerate(gates):
        voltages[gate+'_1'] = xy[i]
        voltages[gate+'_2'] = xy[i]

    linear_ham = sum(
        [
            value * linear_terms[key]
            for key, value in voltages.items()
        ]
    )

    evals = eigsh(linear_ham + num_ham, 6)

    return evals[-1]