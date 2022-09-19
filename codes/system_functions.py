from typing import Dict, Tuple
from codes.trijunction_matrices import make_system
from codes.utils import eigsh


def base_ham(parameters: Dict, trijunction, f_params):

    ham = trijunction.hamiltonian_submatrix(
        sparse=True, params=f_params(**parameters)
    )

    return ham


def adaptive_two_parameters(
    xy: Tuple[float],
    voltages: Dict,
    params: Dict,
    gates: Tuple[str]
):
    """
    Energy of the first non-zero eigenvalue.
    """
    
    zero_potential, linear_terms, trijunction, f_params = make_system()
    
    params.update(potential=zero_potential)
    num_ham = base_ham(params, trijunction, f_params)

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