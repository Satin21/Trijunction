from typing import Dict, Tuple
from codes.trijunction_matrices import make_system
from codes.utils import eigsh
from toolz.functoolz import memoize


def base_ham(trijunction, f_params, parameters):
    """
    Calculate the Hamiltonian of a trijunction using a given set
    of parameters and a position dependent function.

    The result is stored in memory to save time.
    """

    ham = trijunction.hamiltonian_submatrix(
        sparse=True, params=f_params(**parameters)
    )

    return ham


def adaptive_two_parameters(
    xy,
    voltages,
    gates,
    params
):
    """
    Energy of the first non-zero eigenvalue.
    `gates` can be 'left', 'right', 'top', 'accum'.
    """
    
    kwant_params = make_system()

    for i, gate in enumerate(gates):
        if gate in ['left', 'right', 'top']:
            voltages[gate+'_1'] = xy[i]
            voltages[gate+'_2'] = xy[i]            
        elif gate == 'accum':
            voltages[gate] = xy[i]

    evals = diagonalisation(
        kwant_params=kwant_params[1:],
        voltages=voltages,
        params=params,
        new_param={},
        nevals=6
    )
    
    return evals[-1]


def diagonalisation(
    new_param, kwant_params, voltages, params, nevals=20
):
    """
    
    """
    trijunction, f_params, linear_terms = kwant_params
    params.update(new_param)

    linear_ham = sum(
            [
                voltages[key] * linear_terms[key]
                for key, value in voltages.items()
            ]
        )

    num_ham = base_ham(trijunction, f_params, params)

    return eigsh(num_ham + linear_ham,  nevals)