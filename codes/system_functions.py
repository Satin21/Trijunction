from typing import Dict, Tuple
from codes.utils import eigsh
from toolz.functoolz import memoize


def adaptive_two_parameters(xy, gates, params, kwant_params):
    """
    Energy of the first non-zero eigenvalue.
    `gates` can be 'left', 'right', 'top', 'accum'.
    """

    for i, gate in enumerate(gates):
        if gate in ["left", "right", "top"]:
            params[gate + "_1"] = xy[i]
            params[gate + "_2"] = xy[i]
        elif gate == "global_accumul":
            params[gate] = xy[i]

    evals = diagonalisation(
        kwant_params=kwant_params,
        params=params,
        new_param={},
        nevals=6,
    )

    return evals[-1]


def diagonalisation(new_param, kwant_params, params, nevals=20):
    """
    """
    trijunction, f_params, linear_terms = kwant_params
    params.update(new_param)

    linear_ham = sum(
        [params[key] * linear_terms[key] for key in linear_terms.keys()]
    )

    num_ham = trijunction.hamiltonian_submatrix(sparse=True, params=f_params(**params))

    return eigsh(num_ham + linear_ham, nevals)
