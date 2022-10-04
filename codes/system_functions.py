from typing import Dict, Tuple
from codes.utils import eigsh
from kwant.builder import FiniteSystem
from scipy.sparse._coo import coo_matrix
from scipy.sparse._csr import csr_matrix

def adaptive_two_parameters(
    xy, gates, params, trijunction, linear_terms, f_params=None
):
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


def diagonalisation(
    new_param, trijunction, linear_terms, params, f_params=None, nevals=20
):
    """
    Updates parameters dictionary by `new_param` and diagonalises
    the Hamiltonian. In many cases one only changes the linear
    terms or the base Hamiltonian. One can pass the other matrix
    directly as a coo_matrix.

    Parameters
    ----------
    new_param: dict
    trijunction: kwant.Builder or coo_matrix
    linear_terms: set of coo_matrices or single coo_matrix
    params: dict
    f_params: callable
    nevals: int

    Returns
    -------
    lowest `nevals` eigenvalues
    """

    params.update(new_param)

    if isinstance(linear_terms, coo_matrix) or isinstance(linear_terms, csr_matrix):
        linear_ham = linear_terms
    else:
        linear_ham = sum(
            [params[key] * linear_terms[key] for key in linear_terms.keys()]
        )

    if isinstance(trijunction, FiniteSystem):
        num_ham = trijunction.hamiltonian_submatrix(
            sparse=True, params=f_params(**params)
        )
    else:
        num_ham = trijunction

    return eigsh(num_ham + linear_ham, nevals)
