from typing import Dict, Tuple
from codes.utils import eigsh
from kwant.builder import FiniteSystem
from scipy.sparse._coo import coo_matrix
from scipy.sparse._csr import csr_matrix


def adaptive_two_parameters(xy, keys, params, trijunction, linear_terms, f_params=None):
    """
    Sample N parameters in the trijunction Hamiltonian.
    The parameters to be sampled are defined in `keys`.
    One can sample both voltages and hamiltonian parameters
    simultaneously. In case one of those is fixed, the matrix can
    be passed as an argument for `trijuction` or `linear_terms`.

    Parameters
    ----------
    xy: tuple
    trijunction: kwant.Builder or coo_matrix
    linear_terms: set of coo_matrices or single coo_matrix
    params: dict
    f_params: callable
    nevals: int

    Returns
    -------
    lowest `nevals` eigenvalues
    """

    new_param = {}
    for i, key in enumerate(keys):
        if key in ["left", "right", "top"]:
            new_param[key + "_1"] = xy[i]
            new_param[key + "_2"] = xy[i]
        else:
            new_param[key] = xy[i]

    evals = diagonalisation(
        trijunction=trijunction,
        linear_terms=linear_terms,
        f_params=f_params,
        params=params,
        new_param=new_param,
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
    linear_terms: dict of coo_matrices or single coo_matrix
    params: dict
    f_params: callable
    nevals: int

    Returns
    -------
    lowest `nevals` eigenvalues
    """

    params.update(new_param)

    if isinstance(linear_terms, dict):
        linear_ham = sum(
            [params[key] * linear_terms[key] for key in linear_terms.keys()]
        )
    else:
        linear_ham = linear_terms

    if isinstance(trijunction, FiniteSystem):
        num_ham = trijunction.hamiltonian_submatrix(
            sparse=True, params=f_params(**params)
        )
    else:
        num_ham = trijunction

    return eigsh(num_ham + linear_ham, nevals)
