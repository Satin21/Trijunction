import sys, os
import numpy as np
import tinyarray as ta
from tqdm import tqdm
from codes.parameters import junction_parameters
from scipy.sparse._coo import coo_matrix

sys.path.append(os.path.realpath("/home/tinkerer/spin-qubit/"))
from potential import gate_potential


def get_potential(potential):
    def f(x, y):
        return potential[ta.array([x, y])]

    return f


def linear_Hamiltonian(
    poisson_system, poisson_params, kwant_system, kwant_params_fn, kwant_params, gates
):
    """
    Generate the matrix describing the linear contribution of each gate.
    A flat potential is set everywhere, and one gate is varied at a time.

    Parameters
    ----------
    poisson_system
    poisson_params
    kwant_system
    kwant_params_fn
    gates

    Returns
    -------
    base_ham
    hamiltonian_V
    """
    voltages = {}

    for gate in gates:
        voltages[gate] = 0.0

    pp = poisson_params

    # base hamiltonian
    base_ham = kwant_system.hamiltonian_submatrix(
        sparse=True, params=kwant_params_fn(**kwant_params)
    )

    hamiltonian_V = {}
    charges = {}

    # check the effect of varying only one gate
    for gate in tqdm(gates):

        voltages_t = dict.fromkeys(voltages, 0.0)

        voltages_t[gate] = 1.0

        potential = gate_potential(
            poisson_system,
            pp["linear_problem"],
            pp["site_coords"][:, [0, 1]],
            pp["site_indices"],
            voltages_t,
            charges,
            offset=pp["offset"][[0, 1]],
        )

        kwant_params.update(potential=potential)

        hamiltonian = kwant_system.hamiltonian_submatrix(
            sparse=True, params=kwant_params_fn(**kwant_params)
        )

        hamiltonian_V[gate] = hamiltonian - base_ham

    return base_ham, hamiltonian_V


def hamiltonian(
    kwant_system,
    linear_terms,
    params_fn = None,
    **params,
):
    """
    Build Hamiltonian with a linear potential and a potential
    independent part. Parameters for both part are provided by
    `params`.

    Parameters
    ----------
    kwant_system: kwant builder
    linear_terms:
    linear_coefficients: dictionary with voltages for each gate
    params_fn: position dep function describing trijunction
    params: dictionary with parameters for the Hamiltonian

    Returns
    -------
    numerical_hamiltonian
    """
    summed_ham = sum([params[key] * linear_terms[key] for key in linear_terms.keys()])
    
    if isinstance(kwant_system, coo_matrix):
        base_ham = kwant_system
    else:
        base_ham = kwant_system.hamiltonian_submatrix(
            sparse=True, params=params_fn(**params)
        )

    numerical_hamiltonian = base_ham + summed_ham

    return summed_ham, numerical_hamiltonian
