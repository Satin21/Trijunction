import sys, os
import numpy as np
import tinyarray as ta
from tqdm import tqdm
from kwant.builder import FiniteSystem

dirname = os.path.dirname(__file__)
sys.path.append(os.path.realpath(os.path.join(dirname, '../spin-qubit/')))
from potential import gate_potential


def linear_Hamiltonian(
    poisson_system, poisson_params, kwant_system, kwant_params_fn, kwant_params, gates
):
    """
    Find matrices using which the onsite potential energy term in the tight binding Hamiltonian
    can be written as linear combinations with voltages as coefficients. 
    A flat potential is set everywhere, and one gate is varied at a time.

    Parameters
    ----------
    poisson_system: class instance
    Discretized poisson system builder
    
    poisson_params: dict
    Parameters necessary to calculate potential 
        linear problem, site_coords, site_indices
    
    kwant_system: class instance
    Discrete Kwant system builder. 
    
    kwant_params_fn: callable
    Function to update the kwant parameters such as potential energy.
    
    gates: list of strings
    Gate names

    Returns
    -------
    base_ham: scipy sparse coo matrix
    Non-linear part of tight binding Hamiltonian
    
    hamiltonian_V: dict
    sparse coo matrices labelled with the corresponding gate name.
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
    params_fn=None,
    **params,
):
    """
    Build Hamiltonian with a linear potential and a potential
    independent part. Parameters for both part are provided by
    `params`.

    Parameters
    ----------
    kwant_system: class instance
    Discretized Kwant system builder
    
    linear_terms: dict of 
    Sparse coo matrices labelled with the corresponding gate name.
    
    linear_coefficients: dict
    Gate voltages
    
    params_fn: callable
    Function to update the parameters in the Kwant system.
    
    params: dict
    Parameters of the tight binding Hamiltonian.

    Returns
    -------
    numerical_hamiltonian
    """
    if isinstance(linear_terms, dict):
        linear_ham = sum(
            [params[key] * linear_terms[key] for key in linear_terms.keys()]
        )
    else:
        linear_ham = linear_terms

    if not isinstance(kwant_system, FiniteSystem):
        base_ham = kwant_system
    else:
        base_ham = kwant_system.hamiltonian_submatrix(
            sparse=True, params=params_fn(**params)
        )

    numerical_hamiltonian = base_ham + linear_ham

    return linear_ham, numerical_hamiltonian
