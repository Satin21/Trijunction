import sys, os

sys.path.append("/home/tinkerer/spin-qubit/code/")

import numpy as np
import tinyarray as ta
from tqdm import tqdm
import parameters

# sys.path.append(os.path.realpath('./../spin-qubit/'))
from potential import gate_potential

# https://stackoverflow.com/a/3233356
import collections.abc


def find_resonances(energies, n, i=1, sign=1, **kwargs):
    """
    Extract peaks from np.abs(lowest) mode in energies.
    By choosing 'sign' we extract either peaks or dips.
    Parameters:
    -----------
    """
    levels = energies.T
    ground_state = levels[n // 2 + i]
    peaks, properties = find_peaks(sign * np.abs(ground_state), **kwargs)

    return ground_state, peaks


def dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_potential(potential):
    def f(x, y):
        return potential[ta.array([x, y])]

    return f


def linear_Hamiltonian(
    poisson_system, poisson_params, kwant_system, kwant_params_fn, gates
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

    # generate flat potential
    zero_potential = dict(
        zip(
            ta.array(pp["site_coords"][:, [0, 1]] - pp["offset"]),
            np.zeros(len(pp["site_coords"])),
        )
    )

    # base hamiltonian
    mu = parameters.bands[0]
    kwant_params = parameters.junction_parameters(m_nw=[mu, mu, mu])
    kwant_params.update(potential=zero_potential)

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
    linear_coefficients: dict,
    params_fn: callable,
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
    summed_ham = sum(
        [
            linear_coefficients[key] * linear_terms[key]
            for key, value in linear_coefficients.items()
        ]
    )

    base_ham = kwant_system.hamiltonian_submatrix(
        sparse=True, params=params_fn(**params)
    )

    numerical_hamiltonian = base_ham + summed_ham

    return numerical_hamiltonian
