import numpy as np
import sys, os
import kwant

# sys.path.append(os.path.realpath('./../spin-qubit/'))

sys.path.append("/home/tinkerer/spin-qubit/")

from codes.tools import hamiltonian
from codes.constants import scale, majorana_pair_indices, voltage_keys, bands
from codes.parameters import junction_parameters, dict_update, phase_pairs, voltage_dict
from codes.utils import wannierize, svd_transformation, eigsh

from potential import gate_potential, linear_problem_instance
from Hamiltonian import discrete_system_coordinates
from utility import gather_data


def loss(x, *argv):
    """
    x: either list or scalar (float)
        list when optimizing voltages and float when optimizing phases

    """
    pair = argv[0]
    params = argv[1]
    kwant_system, f_params, linear_terms, unperturbed_ml_wfs = argv[2]

    if isinstance(x, list):
        new_parameter = voltage_dict(x)
    elif isinstance(x, float):
        new_parameter = phase_pairs(pair, x * np.pi)

    params.update(new_parameter)

    numerical_hamiltonian = hamiltonian(kwant_system, linear_terms, f_params, **params)

    # shuffle the wavwfunctions based on the Majorana pairs to be optimized
    pair_indices = majorana_pair_indices[pair].copy()
    pair_indices.append(list(set(range(3)) - set(pair_indices))[0])
    shuffle = pair_indices + [-3, -2, -1]
    desired_order = np.array(list(range(2, 5)) + list(range(2)) + [5])[shuffle]

    reference_wave_functions = unperturbed_ml_wfs[desired_order]

    return majorana_loss(numerical_hamiltonian, reference_wave_functions, kwant_system)


def potential_shape_loss(x, *argv):

    if len(x) == 4:
        voltages = {key: x[index] for key, index in voltage_keys.items()}

    elif len(x) == 2:
        voltages = {}

        for arm in ["left", "right", "top"]:
            for i in range(2):
                voltages["arm_" + str(i)] = x[0]

        voltages["global_accumul"] = x[-1]

    for i in range(6):
        voltages["dirichlet_" + str(i)] = 0.0

    poisson_system, poisson_params, mus_nw, dep_points, acc_points = argv

    charges = {}
    potential = gate_potential(
        poisson_system,
        poisson_params["linear_problem"],
        poisson_params["site_coords"][:, [0, 1]],
        poisson_params["site_indices"],
        voltages,
        charges,
        offset=poisson_params["offset"],
    )

    # potential.update((x, y * -1) for x, y in potential.items())

    potential_array = np.array(list(potential.values()))
    potential_keys = np.array(list(potential.keys()))

    loss = []

    for gate, index in acc_points.items():

        channel = potential_array[tuple(acc_points[gate])]

        dgate = potential_array[
            sum([dep_points[gate + "_" + str(i)] for i in range(1, 3)], [])
        ]

        if np.any(dgate < mus_nw):
            # Gated regions not depleted
            loss.append(_depletion_relative_potential(dgate, mus_nw))

        if channel > mus_nw:
            # Channel is depleted relative to nw
            loss.append(np.abs(channel - mus_nw))

        if np.any(channel > dgate):
            # Channel is depleted relative to gated regions
            loss.append(_depletion_relative_potential(dgate, channel))

    for gate in set(["left", "right", "top"]) - set(acc_points.keys()):
        dgate = potential_array[
            sum([dep_points[gate + "_" + str(i)] for i in range(1, 3)], [])
        ]

        if np.any(dgate < mus_nw):
            # Gated regions not depleted relative to nw
            loss.append(_depletion_relative_potential(dgate, mus_nw))

        channel = potential_array[tuple(np.hstack(dep_points[gate]))]

        if np.any(channel < mus_nw):
            # Channel is not depleted relative to nw
            loss.append(np.abs(mus_nw - channel))

    if len(loss):
        return potential, sum(np.hstack(loss))

    return potential, 0


def _depletion_relative_potential(potential, reference):
    return np.abs(potential[np.where(potential < reference)] - reference)


def majorana_loss(
    numerical_hamiltonian, reference_wave_functions, kwant_system, scale=1
):
    """Compute the quality of Majorana coupling in a Kwant system.

    Parameters
    ----------
    x : 1d array
        The vector of parameters to optimize
    numerical_hamiltonian : coo matrix
        A function for returning the sparse matrix Hamiltonian given parameters.
    reference_wave_functions : 2d array
        Majorana wave functions. The first two correspond to Majoranas that
        need to be coupled.
    scale : float
        Energy scale to use.
    """

    energies, wave_functions = eigsh(
        numerical_hamiltonian.tocsc(),
        len(reference_wave_functions),
        sigma=0,
        return_eigenvectors=True,
    )

    transformed_hamiltonian = svd_transformation(
        energies, wave_functions, reference_wave_functions
    )

    desired = np.abs(transformed_hamiltonian[0, 1])
    undesired = np.linalg.norm(transformed_hamiltonian[2:])

    return -desired + np.log(undesired / desired + 1e-3)


def check_grid(A, B):
    if A % B:
        return A % B
    return B
