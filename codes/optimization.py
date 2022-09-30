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
    kwant_system, f_params, linear_terms, reference_wave_functions = argv[2]

    if isinstance(x, (list, np.ndarray)):
        new_parameter = voltage_dict(x)
    elif isinstance(x, float):
        new_parameter = phase_pairs(pair, x * np.pi)

    params.update(new_parameter)

    numerical_hamiltonian = hamiltonian(kwant_system, linear_terms, f_params, **params)
    
    ## Uncomment this in case of soft-thresholding
    # if isinstance(x, (list, np.ndarray)):
    #     potential_shape = soft_threshold(numerical_hamiltonian.A, 
    #                    params['dep_index'],
    #                    params['acc_index'],
    #                    params['mu'])
    #     if potential_shape > 0:
    #         return threshold


    return majorana_loss(numerical_hamiltonian, reference_wave_functions, kwant_system)


def soft_threshold(ham, dep_index:dict, acc_index:dict, mu:float):
    loss = 0
    for index in dep_index.values():
        diff = ham[4*index, 4*index] - mu
        loss += diff if diff < 0 else 0

    for index in acc_index.values():
        diff = ham[4*index, 4*index] - mu
        loss += diff if diff > 0 else 0
    
    return loss


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
