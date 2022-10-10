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
from dask import delayed

def loss(x, *argv):
    """
    x: either list or scalar (float)
        list when optimizing voltages and float when optimizing phases

    """
    loss_args = argv[0]
    
    pair = loss_args[0]
    params = loss_args[1]
    kwant_system, f_params, linear_terms, reference_wave_functions = loss_args[2]
    pot = loss_args[3]

    if isinstance(x, (list, np.ndarray)):
        new_parameter = voltage_dict(x)
    elif isinstance(x, float):
        new_parameter = phase_pairs(pair, x * np.pi)
        
    # print(new_parameter)

    params.update(new_parameter)
    
    linear_ham, numerical_hamiltonian = hamiltonian(kwant_system, linear_terms, f_params, **params)
    
    # Uncomment this in case of soft-thresholding
    if isinstance(x, (list, np.ndarray)):
        potential_shape = soft_threshold(linear_ham, 
                                         params['dep_index'],
                                         params['acc_index'],
                                         params['mus_nw'][0]
                                        )
        if np.abs(potential_shape) > 0 + pot:
            return 1e-3 * np.abs(potential_shape)
    
    cost = majorana_loss(numerical_hamiltonian, reference_wave_functions, kwant_system)

    return cost

def jacobian(x0, *args):

    initial = delayed(loss)(x0, args[0])
    difference = []
    y0 = [x0] * len(x0)
    step_size = args[1]
    y0 += np.eye(len(x0)) * step_size
    
    res = [delayed(objective)(row, args[0]) for row in y0]
    
        
    def difference(yold, ynew):
        return (ynew - yold) / step_size
    
    output = delayed(loss)(initial, res)
    
    return output.compute()



def soft_threshold(ham, dep_index:dict, acc_index:dict, mu:float):
    loss = 0
    for index in np.hstack(list(dep_index.values())):
        diff = ham[4*index, 4*index] - mu
        loss += diff if diff < 0 else 0

    for index in np.hstack(list(acc_index.values())):
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

    return -desired + undesired


def check_grid(A, B):
    if A % B:
        return A % B
    return B
