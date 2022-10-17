import numpy as np
import sys, os
import kwant

# sys.path.append(os.path.realpath('./../spin-qubit/'))

sys.path.append("/home/tinkerer/spin-qubit/")

from codes.tools import hamiltonian
from codes.constants import scale, majorana_pair_indices, voltage_keys, bands, topological_gap
from codes.parameters import junction_parameters, dict_update, phase_pairs, voltage_dict
from codes.utils import wannierize, svd_transformation, eigsh

from potential import gate_potential, linear_problem_instance
from Hamiltonian import discrete_system_coordinates
from utility import gather_data
from dask import delayed


def loss(x, *argv):
    """
        Loss function used to optimise the coupling between pairs of MBSs.
        It can be used to optimize volateges or phases.
        Parameters
        ----------
        x: either list or scalar (float)
            list when optimizing voltages and float when optimizing phases
        argv: argument for the loss functions
            pair: str
                describes the pair to be coupled
            params: dict
                contains parameters for the hamiltonian, voltages, and dep_acc_index
            kwant_system, f_params, linear_terms, reference_wave_functions: tuple
                contains parameters required for `majorana_loss`
        Returns
        -------
        shape_loss: `soft_threshold` if it is not zero
        majorana_loss: if `soft_threshold` is zero
    """

    pair = argv[0]
    params = argv[1]
    kwant_system, f_params, linear_terms, reference_wave_functions = argv[2]


    if isinstance(x, (list, np.ndarray)):
        new_parameter = voltage_dict(x)
    elif isinstance(x, float):
        new_parameter = phase_pairs(pair, x * np.pi)

    params.update(new_parameter)

    linear_ham, numerical_hamiltonian = hamiltonian(
        kwant_system, linear_terms, f_params, **params
    )

    # Uncomment this in case of soft-thresholding
    cost = 0
    if isinstance(x, (list, np.ndarray)):
        potential_shape_loss = soft_threshold(
            params['dep_acc_index'],
            numerical_hamiltonian.diagonal()[::4], # 4 due to spin and particle-hole degrees of freedom.
            pair.split('-'),
            params["mus_nw"][0]
        )
        
        
        cost += np.abs(potential_shape_loss)
        
    cost += majorana_loss(numerical_hamiltonian, 
                          reference_wave_functions, 
                          kwant_system)/topological_gap

    return cost


def jacobian(x0, *args):

    initial = delayed(loss)(x0, *args)
    difference = []
    y0 = [x0] * len(x0)
    step_size = args[-1]
    y0 += np.eye(len(x0)) * step_size

    res = [delayed(loss)(row, *args) for row in y0]

    def difference(yold, ynew):
        return (ynew - yold) / step_size

    output = delayed(loss)(initial, res)

    return output.compute()


def shape_loss(x, *argv):
    """
    Function to optimize the shape of the potential using
    the soft threshold.

    Parameters
    ----------
    x: list of 4 voltages to optimize
    """

    linear_terms = argv[0]
    indices = argv[1]
    pair = argv[2]

    voltages = voltage_dict(x)

    linear_ham = sum(
            [voltages[key] * linear_terms[key] for key in linear_terms.keys()]
        )
    linear_ham = linear_ham.diagonal()[::4]

    potential_shape_loss = soft_threshold(
        indices, linear_ham, pair
    )

    return 1e-3 * np.abs(potential_shape_loss)


def soft_threshold(indices, linear_ham, pair, mu=bands[0]):
    """
    Parameters
    ----------
    indices: dict
        dictionary coming from `dep_acc_indices`
    linear_terms: ndarray
        reduced diagonal of voltage matrix
    pair: ndarray
        array with names of the channels to connect
    mu: float
        reference chemical potential

    Returns
    -------
    loss function of shape
    """
    loss = 0
    
    # loss = {}

    for gate, index in indices.items():
        diff = np.real(linear_ham[index]) - mu

        if gate[:-1] in pair:
            if diff > 0:
                loss += diff
        else:
            if diff < 0:
                loss += diff
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

    transformed_hamiltonian = (
        svd_transformation(energies, wave_functions, reference_wave_functions)
    )

    desired = np.abs(transformed_hamiltonian[0, 1])
    undesired = np.linalg.norm(transformed_hamiltonian[2:], ord=1)

    print(desired, undesired)
    
    return -desired + undesired


def check_grid(A, B):
    if A % B:
        return A % B
    return B
