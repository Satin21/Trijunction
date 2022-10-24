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
from scipy.sparse._coo import coo_matrix


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
    
    # print(x)

    pair = argv[0]
    params = argv[1]
    system, linear_terms, f_params, densityoperator, reference_wavefunctions = argv[2]


    if isinstance(x, (list, np.ndarray)):
        new_parameter = voltage_dict(x)
    elif isinstance(x, float):
        new_parameter = phase_pairs(pair, x * np.pi)

    params.update(new_parameter)
    
    linear_ham, full_hamiltonian = hamiltonian(
            system, linear_terms, f_params, **params
        )
    
    energies, wavefunctions = eigsh(
            full_hamiltonian.tocsc(),
            len(reference_wavefunctions),
            sigma=0,
            return_eigenvectors=True,
        )


    # Uncomment this in case of soft-thresholding
    cost = 0
    if isinstance(x, (list, np.ndarray)):
        
        args = (pair.split('-'),
                densityoperator, 
                params['dep_acc_index'],
                (10, 1)
               )

        cost += wavefunction_loss(wavefunctions, 
                                  *args
                                 )
        
    cost += majorana_loss(energies,
                          wavefunctions,
                          reference_wavefunctions)/topological_gap

    return cost


def shape_loss(x, *argv):
    
    """
    Computes total loss from the potential shape and wavefunction thresholding
    
    Input
    -----
    x: 1xn array
    Input voltages
    
    argv: tuple
    arguments required for soft thresholding such as
    (coupled pair:list, (base hamiltonian, linear ham, kwant density operator), indices:dict)
    
    Returns
    -------
    loss: float
    
    """
    # print(x)
    
    pair = argv[0]
    system, linear_terms, density_operator = argv[1]
    indices = argv[2]
    
    voltages = voltage_dict(x)

    linear_ham, full_hamiltonian = hamiltonian(
        system, linear_terms, **voltages
    )
    linear_ham = linear_ham.diagonal()[::4]
    
    loss = 0.0
    
    for gate, index in indices.items():
        diff = np.real(linear_ham[index]) - bands[0]
        if gate[:-1] in pair:
            if diff > 0:
                loss += diff

        else:
            if diff < 0:
                loss += diff

    return np.abs(loss)



def wavefunction_loss(x, *argv):
    """
    Soft-thresholding with wavefunction
    
    Input
    -----
    x: nx1 array
    Wavefunction
    
    densityoperator: kwant operator
    
    indices: dict
    Values are the indices corresponding to the position at which the and wavefunction probability is evaluated.
    Keys are the region names. Channel indices are represented as `left0`, `right0`, `top0`, whereas regions below 
    the gates are represented as `left_1`, `right_1`, `top_1` (with an underscore).
    
    pair: list
    Channel names those need to be coupled.
    
    ci: int
    Confidence interval (%) for the relative magnitude of the wavefunction density across two channels to be coupled.
    
    
    Returns
    -------
    loss: float
    """

    if len(x.shape) == 1:

        system, params, linear_terms, f_params, densityoperator = argv[0]
        pair, indices, (ci, wf_amp) = argv[1]

        params.update(voltage_dict(x))

        _, full_hamiltonian = hamiltonian(
                system, linear_terms, f_params, **params
            )
        
        _, wfs = eigsh(
                full_hamiltonian.tocsc(),
                6,
                sigma=0,
                return_eigenvectors=True,
            )
        
    else:
        pair, densityoperator, indices, (ci, wf_amp) = argv
        wfs = x
    
    amplitude = lambda i: _amplitude(pair, indices, densityoperator(wfs[:, i]))
    
    desired, undesired = [], []
    for i in range(3):
        des, undes = amplitude(i)
        desired.append(list(des.values()))
        undesired.append(undes)
        
    desired = np.vstack(desired)

    desired = [np.sum(desired[0::2], axis=0), 
               np.sum(desired[1::2], axis=0)
              ]
    
    sum_desired = np.sum(desired, axis = 1)
    rel_amplitude = sum_desired[0]/sum_desired[1]
    
    left_bound = (1-ci/100)
    right_bound = (1+ci/100)
    
    # print(sum_desired, np.abs(np.sum(np.diff(desired, axis=0))), np.sum(undesired))
    
    if np.all(sum_desired > wf_amp) and (left_bound<rel_amplitude<right_bound):
        return -1
    
    return (
        - sum(sum_desired)
        + np.abs(np.sum(np.diff(desired, axis=0)))
        + 1e1 * np.sum(undesired)
    )

def _amplitude(pair, index, wf):
    desired = dict.fromkeys(pair, [])
    undesired = []
    for gate, ind in index.items():
        if gate[:-1] in pair:
            temp = desired[gate[:-1]].copy()
            temp.append(np.abs(wf[ind]))
            desired.update({gate[:-1]: temp})

        else:
            undesired.append(np.abs(wf[ind]))
    return desired, undesired
 

def majorana_loss(
    energies, wavefunctions, reference_wavefunctions
):
    """Compute the quality of Majorana coupling in a Kwant system.

    Parameters
    ----------
    energies : 1d array
        Eigenenergies
    wavefunctions: 2d array
        Eigenvectors
    reference_wavefunctions : 2d array
        Majorana wave functions. The first two correspond to Majoranas that
        need to be coupled.
    scale : float
        Energy scale to use.
    """


    transformed_hamiltonian = (
        svd_transformation(energies, wavefunctions, reference_wavefunctions)
    )

    desired = np.abs(transformed_hamiltonian[0, 1])
    undesired = np.linalg.norm(transformed_hamiltonian[2:], ord=1)

    # print(desired, undesired)
    
    return -desired + undesired

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


def check_grid(A, B):
    if A % B:
        return A % B
    return B
