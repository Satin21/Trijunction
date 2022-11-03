import numpy as np
import sys, os
import kwant
import json

# sys.path.append(os.path.realpath('./../spin-qubit/'))

sys.path.append("/home/tinkerer/spin-qubit/")

from codes.tools import hamiltonian

from codes.constants import scale, majorana_pair_indices, voltage_keys, bands, topological_gap, sides
from codes.parameters import junction_parameters, dict_update, phase_pairs, voltage_dict
from codes.utils import wannierize, svd_transformation, eigsh

from potential import gate_potential, linear_problem_instance
from Hamiltonian import discrete_system_coordinates
from utility import gather_data
from dask import delayed
from scipy.sparse._coo import coo_matrix


def density(wf):
    density = np.zeros(int(len(wf) / 4))
    for i in range(len(density)):
        density[i] = np.sum(np.abs(wf[4 * i : 4 * (i + 1)])**2)
    return density


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
    system, linear_terms, f_params, reference_wavefunctions = argv[2]

    # print(x)

    if isinstance(x, (list, np.ndarray)):
        params.update(voltage_dict(x))
    elif isinstance(x, float):
        params.update(phase_pairs(pair, x * np.pi))

    
    linear_ham, full_hamiltonian = hamiltonian(
            system, linear_terms, f_params, **params
        )
    
    energies, wavefunctions = eigsh(
            full_hamiltonian.tocsc(),
            len(reference_wavefunctions),
            sigma=0,
            return_eigenvectors=True,
        )
    
    desired, undesired = majorana_loss(energies, 
                                       wavefunctions, 
                                       reference_wavefunctions)
    
    desired /= topological_gap
    undesired /= topological_gap
    print(desired, undesired)

    # Uncomment this in case of soft-thresholding
    cost = 0
    if isinstance(x, (list, np.ndarray)):

        args = (pair.split('-'),
                params['dep_acc_index'],
                (10, 1)
               )

        cost += 1e1* wavefunction_loss(wavefunctions, 
                                  *args
                                 )
    
    
    cost += sum([-desired, undesired])

    return cost

def optimizer_status(x, 
                     max_count=50, 
                     file='/home/tinkerer/trijunction-design/data'
                    ):
    """
    If the optimizer is stuck in the optimum for more than max_count, return -1
    """

    with open(file, 'rb') as outfile:
        data = json.load(outfile)
    
    x = str(x)
    key = list(data.keys())
    if len(key) == 0:
        # if empty dictionary add no matter what
        data[x] = 1
    elif x in data:
        # if x already exists in data, then increase it count by 1
        data[x] += 1
    elif key[0] < x:
        # if existing key in the dictionary is lower than the current x, replace it with x.
        del data[key[0]]
        data[x] = 1

    with open(file , 'w') as outfile:
        json.dump(data, outfile)

    if list(data.values())[0] > max_count:
        return -1
    return 0


def soft_threshold_loss(x, *argv):

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
    print(x)
    pair = argv[0]
    system, linear_terms, (ci, wf_amp) = argv[1]
    indices = argv[2]

    voltages = voltage_dict(x)

    linear_ham, full_hamiltonian = hamiltonian(system, linear_terms, **voltages)
    linear_ham = linear_ham.diagonal()[::4]
    potential_shape_loss = shape_loss(x, *argv)

    argv = (pair, indices, (ci, wf_amp))
    evals, evecs = eigsh(full_hamiltonian, k=6, sigma=0, return_eigenvectors=True)

    return 1e2 * potential_shape_loss + wavefunction_loss(evecs, *argv)


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
    system, linear_terms = argv[1]
    indices = argv[2]

    voltages = voltage_dict(x)

    linear_ham, full_hamiltonian = hamiltonian(system, linear_terms, **voltages)
    linear_ham = linear_ham.diagonal()[::4]

    loss = 0.0

    for gate, index in indices.items():
        diff = np.real(linear_ham[index]) - bands[0]
        if gate in pair:
            if np.any(diff > 0):
                loss += sum(diff[diff > 0])

        else:
            if np.any(diff < 0):
                loss += sum(diff[diff<0])

    return np.abs(loss)


def wavefunction_loss(x, *argv):
    """
    Soft-thresholding with wavefunction

    Input
    -----
    x: nx1 array
    Wavefunction


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
    # unpack arguments
    if len(x.shape) == 1:
        print(x)


        system, params, linear_terms, f_params, reference_wavefunctions = argv[0]
        pair, indices, (ci, wf_amp) = argv[1]

        params.update(voltage_dict(x))


        _, full_hamiltonian = hamiltonian(
                system, linear_terms, f_params, **params
            )
        
        energies, wfs = eigsh(
                full_hamiltonian.tocsc(),
                6,
                sigma=0,
                return_eigenvectors=True,
            )
        
    else:
        pair, indices, (ci, wf_amp) = argv
        wfs = x

    amplitude = lambda i: _amplitude(pair, indices, density(wfs[:, i]))

    desired, undes = [], []
    for i in range(3):
        x, y = amplitude(i)
        desired.append(list(x.values()))
        undes.append(y)
        
    
    undesired = dict.fromkeys(undes[0].keys(), 0.0)
    for key, _ in undes[0].items(): 
        for i in range(3): undesired[key] += undes[i][key]

    desired = np.vstack(desired)

    desired = np.array([np.sum(desired[0::2], axis=0), 
                        np.sum(desired[1::2], axis=0)]
                      )


    sum_desired = np.abs(np.sum(desired, axis=1))
    
    rel_amplitude = sum_desired[0] / sum_desired[1]
    

    rel_des_undes = []
    for i, gate in enumerate(pair):
        rel_des_undes.append([sum_desired[i]/undesired[gate + '_' + str(j+1)] for j in range(2)])
    
    undesired = list(undesired.values())
    uniformity = np.abs(np.sum(np.diff(desired, axis=0)))

    
    # print(sum_desired, np.sum(np.hstack(undesired)), rel_amplitude, np.hstack(rel_des_undes))
    
    if (
        (np.abs(1 - np.sum(rel_amplitude)) < ci / 100) 
        and np.all(np.hstack(rel_des_undes) > 10)
       ):
        try:
            desired_coupling, _ = majorana_loss(energies,
                                                wfs, 
                                                reference_wavefunctions
                                               )
            if desired_coupling>(topological_gap*5/100):
                return -1
        except (AttributeError, UnboundLocalError):
            pass

    
    return (
        - sum(sum_desired)
        + uniformity
        + 1e1*np.sum(np.hstack(undesired))
    )


def _amplitude(pair, index, wf):
    desired = dict.fromkeys(pair, [])
    depleted_channel = set(sides)-set(pair)
    undesired = {}
    for gate, ind in index.items():
        if gate in pair:
            desired[gate] = np.abs(wf[ind])
        elif gate in depleted_channel:
            x = np.abs(wf[ind])
            undesired[list(depleted_channel)[0]] = x[:int(len(x)*50/100)]
        else:
            undesired[gate] = np.abs(wf[ind]) # underneath the gates
    ## remove 50% of the depleted channel which is closer to the center
    
    
    return desired, undesired


def majorana_loss(energies, wavefunctions, reference_wavefunctions):
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

    transformed_hamiltonian = svd_transformation(
        energies, wavefunctions, reference_wavefunctions
    )

    desired = np.abs(transformed_hamiltonian[0, 1])
    undesired = np.linalg.norm(transformed_hamiltonian[2:], ord=1)

    return desired,  undesired


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
