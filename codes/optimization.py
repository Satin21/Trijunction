import numpy as np
import sys, os
import kwant

# sys.path.append(os.path.realpath('./../spin-qubit/'))

sys.path.append("/home/tinkerer/spin-qubit/")

from codes.tools import hamiltonian
from codes.constants import (
    scale,
    majorana_pair_indices,
    voltage_keys,
    bands,
    topological_gap,
)
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
    system, linear_terms, f_params, density_operator, reference_wave_functions = argv[2]

    if isinstance(x, (list, np.ndarray)):
        new_parameter = voltage_dict(x)
    elif isinstance(x, float):
        new_parameter = phase_pairs(pair, x * np.pi)

    params.update(new_parameter)

    # Uncomment this in case of soft-thresholding
    cost = 0
    if isinstance(x, (list, np.ndarray)):
        args = (
            pair.split("-"),
            (system, linear_terms, density_operator),
            params["dep_acc_index"],
        )
        potential_shape_loss = soft_threshold_loss(x, *args)

        cost += potential_shape_loss

    linear_ham, full_hamiltonian = hamiltonian(system, linear_terms, f_params, **params)

    cost += majorana_loss(full_hamiltonian, reference_wave_functions) / topological_gap

    return cost


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
    system, linear_terms, density_operator = argv[1]
    indices = argv[2]

    voltages = voltage_dict(x)

    linear_ham, full_hamiltonian = hamiltonian(system, linear_terms, **voltages)
    linear_ham = linear_ham.diagonal()[::4]
    potential_shape_loss = shape_loss(indices, linear_ham, pair)

    if potential_shape_loss < 1e-9:
        evals, evecs = eigsh(full_hamiltonian, k=6, sigma=0, return_eigenvectors=True)
        return wavefunction_loss(evecs, density_operator, indices, pair)
    return potential_shape_loss


def shape_loss(indices, linear_ham, pair, mu=bands[0]):
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
    return np.abs(loss)


def wavefunction_loss(wfs, densityoperator, indices, pair, ci=50):
    """
    Soft-thresholding with wavefunction

    Input
    -----
    wfs: nx1 array
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

    def return_gain(pair, index, wf):
        gain = dict.fromkeys(pair, [])
        for gate, ind in index.items():
            if gate[:-1] in pair:
                a = gain[gate[:-1]].copy()
                a.append(np.abs(wf[ind]))
                gain.update({gate[:-1]: a})
        return gain

    gain = np.vstack(
        [
            list(return_gain(pair, indices, densityoperator(wfs[:, i])).values())
            for i in range(3)
        ]
    )
    gain = [np.sum(gain[0::2], axis=0), np.sum(gain[1::2], axis=0)]
    sum_gain = np.sum(gain, axis=1)

    if np.all(sum_gain > 1e-5) and (
        (1 - ci / 100) < sum_gain[0] / sum_gain[1] < (1 + ci / 100)
    ):
        return -1
    return -(sum(sum_gain)) + np.sum(np.diff(gain, axis=0))


def majorana_loss(numerical_hamiltonian, reference_wave_functions):
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
    undesired = np.linalg.norm(transformed_hamiltonian[2:], ord=1)

    print(desired, undesired)

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
