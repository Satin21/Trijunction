import numpy as np
import sys, os
from dask import delayed

from .tools import hamiltonian
from .constants import (
    scale,
    majorana_pair_indices,
    voltage_keys,
    bands,
    topological_gap,
    sides,
)
from .parameters import phase_pairs, voltage_dict
from .utils import svd_transformation, eigsh


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
        kwant_system, linear_terms, f_params, reference_wave_functions: tuple
            contains parameters required for `majorana_loss`
    Returns
    -------
    cost: float
    A sum of the loss from potential shape, wavefunctions and majorana couplings.
    """

    pair = argv[0]
    params = argv[1]
    system, linear_terms, f_params, reference_wavefunctions = argv[2]

    if isinstance(x, (list, np.ndarray)):
        params.update(voltage_dict(x))
    else:
        params.update(phase_pairs(pair, x * np.pi))

    linear_ham, full_hamiltonian = hamiltonian(system, linear_terms, f_params, **params)

    energies, wavefunctions = eigsh(
        full_hamiltonian.tocsc(),
        len(reference_wavefunctions),
        sigma=0,
        return_eigenvectors=True,
    )

    desired, undesired = majorana_loss(energies, wavefunctions, reference_wavefunctions)

    desired /= topological_gap
    undesired /= topological_gap

    cost = 0
    if isinstance(x, (list, np.ndarray)):

        args = (
            pair.split("-"),
            (system, linear_terms),
            params["dep_acc_index"],
        )

        cost += shape_loss(x, *args)

        weights = [1, 1, 1e2]
        args = (pair.split("-"), params["dep_acc_index"], weights)

        cost += wavefunction_loss(wavefunctions, *args)

    cost += sum([-desired, undesired])

    return cost


def shape_loss(x, *argv):

    """
    Checks whether the potential shape is optimum and return the difference to the reference potential
    otherwise

    Parameters
    ----------
    x: 1xn array
    Input voltages

    argv: tuple
        coupled pair: str
        system: Sparse coo matrix
        Kwant tight binding Hamiltonian.
        indices: dict
        Indices of the Kwant system coordinates where the potential  is checked whether depleted or accumulated.


    Returns
    -------
    loss: float
    Difference between the potential at the indices and the reference potential.

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
                loss += sum(np.abs(diff[diff > 0]))

        else:
            if np.any(diff < 0):
                loss += sum(np.abs(diff[diff < 0]))
    return loss


def wavefunction_loss(x, *argv):
    """
    Loss function based on the amplitude of wavefunctions.

    Parameters
    ----------
    x: 1xn array or nx3 array

    When x is a 1d array, it is considered to be the gate voltages.
    Arguments needed specific to this case are as follows:
        system: Sparse coo matrix
        Kwant tight binding Hamiltonian.

        linear_terms: list of sparse coo matrices
        Each matrix contains along the diagonal the change in the potential energy for a unit change in
        voltage of a gate.

        reference_wavefunctions: nx6 array
        Maximally localized Wannier functions that acts as a good orthogonal basis to compute an
        effective Hamiltonian for Majoranas.



    When x is a nx3 array, it is considered to be wavefunctions. In this case, the function
    needs three wavefunctions corresponding to the energies closest to zero which are nevertheless Majoranas.
    Arguments needed specific to this case are as follows:

        wfs: nx3 array
        Majorana wavefunctions


    pair: list
        List containing strings of the sides to be coupled, e.g. `['left', 'right']`

    indices: dict
        Values are the indices corresponding to the position at which the and wavefunction probability is evaluated.
        Keys are the region names. Channel indices are represented as `left0`, `right0`, `top0`  whereas regions below the
        gates  are represented as `left_1`, `right_1`, `top_1` (with an underscore). Please make sure that the points along
        the channel to be disconnected is not very close to the center of the trijunction so that it doesn't conflict with
        the points along the channels to be connected.

    weights: list
        scaling coefficient for elements in the loss function

    Other arguments needed commonly for the above two cases include:

    indices: dict


    weights: tuple
    Weights for the elements in the cost function

    Returns
    -------
    wf_cost: float
    """
    # unpack arguments
    if len(x.shape) == 1:

        # print(x)

        pair = argv[0]
        system, linear_terms, reference_wavefunctions = argv[1]
        indices = argv[2]
        weights = argv[3]

        _, full_hamiltonian = hamiltonian(system, linear_terms, None, **voltage_dict(x))

        energies, wfs = eigsh(
            full_hamiltonian.tocsc(),
            6,
            sigma=0,
            return_eigenvectors=True,
        )

    else:
        pair, indices, weights = argv
        wfs = x

    amplitude = lambda i: _amplitude(pair, indices, density(wfs[:, i]))

    desired, undes = [], []
    for i in range(3):
        a, b = amplitude(i)
        desired.append(list(a.values()))
        undes.append(b)

    undesired = dict.fromkeys(undes[0].keys(), 0.0)
    for key, _ in undes[0].items():
        for i in range(3):
            undesired[key] += undes[i][key]

    desired = np.vstack(desired)

    desired = np.array([np.sum(desired[0::2], axis=0), np.sum(desired[1::2], axis=0)])

    sum_desired = np.sum(desired, axis=1)

    undesired = np.hstack(list(undesired.values()))
    uniformity = np.abs(np.diff(desired, axis=0))

    wf_cost = (
        -weights[0] * sum(sum_desired)
        + weights[1] * np.sum(uniformity)
        + weights[2] * np.sum(undesired)
    )

    return wf_cost


def _amplitude(pair, index, wf):
    """
    Returns the amplitude of wavefunction at the positions along the channels and underneath the gates

    Input
    -----
    pair: str
        Pair to be coupled. Either 'left-right' or 'right-top' or 'left-top'

    indices: dict
        Indices of the Kwant system coordinates where the potential  is checked whether depleted or accumulated.

    wf: nx1 array
    Wavefunctions

    Returns
    -------
    desired, undesired: two 1d array
    Wavefunction amplitude at the positions to be depleted (also undesired) and accumulated (also desired)
    """
    desired = dict.fromkeys(pair, [])
    depleted_channel = set(sides) - set(pair)
    undesired = {}
    for gate, ind in index.items():
        if gate in pair:
            desired[gate] = np.abs(wf[ind])
        elif gate in depleted_channel:
            undesired[list(depleted_channel)[0]] = np.abs(wf[ind])
        else:
            undesired[gate] = np.abs(wf[ind])  # underneath the gates
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
    """

    transformed_hamiltonian = svd_transformation(
        energies, wavefunctions, reference_wavefunctions
    )

    desired = np.abs(transformed_hamiltonian[0, 1])
    undesired = np.linalg.norm(transformed_hamiltonian[2:], ord=1)

    return desired, undesired


def jacobian(x0, *args):
    """
    Jacobian matrix using finite difference approximation. This gives same results as approx_fprime function
    from scipy.optimize. However, this function has an advantage to be computed parallely using dask.delayed.
    """

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
