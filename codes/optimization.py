import numpy as np
import sys, os
from dask import delayed
from codes.tools import hamiltonian

from codes.constants import (
    scale,
    majorana_pair_indices,
    voltage_keys,
    bands,
    topological_gap,
    sides,
)
from codes.parameters import phase_pairs, voltage_dict
from codes.utils import svd_transformation, eigsh


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

    print(x)

    if isinstance(x, (list, np.ndarray)):
        params.update(voltage_dict(x))
    elif isinstance(x, float):
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
    # prints(desired, undesired)

    # Uncomment this in case of soft-thresholding
    cost = 0
    if isinstance(x, (list, np.ndarray)):

        args = (
            pair.split("-"),
            (system, linear_terms),
            params["dep_acc_index"],
        )

        cost += shape_loss(x, *args)

        weights = [1, 1, 1e2]
        args = (pair.split("-"), params["dep_acc_index"], (10, weights))

        cost += wavefunction_loss(wavefunctions, *args)

    cost += sum([-desired, undesired])

    return cost


def shape_loss(x, *argv):

    """
    Checks whether the potential shape is optimum and return the difference to the reference potential
    otherwise

    Input
    -----
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
    system, linear_terms, _ = argv[1]
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
    # print(f"shape:{loss}")
    return loss


def wavefunction_loss(x, *argv):
    """
    Loss function based on the amplitude of wavefunctions.

    Input
    -----
    x: 1xn array or nx3 array
    When x is a 1d array, it is considered to be the gate voltages.
    Arguments needed specific to this case as follows:
        system: Sparse coo matrix
        Kwant tight binding Hamiltonian.

        params: dict
        Parameters of the Majorana Hamiltonian

        linear_terms: list of sparse coo matrices
        Each matrix contains along the diagonal the change in the potential energy for a unit change in
        voltage of a gate.

        f_params: callable
        Updates the parameters in the Kwant Hamiltonian such as potential and the phases.

        reference_wavefunctions: nx6 array
        Maximally localized Wannier functions that acts as a good orthogonal basis to compute an
        effective Hamiltonian for Majoranas.


    When x is a nx3 array, it is considered to be wavefunctions. In this case, the function
    needs three wavefunctions corresponding to the energies closest to zero which are nevertheless Majoranas.
    Arguments needed specific to this case are as follows:
        indices: dict
        Indices of the Kwant system coordinates where the potential  is checked whether depleted or accumulated.

    pair: str
        Pair to be coupled. Either 'left-right' or 'right-top' or 'left-top'

    Other arguments needed commonly for the above two cases include:

    ci: int
    Confidence interval (%) for the relative magnitude of the wavefunction density across two channels to be coupled.

    indices: dict
    Values are the indices corresponding to the position at which the and wavefunction probability is evaluated.
    Keys are the region names. Channel indices are represented as `left0`, `right0`, `top0`, whereas regions below
    the gates are represented as `left_1`, `right_1`, `top_1` (with an underscore).

    weights: tuple
    Weights for the elements in the cost function

    Returns
    -------
    wf_cost: float
    """
    # unpack arguments
    if len(x.shape) == 1:
        print(x)

        system, params, linear_terms, f_params, reference_wavefunctions = argv[0]
        pair, ci, weights = argv[1]

        indices = params["dep_acc_index"]

        params.update(voltage_dict(x))

        _, full_hamiltonian = hamiltonian(system, linear_terms, f_params, **params)

        energies, wfs = eigsh(
            full_hamiltonian.tocsc(),
            6,
            sigma=0,
            return_eigenvectors=True,
        )

    else:
        pair, indices, (ci, weights) = argv
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

    rel_amplitude = sum_desired[0] / sum_desired[1]

    rel_des_undes = []
    for i, gate in enumerate(pair):
        rel_des_undes.append(
            [sum_desired[i] / undesired[gate + "_" + str(j + 1)] for j in range(2)]
        )

    undesired = np.hstack(list(undesired.values()))
    uniformity = np.abs(np.diff(desired, axis=0))

    ## avoid optimizing the following elements more than needed
    undesired[np.where(undesired > 1e3)] = 1e3
    uniformity[np.where(uniformity < 1e-6)] = 1e-6

    # print(sum_desired, uniformity, np.hstack(rel_des_undes))

    if (np.abs(1 - np.sum(rel_amplitude)) < ci / 100) and np.all(
        np.hstack(rel_des_undes) > 10
    ):
        try:
            desired_coupling, _ = majorana_loss(energies, wfs, reference_wavefunctions)
            # print(desired_coupling)
            # print(f"coupling:{desired_coupling:.3e}")
            if desired_coupling > (topological_gap * 1 / 100):
                return -1
        except UnboundLocalError:
            pass

    wf_cost = (
        -weights[0] * sum(sum_desired)
        + weights[1] * np.sum(uniformity)
        + weights[2] * np.sum(undesired)
    )

    return wf_cost

def density(wf):
    """
    Works similar to the Kwant density operator; takes particle-hole and spin degrees of freedom into account.
    """
    density = np.zeros(int(len(wf) / 4))
    for i in range(len(density)):
        density[i] = np.sum(np.abs(wf[4 * i : 4 * (i + 1)]) ** 2)
    return density


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

    return potential_shape_loss + wavefunction_loss(evecs, *argv)


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
