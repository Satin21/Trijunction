from scipy.sparse import csc_matrix
import scipy.linalg as la
import kwant
import numpy as np


def sort_eigen(ev):
    """
    Sort eigenvectors and eigenvalues from lowest to max by absolute value.
    """
    evals, evecs = ev
    idx = np.argsort(np.abs(evals))
    evals = evals[idx]
    evecs = evecs[:, idx]
    return evals, evecs


def prepare_perturbation(
    system, key, val, x_cut, unperturbed_hamiltonian, f_params, n_wire, n_dot, **parameters
):

    parameters[key] = val
    perturbed_hamiltonian = system.hamiltonian_submatrix(params=f_params(**parameters))

    energies, modes = sort_eigen(np.linalg.eigh(perturbed_hamiltonian))

    block_hamiltonian, _ = lowdin_partition(system,
                                            perturbed_hamiltonian,
                                            x_cut,
                                            n_wire=n_wire,
                                            n_dot=n_dot)

    barrier_perturbation = unperturbed_hamiltonian - perturbed_hamiltonian
    hopping_perturbation = unperturbed_hamiltonian - block_hamiltonian

    return energies, modes, np.array(barrier_perturbation), np.array(hopping_perturbation)


def two_sided(system, x_cut, l_cut):
    # Projectors of each subspace
    PL = kwant.operator.Density(system,
                                where=lambda site: site.pos[0] < x_cut or site.pos[0] >= x_cut + l_cut,
                                sum=True).tocoo()
    PR = kwant.operator.Density(system,
                                where=lambda site: x_cut + l_cut > site.pos[0] >= x_cut,
                                sum=True).tocoo()
    return PL, PR


def one_sided(system, x_cut):
    # Projectors of each subspace
    PL = kwant.operator.Density(system,
                                where=lambda site: site.pos[0] < x_cut,
                                sum=True).tocoo()
    PR = kwant.operator.Density(system,
                                where=lambda site: site.pos[0] >= x_cut,
                                sum=True).tocoo()
    return PL, PR


def lowdin_partition(
    system, hamiltonian, x_cut, n_wire, n_dot, solve=False, sides=1, l_cut=None
):
    if sides == 1:
        PL, PR = one_sided(system, x_cut)
    elif sides == 2:
        PL, PR = two_sided(system, x_cut, l_cut)

    PL = csc_matrix(PL)[PL.getnnz(1) > 0]
    PR = csc_matrix(PR)[PR.getnnz(1) > 0]

    # Extract uncoupled sectors
    HL = PL @ hamiltonian @ PL.T
    HR = PR @ hamiltonian @ PR.T
    # Write uncoupled Hamiltonian
    H_partition = PR.T @ HR @ PR + PL.T @ HL @ PL
    vectors = np.array([])

    if solve is True:
        # Get eigenvectors and eigenvalues of each subspace
        evL, evecL = sort_eigen(la.eigh(HL))
        evR, evecR = sort_eigen(la.eigh(HR))

        # Project them onto the full Hamiltonian
        evecL = PL.T @ evecL
        evecR = PR.T @ evecR
        # Order them as desired
        vectors = np.hstack([evecL[:, :n_wire], evecR[:, :n_dot], evecL[:, n_wire:], evecR[:, n_dot:]])
        #vectors = np.hstack([evecL, evecR])
    return H_partition, vectors