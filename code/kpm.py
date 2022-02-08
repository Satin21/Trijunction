import sys
sys.path.append("/home/tinkerer/hybrid_kpm/")

import codes.higher_order_lowdin as effective

import numpy as np
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as sla
import kwant
import scipy.linalg as la
import sympy
from .solvers import bands, sort_eigen
from .tools import get_potential

mu = bands[0]

def two_sided(system, l_cut):
    # Projectors of each subspace
    PL = kwant.operator.Density(system,
                                where=lambda site: site.pos[1] < 0 or site.pos[1] >= l_cut,
                                sum=True).tocoo()
    PR = kwant.operator.Density(system,
                                where=lambda site: l_cut > site.pos[1] >= 0,
                                sum=True).tocoo()
    return PL, PR


def one_sided(system):
    # Projectors of each subspace
    PL = kwant.operator.Density(system,
                                where=lambda site: site.pos[1] < 0,
                                sum=True).tocoo()
    PR = kwant.operator.Density(system,
                                where=lambda site: site.pos[1] >= 0,
                                sum=True).tocoo()
    return PL, PR


def lowdin_partition(
    system, hamiltonian, sides, l_cut
):
    if sides == 1:
        PL, PR = one_sided(system)
    elif sides == 2:
        PL, PR = two_sided(system, l_cut)

    PL = csc_matrix(PL)[PL.getnnz(1) > 0]
    PR = csc_matrix(PR)[PR.getnnz(1) > 0]

    # Extract uncoupled sectors
    HL = PL @ hamiltonian @ PL.T
    HR = PR @ hamiltonian @ PR.T
    # Write uncoupled Hamiltonian
    H_partition = PR.T @ HR @ PR + PL.T @ HL @ PL
    
    return csc_matrix(H_partition), (HL, HR, PL, PR)

def majorana_dot_basis(system, f_params_potential, potential, params, n_dot, n_extra, sides, l_cut):
    orthogonal_evecL = []
    wires_isolated = np.array([[mu, -2, -2],[-2, mu, -2],[-2, -2, mu]])

    for w in wires_isolated:
        params.update(mus_nw=w)
        f_potential = f_params_potential(potential=get_potential(potential), params=params)
        hamiltonian = system.hamiltonian_submatrix(params=f_potential,
                                                   sparse=True)
        _, projectors = lowdin_partition(system, hamiltonian, sides, l_cut)
        HL, HR, PL, PR = projectors

        evL, evecL = sort_eigen(sla.eigsh(HL, sigma=0, k=1))
        orthogonal_evecL.append(evecL[0])

    k = 2*(n_dot + n_extra)
    evR, evecR = sort_eigen(sla.eigsh(HR, sigma=0, k=k))
    
    orthogonal_evecL = np.array(orthogonal_evecL)
    majorana_states = PL.T@orthogonal_evecL.T
    evecR = PR.T@evecR.T
    
    dot_states = evecR[:, k//2-n_dot:k//2+n_dot]
    extra_dot_states_1 = evecR[:, :k//2-n_dot]
    extra_dot_states_2 = evecR[:, k//2+n_dot:]
    
    A_basis = np.hstack([majorana_states, dot_states])
    B_basis = np.hstack([extra_dot_states_1, extra_dot_states_2])

    return A_basis, B_basis


def prepare_perturbation(
    system, unperturbed_hamiltonian, f_params_potential, potential, params,sides ,l_cut
):
    f_potential = f_params_potential(potential=get_potential(potential), params=params)
    perturbed_hamiltonian = system.hamiltonian_submatrix(params=f_potential, sparse=True)

    block_hamiltonian, _  = lowdin_partition(system, perturbed_hamiltonian, sides, l_cut)

    barrier_perturbation = unperturbed_hamiltonian - perturbed_hamiltonian
    hopping_perturbation = unperturbed_hamiltonian - block_hamiltonian

    return csc_matrix(barrier_perturbation), csc_matrix(hopping_perturbation)


def calculate_effective_models(H0, Hgate, Hcoup, A_basis, B_basis):

    def solver(index, order):
        eff = effective.effective_model(H0=H0, H1={'gate': Hgate[index], 'p': Hcoup[index]},
                                        evec_A=A_basis,
                                        evec_B=B_basis,                                     
                                        order=order,
                                        kpm_params=dict(num_moments=1000))

        model = kwant.continuum.lambdify(sympy.expand(eff.tosympy()))

        return model

    return solver