import kwant
import kwant.continuum
import tinyarray as ta
import numpy as np
import scipy.sparse.linalg as sla
from scipy.constants import electron_mass, hbar

# data extracted from sub_bands.ipynb
# Zeeman fields for the topological transition at each sub-band


# Bottom of each transverse band
bands = [
    0.0023960204649275973,
    0.009605416498312178,
    0.020395040147213304,
    0.03312226901926766,
    0.045849497891322026,
    0.056639121540223145,
    0.06384851757360771
]

b = 0.001

phi12 = 1.23232323*np.pi
phi13 = 0.02020202*np.pi
phi23 = 1.97979798*np.pi
phis = np.array([1.23232323, 0.02020202, 1.97979798])*np.pi

def finite_coupling_parameters(index, sigma=0):
    mu = bands[index]
    params_12 = {'mus_nw': np.array([mu, mu, -2]), 'phi1': phi12, 'sigma': sigma}
    params_13 = {'mus_nw': np.array([mu, -2, mu]), 'phi2': phi13, 'sigma': sigma}
    params_23 = {'mus_nw': np.array([-2, mu, mu]), 'phi2': phi23, 'sigma': sigma}
    params = [params_12, params_13, params_23]
    return params


def phase(pair):

    if pair == 0:
        key_phi = 'phi1'
    else:
        key_phi = 'phi2'

    extra_params = {key_phi: phis[pair]}
    return extra_params


def phase_params(key, param, band_index=0, n=100):
    wires = finite_coupling_parameters(band_index)
    phases = np.linspace(0, 2*np.pi, n)
    params = []
    for phase in phases:
        i = 0
        for wire in wires:
            if i < 1:
                updated_params = {key: param, 'phi1': phase, 'phi2': 0}
            else:
                updated_params = {key: param, 'phi2': phase, 'phi1': 0}
            params.append(wire | updated_params)
            i += 1
    return params


# helper functions
def lead_parameters(m_nw, m_qd, B):
    a = 10E-9
    t = hbar**2/(2*0.023*electron_mass)*(6.24E18)
    alpha = 0.4E-10
    Delta = 5E-3
    parameters = dict(mu_nw_1=m_nw[0],
                      mu_nw_2=m_nw[1],
                      mu_nw_3=m_nw[2],
                      mu_qd=m_qd,
                      t=t,
                      Delta=Delta,
                      alpha=alpha,
                      B_x=B,
                      phi=0,
                      a=a)
    return parameters


def junction_parameters(m_nw, m_qd, bx=b):
    """
    Typical parameters
    """

    a = 10E-9
    t = hbar**2/(2*0.023*electron_mass)*(6.24E18)
    alpha = 0.3E-10
    Delta = 5E-4
    parameters = dict(mus_nw=m_nw,
                      mu_qd=m_qd,
                      t=t,
                      Delta=Delta,
                      alpha=alpha,
                      B_x=bx,
                      phi1=0,
                      phi2=0,
                      sigma=0,
                      a=a)
    return parameters


def get_potential(potential):
    def f(x, y):
        return potential[ta.array([x, y])]
    return f


def solver(geometries, n, key, eigenvecs=False):
    """
    Return a function that diagonalizes the Hamiltonian for different geometries.
    The parameters for the Hamiltonian are set from the beginning, and only one
    parameter is varied via key and mu.
    """
    def eigensystem_sla(geometry_index, mu, extra_params):

        params = junction_parameters(m_nw=np.array([-2, -2, -2]), m_qd=0)
        params.update(extra_params)
        params[key] = mu

        system, params_func = geometries[geometry_index]

        ham_mat = system.hamiltonian_submatrix(sparse=True, params=params_func(**params))

        if not eigenvecs:
            evals = np.sort(sla.eigsh(ham_mat.tocsc(), k=n, sigma=0, return_eigenvectors=eigenvecs))
            evecs = []
        else:
            evals, evecs = sort_eigen(sla.eigsh(ham_mat.tocsc(), k=n, sigma=0))

        return evals, evecs

    return eigensystem_sla


def solver_potential(tj_system, n, potentials, eigenvecs=False, band=0):
    """
    Return a function that diagonalizes the Hamiltonian for a single geometry.
    The potential is defined as a list of dictionaries, each contains the
    potential for every site in the system.
    """

    def eigensystem_sla(potential_index, voltage, pair):

        mu = bands[band]
        params = junction_parameters(m_nw=np.array([mu, mu, mu]), m_qd=0)
        params.update(phase(pair))
        system, f_params_potential = tj_system

        potential = potentials[potential_index]
        f_potential = get_potential(potential[voltage])

        ham_mat = system.hamiltonian_submatrix(sparse=True,
                                               params=f_params_potential(potential=f_potential, params=params))
        if eigenvecs:
            evals, evecs = sort_eigen(sla.eigsh(ham_mat.tocsc(), k=n, sigma=0))
        else:
            evals = np.sort(sla.eigsh(ham_mat.tocsc(), k=n, sigma=0, return_eigenvectors=eigenvecs))
            evecs = []

        return evals, evecs

    return eigensystem_sla


def general_solver(geometries, n, eigenvecs=False):
    """
    Return a function that diagonalizes the Hamiltonian for different geometries.
    The parameters for the Hamiltonian are varied in a list of dictionaries where
    multiple parameters can be varied simultaneously via extra_params.
    """
    def solver(index, extra_params):
        system, f_params = geometries[index]

        params = junction_parameters([-2, -2, -2], 2.5e-3)
        params.update(extra_params)
        ham_mat = system.hamiltonian_submatrix(sparse=True, params=f_params(**params))

        if not eigenvecs:
            evals = np.sort(sla.eigsh(ham_mat.tocsc(), k=n, sigma=0, return_eigenvectors=eigenvecs))
            evecs = []
        else:
            evals, evecs = sort_eigen(sla.eigsh(ham_mat.tocsc(), k=n, sigma=0))

        return evals, evecs
    return solver


def sort_eigen(ev):
    """
    Sort eigenvectors and eigenvalues.
    """
    evals, evecs = ev
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    return evals, evecs.T