import kwant
import kwant.continuum
import tinyarray as ta
import numpy as np
import scipy.sparse.linalg as sla




def get_potential(potential):
    def f(x, y):
        return potential[ta.array([x, y])]
    return f


def solver(geometries, n, key, eigenvecs=False):
    """
    Return a function that diagonalizes the Hamiltonian for different geometries.
    The parameters for the Hamiltonian are set from the beginning, and only one
    parameter is varied via key and mu.

    Paramters:
    ----------
        geometries: array where each element is (kwant.Builder, paramters_function(x, y))
        n: number of eigenvalues to be extracted
        key: single parameter to be varied
        eigenvecs: bool telling if we extract or not eigenvectors

    Returns:
    --------
        eigensystem_sla: function that returns eigenvalues and eigenvectors per geometry per mu
    """
    def eigensystem_sla(geometry_index, mu, extra_params):

        system, params_func = geometries[geometry_index]

        params = junction_parameters(m_nw=np.array([-2, -2, -2]), m_qd=0)
        params.update(extra_params)
        params[key] = mu

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

    Parameters:
    -----------
        tj_system: pair (kwant.Builder, parameters_function(x,y))
        n: number of eigenvalues to be extracted
        potentials: list of dictionaries containing the potential configuration for each voltage

    Returns:
    --------
        
    """

    def eigensystem_sla(potential_index, voltage, pair):
        """
        Paramters:
        ----------
            potential_index: int that tells what element to extract from potentials
            voltage: voltage associated 
        """

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


def general_solver(geometries, n, base_parameters, eigenvecs=False):
    """
    Return a function that diagonalizes the Hamiltonian for different geometries.
    The parameters for the Hamiltonian are varied in a list of dictionaries where
    multiple parameters can be varied simultaneously via extra_params.

    Parameters:
    -----------
        geometries: array where each element is (kwant.Builder, paramters_function(x, y))
        n: number of eigenvalues to be extracted
        base_paramters: set of paramters that won't be changed during simulations
        eigenvecs: bool telling if we extract or not eigenvectors

    Returns:
    --------
        solver: solver function that computes eigenvalues and eigenvectors
    """
    def solver(index, extra_params):
        """
        Parameters:
        -----------
            index: geometry index
            extra_params: parameters to be varied in a given geometry
        Returns:
        --------
            evals: eigenvalues
            evecs: eigenvectors if requested, empty array otherwise
        """
        system, f_params = geometries[index]

        base_parameters.update(extra_params)
        ham_mat = system.hamiltonian_submatrix(sparse=True, params=f_params(**base_parameters))

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