import numpy as np
from constants import voltage_keys, scale
from scipy.linalg import svd
import scipy.sparse.linalg as sla
from shapely.geometry.polygon import Polygon
from alphashape import alphashape
import kwant
import kwant.linalg.mumps as mumps
import sys, os
from scipy.sparse import identity
import collections.abc
import tinyarray as ta
import parameters
from tqdm import tqdm


# sys.path.append(os.path.realpath('./../spin-qubit/'))
from potential import gate_potential

sys.path.append("/home/srangaswamykup/trijunction_design/spin-qubit/")

from utility import wannier_basis


def voltage_dict(x, dirichlet=False):
    """Return dictionary of gate voltages
    x: list
    voltages

    dirichlet: bool
    Whether to add dirichlet gates
    """

    voltages = {key: x[index] for key, index in voltage_keys.items()}

    if dirichlet:
        for i in range(6):
            voltages["dirichlet_" + str(i)] = 0.0

    return voltages


# https://stackoverflow.com/a/3233356
def dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class LuInv(sla.LinearOperator):
    def __init__(self, A):
        inst = mumps.MUMPSContext()
        inst.analyze(A, ordering="pord")
        inst.factor(A)
        self.solve = inst.solve
        sla.LinearOperator.__init__(self, A.dtype, A.shape)

    def _matvec(self, x):
        return self.solve(x.astype(self.dtype))


def hamiltonian(
    kwant_system,
    linear_coefficients: dict,
    params_fn: callable,
    **params,
):
    summed_ham = sum(
        [
            linear_coefficients[key] * params[key]
            for key, value in linear_coefficients.items()
        ]
    )

    base_ham = kwant_system.hamiltonian_submatrix(
        sparse=True, params=params_fn(**params)
    )

    return base_ham + summed_ham


def linear_Hamiltonian(
    poisson_system, poisson_params, kwant_system, kwant_params_fn, gates
):

    ## non-linear part of the Hamiltonian

    voltages = {}

    for gate in gates:
        voltages[gate] = 0.0

    pp = poisson_params

    zero_potential = dict(
        zip(
            ta.array(pp["site_coords"][:, [0, 1]] - pp["offset"]),
            np.zeros(len(pp["site_coords"])),
        )
    )

    mu = parameters.bands[0]
    kwant_params = parameters.junction_parameters(m_nw=[mu, mu, mu])
    kwant_params.update(potential=zero_potential)
    #     general_params["phi1"] = phis[0]
    #     general_params["phi2"] = phis[1]

    base_ham = kwant_system.hamiltonian_submatrix(
        sparse=True, params=kwant_params_fn(**kwant_params)
    )

    hamiltonian_V = {}
    charges = {}
    for gate in tqdm(gates):

        voltages_t = dict.fromkeys(voltages, 0.0)

        voltages_t[gate] = 1.0

        potential = gate_potential(
            poisson_system,
            pp["linear_problem"],
            pp["site_coords"][:, [0, 1]],
            pp["site_indices"],
            voltages_t,
            charges,
            offset=pp["offset"][[0, 1]],
        )

        kwant_params.update(potential=potential)

        hamiltonian = kwant_system.hamiltonian_submatrix(
            sparse=True, params=kwant_params_fn(**kwant_params)
        )

        hamiltonian_V[gate] = hamiltonian - base_ham

    return base_ham, hamiltonian_V


def eigsh(
    A,
    k,
    sigma=0,
    return_eigenvectors=False,
    **kwargs,
):
    """Call sla.eigsh with mumps support and sorting.

    Please see scipy.sparse.linalg.eigsh for documentation.
    """

    opinv = LuInv(A - sigma * identity(A.shape[0]))
    out = sla.eigsh(
        A,
        k,
        sigma=sigma,
        OPinv=opinv,
        return_eigenvectors=return_eigenvectors,
        **kwargs,
    )

    if not return_eigenvectors:
        return np.sort(out)

    evals, evecs = out
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    return evals, evecs


def projected_operator(operator, eigenstates):
    if not isinstance(operator, np.ndarray):
        operator_diagonal = operator.tocoo().diagonal()
    else:
        operator_diagonal = operator

    projected_operator = np.array(
        [
            [
                sum(psi_1.conjugate() * operator_diagonal * psi_2)
                for psi_2 in eigenstates.T
            ]
            for psi_1 in eigenstates.T
        ]
    )

    return projected_operator


def wannierize(tightbindingsystem, eigenstates):

    X_operator = kwant.operator.Density(
        tightbindingsystem, onsite=lambda site: np.eye(4) * site.pos[0]
    )

    Y_operator = kwant.operator.Density(
        tightbindingsystem, onsite=lambda site: np.eye(4) * site.pos[1]
    )

    projected_X_operator = projected_operator(X_operator, eigenstates.T)

    projected_Y_operator = projected_operator(Y_operator, eigenstates.T)

    w_basis = wannier_basis([projected_X_operator, projected_Y_operator])

    mlwf = w_basis.T @ eigenstates

    return mlwf


def svd_transformation(energies, wave_functions, reference_wave_functions):
    """SVD unitary tranformation of the coupled Hamiltonian in the Wannierized Majorana basis"""
    S = wave_functions.T @ reference_wave_functions.T.conj()
    # Unitarize the overlap matrix
    U, _, Vh = svd(S)
    S = U @ Vh
    return S.T.conj() @ np.diag(energies) @ S


def _closest_node(node, nodes):
    """Euclidean distance between a node and array of nodes"""
    nodes = np.asarray(nodes)
    dist = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist)


def phase_spectrum(
    Cluster,
    nnodes,
    cluster_options,
    cluster_dashboard_link,
    voltages,
    no_eigenvalues,
    kwant_sys,
    kwant_params_fn,
    general_params,
    linear_terms,
    depleteV=[],
    acumulateV=[],
    closeV=[],
):

    potentials = []
    arms = ["left", "right", "top"]

    if not len(voltages):
        voltages = [
            _voltage_dict(depleteV, acumulateV, close=closeV, arm=arms[i])
            for i in range(3)
        ]
    elif not isinstance(depleteV, list):
        voltages = voltages

    phases = np.linspace(0, 2, 100) * np.pi
    phis1 = [{"phi1": phi, "phi2": 0} for phi in phases]
    phis2 = [{"phi2": phi, "phi1": 0} for phi in phases]
    phis = [phis2, phis2, phis1]

    phase_results = []

    with Cluster(cluster_options) as cluster:

        cluster.scale(n=nnodes)
        client = cluster.get_client()
        print(cluster_dashboard_link + cluster.dashboard_link[17:])

        for voltage, phi in zip(voltages, phis):
            args = (
                voltage,
                no_eigenvalues,
                kwant_sys,
                kwant_params_fn,
                general_params,
                linear_terms,
            )

            arg_db = db.from_sequence(phi)
            result = db.map(tune_phase, *args, phase=arg_db).compute()

            energies = []
            for energy in result:
                energies.append(energy)
            phase_results.append(energies)

    max_phis_id = []
    for pair in phase_results:
        max_phis_id.append(
            find_resonances(energies=np.array(pair), n=no_eigenvalues, sign=1, i=2)[1]
        )
    max_phis_id = np.array(max_phis_id).flatten()
    max_phis = phases[max_phis_id] / np.pi

    assert (
        np.abs(sum([1 - max_phis[0], 1 - max_phis[1]])) < 1e-9
    )  # check whether the max phases are symmetric for LC and RC pairs

    return max_phis, phase_results


##TODO: rename it to soft_threshold or piecewise_linear
def dep_acc_regions(
    poisson_system, site_indices: np.ndarray, nw_centers: dict, pair: str
):

    """Return indices from the poisson system grid corresponding to regions that
    needs to de depleted according to the desired majorana pair
    """

    dep_indices = {}
    acc_indices = {}

    voltage_regions = poisson_system.regions.voltage.tag_points
    grid_points = poisson_system.grid.points

    dep_regions = np.array(
        list(
            filter(
                None,
                [
                    x if x.split("_")[0] not in ["global", "dirichlet"] else None
                    for x in voltage_regions.keys()
                ],
            )
        )
    )

    twodeg_grid = site_indices

    for gate in dep_regions:
        indices = voltage_regions[gate]
        points = np.unique(grid_points[indices][:, [0, 1]], axis=0)
        center = list(alphashape(points, alpha=0.01).centroid.coords)[0]

        closest_coord_index = _closest_node(
            center,
            grid_points[twodeg_grid][
                :, [0, 1]
            ],  ## Orthogonal projection of a gate coordinate to 2DEG
        )
        dep_indices[gate] = [closest_coord_index]

    #         boundaries = np.array(alphashape(points, alpha = 0.1).exterior.coords)

    #         temp = []
    #         for point in boundaries:
    #             closest_coord_index = _closest_node(
    #                 point, grid_points[twodeg_grid][:, [0, 1]]  ## Orthogonal projection of a gate coordinate to 2DEG
    #             )
    #             temp.append(closest_coord_index)

    # dep_indices[gate] = temp.copy()

    depletion = {}
    for gate in set(["left", "right", "top"]) - set(pair.split("-")):
        closest_coord_index = _closest_node(
            nw_centers[gate], grid_points[twodeg_grid][:, [0, 1]]
        )
        depletion[gate] = [closest_coord_index]

    accumulation = {}
    for gate in pair.split("-"):
        closest_coord_index = _closest_node(
            nw_centers[gate], grid_points[twodeg_grid][:, [0, 1]]
        )
        accumulation[gate] = [[closest_coord_index]]

    for x in dep_regions:
        depletion[x] = dep_indices[x]

    return depletion, accumulation


def find_resonances(energies, n, i=1, sign=1, **kwargs):
    """
    Extract peaks from np.abs(lowest) mode in energies.
    By choosing 'sign' we extract either peaks or dips.
    Parameters:
    -----------
    """
    levels = energies.T
    ground_state = levels[n // 2 + i]
    peaks, properties = find_peaks(sign * np.abs(ground_state), **kwargs)

    return ground_state, peaks
