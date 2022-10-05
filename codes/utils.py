import sys, os
import numpy as np
from codes.constants import voltage_keys, scale, majorana_pair_indices
from scipy.linalg import svd
import scipy.sparse.linalg as sla
from shapely.geometry.polygon import Polygon
import kwant
import kwant.linalg.mumps as mumps
from scipy.sparse import identity
import collections
from alphashape import alphashape
import matplotlib.pyplot as plt

sys.path.append(os.path.realpath("./../spin-qubit/"))
from utility import wannier_basis

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

def order_wavefunctions(pair):
    # shuffle the wavwfunctions based on the Majorana pairs to be optimized
    pair_indices = majorana_pair_indices[pair].copy()
    pair_indices.append(list(set(range(3)) - set(pair_indices))[0])
    shuffle = pair_indices + [-3, -2, -1]
    desired_order = np.array(list(range(2, 5)) + list(range(2)) + [5])[shuffle]
    return desired_order


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
    """
    SVD unitary tranformation of the coupled Hamiltonian
    in the Wannierized Majorana basis
    """
    S = wave_functions.T @ reference_wave_functions.T.conj()
    # Unitarize the overlap matrix
    U, _, Vh = svd(S)
    S = U @ Vh
    return S.T.conj() @ np.diag(energies) @ S


def _closest_node(node, nodes):
    """
    Euclidean distance between a node and array of nodes
    """
    nodes = np.asarray(nodes)
    dist = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist)


def dep_acc_index(
    poisson_system, site_indices: np.ndarray, nw_centers: dict, pair: str, plot_points=False
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
        accumulation[gate] = [closest_coord_index]

    for x in dep_regions:
        depletion[x] = dep_indices[x]
        
    if plot_points:
        site_coords = grid_points[site_indices]
        for index in np.hstack(list(accumulation.values())):
            point = site_coords[index]
            print(point)
            plt.scatter(point[0], point[1], c='b')

        for index in np.hstack(list(depletion.values())):
            point = site_coords[index]
            print(point)
            plt.scatter(point[0], point[1], c = 'r')

        for key, value in voltage_regions.items():
            if not key.startswith(('global', 'dirichlet')):
                coords = grid_points[value]
                plt.scatter(coords[:, 0], coords[:, 1], s = 0.1)
        

    return depletion, accumulation

