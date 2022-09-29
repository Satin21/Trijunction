import sys, os
import numpy as np
from codes.constants import voltage_keys, scale
from scipy.linalg import svd
import scipy.sparse.linalg as sla
from shapely.geometry.polygon import Polygon
import kwant
import kwant.linalg.mumps as mumps
from scipy.sparse import identity

sys.path.append(os.path.realpath("./../spin-qubit/"))
from utility import wannier_basis


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


def dep_acc_regions(
    poisson_system, site_indices: np.ndarray, kwant_geometry: dict, pair: str
):
    """
    Return indices from the poisson system grid corresponding to regions that
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
        center = Polygon(grid_points[indices][:, [0, 1]]).centroid.coords
        closest_coord_index = _closest_node(
            list(center)[0], grid_points[twodeg_grid][:, [0, 1]]
        )
        dep_indices[gate] = [closest_coord_index]

    geometry = kwant_geometry

    nw_centers = {}
    nw_centers["left"] = np.array(geometry["centers"][0]) / scale
    nw_centers["right"] = np.array(geometry["centers"][1]) / scale
    nw_centers["top"] = np.array(geometry["centers"][2])
    nw_centers["top"][1] -= geometry["nw_l"]
    nw_centers["top"][1] /= scale

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

