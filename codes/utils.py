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

sys.path.append(os.path.realpath("/home/tinkerer/spin-qubit/"))
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


def dep_acc_indexes(
    gates_dict, centers_dict, kwant_sites, angle, a=10e-9, shift=2
):
    """
    Parameters
    ----------
    gates_dict: dict
        Dictionary with gate names as key and vertices as values
    centers_dict: dict
        Dictionary with the position of the nanowires from `gate_coords`
    kwant_sites: nd.array
        Array containg positoons of kwant sites in proper order
    angle: float
        Angle of the trijunction arms
    a: float
        Lattice constant
    shift: float
        How deep we go into the channels

    Returns
    -------
    dict with structure name as key and index on the kwant system as value
    """
    centroids = {}
    # centroids of the gates
    for gate_name, gate_pos in gates_dict:
        x = gate_pos.T[0]
        y = gate_pos.T[1]
        centroid = np.array([sum(x) / len(x), sum(y) / len(y)])
        centroids[gate_name] = centroid * a

    # positions along the channels
    centroids["right"] = a * (
        centers_dict["left"] + shift * np.array([np.sin(angle), np.cos(angle)])
    )
    centroids["left"] = centers_dict["right"] * a + shift * a * np.array(
        [-np.sin(angle), np.cos(angle)]
    )
    centroids["top"] = centers_dict["top"] * a - shift * a * np.array([0, 1])

    # get indexes in the kwant system
    for key, val in centroids.items():
        centroids[key] = _closest_node(val, kwant_sites)

    return centroids
