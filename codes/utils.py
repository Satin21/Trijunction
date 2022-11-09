import sys, os
import numpy as np
from codes.constants import voltage_keys, scale, majorana_pair_indices
from scipy.linalg import svd
import scipy.sparse.linalg as sla
import kwant
import kwant.linalg.mumps as mumps
from scipy.sparse import identity
from collections.abc import Mapping

dirname = os.path.dirname(__file__)
sys.path.append(os.path.realpath(os.path.join(dirname, '../spin-qubit/')))
from utility import wannier_basis

# https://stackoverflow.com/a/3233356
def dict_update(d, u):
    """
    Update parent dictionary with many child branches inside
    
    d: dict
    Parent dictionary
    u: dict
    Child dictionary 
    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class LuInv(sla.LinearOperator):
    """Inverse of a matrix using LU decomposition"""
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
    """Projects the eigenstates on to the positions of the Kwant lattice which is chosen by the user.
    
    operator: kwant.operator.Density or ndarray
    eigenstates: ndarray
    """
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
    """
    Return indices to shuffle the Majorana wavefunction in Wannier basis based on the pair to be coupled.
    
    Pair: str
    For instance 'left-right', 'left-top' or 'right-top'
    
    """
    # shuffle the wavwfunctions based on the Majorana pairs to be optimized
    pair_indices = majorana_pair_indices[pair].copy()
    pair_indices.append(list(set(range(3)) - set(pair_indices))[0])
    shuffle = pair_indices + [-3, -2, -1]
    desired_order = np.array(list(range(2, 5)) + list(range(2)) + [5])[shuffle]
    return desired_order


def wannierize(tightbindingsystem, eigenstates):
    """
    Return the Majorana wavefunctions in the Wannier basis which are maximally localized orthogonal functions.
    
    Parameters:
    ----------
    tightbindingsystem: class instance
    Discretized Kwant system
    
    eigenstates: ndarray
    Minimum 6 wavefunctions which are particle and hole-symmetric MBS.
    """

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
    Return the index of nodes that has shortest euclidean distance to a node.
    
    Parameters:
    ----------
    node: 1x2 array
    nodes: nx2 array
    
    
    """
    nodes = np.asarray(nodes)
    dist = np.sum((nodes - node) ** 2, axis=1)
    closest = np.where(dist == dist.min())[0]
    # Previously we used np.argmin which returns the indices corresponding to the first occurrence in case of multiple occurrences of the minimum values. So we changed to np.where. Though we need to choose the extreme nodes no matter x < 0 or x > 0.
    if len(closest) == 2:
        nns = nodes[closest]
        if np.all(nns[:, 0] < 0):
            return closest[0]
        else:
            return closest[-1]
    return closest[0]


def dep_acc_index(
    gates_dict, centers_dict, kwant_sites, angle, a=10e-9, shift=2, spacing=2, npts=5
):
    """
    Returns the indices of the Kwant sites with one site under every gate and a list of sites along 
    each channel.
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
    shift: int
        No. of unit cells to shift away from the edges of the scattering region
    spacing: float
        How deep we go into the channels
    n_pts: int
        Number of points along each channel

    Returns
    -------
    dict with structure name as key and index on the kwant system as value
    """
    centroids = {}
    sides = ["left", "right", "top"]

    rotation = spacing * np.array(
        [[np.sin(angle), np.cos(angle)], [-np.sin(angle), np.cos(angle)], [0, -1]]
    )

    vector_shift = np.mgrid[0:3, shift : npts + shift, 0:2]

    vector_shift = a * vector_shift[1]

    for i, side in enumerate(sides):

        centroids[f"{side}"] = (
            a * centers_dict[f"{side}"] * np.ones((npts, 2))
            + vector_shift[i] * rotation[i]
        )

    pts = int(npts / 2 / 2)
    vector_shift = a * np.mgrid[0:3, -pts:pts, 0:2][1]
    rotation = spacing * np.array(
        [[np.sin(angle), np.cos(angle)], [-np.sin(angle), np.cos(angle)], [0, -1]]
    )
    pts = vector_shift[0].shape[0]

    # centroids of the gates
    shift_index = [0, 0, 1, 1, 2, 2]
    for i, (gate_name, gate_pos) in enumerate(gates_dict):
        x = gate_pos.T[0][
            :-1
        ]  # -1 because the first and last vertex in gate_vertices are the same.
        y = gate_pos.T[1][
            :-1
        ]  # and we avoid counting them twice by neglecting the last vertex.
        centroid = np.array([sum(x) / len(x), sum(y) / len(y)])
        centroids[gate_name] = a * centroid


    # get indexes in the kwant system
    for key, val in centroids.copy().items():
        if key in sides:
            centroids[f"{key}"] = [
                _closest_node(val[i], kwant_sites) for i in range(npts)
            ]
        else:
            centroids[f"{key}"] = [_closest_node(val, kwant_sites)]

    return centroids


def optimizer_status(
    x, max_count=50, filepath="/home/tinkerer/trijunction-design/data"
):
    """
    Checks whether the optimizer is stuck in the optimum for more than max_count, then return -1.


    """
    # It reads the json file containing the result of previous function evaluation in the optimization
    # step. Checks whether the value in the dictionary is higher than max_count and then return -1.

    with open(filepath, "rb") as outfile:
        data = json.load(outfile)

    x = str(x)
    key = list(data.keys())
    if len(key) == 0:
        # if empty dictionary add no matter what
        data[x] = 1
    elif x in data:
        # if x already exists in data, then increase it count by 1
        data[x] += 1
    elif key[0] < x:
        # if existing key in the dictionary is lower than the current x, replace it with x.
        del data[key[0]]
        data[x] = 1

    with open(filepath, "w") as outfile:
        json.dump(data, outfile)

    if list(data.values())[0] > max_count:
        return -1
    return 0
