import numpy as np
import matplotlib.pyplot as plt
import json
import sys, os
import kwant
import scipy.sparse.linalg as sla
import dask.bag as db
from dask_quantumtinkerer import Cluster
from shapely.geometry.polygon import Polygon
from scipy.linalg import svd
from scipy.optimize import minimize
from discretize import discretize_heterostructure
from finite_system import finite_system
from tools import dict_update, find_resonances
from constants import length_unit
from solvers import sort_eigen
import parameters

import importlib
import tools

importlib.reload(tools)


# pre-defined functions from spin-qubit repository
sys.path.append("/home/tinkerer/spin-qubit/")
from potential import gate_potential, linear_problem_instance
from Hamiltonian import discrete_system_coordinates
from utility import wannier_basis
from tools import linear_Hamiltonian


global optimization_args
optimization_args = None

sys.path.append(os.path.realpath(sys.path[0] + "/.."))
from rootpath import ROOT_DIR


def _closest_node(node, nodes):
    """Euclidean distance between a node and array of nodes"""
    nodes = np.asarray(nodes)
    dist = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist)


def configuration(config, length_unit=1e-8, change_config=[]):

    if len(change_config):
        for local_config in change_config:
            config = dict_update(config, local_config)

    device_config = config["device"]
    gate_config = config["gate"]

    L = config["gate"]["L"]
    R = L / np.sqrt(2)

    # Boundaries within Poisson region
    xmax = R
    xmin = -xmax
    ymin = 0
    ymax = R + gate_config["L"] - gate_config["width"]

    boundaries = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}

    poisson_system = discretize_heterostructure(config, boundaries)
    linear_problem = linear_problem_instance(poisson_system)

    return config, boundaries, poisson_system, linear_problem


def kwantsystem(config, boundaries, length_unit=1e-8):

    L = config["gate"]["L"]
    R = L / np.sqrt(2)

    a = length_unit
    R_a = R * a
    width_a = config["gate"]["width"] * a
    l = config["kwant"]["nwl"] * a
    w = config["kwant"]["nww"] * a

    boundaries = list(boundaries.values())

    boundaries = np.array(boundaries) * a

    geometry = {
        "nw_l": l,
        "nw_w": w,
        "s_w": boundaries[1] - boundaries[0],
        "s_l": boundaries[3] - boundaries[2],
        "centers": [
            [-R_a + width_a / np.sqrt(2), 0],
            [-(-R_a + width_a / np.sqrt(2)), 0],
            [0, boundaries[3] + l - a],
        ],
    }

    ## Discretized Kwant system

    trijunction, f_params = finite_system(**geometry)
    trijunction = trijunction.finalized()

    trijunction = trijunction
    f_params = f_params

    return geometry, trijunction, f_params


class Optimize:
    def __init__(
        self,
        config,
        poisson_system,
        linear_problem,
        boundaries=[],
        length_unit=1e-8,
        arm=(
            "left_1",
            "left_2",
            "right_1",
            "right_2",
            "top_1",
            "top_2",
            "global_accumul",
        ),
        volt=(-1e-3, -1e-3, -1e-3, -1e-3, -1e-3, -1e-3, 4e-3),
    ):

        self.config = config
        self.length_unit = length_unit
        self.boundaries = boundaries

        self.poisson_system = poisson_system
        self.linear_problem = linear_problem

        self.voltages = dict(zip(arm, volt))
        for i in range(6):
            self.voltages["dirichlet_" + str(i)] = 0.0

        self.setconfig()

    def setconfig(self):

        device_config = self.config["device"]

        self.site_coords, self.site_indices = discrete_system_coordinates(
            self.poisson_system, [("charge", "twoDEG")], boundaries=None
        )

        self.grid_points = self.poisson_system.grid.points
        voltage_regions = self.poisson_system.regions.voltage.tag_points

        self.voltage_regions = {}
        for key, value in voltage_regions.items():
            if key.split("_")[0] not in ["dirichlet"]:
                self.voltage_regions[key] = value
        self.charge_regions = self.poisson_system.regions.charge.tag_points

        crds = self.site_coords[:, [0, 1]]
        grid_spacing = device_config["grid_spacing"]["twoDEG"]
        self.offset = crds[0] % grid_spacing

        self.geometry, self.trijunction, self.f_params = kwantsystem(
            self.config, self.boundaries, self.length_unit
        )
        self.densityoperator = kwant.operator.Density(self.trijunction, np.eye(4))

    def changeconfig(self, change_config):
        (
            self.config,
            self.boundaries,
            self.poisson_system,
            self.linear_problem,
        ) = configuration(
            self.config, length_unit=self.length_unit, change_config=change_config
        )
        self.setconfig()

        return self.config, self.boundaries, self.poisson_system, self.linear_problem

    def set_voltages(self, newvoltages):
        self.voltages.update(dict(zip(self.voltage_regions, newvoltages)))
        _, self.eigenstates = energyspectrum(
            self.base_hamiltonian, self.linear_ham, self.voltages
        )
        self.mlwf = _wannierize(self.trijunction, self.eigenstates)

    def optimize_gate(self, pair: str, initial_condition: list, optimal_phis=None):

        if optimal_phis is not None:
            self.optimal_phis = optimal_phis
        if hasattr(self, "optimal_phis"):
            args = self.params(pair, self.optimal_phis)
            sol1 = minimize(
                cost_function,
                initial_condition,
                args=args,
                # ftol = 1e-3,
                # verbose = 2,
                # max_nfev= 15
                # bounds = bounds,
                method="trust-constr",
                options={
                    "disp": True,
                    "verbose": 2,
                    "initial_tr_radius": 1e-3,
                    "gtol": 1e0,
                },
            )
            return sol1.x

        else:
            print(
                "Please calculate optimal phases for the nanowires before optimizing the gates"
            )

    def params(self, pair: str, optimal_phis=None):

        if optimal_phis is not None:
            self.optimal_phis = optimal_phis
        if hasattr(self, "optimal_phis"):
            self.pair = pair

            poisson_params = {
                "poisson_system": self.poisson_system,
                "linear_problem": self.linear_problem,
                "site_coords": self.site_coords,
                "site_indices": self.site_indices,
            }

            crds = self.site_coords
            grid_spacing = self.config["device"]["grid_spacing"]["twoDEG"]
            offset = crds[0] % grid_spacing

            kwant_params = {
                "offset": offset,
                "grid_spacing": self.length_unit,
                "finite_system_object": self.trijunction,
                "finite_system_params_object": self.f_params,
            }

            twodeg_grid = self.site_indices

            mu = parameters.bands[0]

            params = parameters.junction_parameters(m_nw=[mu, mu, mu], m_qd=0)

            voltage_regions = list(
                self.poisson_system.regions.voltage.tag_points.keys()
            )

            self.base_hamiltonian, self.linear_ham = tools.linear_Hamiltonian(
                poisson_params,
                kwant_params,
                params,
                voltage_regions,
                phis=self.optimal_phis[pair],
            )
            depletion, accumulation, uniform = self.dep_acc_regions(twodeg_grid)

            pair_indices = {
                "left-right": [0, 1],
                "left-top": [0, 2],
                "right-top": [1, 2],
            }
            coupled_pair = pair_indices[pair]
            uncoupled_pairs = [
                pair_indices[index] for index in set(pair_indices) - set([pair])
            ]

            _, self.eigenstates = energyspectrum(
                self.base_hamiltonian, self.linear_ham, self.voltages
            )
            self.mlwf = _wannierize(self.trijunction, self.eigenstates)

            optimize_args = (
                poisson_params,
                kwant_params,
                params,
                depletion,
                accumulation,
                uniform,
                coupled_pair,
                uncoupled_pairs,
                self.base_hamiltonian,
                self.linear_ham,
                self.mlwf,
            )

            return optimize_args

        else:
            print(
                "Please calculate optimal phases for the nanowires before optimizing the gates"
            )

    def dep_acc_regions(self, twodeg_grid):

        dep_indices = {}
        acc_indices = {}

        dep_regions = np.array(
            list(
                filter(
                    None,
                    [
                        x if x.split("_")[0] not in ["global", "dirichlet"] else None
                        for x in self.voltage_regions.keys()
                    ],
                )
            )
        )

        for gate in dep_regions:
            indices = self.voltage_regions[gate]
            center = Polygon(self.grid_points[indices][:, [0, 1]]).centroid.coords
            closest_coord_index = _closest_node(
                list(center)[0], self.grid_points[twodeg_grid][:, [0, 1]]
            )
            dep_indices[gate] = [closest_coord_index]

        depletion = np.array(
            [
                [dep_indices[pair[0]], dep_indices[pair[1]]]
                for pair in np.split(dep_regions, 3)
            ]
        )

        nw_centers = {}
        nw_centers["left"] = np.array(self.geometry["centers"][0]) / self.length_unit
        nw_centers["right"] = np.array(self.geometry["centers"][1]) / self.length_unit
        nw_centers["top"] = np.array(self.geometry["centers"][2])
        nw_centers["top"][1] -= self.geometry["nw_l"]
        nw_centers["top"][1] /= self.length_unit

        accumulation = []
        for gate in self.pair.split("-"):
            closest_coord_index = _closest_node(
                nw_centers[gate], self.grid_points[twodeg_grid][:, [0, 1]]
            )
            accumulation.append([[closest_coord_index]])

        uniform = []

        return depletion, accumulation, uniform

    def optimalphase(
        self,
        depleteV,
        acumulateV,
        closeV,
        Cluster,
        nnodes,
        cluster_options,
        cluster_dashboard_link,
    ):

        params = parameters.junction_parameters(m_nw=parameters.bands[0] * np.ones(3))

        potentials = []
        arms = ["left", "right", "top"]
        for arm in arms:
            voltages = _voltage_dict(depleteV, acumulateV, close=closeV, arm=arm)
            charges = {}
            potential = gate_potential(
                self.poisson_system,
                self.linear_problem,
                self.site_coords[:, [0, 1]],
                self.site_indices,
                voltages,
                charges,
                offset=self.offset[[0, 1]],
                grid_spacing=self.length_unit,
            )

            # potential.update((x, y*-1) for x, y in potential.items())
            potentials.append(potential)

        self.phases = np.linspace(0, 2, 100) * np.pi

        phis1 = [{"phi1": phi, "phi2": 0} for phi in self.phases]
        phis2 = [{"phi2": phi, "phi1": 0} for phi in self.phases]
        self.phis = [phis2, phis2, phis1]

        self.phase_results = []

        if Cluster is not None:
            with Cluster(cluster_options) as cluster:

                cluster.scale(n=nnodes)
                client = cluster.get_client()
                print(cluster_dashboard_link + cluster.dashboard_link[17:])

                i = 0

                for potential in potentials:
                    params.update(potential=potential)
                    solver = _fixed_potential_solver(
                        self.trijunction, self.f_params, params, eigenvecs=False
                    )
                    args_db = db.from_sequence(self.phis[i])
                    result = args_db.map(solver).compute()

                    i += 1

                    energies = []
                    for aux, _ in result:
                        energies.append(aux)
                    self.phase_results.append(energies)

            max_phis_id = []
            for pair in self.phase_results:
                max_phis_id.append(
                    find_resonances(energies=np.array(pair), n=20, sign=-1, i=-1)[1]
                )
            max_phis_id = np.array(max_phis_id).flatten()
            max_phis = self.phases[max_phis_id] / np.pi
            self.max_phis = max_phis

            self.optimal_phis = {}
            self.optimal_phis["left-right"] = [self.max_phis[2] * np.pi, 0]
            self.optimal_phis["left-center"] = [self.max_phis[1] * np.pi, 0]
            self.optimal_phis["center-right"] = [0, self.max_phis[0] * np.pi]

            return max_phis

        else:
            print(
                "Do you really want to run the simulations locally in a serial fashion or do you have access to a cluster to run the simulations remotely in parallel fashion ? "
            )

    def plot(self, to_plot="POTENTIAL"):

        if to_plot == "GATES":
            for name, indices in self.voltage_regions.items():
                grid_to_plot = self.grid_points[indices][:, [0, 1]]
                plt.scatter(grid_to_plot[:, 0], grid_to_plot[:, 1], s=0.5)

        if to_plot == "KWANT_SYSTEM":
            kwant.plot(self.trijunction, lead_site_size=4)
            # ax.set_ylim(-10*a, boundaries[3]+10*a)

        if to_plot == "POTENTIAL":
            charges = {}
            clean_potential = gate_potential(
                self.poisson_system,
                self.linear_problem,
                self.site_coords[:, [0, 1]],
                self.site_indices,
                self.voltages,
                charges,
                offset=self.offset,
                grid_spacing=self.length_unit,
            )

            coordinates = np.array(list(clean_potential.keys()))
            x = coordinates[:, 0]
            width_plot = np.unique(x).shape[0]
            Z = np.round(
                np.array(list(clean_potential.values())).reshape(width_plot, -1) * -1, 4
            )
            plt.figure()
            plt.imshow(
                np.rot90(Z),
                extent=np.array(list(self.boundaries.values())) * length_unit,
                cmap="magma_r",
            )
            plt.colorbar()

        if to_plot == "DECOUPLED":

            try:
                fig, ax = plt.subplots(
                    1, len(self.eigenstates), figsize=(5, 5), sharey=True
                )
                for i, state in enumerate(self.eigenstates):
                    cax = kwant.plotter.density(
                        self.trijunction, self.densityoperator(state), ax=ax[i]
                    )

            except AttributeError as error:
                print(
                    "Please calculate the tight binding Hamiltonian and separate in into linear and non-linear part before this step."
                )

        if to_plot == "PHASE_DIAGRAM":
            fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
            fig.tight_layout(w_pad=5)
            i = 0
            titles = ["left arm depleted", "right arm depleted", "top arm depleted"]
            phis_labels = [r"$\phi_{center}$", r"$\phi_{center}$", r"$\phi_{right}$"]
            peaks = []

            params = parameters.junction_parameters(
                m_nw=parameters.bands[0] * np.ones(3)
            )
            solver = _fixed_potential_solver(
                self.trijunction, self.f_params, params, eigenvecs=False
            )
            self.topo_gap = solver(self.phis[0][0])[0][-1]

            try:
                for energies in self.phase_results:
                    energies = np.array(energies)
                    for level in energies.T:
                        ax[i].plot(self.phases / np.pi, level / self.topo_gap)
                    ax[i].vlines(x=max_phis[i], ymin=-1, ymax=1)
                    ax[i].set_title(titles[i])
                    ax[i].set_ylabel(r"E[$\Delta^*]$")
                    ax[i].set_xlabel(phis_labels[i])
                    i += 1
            except AttributeError:
                print("Please calculate optimal phases first")

        if to_plot == "WANNIER_FUNCTIONS":
            fig, ax = plt.subplots(1, 6, figsize=(5, 5), sharey=True)
            for i in range(6):
                cax = kwant.plotter.density(
                    self.trijunction, self.densityoperator(self.mlwf[i]), ax=ax[i]
                )


def _wannierize(tightbindingsystem, eigenstates):

    X_operator = kwant.operator.Density(
        tightbindingsystem, onsite=lambda site: np.eye(4) * site.pos[0]
    )

    Y_operator = kwant.operator.Density(
        tightbindingsystem, onsite=lambda site: np.eye(4) * site.pos[1]
    )

    projected_X_operator = _projected_operator(X_operator, eigenstates.T)

    projected_Y_operator = _projected_operator(Y_operator, eigenstates.T)

    w_basis = wannier_basis([projected_X_operator, projected_Y_operator])

    mlwf = w_basis.T @ eigenstates

    return mlwf


def _projected_operator(operator, eigenstates):
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


def _voltage_dict(deplete, accumulate, close=0.0, arm="left"):

    voltages = {}
    if arm != "":
        for i in range(1, 3):
            voltages[arm + "_" + str(i)] = close

    for channel in set(["left", "right", "top"]) - set([arm]):
        for i in range(1, 3):
            voltages[channel + "_" + str(i)] = deplete

    voltages["global_accumul"] = accumulate

    for i in range(6):
        voltages["dirichlet_" + str(i)] = 0.0

    return voltages


def _fixed_potential_solver(kwant_syst, f_params, base_params, eigenvecs=False, n=20):
    def solver(extra_params):

        base_params.update(extra_params)
        ham_mat = kwant_syst.hamiltonian_submatrix(
            sparse=True, params=f_params(**base_params)
        )

        if eigenvecs:
            evals, evecs = sort_eigen(sla.eigsh(ham_mat.tocsc(), k=n, sigma=0))
        else:
            evals = np.sort(
                sla.eigsh(ham_mat.tocsc(), k=n, sigma=0, return_eigenvectors=eigenvecs)
            )
            evecs = []

        return evals, evecs

    return solver


def energyspectrum(base_ham, linear_ham, voltages):
    summed_ham = sum(
        [linear_ham[key] * voltages[key] for key, value in linear_ham.items()]
    )

    tight_binding_hamiltonian = base_ham + summed_ham

    eigval, eigvec = sort_eigen(
        sla.eigsh(tight_binding_hamiltonian.tocsc(), k=12, sigma=0)
    )

    lowest_e_indices = np.argsort(np.abs(eigval))[:6]
    eigenenergies = eigval[lowest_e_indices]
    eigenstates = eigvec.T[:, lowest_e_indices].T

    return eigenenergies, eigenstates


def cost_function(x, *argv):

    voltages = {}

    voltages["left_1"] = x[0]
    voltages["left_2"] = voltages["left_1"]
    voltages["right_1"] = x[1]
    voltages["right_2"] = voltages["right_1"]
    voltages["top_1"] = x[2]
    voltages["top_2"] = voltages["top_1"]
    voltages["global_accumul"] = x[3]

    for i in range(6):
        voltages["dirichlet_" + str(i)] = 0.0

    poisson_params, kwant_params, general_params = argv[:3]
    dep_points, acc_points, uniform = argv[3:6]
    coupled_pair, uncoupled_pairs = argv[6:8]
    base_hamiltonian, linear_ham = argv[8:10]
    mlwf = argv[10]

    pp = poisson_params
    kp = kwant_params

    charges = {}
    potential = gate_potential(
        pp["poisson_system"],
        pp["linear_problem"],
        pp["site_coords"],
        pp["site_indices"],
        voltages,
        charges,
        offset=kp["offset"],
        grid_spacing=kp["grid_spacing"],
    )

    potential.update((x, y * -1) for x, y in potential.items())

    potential_array = np.array(list(potential.values()))

    dep_acc_cost = []
    barrier_height = []

    for i, _ in enumerate(acc_points):
        dep_potential = potential_array[np.hstack(dep_points[i])]
        acc_potential = potential_array[acc_points[i]]

        if acc_potential > general_params["mus_nw"][0]:
            dep_acc_cost.append(np.abs(acc_potential - general_params["mus_nw"][0]))

        if np.any(dep_potential < acc_potential):
            dep_acc_cost.append(
                np.abs(
                    dep_potential[np.where(dep_potential < acc_potential)]
                    - acc_potential
                )
            )

        # barrier_height.append(sum(np.abs(dep_potential - acc_potential)))

    if len(dep_points) > len(acc_points):
        dep_potential = potential_array[np.hstack(dep_points[-1])]

        if np.any(dep_potential < general_params["mus_nw"][0]):
            dep_acc_cost.append(
                np.abs(
                    dep_potential[np.where(dep_potential < general_params["mus_nw"][0])]
                    - general_params["mus_nw"][0]
                )
            )

    coupling_cost = 0.0

    if len(dep_acc_cost):
        # print(dep_acc_cost)
        return sum(np.hstack(dep_acc_cost))
    else:
        uniformity = 0.0
        if len(uniform):
            uniformity = np.abs(
                potential_array[uniform[0]] - potential_array[uniform[1]]
            )

        energies, coupled_states = energyspectrum(
            base_hamiltonian, linear_ham, voltages
        )

        # Overlap matrix
        decoupled_states = mlwf
        S = coupled_states @ decoupled_states.T.conj()

        # Unitary matrix using SVD
        U, _, Vh = svd(S)
        S_prime = U @ Vh

        # Transform coupled Hamiltonian to Majorana basis
        coupled_ham = S_prime.T.conj() @ np.diag(energies) @ S_prime

        coupled_ham = coupled_ham[2:5, 2:5] / general_params["Delta"]

        print(np.abs(coupled_ham[coupled_pair[0], coupled_pair[1]]))

        coupled_cost = np.abs(coupled_ham[coupled_pair[0], coupled_pair[1]])

        uncoupled_cost = np.abs(
            coupled_ham[uncoupled_pairs[0][0], uncoupled_pairs[0][1]]
        ) + np.abs(coupled_ham[uncoupled_pairs[1][0], uncoupled_pairs[1][1]])

        return (-1 * coupled_cost) + uncoupled_cost + uniformity
