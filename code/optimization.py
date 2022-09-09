import numpy as np
import matplotlib.pyplot as plt
import json
import sys, os
import kwant
import dask.bag as db
from scipy.optimize import minimize, minimize_scalar, approx_fprime


# sys.path.append(os.path.realpath('./../spin-qubit/'))

sys.path.append("/home/tinkerer/trijunction_design/spin-qubit/")

from discretize import discretize_heterostructure
from gate_design import gate_coords

from finite_system import finite_system
from tools import dict_update, find_resonances
from constants import scale, majorana_pair_indices, voltage_keys, phase_pairs
import parameters
import tools
from collections import OrderedDict
import tinyarray as ta
from utils import wannierize, svd_transformation, eigsh

from potential import gate_potential, linear_problem_instance
from Hamiltonian import discrete_system_coordinates
from utility import gather_data
from tools import linear_Hamiltonian


class Optimize:
    def __init__(
        self,
        config,
        poisson_system,
        linear_problem,
        boundaries=[],
        scale=1e-8,
        arm=(
            "left_1",
            "left_2",
            "right_1",
            "right_2",
            "top_1",
            "top_2",
            "global_accumul",
        ),
        volt=(-0.0014, -0.0014, -0.0014, -0.0014, -0.0014, -0.0014, 3e-3),
    ):

        self.config = config
        self.scale = scale
        self.boundaries = boundaries

        self.poisson_system = poisson_system
        self.linear_problem = linear_problem

        self.voltages = dict(zip(arm, volt))
        for i in range(6):
            self.voltages["dirichlet_" + str(i)] = 0.0

        if self.poisson_system:
            self.set_params()

    def set_params(self):

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

        poisson_params = {
            "linear_problem": self.linear_problem,
            "site_coords": self.site_coords,
            "site_indices": self.site_indices,
            "offset": self.offset,
        }
        self.geometry, self.trijunction, self.f_params = kwantsystem(
            self.config, self.boundaries, self.nw_centers, self.scale
        )

        self.check_symmetry()

        voltage_regions = list(self.poisson_system.regions.voltage.tag_points.keys())

        print("Finding linear part of the tight-binding Hamiltonian")

        base_ham, linear_terms = linear_Hamiltonian(
            self.poisson_system,
            poisson_params,
            self.trijunction,
            self.f_params,
            voltage_regions,
        )

        self.set_voltages([-7.0e-3, -7.0e-3, -6.8e-3, -6.8e-3, -7.0e-3, -7.0e-3, 3e-3])

        summed_ham = sum(
            [
                self.voltages[key] * linear_terms[key]
                for key, value in linear_terms.items()
            ]
        )
        numerical_hamiltonian = base_ham + summed_ham

        eigval, eigvec = eigsh(
            numerical_hamiltonian, 6, sigma=0, return_eigenvectors=True
        )

        lowest_e_indices = np.argsort(np.abs(eigval))
        self.eigenstates = eigvec[:, lowest_e_indices].T

        self.set_voltages([-7.0e-3, -7.0e-3, -7.0e-3, -7.0e-3, -7.0e-3, -7.0e-3, 3e-3])

        summed_ham = sum(
            [
                self.voltages[key] * linear_terms[key]
                for key, value in linear_terms.items()
            ]
        )
        numerical_hamiltonian = base_ham + summed_ham

        eigval = np.sort(eigsh(numerical_hamiltonian, 20, sigma=0))

        self.topological_gap = eigval[-1]

        assert np.allclose(
            self.eigenstates @ self.eigenstates.T.conj(), np.eye(len(self.eigenstates))
        )

        self.mlwf = wannierize(self.trijunction, self.eigenstates)

        assert np.allclose(self.mlwf @ self.mlwf.T.conj(), np.eye(len(self.mlwf)))

        self.optimizer_args = OrderedDict(
            site_coords=self.site_coords,
            kwant_system=self.trijunction,
            kwant_params_fn=self.f_params,
            linear_terms=linear_terms,
            mlwf=self.mlwf,
        )

        self.densityoperator = kwant.operator.Density(self.trijunction, np.eye(4))

    def changeconfig(self, change_config, poisson_system=[]):
        (
            self.config,
            self.boundaries,
            self.nw_centers,
            self.poisson_system,
            self.linear_problem,
        ) = configuration(
            self.config, change_config=change_config, poisson_system=poisson_system
        )
        self.set_params()

        return self.config, self.boundaries, self.poisson_system, self.linear_problem

    def check_symmetry(self):
        """
        Check that the potential is symmetric in the kwant and poisson systems.
        """

        unique_indices = self.site_coords[:, 2] == 0
        coords = self.site_coords[unique_indices]
        indices = self.site_indices[unique_indices]

        charges = {}
        pot = gate_potential(
            self.poisson_system,
            self.linear_problem,
            coords[:, [0, 1]],
            indices,
            self.voltages,
            charges,
            offset=self.offset[[0, 1]],
        )

        poisson_sites = np.array(list(pot.keys()))

        def diff_pot(x, y):
            return pot[ta.array((x, y))] - pot[ta.array((-x, y))]

        to_check = [diff_pot(*site) for site in poisson_sites]

        assert max(to_check) < 1e-9

        mu = parameters.bands[0]
        params = parameters.junction_parameters(m_nw=[mu, mu, mu])
        params.update(potential=pot)

        f_mu = self.f_params(**params)["mu"]

        def diff_f_mu(x, y):
            return f_mu(x, y) - f_mu(-x, y)

        kwant_sites = np.array(list(site.pos for site in self.trijunction.sites))

        to_check = [diff_f_mu(*site) for site in kwant_sites]

        assert max(to_check) < 1e-9

    def set_voltages(self, newvoltages: np.ndarray):
        """
        Update voltages dictionary to `newvoltages`.
        """
        self.voltages.update(dict(zip(self.voltage_regions, newvoltages)))

    def dep_acc_voltages(self, pair, initial_condition):

        unique_indices = np.unique(
            self.site_coords[:, [0, 1]], axis=0, return_index=True
        )[1]
        depletion, accumulation = dep_acc_regions(
            self.poisson_system, self.site_indices[unique_indices], self.geometry, pair
        )
        self.optimizer_args["depletion"] = depletion
        self.optimizer_args["accumulation"] = accumulation

        args = tuple(
            self.poisson_system,
            self.optimizer_args["poisson"],
            parameters.bands[0],
            self.optimizer_args["dep_region"],
            self.optimizer_args["acc_region"],
        )

        sol1 = minimize(
            potential_shape_loss,
            initial_condition,
            args=args,
            method="trust-constr",
            options={
                "disp": True,
                # "verbose": 2,
                "initial_tr_radius": 1e-3,
                "gtol": 1e0,
            },
        )
        return sol1.x

    def plot(self, to_plot="POTENTIAL", optimal_phis=[], phase_results=[]):

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
                extent=np.array(list(self.boundaries.values())) * scale,
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

        if (
            to_plot == "PHASE_DIAGRAM"
            and hasattr(self, optimal_phis)
            and len(phase_results)
        ):

            fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
            fig.tight_layout(w_pad=5)
            i = 0
            titles = ["left arm depleted", "right arm depleted", "top arm depleted"]
            phis_labels = [r"$\phi_{center}$", r"$\phi_{center}$", r"$\phi_{right}$"]

            for energies in phase_results:
                energies = np.array(energies)
                for level in energies.T:
                    ax[i].plot(phases / np.pi, level / self.topo_gap)
                ax[i].vlines(x=max_phis[i], ymin=-1, ymax=1)
                ax[i].set_title(titles[i])
                ax[i].set_ylabel(r"E[$\Delta^*]$")
                ax[i].set_xlabel(phis_labels[i])
                i += 1

        if to_plot == "WANNIER_FUNCTIONS":
            fig, ax = plt.subplots(1, len(self.mlwf), figsize=(5, 5), sharey=True)
            for i, fn in enumerate(self.mlwf):
                cax = kwant.plotter.density(
                    self.trijunction, self.densityoperator(fn), ax=ax[i]
                )


def configuration(config, change_config=[], poisson_system=[], boundaries=[]):

    if len(change_config):
        for local_config in change_config:
            config = dict_update(config, local_config)

    grid_spacing = config["device"]["grid_spacing"]["gate"]
    if not poisson_system:
        gate_vertices, gate_names, boundaries, nw_centers = gate_coords(
            grid_spacing, **config["gate"]
        )
        poisson_system = discretize_heterostructure(
            config, boundaries, gate_vertices, gate_names
        )

    linear_problem = linear_problem_instance(poisson_system)

    return config, boundaries, nw_centers, poisson_system, linear_problem


def optimize_gate(
    pairs: list,
    initial_conditions: list,
    optimal_phases: list,
    optimizer_args=None,
    scale=1e-3,
):

    site_coords = optimizer_args["site_coords"]

    unique_indices = np.unique(site_coords[:, [0, 1]], axis=0, return_index=True)[1]
    optimal_voltages = {}
    for pair, initial, phase in zip(pairs, initial_conditions, optimal_phases):
        # depletion, accumulation = dep_acc_regions(self.poisson_system,
        #                                           self.site_indices[unique_indices],
        #                                           self.geometry,
        #                                           pair
        #                                          )
        optimizer_args["optimal_phase"] = phase
        optimizer_args["desired_pair"] = pair
        # self.optimizer_args['depletion'] = depletion
        # self.optimizer_args['accumulation'] = accumulation
        optimizer_args["energy_scale"] = scale

        print(f"Optimizing pair {pair}")

        # print(approx_fprime(initial,
        #               voltage_loss,
        # 1e-8,
        #               *list(self.optimizer_args.values())
        #              )
        #      )

        sol1 = minimize(
            voltage_loss,
            initial,
            args=tuple(list(optimizer_args.values())),
            method="trust-constr",
            options={
                # "verbose": 2,
                "initial_tr_radius": 1e-4,
                "gtol": 1e-1,
            },
        )

        optimal_voltages[pair] = sol1

    return optimal_voltages


def _voltage_dict(
    deplete,
    accumulate,
    close=0.0,
    arm="left",
):

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


def potential_shape_loss(x, *argv):

    if len(x) == 4:
        voltages = {key: x[index] for key, index in voltage_keys.items()}

    elif len(x) == 2:
        voltages = {}

        for arm in ["left", "right", "top"]:
            for i in range(2):
                voltages["arm_" + str(i)] = x[0]

        voltages["global_accumul"] = x[-1]

    for i in range(6):
        voltages["dirichlet_" + str(i)] = 0.0

    poisson_system, poisson_params, mus_nw, dep_points, acc_points = argv

    charges = {}
    potential = gate_potential(
        poisson_system,
        poisson_params["linear_problem"],
        poisson_params["site_coords"][:, [0, 1]],
        poisson_params["site_indices"],
        voltages,
        charges,
        offset=poisson_params["offset"],
    )

    # potential.update((x, y * -1) for x, y in potential.items())

    potential_array = np.array(list(potential.values()))
    potential_keys = np.array(list(potential.keys()))

    loss = []

    for gate, index in acc_points.items():

        channel = potential_array[tuple(acc_points[gate])]

        dgate = potential_array[
            sum([dep_points[gate + "_" + str(i)] for i in range(1, 3)], [])
        ]

        if np.any(dgate < mus_nw):
            # Gated regions not depleted
            loss.append(_depletion_relative_potential(dgate, mus_nw))

        if channel > mus_nw:
            # Channel is depleted relative to nw
            loss.append(np.abs(channel - mus_nw))

        if np.any(channel > dgate):
            # Channel is depleted relative to gated regions
            loss.append(_depletion_relative_potential(dgate, channel))

    for gate in set(["left", "right", "top"]) - set(acc_points.keys()):
        dgate = potential_array[
            sum([dep_points[gate + "_" + str(i)] for i in range(1, 3)], [])
        ]

        if np.any(dgate < mus_nw):
            # Gated regions not depleted relative to nw
            loss.append(_depletion_relative_potential(dgate, mus_nw))

        channel = potential_array[tuple(np.hstack(dep_points[gate]))]

        if np.any(channel < mus_nw):
            # Channel is not depleted relative to nw
            loss.append(np.abs(mus_nw - channel))

    if len(loss):
        return potential, sum(np.hstack(loss))

    return potential, 0


def _depletion_relative_potential(potential, reference):
    return np.abs(potential[np.where(potential < reference)] - reference)


def tune_phase(
    *argv,
    phase={"phi1": np.pi, "phi2": 0.0},
):

    voltage, n_eval, kwant_sys, kwant_params_fn, gparams, linear_terms = argv

    gparams.update(phase)

    params = {**gparams, **linear_terms}

    numerical_hamiltonian = hamiltonian(kwant_sys, voltage, kwant_params_fn, **params)

    return eigsh(numerical_hamiltonian, n_eval)


def phase_loss(phi, *argv):

    pair = argv[0]
    phase = phase_pairs(pair, phi * np.pi)

    energies = tune_phase(*argv[1:], phase=phase)

    no_eigenvalues = argv[2]
    first_excited_state_index = 2
    kwant_params = argv[-2]
    scale = kwant_params["Delta"]

    return (
        -1 * energies[no_eigenvalues // 2 + first_excited_state_index] / scale
    )  # energy of the first excited state


def voltage_loss(x, *argv):
    # Unpack argv
    voltages = {key: x[index] for key, index in voltage_keys.items()}

    # Boundary conditions on system sides.
    for i in range(6):
        voltages["dirichlet_" + str(i)] = 0.0

    site_coords, kwant_system, kwant_params_fn = argv[:3]
    linear_terms = argv[3]
    mlwf = argv[4]
    optimal_phase = argv[5]
    pair = argv[6]
    # dep_points, acc_points = argv[-3:-1]
    energy_scale = argv[-1]

    mu = parameters.bands[0]
    params = parameters.junction_parameters(m_nw=[mu, mu, mu])

    # potential, potential_loss = potential_shape_loss(
    #     x,
    #     poisson_system,
    #     poisson_params,
    #     mu,
    #     dep_points,
    #     acc_points
    # )
    # if potential_loss:
    #     return potential_loss

    potential = dict(zip(ta.array(site_coords[:, [0, 1]]), np.zeros(len(site_coords))))
    params.update(potential=potential)

    params.update(optimal_phase)

    kwant_params = {**params, **linear_terms}

    numerical_hamiltonian = hamiltonian(
        kwant_system, voltages, kwant_params_fn, **kwant_params
    )

    # shuffle the wavwfunctions based on the Majorana pairs to be optimized

    pair_indices = majorana_pair_indices[pair].copy()
    pair_indices.append(list(set(range(3)) - set(pair_indices))[0])
    shuffle = pair_indices + [-3, -2, -1]
    desired_order = np.array(list(range(2, 5)) + list(range(2)) + [5])[shuffle]

    reference_wave_functions = mlwf[desired_order]

    return majorana_loss(
        numerical_hamiltonian, reference_wave_functions, energy_scale, kwant_system
    )


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


def majorana_loss(numerical_hamiltonian, reference_wave_functions, scale, kwant_system):
    """Compute the quality of Majorana coupling in a Kwant system.

    Parameters
    ----------
    x : 1d array
        The vector of parameters to optimize
    numerical_hamiltonian : coo matrix
        A function for returning the sparse matrix Hamiltonian given parameters.
    reference_wave_functions : 2d array
        Majorana wave functions. The first two correspond to Majoranas that
        need to be coupled.
    scale : float
        Energy scale to use.
    """

    energies, wave_functions = eigsh(
        numerical_hamiltonian.tocsc(),
        len(reference_wave_functions),
        sigma=0,
        return_eigenvectors=True,
    )

    #     fig, ax = plt.subplots(1, len(wave_functions.T), figsize = (10, 5), sharey= True)

    #     density = kwant.operator.Density(kwant_system, np.eye(4))
    #     for i, vec in enumerate(wave_functions.T):
    #         kwant.plotter.density(kwant_system, density(vec), ax = ax[i]);

    #     filepath = '/home/tinkerer/trijunction-design/data/optimization/'
    #     seed = gather_data(filepath)

    #     if ".ipynb_checkpoints" in seed:
    #         os.system("rm -rf .ipynb_checkpoints")

    #     if len(seed):
    #         file_name = filepath + "plt_" + str(max(seed) + 1) + "_.png"
    #     else:
    #         file_name = filepath + "plt_" + str(0) + "_.png"

    #     plt.savefig(file_name, format="png", bbox_inches="tight", pad_inches=0.0)

    #     plt.close()

    transformed_hamiltonian = (
        svd_transformation(energies, wave_functions, reference_wave_functions) / scale
    )

    desired = np.abs(transformed_hamiltonian[0, 1])
    undesired = np.linalg.norm(transformed_hamiltonian[2:])

    # print(desired , undesired)

    return -desired + np.log(undesired / desired + 1e-3)


def optimize_phase_fn(voltages, pairs, kwant_params, no_eigenvalues = 10):
    optimal_phases = {}
    for voltage, pair in zip(voltages, pairs):
        args = [pair, voltage, no_eigenvalues]
        args = args + list(kwant_params.values())

        sol = minimize_scalar(phase_loss, args=tuple(args), bounds=(0, 2), 
                              method='bounded')

        optimal_phases[pair] = phase_pairs(pair, sol.x * np.pi)

    return optimal_phases


def optimize_gate_fn(
    pairs, initial_condition, optimal_phases, optimizer_args, energy_scale
):
    return optimize_gate(
        pairs,
        list(initial_condition.values()),
        list(optimal_phases.values()),
        optimizer_args=optimizer_args,
        scale=energy_scale,
    )
