import numpy as np
import matplotlib.pyplot as plt
import json
import sys, os
import kwant
import dask.bag as db
from scipy.optimize import minimize, minimize_scalar, approx_fprime


# sys.path.append(os.path.realpath('./../spin-qubit/'))

sys.path.append("/home/srangaswamykup/trijunction_design/spin-qubit/")

from discretize import discretize_heterostructure
from gate_design import gate_coords

from finite_system import finite_system, kwantsystem
from constants import scale, majorana_pair_indices, voltage_keys, phase_pairs
import parameters
from collections import OrderedDict
import tinyarray as ta
from utils import (
    wannierize,
    svd_transformation,
    eigsh,
    dep_acc_regions,
    hamiltonian,
    linear_Hamiltonian,
    dict_update,
)

from potential import gate_potential, linear_problem_instance
from Hamiltonian import discrete_system_coordinates
from utility import gather_data


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

        ## Check whether the two sides of the device are symmetric around x = zero

        device_config = self.config["device"]

        site_coords, site_indices = discrete_system_coordinates(
            self.poisson_system, [("mixed", "twoDEG")], boundaries=None
        )

        unique_indices = site_coords[:, 2] == 0
        self.site_coords = site_coords[unique_indices]
        self.site_indices = site_indices[unique_indices]

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
            poisson_system=self.poisson_system,
            poisson_params=poisson_params,
            kwant_system=self.trijunction,
            kwant_params_fn=self.f_params,
            geometry=self.geometry,
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

        charges = {}
        pot = gate_potential(
            self.poisson_system,
            self.linear_problem,
            self.site_coords[:, [0, 1]],
            self.site_indices,
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

    def set_voltages(self, newvoltages):
        self.voltages.update(dict(zip(self.voltage_regions, newvoltages)))

    def dep_acc_voltages(self, pair, initial_condition):

        unique_indices = np.unique(
            self.site_coords[:, [0, 1]], axis=0, return_index=True
        )[1]
        depletion, accumulation = dep_acc_regions(
            self.poisson_system,
            self.site_indices[unique_indices],
            self.nw_centers,
            pair,
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


def check_grid(A, B):
    if A % B:
        return A % B
    return B


def configuration(config, change_config=[], poisson_system=[], boundaries=[]):

    if len(change_config):
        for local_config in change_config:
            config = dict_update(config, local_config)

    grid_spacing = config["device"]["grid_spacing"]
    thickness = config["device"]["thickness"]
    if not poisson_system:

        grid_spacing = check_grid(thickness["gates"], grid_spacing["gate"])

        gate_vertices, gate_names, boundaries, nw_centers = gate_coords(
            **config["gate"]
        )

        poisson_system = discretize_heterostructure(
            config, boundaries, gate_vertices, gate_names
        )

    linear_problem = linear_problem_instance(poisson_system)

    return config, boundaries, nw_centers, poisson_system, linear_problem


# -------------- Cost function for optimizing Majorana pair(s) coupling through gate tuning------------


def optimize_voltage(
    pairs: list,
    initial_conditions: list,
    optimal_phases: list,
    optimizer_args=None,
    scale=1e-3,
):

    optimal_voltages = {}
    for pair, initial, phase in zip(pairs, initial_conditions, optimal_phases):
        depletion, accumulation = dep_acc_regions(
            optimizer_args["poisson_system"],
            optimizer_args["poisson_params"]["site_indices"],
            optimizer_args["nw_centers"],
            pair,
        )
        optimizer_args["optimal_phase"] = phase
        optimizer_args["desired_pair"] = pair
        optimizer_args["depletion"] = depletion
        optimizer_args["accumulation"] = accumulation
        optimizer_args["energy_scale"] = scale

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
                "verbose": 2,
                "initial_tr_radius": 1e-3,
                # "gtol": 1e-1,
            },
        )

        optimal_voltages[pair] = sol1

    return optimal_voltages



def potential_shape(potential, dep_points, acc_points):


    potential_array = np.array(list(potential.values()))
    potential_keys = np.array(list(potential.keys()))
    
    chemical_potential = constants.bands[0]

    loss = []

    for gate, index in acc_points.items():

        channel = potential_array[tuple(acc_points[gate])]

        dgate = potential_array[
            sum([dep_points[gate + "_" + str(i)] for i in range(1, 3)], [])
        ]

        if np.any(dgate < chemical_potential):
            # Gated regions not depleted
            loss.append(_depletion_relative_potential(dgate, chemical_potential))

        if channel > chemical_potential:
            # Channel is depleted relative to nw
            loss.append(np.abs(channel - chemical_potential))

        if np.any(channel > dgate):
            # Channel is depleted relative to gated regions
            loss.append(_depletion_relative_potential(dgate, channel))

    for gate in set(["left", "right", "top"]) - set(acc_points.keys()):
        dgate = potential_array[
            sum([dep_points[gate + "_" + str(i)] for i in range(1, 3)], [])
        ]

        if np.any(dgate < chemical_potential):
            # Gated regions not depleted relative to nw
            loss.append(_depletion_relative_potential(dgate, chemical_potential))

        channel = potential_array[tuple(np.hstack(dep_points[gate]))]

        if np.any(channel < chemical_potential):
            # Channel is not depleted relative to nw
            loss.append(np.abs(chemical_potential - channel))

    if len(loss):
        return sum(np.hstack(loss))

    return 0


def _depletion_relative_potential(potential, reference):
    return np.abs(potential[np.where(potential < reference)] - reference)


def majorana_loss(numerical_hamiltonian, reference_wave_functions, scale):
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

    transformed_hamiltonian = svd_transformation(
        energies, wave_functions, reference_wave_functions
    )

    desired = np.abs(transformed_hamiltonian[0, 1])
    undesired = np.linalg.norm(transformed_hamiltonian[2:])

    return -desired + np.log(undesired / desired + 1e-3)


# --------- Cost function for optimizing phase between different nanowires-------

def loss(x, *argv):
    
    """
    x: either list or scalar (float)
        list when optimizing voltages and float when optimizing phases
    
    """
    
    ## 
    params = argv[0]
    kwant_system, kwant_params_fn = argv[2:5]
    linear_terms = argv[5]
    mlwf = argv[6]
    pair = argv[9]
    dep_points, acc_points = argv[10:12]
    energy_scale = argv[-1]
    
    if isinstance(x, list):
        # Unpack argv
        new_parameter = voltage_dict(x)
    elif isinstance(x, float):
        new_parameter = phase_pairs(pair, x * np.pi)
        
    ##TODO
    potential = return_potential()

    potential_loss = potential_shape_loss(
        potential, dep_points, acc_points
    )
    if potential_loss:
        return potential_loss + energy_scale

    params.update(new_parameter)

    numerical_hamiltonian = hamiltonian(
        kwant_system, linear_terms, kwant_params_fn, **params
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