import kwant
import numpy as np
import sys
import tinyarray as ta
from collections import OrderedDict
from scipy.optimize import minimize_scalar, minimize

from codes.constants import scale, pairs
from codes.tools import linear_Hamiltonian
from codes.utils import eigsh, wannierize, dep_acc_index, order_wavefunctions
from codes.parameters import (
    voltage_dict,
    junction_parameters,
    pair_voltages,
    phase_pairs,
    bands,
)
from codes.gate_design import gate_coords
from codes.finite_system import kwantsystem
from codes.discretize import discretize_heterostructure
from codes.optimization import loss, soft_threshold_loss

sys.path.append("/home/tinkerer/spin-qubit/")
from potential import gate_potential, linear_problem_instance
from Hamiltonian import discrete_system_coordinates


class Trijunction:
    """
    Class wrapping all objects associated to a trijunction
    """

    def __init__(self, config, optimize_phase_pairs=["left-right"]):
        """
        Initialisation requires only a `config` dictionary.
        """
        self.scale = scale
        self.config = config

        (
            self.gates_vertex,
            self.gate_names,
            self.boundaries,
            self.nw_centers,
        ) = gate_coords(self.config)

        self.initialize_kwant()
        self.initialize_poisson()

        self.base_params = junction_parameters()
        self.base_params.update(potential=self.flat_potential())

        self.create_base_matrices()
        self.generate_wannier_basis([-7.0e-3, -6.8e-3, -7.0e-3, 3e-3])
        self.optimize_phase_pairs = optimize_phase_pairs

        self.dep_acc_indices()

        if len(optimize_phase_pairs):
            self.voltage_initial_conditions()
            self.optimal_phases()
            self.optimal_base_hams = {}
            for pair in self.optimize_phase_pairs:
                self.base_params.update(self.optimal_phases[pair])
                ham = self.trijunction.hamiltonian_submatrix(
                    sparse=True, params=self.f_params(**self.base_params)
                )
                self.optimal_base_hams[pair] = ham

    def initialize_kwant(self):
        """
        Create kwant system
        """

        self.geometry, self.trijunction, self.f_params = kwantsystem(
            self.config, self.boundaries, self.nw_centers, self.scale
        )

        self.densityoperator = kwant.operator.Density(self.trijunction, np.eye(4))

    def initialize_poisson(self):
        """
        Create poisson system
        """
        self.poisson_system = discretize_heterostructure(
            self.config, self.boundaries, self.gates_vertex, self.gate_names
        )

        self.linear_problem = linear_problem_instance(self.poisson_system)

        self.site_coords, self.site_indices = discrete_system_coordinates(
            self.poisson_system, [("mixed", "twoDEG")], boundaries=None
        )

        unique_indices = self.site_coords[:, 2] == 0
        self.site_coords = self.site_coords[unique_indices]
        self.site_indices = self.site_indices[unique_indices]

        self.grid_points = self.poisson_system.grid.points

        crds = self.site_coords[:, [0, 1]]
        grid_spacing = self.config["device"]["grid_spacing"]["twoDEG"]
        self.offset = crds[0] % grid_spacing
        self.check_symmetry([-7.0e-3, -7.0e-3, -7.0e-3, 3e-3])

    def create_base_matrices(self):
        """
        Create base hamiltonian and linearize voltage dependence
        """
        voltage_regions = self.poisson_system.regions.voltage.tag_points

        poisson_params = {
            "linear_problem": self.linear_problem,
            "site_coords": self.site_coords,
            "site_indices": self.site_indices,
            "offset": self.offset,
        }

        self.base_ham, self.linear_terms = linear_Hamiltonian(
            poisson_system=self.poisson_system,
            poisson_params=poisson_params,
            kwant_system=self.trijunction,
            kwant_params_fn=self.f_params,
            kwant_params=self.base_params,
            gates=voltage_regions,
        )

    def flat_potential(self, value=0):
        flat_potential = dict(
            zip(
                ta.array(self.site_coords[:, [0, 1]] - self.offset),
                np.ones(len(self.site_coords)) * value,
            )
        )
        return flat_potential

    def optimal_phases(
        self, voltages=(-3.0e-3, -3.0e-3, -3.0e-3, 3e-3), depleted=-7.0e-3
    ):
        self.optimal_phases = {}
        voltages = pair_voltages(initial=voltages, depleted=depleted)

        for pair in self.optimize_phase_pairs:
            opt_args = tuple(
                [
                    pair.split("-"),
                    (self.base_ham, self.linear_terms, self.densityoperator),
                    self.indices,
                ]
            )
            self.base_params.update(voltage_dict(self.initial_conditions[pair]))
            opt_args = tuple(
                [pair, self.base_params, list(self.optimiser_arguments(pair).values())]
            )

            phase_sol = minimize_scalar(
                loss, args=opt_args, bounds=(0, 2), method="bounded"
            )

            self.optimal_phases[pair] = phase_pairs(pair, np.pi * phase_sol.x)
        
    def voltage_initial_conditions(self):
        """
        Find initial condition for the voltages based on the soft-threshold.
        """
        self.initial_conditions = {}

        for pair in self.optimize_phase_pairs:
            opt_args = tuple(
                [
                    pair.split("-"),
                    (self.base_ham, self.linear_terms, self.densityoperator),
                    self.indices,
                ]
            )
            vol_sol = minimize(
                soft_threshold_loss,
                x0=(-1.5e-3, -1.5e-3, -1.5e-3, 3e-3),
                args=opt_args,
                method="trust-constr",
                options={
                    "initial_tr_radius": 1e-3,
                },
            )
            if vol_sol.success:
                self.initial_conditions[pair] = vol_sol.x

    def dep_acc_indices(self, indices=None):
        """
        Calculate the indexes of sites along the depletion and accumulation regions.
        """
        if not indices:
            self.indices = dep_acc_index(
                zip(self.gate_names, self.gates_vertex),
                self.nw_centers,
                [site.pos for site in self.trijunction.sites],
                self.config["gate"]["angle"],
                shift=3,
                spacing=3,
                npts=5,
            )
        else:
            self.indices = indices

    def potential(self, voltage_list, charges={}):
        """
        Wrap potential function to require only voltages
        """
        voltages = voltage_dict(voltage_list)
        return gate_potential(
            self.poisson_system,
            self.linear_problem,
            self.site_coords[:, [0, 1]],
            self.site_indices,
            voltages,
            charges,
            offset=self.offset,
        )

    def generate_wannier_basis(self, voltage_list):
        """
        Create basis of MLWFs corresponding to individual Majoranas
        """
        voltages = voltage_dict(voltage_list)

        summed_ham = sum(
            [
                voltages[key] * self.linear_terms[key]
                for key, value in self.linear_terms.items()
            ]
        )
        numerical_hamiltonian = self.base_ham + summed_ham

        eigval, eigvec = eigsh(
            numerical_hamiltonian, 6, sigma=0, return_eigenvectors=True
        )

        lowest_e_indices = np.argsort(np.abs(eigval))
        self.eigenstates = eigvec[:, lowest_e_indices].T

        # check that they are orthogonal
        assert np.allclose(
            self.eigenstates @ self.eigenstates.T.conj(), np.eye(len(self.eigenstates))
        )

        self.mlwf = wannierize(self.trijunction, self.eigenstates)

        # check orthogonality again
        assert np.allclose(self.mlwf @ self.mlwf.T.conj(), np.eye(len(self.mlwf)))

    def optimiser_arguments(self, pair):
        """
        Organise the arguments for `optimization.loss` method.

        Parameters
        ----------
        pair: str
            Determines the order of the mlwf
        """
        return OrderedDict(
            kwant_system=self.trijunction,
            linear_terms=self.linear_terms,
            kwant_params_fn=self.f_params,
            density_operator=self.densityoperator,
            mlwf=self.mlwf[order_wavefunctions(pair)],
        )

    def check_symmetry(self, voltages_list):
        """
        Check that the potential is symmetric in the kwant and poisson systems.
        """

        voltages = voltage_dict(voltages_list)
        unique_indices = self.site_coords[:, 2] == 0
        coords = self.site_coords[unique_indices]
        indices = self.site_indices[unique_indices]

        charges = {}
        pot = gate_potential(
            self.poisson_system,
            self.linear_problem,
            coords[:, [0, 1]],
            indices,
            voltages,
            charges,
            offset=self.offset[[0, 1]],
        )

        poisson_sites = np.array(list(pot.keys()))

        def diff_pot(x, y):
            return pot[ta.array((x, y))] - pot[ta.array((-x, y))]

        to_check = [diff_pot(*site) for site in poisson_sites]

        assert max(to_check) < 1e-9

        params = junction_parameters()
        params.update(potential=pot)

        f_mu = self.f_params(**params)["mu"]

        def diff_f_mu(x, y):
            return f_mu(x, y) - f_mu(-x, y)

        kwant_sites = np.array(list(site.pos for site in self.trijunction.sites))

        to_check = [diff_f_mu(*site) for site in kwant_sites]

        assert max(to_check) < 1e-9
