import kwant
import numpy as np
import sys
import tinyarray as ta
from collections import OrderedDict
from copy import copy
from scipy.optimize import minimize_scalar, minimize

from codes.constants import scale, pairs, default
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
from codes.optimization import loss, shape_loss, soft_threshold_loss
from codes.constants import rounding_limit

sys.path.append("/home/tinkerer/spin-qubit/")
from potential import gate_potential, linear_problem_instance
from Hamiltonian import discrete_system_coordinates


class Trijunction:
    """
    Class wrapping all objects associated to a trijunction
    """

    def __init__(self, config, optimize_phase_pairs=["left-right"], solve_poisson=True):
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

        if solve_poisson:
            self.initialize_poisson()

            self.base_params = junction_parameters()
            self.base_params.update(potential=self.flat_potential())
            self.create_base_matrices()
            self.generate_wannier_basis([-7.0e-3, -6.8e-3, -7.0e-3, 3e-3])
            self.optimize_phase_pairs = optimize_phase_pairs
            self.dep_acc_indices()

        if len(optimize_phase_pairs):
            self.voltage_initial_conditions()
            self.optimize_phases()
            self.optimal_base_hams = {}
            for pair in self.optimize_phase_pairs:
                self.base_params.update(self.optimal_phases[pair])
                ham = self.trijunction.hamiltonian_submatrix(
                    sparse=True, params=self.f_params(**self.base_params)
                )
                self.optimal_base_hams[pair] = ham

        self.compute_topological_gap()
        # This step cannot be done ahead of generating Wannier basis.
        # Otherwise, it results in assertion error for some reason when checking whether the
        # eigenstates are orthogonal. TODO.

    def initialize_kwant(self):
        """
        Create kwant system
        """

        self.geometry, self.trijunction, self.f_params = kwantsystem(
            self.config, self.boundaries, self.nw_centers, self.scale
        )

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
        """
        Dictionary with a flat potential in the scattering region.
        """
        scattering_sites = {}
        for site in self.trijunction.sites:
            x, y = site.pos
            if y >= 0 and y <= self.geometry["s_l"]:
                scattering_sites[
                    ta.array(np.round(site.pos / scale, rounding_limit))
                ] = value
        return scattering_sites

    def optimize_phases(self):
        """
        Find phase at which coupling is maximum for each pair.
        """
        self.optimal_phases = {}

        for pair in self.optimize_phase_pairs:

            self.base_params.update(voltage_dict(self.initial_conditions[pair]))
            opt_args = tuple(
                [pair, self.base_params, list(self.optimiser_arguments(pair).values())]
            )

            phase_sol = minimize_scalar(
                loss, args=opt_args, bounds=(0, 2), method="bounded"
            )

            self.optimal_phases[pair] = phase_pairs(pair, np.pi * phase_sol.x)

    def voltage_initial_conditions(
        self, guess=(-3e-3, -3e-3, -3e-3, 10e-3), ci=10, wf_amp=1
    ):
        """
        Find initial condition for the voltages based on the soft-threshold.
        """
        initial_conditions = {}
        for pair in self.optimize_phase_pairs:
            args = (
                pair.split("-"),
                (self.base_ham, self.linear_terms, (ci, wf_amp)),
                self.indices,
            )

            sol = minimize(
                fun=soft_threshold_loss,
                x0=guess,
                args=args,
                method="COBYLA",
                options={"rhobeg": 1e-2},
            )

            initial_conditions[pair] = sol.x
        self.initial_conditions = initial_conditions

    def dep_acc_indices(self, indices=None):
        """
        Calculate the indexes of sites along the depletion and accumulation regions.
        """
        if not indices:
            L = self.config["gate"]["L"]
            gap = self.config["gate"]["gap"]
            shift = 1
            spacing = (self.config["device"]["grid_spacing"]["twoDEG"] * scale) / 1e-9
            npts = np.rint(
                (L + gap + (spacing - gap % spacing) - (shift * spacing)) / spacing
            ).astype(int)
            self.indices = dep_acc_index(
                zip(self.gate_names, self.gates_vertex),
                self.nw_centers,
                [site.pos for site in self.trijunction.sites],
                self.config["gate"]["angle"],
                shift=shift,
                spacing=spacing,
                npts=npts,
            )
        else:
            self.indices = indices

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
            mlwf=self.mlwf[order_wavefunctions(pair)],
        )

    def compute_topological_gap(self):
        ## Check whether all the nanowires has equal width and topological gap
        ## If not, most probably the kwant system is built wrongly.
        base_params = junction_parameters()
        base_params.update(potential=self.flat_potential(2))
        ham = self.trijunction.hamiltonian_submatrix(
            sparse=True, params=self.f_params(**base_params)
        )
        # 12 eigenvalues, 6 for MBS, 6 for excited states
        evals = eigsh(ham, k=12, sigma=0)

        assert np.all(np.diff(evals[:3]) < 1e-9)
        assert np.all(np.diff(evals[3:6]) < 1e-9)
        self.topological_gap = evals[0]

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
