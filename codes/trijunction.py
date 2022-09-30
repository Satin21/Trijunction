import kwant
import numpy as np
from codes.constants import scale, voltage_keys
from codes.tools import linear_Hamiltonian
from codes.utils import eigsh, wannierize
from codes.parameters import voltage_dict, junction_parameters
from codes.gate_design import gate_coords
from codes.finite_system import kwantsystem
from codes.discretize import discretize_heterostructure
import sys
import tinyarray as ta
from collections import OrderedDict

sys.path.append("/home/tinkerer/spin-qubit/")
from potential import gate_potential, linear_problem_instance
from Hamiltonian import discrete_system_coordinates


class Trijunction:
    """
    Class wrapping all objects associated to a trijunction
    """

    def __init__(self, config):
        """
        Initialisation requires only a `config` dictionary
        """
        self.scale = scale
        self.config = config

        (
            self.gates_vertex,
            self.gate_names,
            self.boundaries,
            self.nw_centers,
        ) = gate_coords(self.config)

    def make_system(self):
        self.initialize_kwant()
        self.initialize_poisson()
        self.create_base_matrices()
        self.generate_wannier_basis([-7.0e-3, -6.8e-3, -7.0e-3, 3e-3])

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
        # symmetry check fails, debug
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
            self.poisson_system,
            poisson_params,
            self.trijunction,
            self.f_params,
            voltage_regions,
        )

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

    def optimiser_arguments(self):
        """
        Organise the arguments for `optimization.loss` method
        """
        return OrderedDict(
            kwant_system=self.trijunction,
            kwant_params_fn=self.f_params,
            linear_terms=self.linear_terms,
            mlwf=self.mlwf,
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


"""
if len(change_config):
for local_config in change_config:
config = dict_update(config, local_config)
"""
