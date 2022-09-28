import kwant
import numpy as np
from constants import scale
from parameters import voltage_dict
from gate_design import gate_coords
from finite_system import kwantsystem
from discretize import discretize_heterostructure
import sys
sys.path.append("/home/tinkerer/spin-qubit/")
from potential import gate_potential, linear_problem_instance
from Hamiltonian import discrete_system_coordinates
 
    
class Trijunction:
    
    def __init__(self, config):
        self.scale = scale
        self.config = config
        self.gates_vertex, self.gate_names, self.boundaries, self.nw_centers = gate_coords(
            self.config
        )
    
    def make_system(self):
        print('kwant')
        self.initialize_kwant()
        print('poisson')
        self.initialize_poisson()
        self.create_base_matrices()
        self.generate_wannier_basis()
            
    def initialize_kwant(self):

        self.geometry, self.trijunction, self.f_params = kwantsystem(
            self.config, self.boundaries, self.nw_centers, self.scale
        )
        
        self.densityoperator = kwant.operator.Density(self.trijunction, np.eye(4))
        
    def initialize_poisson(self):
        
        self.poisson_system = discretize_heterostructure(
            self.config, self.boundaries, self.gates_vertex, self.gate_names
        )

        self.linear_problem = linear_problem_instance(self.poisson_system)

        self.site_coords, self.site_indices = discrete_system_coordinates(
            self.poisson_system, [("charge", "twoDEG")], boundaries=None
        )

        self.grid_points = self.poisson_system.grid.points
        
        crds = self.site_coords[:, [0, 1]]
        grid_spacing = self.config["device"]["grid_spacing"]["twoDEG"]
        self.offset = crds[0] % grid_spacing

        self.check_symmetry()

    def create_base_matrices(self):
        
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

    def generate_wannier_basis(self):
        
        voltages = voltage_dict([-7.0e-3, -6.8e-3, -7.0e-3, 3e-3])

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

        return OrderedDict(
            site_coords=self.site_coords,
            kwant_system=self.trijunction,
            kwant_params_fn=self.f_params,
            linear_terms=linear_terms,
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

    
    def get_potential(self, voltages):
        potential = gate_potential(
            poisson_system,
            poisson_params["linear_problem"],
            poisson_params["site_coords"][:, [0, 1]],
            poisson_params["site_indices"],
            voltages,
            charges,
            offset=poisson_params["offset"],
        )


    def plot(self, voltages, to_plot="POTENTIAL", optimal_phis=[], phase_results=[]):

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
                voltages,
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

"""
if len(change_config):
for local_config in change_config:
config = dict_update(config, local_config)
"""