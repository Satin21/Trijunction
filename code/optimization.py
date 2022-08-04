import numpy as np
import matplotlib.pyplot as plt
import json
import sys, os
import kwant
import scipy.sparse.linalg as sla
import dask.bag as db
from shapely.geometry.polygon import Polygon
from scipy.linalg import svd
from scipy.optimize import minimize
from discretize import discretize_heterostructure
from finite_system import finite_system
from tools import dict_update, find_resonances
from constants import scale, majorana_pair_indices, voltage_keys
from solvers import sort_eigen
import parameters
import tools
from mumps_sparse_diag import sparse_diag

# ROOT_DIR = '/home/srangaswamykup/trijunction_design'
# 
sys.path.append(os.path.realpath(sys.path[0] + '/..'))
from rootpath import ROOT_DIR

# pre-defined functions from spin-qubit repository
sys.path.append(os.path.join(ROOT_DIR + '/spin-qubit/'))
from potential import gate_potential, linear_problem_instance
from Hamiltonian import discrete_system_coordinates
from utility import wannier_basis
from tools import linear_Hamiltonian





def _closest_node(node, nodes):
    """Euclidean distance between a node and array of nodes"""
    nodes = np.asarray(nodes)
    dist = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist)


def configuration(config, change_config=[], poisson_system = []):

    if len(change_config):
        for local_config in change_config:
            config = dict_update(config, local_config)

    device_config = config["device"]
    gate_config = config["gate"]

    L = config["gate"]["L"]
    R = np.round(L / np.sqrt(2))

    # Boundaries of Poisson region
    xmax = R
    xmin = -xmax
    ymin = 0
    ymax = R + gate_config["L"] - gate_config["width"]

    boundaries = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
    
    if not poisson_system:
        poisson_system = discretize_heterostructure(config, boundaries)
        
    linear_problem = linear_problem_instance(poisson_system)

    return config, boundaries, poisson_system, linear_problem


def kwantsystem(config, boundaries, scale=1e-8):

    L = config["gate"]["L"]
    R = np.round(L / np.sqrt(2))

    a = scale
    width = config["gate"]["width"]
    l = config["kwant"]["nwl"]
    w = config["kwant"]["nww"]

    boundaries = np.array(list(boundaries.values()))

    geometry = {
        "nw_l": l * a,
        "nw_w": w * a,
        "s_w": (boundaries[1] - boundaries[0]) * a,
        "s_l": (boundaries[3] - boundaries[2]) * a,
        "centers": [
            [np.round(-R + width / np.sqrt(2)) * a, 0],
            [np.round(-(-R + width / np.sqrt(2))) * a, 0],
            [0, ((boundaries[3] + l) * a) - a],
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
            self.config, self.boundaries, self.scale
        )
        
        ## Check whether the two sides of the device are symmetric around x = zero
        self.check_symmetry()
        
        self.densityoperator = kwant.operator.Density(self.trijunction, np.eye(4))

    def changeconfig(self, change_config, poisson_system = []):
        (
            self.config,
            self.boundaries,
            self.poisson_system,
            self.linear_problem,
        ) = configuration(
            self.config, change_config=change_config, poisson_system = poisson_system
        )
        self.setconfig()

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
                offset = self.offset[[0, 1]],
                scale = 1
            )
        pot.update((x, y*-1) for x, y in pot.items())
        
        mu = parameters.bands[0]
        params = parameters.junction_parameters(m_nw=[mu, mu, mu], m_qd=0)
        params.update(potential=pot)

        f_mu = self.f_params(**params)['mu']

        def diff_f_mu(x, y):
            return f_mu(x, y) - f_mu(-x, y)

        kwant_sites = np.array(list(site.pos for site in self.trijunction.sites))

        assert max([diff_f_mu(*site) for site in kwant_sites]) < 1e-9

    def set_voltages(self, newvoltages):
        self.voltages.update(dict(zip(self.voltage_regions, newvoltages)))
#         if find_mlwf:
#             eigval, eigvec = diagonalize(
#                 hamiltonian,
#                 {self.base_ham, self.linear_ham}, 
#                 self.voltages,
#                 no_eigenvalues = 6 
#             )
#             lowest_e_indices = np.argsort(np.abs(eigval))
#             self.eigenstates = eigvec.T[:, lowest_e_indices].T
            
#             self.mlwf = _wannierize(self.trijunction, self.eigenstates)

    def optimize_gate(self, pairs: list, initial_condition: list, optimal_phis=None):

        if optimal_phis is not None:
            self.optimal_phis = optimal_phis
        if hasattr(self, "optimal_phis"):
            optimal_voltages = {}
            for initial, pair in zip(initial_condition, pairs):
                args = list(self.params(pair, self.optimal_phis).values())
                print(f'Optimizing pair {pair}')
                sol1 = minimize(
                    cost_function,
                    initial,
                    args=tuple(args),
                    # ftol = 1e-3,
                    # verbose = 2,
                    # max_nfev= 15
                    # bounds = bounds,
                    method="trust-constr",
                    options={
                        "disp": True,
                        # "verbose": 2,
                        "initial_tr_radius": 1e-3,
                        "gtol": 1e0,
                    },
                )
                
                optimal_voltages[pair] = sol1
            
            return optimal_voltages

        else:
            print(
                "Please calculate optimal phases for the nanowires before optimizing the gates"
            )
            
    def dep_acc_voltages(self, pair, initial_condition):
        
        self.optimize_args['dep_region'], self.optimize_args['acc_region'] = dep_acc_regions(self.poisson_system, 
                                                                       self.site_indices, 
                                                                       self.geometry, 
                                                                       pair)

        args = tuple(self.optimize_args['poisson'],
                     self.optimize_args['general_param']['mus_nw'],
                     self.optimize_args['dep_region'],
                     self.optimize_args['acc_region']
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

    def params(self, pair: str, optimal_phis=None, voltages = None):
        
        self.pair = pair
        
        crds = self.site_coords
        grid_spacing = self.config['device']['grid_spacing']['twoDEG']
        offset = crds[0]%grid_spacing

        poisson_params = {
            "poisson_system": self.poisson_system,
            "linear_problem": self.linear_problem,
            "site_coords": self.site_coords,
            "site_indices": self.site_indices,
            "offset": offset
        }

        kwant_params = {
            "grid_spacing": self.scale,
            "finite_system_object": self.trijunction,
            "finite_system_params_object": self.f_params,
        }

        mu = parameters.bands[0]

        param = parameters.junction_parameters(m_nw=[mu, mu, mu], m_qd=0)
        

        depletion, accumulation = dep_acc_regions(self.poisson_system, 
                        self.site_indices, 
                        self.geometry, 
                        pair)

        self.optimize_args = {
            'poisson': poisson_params,
            'kwant': kwant_params,
            'general_param': param,
            'majorana_pair': pair,
            'dep_region': depletion,
            'acc_region': accumulation
            }
            
        if optimal_phis is not None:
            
            self.optimal_phis = optimal_phis
            
            voltage_regions = list(
                self.poisson_system.regions.voltage.tag_points.keys()
            )
            
            
            print('Finding linear part of the tight-binding Hamiltonian')
            base_ham, linear_ham = linear_Hamiltonian(
                poisson_params,
                kwant_params,
                param,
                voltage_regions,
                phis=self.optimal_phis[pair],
            )
            
            if voltages is not None: self.voltages = voltages
            
            
            eigval, eigvec = diagonalize(
                hamiltonian,
                self.voltages,
                **{'base_ham': base_ham, 'linear_ham': linear_ham},
                no_eigenvalues = 6
            )
            
            lowest_e_indices = np.argsort(np.abs(eigval))
            self.eigenstates = eigvec.T[:, lowest_e_indices].T

            
            if not hasattr(self, "mlwf"):
                self.mlwf = _wannierize(self.trijunction, self.eigenstates)

            self.optimize_args['base_ham'] = base_ham
            self.optimize_args['linear_ham'] = linear_ham
            self.optimize_args['mlwf'] = self.mlwf

        
        return self.optimize_args
        
    
    def plot(self, to_plot="POTENTIAL", phase_results = []):

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

        if to_plot == "PHASE_DIAGRAM" and hasattr(self, optimal_phis) and len(phase_results):
            
            fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
            fig.tight_layout(w_pad=5)
            i = 0
            titles = ["left arm depleted", "right arm depleted", "top arm depleted"]
            phis_labels = [r"$\phi_{center}$", r"$\phi_{center}$", r"$\phi_{right}$"]
            
            phases = np.linspace(0, 2, 100) * np.pi

            phis1 = [{"phi1": phi, "phi2": 0} for phi in phases]
            phis2 = [{"phi2": phi, "phi1": 0} for phi in phases]
            phis = [phis2, phis2, phis1]

            params = parameters.junction_parameters(
                m_nw=parameters.bands[0] * np.ones(3)
            )
            solver = _fixed_potential_solver(
                self.trijunction, self.f_params, params, eigenvecs=False, n = 6
            )
            topo_gap = solver(phis[0][0])[0][-1]


            for energies in phase_results:
                energies = np.array(energies)
                for level in energies.T:
                    ax[i].plot(phases / np.pi, level / topo_gap)
                ax[i].vlines(x=max_phis[i], ymin=-1, ymax=1)
                ax[i].set_title(titles[i])
                ax[i].set_ylabel(r"E[$\Delta^*]$")
                ax[i].set_xlabel(phis_labels[i])
                i += 1
    
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


def _voltage_dict(deplete, accumulate, close=0.0, arm="left",):
    
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


def _fixed_potential_solver(kwant_syst, f_params, base_params, eigenvecs=False, n=6):
    def solver(extra_params):

        base_params.update(extra_params)
        ham_mat = kwant_syst.hamiltonian_submatrix(
            sparse=True, params=f_params(**base_params)
        )
    
        if eigenvecs:
            evals, evecs = sort_eigen(sparse_diag(
                ham_mat.tocsc(), 
                k=n, 
                sigma=0)
                                     )
        else:
            evals = np.sort(sparse_diag(
                ham_mat.tocsc(), k=n, sigma=0, return_eigenvectors = eigenvecs)
                           )
            evecs = []

        return evals, evecs

    return solver


def optimalphase(
    voltages,
    no_eigenvalues,
    poisson_system, 
    linear_problem, 
    site_coords,
    site_indices,
    offset,
    kwant_sys,
    kwant_params,
    general_params,
    Cluster = None,
    nnodes = None,
    cluster_options = None,
    cluster_dashboard_link = None,
    depleteV = [],
    acumulateV = [],
    closeV = [],
):


    potentials = []
    arms = ["left", "right", "top"]

    if not len(voltages): 
        voltages = [_voltage_dict(depleteV, acumulateV, close=closeV, arm=arms[i]) for i in range(3)]
    elif not isinstance(depleteV, list):
        voltages = voltages
    
    
    for voltage in voltages:
        charges = {}
        potential = gate_potential(
            poisson_system,
            linear_problem,
            site_coords[:, [0, 1]],
            site_indices,
            voltage,
            charges,
            offset=offset[[0, 1]]
        )

        # potential.update((x, y*-1) for x, y in potential.items())
        potentials.append(potential)

    phases = np.linspace(0, 2, 100) * np.pi

    phis1 = [{"phi1": phi, "phi2": 0} for phi in phases]
    phis2 = [{"phi2": phi, "phi1": 0} for phi in phases]
    phis = [phis2, phis2, phis1]

    phase_results = []

    if Cluster is not None:
        with Cluster(cluster_options) as cluster:

            cluster.scale(n=nnodes)
            client = cluster.get_client()
            print(cluster_dashboard_link + cluster.dashboard_link[17:])

            for potential, phi in zip(potentials, phis):
                general_params.update(potential=potential)
                solver = _fixed_potential_solver(
                    kwant_sys, kwant_params, general_params, eigenvecs=False, n = no_eigenvalues
                )
                args_db = db.from_sequence(phi)
                result = args_db.map(solver).compute()


                energies = []
                for aux, _ in result:
                    energies.append(aux)
                phase_results.append(energies)


    else:

        print(
            "Optimizing phase for three channels in serial fashion. Do you have access to cluster for parallel calculations? "
        )


        for i, data  in enumerate(zip(potentials, phis)):
            potential, phi = data
            print(f'optimizing phase for pair {i}')
            general_params.update(potential=potential)
            solver = _fixed_potential_solver(
                trijunction, f_params, general_params, eigenvecs=False, n = no_eigenvalues
            )
            energies = []

            for p in phi:
                result = solver(p)
                energies.append(result[0])

            phase_results.append(energies)

    max_phis_id = []
    for pair in phase_results:
        max_phis_id.append(
            find_resonances(energies=np.array(pair), n=no_eigenvalues, sign=1, i=2)[1]
        )
    max_phis_id = np.array(max_phis_id).flatten()
    max_phis = phases[max_phis_id] / np.pi

    assert np.abs(sum([1 - max_phis[0], 1 - max_phis[1]])) < 1e-9 # check whether the max phases are symmetric for LC and RC pairs

    return max_phis, phase_results
                    

def hamiltonian(base_ham, linear_ham, voltages):
    summed_ham = sum(
        [linear_ham[key] * voltages[key] for key, value in linear_ham.items()]
    )

    return base_ham + summed_ham

def diagonalize(hamiltonian,
                voltages,
                base_ham,
                linear_ham,
                no_eigenvalues = 3,
               ):
    numerical_hamiltonian = hamiltonian(
        base_ham,
        linear_ham,
        voltages   
    )
    eigval, eigvec = sort_eigen(
        sparse_diag(numerical_hamiltonian.tocsc(),
                    k=no_eigenvalues,
                    sigma=0, 
                    return_eigenvectors = eigenvecs
                   )
    )
    
    return eigval, eigvec

def dep_acc_regions(poisson_system, 
                    site_indices: np.ndarray, 
                    kwant_geometry:dict, 
                    pair: str):

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

    depletion = [dep_indices[x] for x in dep_regions]
    
    geometry = kwant_geometry
    
    nw_centers = {}
    nw_centers["left"] = np.array(geometry["centers"][0]) / scale
    nw_centers["right"] = np.array(geometry["centers"][1]) / scale
    nw_centers["top"] = np.array(geometry["centers"][2])
    nw_centers["top"][1] -= geometry["nw_l"]
    nw_centers["top"][1] /= scale


    for gate in (set(['left', 'right', 'top']) - set(pair.split("-"))):
        closest_coord_index = _closest_node(nw_centers[gate], grid_points[twodeg_grid][:, [0, 1]])
        depletion.append([closest_coord_index])

    accumulation = []
    for gate in pair.split("-"):
        closest_coord_index = _closest_node(
            nw_centers[gate], grid_points[twodeg_grid][:, [0, 1]]
        )
        accumulation.append([[closest_coord_index]])


    return depletion, accumulation


def potential_shape_loss(x, *argv):
    
    if len(x) == 4 :
        voltages = {key: x[index] for key, index in voltage_keys.items()}
        
    elif len(x) == 2:
        voltages = {}
        
        for arm in ['left', 'right', 'top']:
            for i in range(2): voltages["arm_"+str(i)] = x[0]
        
        voltages["global_accumul"] = x[-1]
    
    for i in range(6):
        voltages["dirichlet_" + str(i)] = 0.0
    
    pp, mus_nw, dep_points, acc_points = argv

    charges = {}
    potential = gate_potential(
        pp["poisson_system"],
        pp["linear_problem"],
        pp["site_coords"],
        pp["site_indices"],
        voltages,
        charges,
        offset=pp["offset"],
    )
    

    potential.update((x, y * -1) for x, y in potential.items())

    potential_array = np.array(list(potential.values()))

    dep_acc_cost = []

    for i, _ in enumerate(acc_points):
        dep_potential = potential_array[np.hstack(dep_points[i])]
        acc_potential = potential_array[acc_points[i]]

        check_potential = (acc_potential > mus_nw[0])
        if check_potential:
            print(f'2DEG is at higher potential than the nanowires: {np.where(check_potential)}')
            dep_acc_cost.append(np.abs(acc_potential - mus_nw[0]))

        if np.any(dep_potential < acc_potential):
            print("Channel not formed as the potential there is higher than elsewhere")
            dep_acc_cost.append(
                np.abs(
                    dep_potential[np.where(dep_potential < acc_potential)]
                    - acc_potential
                )
            )

    if len(dep_acc_cost):
        return sum(np.hstack(dep_acc_cost))

    return 0



def cost_function(x, *argv):
    # Unpack argv
    
    voltages = {key: x[index] for key, index in voltage_keys.items()}

    # Boundary conditions on system sides.
    for i in range(6):
        voltages["dirichlet_" + str(i)] = 0.0

    poisson_params, kwant_params, general_params, pair, dep_points, acc_points = argv[:6]
    
    potential_cost = potential_shape_loss(
        x,
        poisson_params,
        general_params['mus_nw'],
        dep_points,
        acc_points
    )

    if potential_cost:
        return potential_cost
    else:
        other_params = {'base_ham': argv[6], 'linear_ham': argv[7]}
        mlwf = argv[8]
        
        index = pair_indices[pair]
        index.append(list(set(range(3)) - set(index))[0])
        
        #shuffle the wavwfunctions based on the Majorana pairs to be optimized
        reference_wave_functions = mlwf[:, index] 
        
        return majorana_loss(
            x,
            hamiltonian,
            other_params,
            voltage_keys,
            reference_wave_functions,
            general_params["Delta"]
        )


def majorana_loss(
    x,
    hamiltonian,
    other_params,
    x_to_params,
    reference_wave_functions,
    scale,
):
    """Compute the quality of Majorana coupling in a Kwant system.

    Parameters
    ----------
    x : 1d array
        The vector of parameters to optimize
    hamiltonian : callable
        A function for returning the sparse matrix Hamiltonian given parameters.
    other_params : dict
        All the extra inputs to the Hamiltonian
    x_to_params : dict
        Conversion from x to the inputs to the Hamiltonian
    reference_wave_functions : 2d array
        Majorana wave functions. The first two correspond to Majoranas that
        need to be coupled.
    scale : float
        Energy scale to use.
    """
    
    energies, wave_functions = diagonalize(hamiltonian,
                                           {key: x[index] for key, index in x_to_params.items()},
                                           **other_params, 
                                           no_eigenvalues = len(reference_wave_functions)
                                          )
    
    
    
    S = wave_functions @ reference_wave_functions.T.conj()
    # Unitarize the overlap matrix
    U, _, Vh = svd(S)
    S = U @ Vh
    transformed_hamiltonian = S.T.conj() @ np.diag(energies / scale) @ S
    return (
        # Desired coupling
        - np.abs(transformed_hamiltonian[0, 1])
        # Undesired couplings
        + np.log(np.linalg.norm(transformed_hamiltonian[2:]))
    )

