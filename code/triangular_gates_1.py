import sys
sys.path.append("/home/tinkerer/spin-qubit/codes/")
sys.path.insert(0, "/home/tinkerer/Poisson_Solver")

import kwant
import numpy as np
from scipy import constants
import itertools as it
import scipy.sparse.linalg as sla

from potential import gate_potential, linear_problem_instance
from .gates_trijunction import triangular_gates_2
from .finite_system import finite_system
from .solvers import junction_parameters, phase, bands
from .tools import get_potential


from utility import prepare_voltages
from plotting import plot_potential
from layout import (
    Layout,
    OverlappingGateLayer,
    PlanarGateLayer,
    SimpleChargeLayer,
    SimpleVoltageLayer,
    TwoDEGLayer,
)
from Hamiltonian import discrete_system_coordinates, kwant_system, tight_binding_Hamiltonian

mu = bands[0]
# Set up system paramters
thickness_GaAs = 6
thickness_twoDEG = 4
thickness_Al2O3 = 4
thickness_gate = 2
thickness_self_Al2O3 = 0

meff = 0.023 * constants.m_e  # in Kg
eV = 1.0
bandgap_GaAs = 1.519 * eV

permittivity_metal = 5000
permittivity_GaAs = 12.18
permittivity_twoDEG = 15  # InAs
permittivity_air = 1.0
permittivity_Al2O3 = 9.1

grid_spacing_twoDEG = 1
grid_spacing_normal = 1
grid_spacing_dielectric = 5
grid_spacing_air = 5
grid_spacing_GaAs = grid_spacing_normal
grid_spacing_gate = grid_spacing_twoDEG
grid_spacing = grid_spacing_twoDEG

area = 600
angle = 0.68
wire_width = 7
gap = 4

triangle_length = np.sqrt(area*np.tan(angle))
triangle_width = np.abs((triangle_length/np.tan(angle)))
top_shift = np.tan(angle)*(wire_width/2)
tunnel_length = 3
tunnel_width = wire_width

total_length = triangle_length + 2 * tunnel_length + 2 * gap - top_shift
extra_width = 10
total_width = 2 * extra_width + triangle_width
total_width = 2*total_width

# Set up gates
gates = triangular_gates_2(area, angle, wire_width, tunnel_length, gap, extra_width)

zmin = -0.5
zmax = 0.5
xmax = triangle_width
xmin = -xmax
ymin = 0
ymax = total_length
boundaries = [xmin, xmax, ymin, ymax, zmin, zmax]

a = 1
L = boundaries[3] - boundaries[2]
W = boundaries[1] - boundaries[0]

# Solve for Poisson system
layout = Layout(total_width,
                total_length,
                grid_width_air=grid_spacing_air,
                margin=(50, 50, 50),
                shift=(0, total_length/2, 0))

layout.add_layer(
    TwoDEGLayer(
        "twoDEG",
        thickness_twoDEG,
        permittivity_twoDEG,
        grid_spacing_twoDEG
    ),
    center=True,
)

layout.add_layer(
    SimpleChargeLayer(
        "GaAs",
        thickness_GaAs,
        permittivity_GaAs,
        grid_spacing_GaAs,
    )
)

vertex = (
    list(gates["plunger_gates"].values()),
    list(gates["screen_gates"].values()),
    list(gates["tunel_gates"].values()),
        )

layout.add_layer(
    OverlappingGateLayer(
        vertex,
        np.hstack([list(gates[key].keys()) for key, _ in gates.items()]),
        thickness_gate,
        thickness_self_Al2O3,
        permittivity_metal,
        grid_spacing_gate,
    )
)

poisson_system = layout.build()
linear_problem = linear_problem_instance(poisson_system)


site_coords, site_indices = discrete_system_coordinates(
    poisson_system, [('mixed', 'twoDEG')], boundaries=boundaries
)

crds = site_coords[:, [0, 1]]
offset = crds[0]%grid_spacing

# Build kwant system
a = 10e-9
center = W*a/4
centers = [center, -center]
geometry = {
    "l": 130*a,
    "w": 7*a,
    "a": a,
    "side": 'up',
    "shape": 'rectangle',
    "L": L*a,
    "W": W*a,
    "centers": centers
}

# tunnel positions
x_r, y_r = int(W/4)*a, 0
x_l, y_l = -int(W/4)*a, 0
x_c, y_c = 0, int(L-1)*a

trijunction, f_params, f_params_potential = finite_system(**geometry)
trijunction = trijunction.finalized()

# Functions used for adaptive calculations
def potential(voltage_setup, offset=offset, grid_spacing=1):
    charges = {}
    clean_potential = gate_potential(
        poisson_system,
        linear_problem,
        site_coords[:, [0, 1]],
        site_indices,
        voltage_setup,
        charges,
        offset,
        grid_spacing
    )
    return clean_potential

def get_hamiltonian(voltage_setup, key_1, val_1, key_2, val_2, params, points):
    voltage_setup[key_1] = val_1
    voltage_setup[key_2] = val_2    
    f_pot = get_potential(potential(voltage_setup=voltage_setup, grid_spacing=10e-9))

    ham_mat = trijunction.hamiltonian_submatrix(sparse=True,
                                                params=f_params_potential(potential=f_pot,
                                                                          params=params))
    
    return ham_mat, potential_at_gates(f_pot, points)

def potential_at_gates(f_pot, points):
    data = []
    for point in points:
        x, y = point
        data.append(f_pot(x, y))
    return np.array(data)

def solver_electrostatics(tj_system, voltage_setup, pair, key, n, eigenvecs):

    params = junction_parameters(m_nw=np.array([mu, mu, mu]), m_qd=0)
    params.update(phase(pair))

    def eigensystem_sla(val):
        
        system, f_params_potential = tj_system

        voltage_setup[key] = val
        f_potential = get_potential(potential(voltage_setup=voltage_setup, grid_spacing=10e-9))
        f_params = f_params_potential(potential=f_potential, params=params)
        ham_mat = system.hamiltonian_submatrix(sparse=True, params=f_params)

        if eigenvecs:
            evals, evecs = sort_eigen(sla.eigsh(ham_mat.tocsc(), k=n, sigma=0))
        else:
            evals = np.sort(sla.eigsh(ham_mat.tocsc(), k=n, sigma=0, return_eigenvectors=eigenvecs))
            evecs = []

        return evals, evecs

    return eigensystem_sla 