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


mu = pm.bands[0]
a = 10e-9

R = 60
r = 51
gap = 4
tunel_length = 5
wire_width = 13
y_shift = tunel_length + gap

zmin = -0.5
zmax = 0.5
xmax = R + 2*gap + 2*wire_width
xmin = -xmax
ymin = 0
ymax = R + gap +tunel_length + y_shift
boundaries = [xmin, xmax, ymin, ymax, zmin, zmax]
total_width = 2*xmax
total_length = ymax

L = boundaries[3] - boundaries[2]
W = boundaries[1] - boundaries[0]

thickness_barrier = 4
thickness_twoDEG = 4
thickness_gates = 6

permittivity_metal = 5000
permittivity_twoDEG = 15  # InAs
permittivity_air = 1.0
permittivity_Al2O3 = 9.1

grid_spacing_twoDEG = 1
grid_spacing_normal = 1
grid_spacing_dielectric = 5
grid_spacing_air = 5
grid_spacing_barrier = grid_spacing_normal
grid_spacing_gate = grid_spacing_twoDEG

gates_name, gates_vertex = gt.ring_gates(R, wire_width, gap, tunel_length)

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
        grid_spacing_twoDEG,
        add_to_previous_layer=False,
        fix_overlap=False,
        z_bottom=None
    ),
    center=True,
)

height = thickness_twoDEG / 2

layout.add_layer(
    SimpleChargeLayer(
        "Al2O3",
        thickness_barrier,
        permittivity_Al2O3,
        grid_spacing_barrier,
        add_to_previous_layer=False,
        fix_overlap=False,
        z_bottom=None
    )
)

height += thickness_barrier

layout.add_layer(OverlappingGateLayer(thickness_gates,
                                      permittivity_metal,
                                      grid_spacing_gate,
                                      layer_name=gates_name,
                                      gate_objects=gates_vertex,
                                      remove_points=False,
                                      add_to_previous_layer=False,
                                      z_bottom=height,
                                      fix_overlap=True
                                     )
            )

poisson_system = layout.build()
linear_problem = linear_problem_instance(poisson_system)

site_coords, site_indices = discrete_system_coordinates(
    poisson_system, [('mixed', 'twoDEG')], boundaries=boundaries
)

crds = site_coords[:, [0, 1]]
offset = crds[0]%grid_spacing_twoDEG

a = 10e-9
center = R - (R-r)/2
centers = [center*a, -center*a]
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

