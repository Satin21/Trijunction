import sys

sys.path.append("/home/tinkerer/spin-qubit/codes/")

from .gates import rectangular_gate, half_disk_gate
import numpy as np
from shapely.geometry.polygon import Polygon
from Hamiltonian import discrete_system_coordinates
from shapely.ops import unary_union
import poisson
from potential import gate_potential, linear_problem_instance
from layout import (
    Layout,
    OverlappingGateLayer,
    PlanarGateLayer,
    SimpleChargeLayer,
    TwoDEGLayer,
)

# Geometry parameters

a = 10e-9
L = 41
width = 12
gap = 4

R = L / np.sqrt(2)

zmin = -0.5
zmax = 0.5
xmax = R
xmin = -xmax
ymin = 0
ymax = R + L - width
total_width = 2 * xmax
total_length = ymax

# Poisson parameters

thickness_dielectric = 1
thickness_twoDEG = 1
thickness_gates = 3

grid_spacing_twoDEG = 0.5
grid_spacing_normal = 1
grid_spacing_dielectric = 1
grid_spacing_air = 5
grid_spacing_gate = grid_spacing_twoDEG

permittivity_metal = 5000
permittivity_twoDEG = 15  # InAs
permittivity_air = 1.0
permittivity_Al2O3 = 9.1

# Make gates

Y = unary_union(
    (
        Polygon(half_disk_gate(R=R, npts=3)).difference(
            Polygon(half_disk_gate(R=R - width * np.sqrt(2), npts=3))
        ),
        Polygon(rectangular_gate(center=(0, R + L / 2 - width), width=width, length=L)),
    )
)

gates = Polygon(
    rectangular_gate(
        center=(0, (R + L - width) / 2), length=R + L - width - 1, width=2 * R
    )
).difference(Y)

aux_rectangle_1 = rectangular_gate(
    length=R + 2 * gap, width=R + gap, center=(R / 2, R / 2 - width / 2)
)
aux_rectangle_2 = rectangular_gate(
    length=R + 2 * gap, width=R + gap, center=(-R / 2, R / 2 - width / 2)
)


def gate_coords(obj, difference=None, common=None, gap=None):

    if type(common) == np.ndarray:
        return np.round(
            np.array(list(obj.intersection(Polygon(common)).exterior.coords)), 2
        )

    else:
        if gap is not None:
            return np.round(
                np.array(
                    list(
                        obj.difference(Polygon(difference).buffer(gap)).exterior.coords
                    )
                ),
                2,
            )
        else:
            return np.round(
                np.array(list(obj.difference(Polygon(difference)).exterior.coords)), 2
            )


gates_vertex = [
    gate_coords(gates[0], common=aux_rectangle_2),
    gate_coords(gates[2], difference=aux_rectangle_1),
    gate_coords(gates[2], difference=aux_rectangle_2),
    gate_coords(gates[1], common=aux_rectangle_1),
    gate_coords(gates[0], difference=aux_rectangle_2, gap=gap),
    gate_coords(gates[1], difference=aux_rectangle_1, gap=gap),
]

gates_name = ["left_1", "left_2", "right_1", "right_2", "top_1", "top_2"]


# Solve for Poisson system

layout = Layout(
    total_width,
    total_length,
    grid_width_air=grid_spacing_air,
    margin=(50, 50, 50),
    shift=(0, total_length / 2, 0),
)

layout.add_layer(
    SimpleChargeLayer(
        "twoDEG",
        thickness_twoDEG,
        permittivity_twoDEG,
        grid_spacing_twoDEG,
        add_to_previous_layer=False,
        fix_overlap=False,
        z_bottom=None,
    ),
    center=True,
)

height = thickness_twoDEG / 2

layout.add_layer(
    SimpleChargeLayer(
        "Al2O3",
        thickness_dielectric,
        permittivity_Al2O3,
        grid_spacing_dielectric,
        add_to_previous_layer=False,
        fix_overlap=False,
        z_bottom=None,
    )
)

height += thickness_dielectric


layout.add_layer(
    OverlappingGateLayer(
        thickness_gates,
        permittivity_metal,
        grid_spacing_gate,
        layer_name=gates_name,
        gate_objects=gates_vertex,
        remove_points=False,
        add_to_previous_layer=False,
        z_bottom=height,
        fix_overlap=True,
    )
)


height += thickness_gates + 1
layout.add_layer(
    SimpleChargeLayer(
        "Al2O3_2",
        thickness_dielectric,
        permittivity_Al2O3,
        grid_spacing_dielectric,
        add_to_previous_layer=False,
        z_bottom=height,
        fix_overlap=False,
    )
)

height += thickness_dielectric + 1
thickness_accumulation_gate = 2
layout.add_layer(
    PlanarGateLayer(
        "global_accumul",
        thickness_accumulation_gate,
        permittivity_metal,
        grid_spacing_gate,
        gate_coords=[],
        second_layer=[],
        fix_overlap=False,
        z_bottom=height,
    )
)

poisson_system = layout.build()
linear_problem = linear_problem_instance(poisson_system)

site_coords, site_indices = discrete_system_coordinates(
    poisson_system, [("charge", "twoDEG")], boundaries=None
)

crds = site_coords[:, [0, 1]]
grid_spacing = grid_spacing_twoDEG
offset = crds[0] % grid_spacing


def poisson_potential(voltages):
    charges = {}
    clean_potential = gate_potential(
        poisson_system,
        linear_problem,
        site_coords[:, [0, 1]],
        site_indices,
        voltages,
        charges,
        offset=offset,
        grid_spacing=a,
    )
    return clean_potential
