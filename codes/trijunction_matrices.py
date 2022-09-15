import sys, os
import numpy as np
import matplotlib.pyplot as plt
import tinyarray as ta
import kwant
from collections import OrderedDict
from toolz.functoolz import memoize

import codes.finite_system
import codes.parameters
from codes.tools import linear_Hamiltonian, hamiltonian
from codes.utils import eigsh, voltage_dict
from codes.discretize import discretize_heterostructure
from codes.gate_design import gate_coords
from codes.optimization import phase_loss

sys.path.append(os.path.realpath('./../spin-qubit/'))
from Hamiltonian import discrete_system_coordinates
from potential import linear_problem_instance



config = {
    "device":
        {"thickness":
            {"dielectric": 3.0, "twoDEG": 2, "gates": 2, "substrate": 5},
        "grid_spacing":
            {"twoDEG": 0.5, "normal": 1, "dielectric": 1, "air": 5, "gate": 0.5, "substrate": 1},
        "permittivity":
            {"metal": 5000, "twoDEG": 15, "air": 1.0, "Al2O3": 9.1, "substrate": 16}},
    "gate":
        {"L": 41.0, "channel_width": 7, "gap": 4.0, "angle": 0.7853981633974483},
    "kwant":
        {"nwl": 150, "nww": 7}
}


gates_vertex, gate_names, boundaries, nw_centers = gate_coords(
    grid_spacing=1
)

geometry, trijunction, f_params = codes.finite_system.kwantsystem(
    config, boundaries, nw_centers
)

poisson_system = discretize_heterostructure(
    config, boundaries, gates_vertex, gate_names
)

linear_problem = linear_problem_instance(poisson_system)

site_coords, site_indices = discrete_system_coordinates(
    poisson_system, [("charge", "twoDEG")], boundaries=None
)

crds = site_coords[:, [0, 1]]
grid_spacing = config["device"]["grid_spacing"]["twoDEG"]
offset = crds[0] % grid_spacing

poisson_params = {
    "linear_problem": linear_problem,
    "site_coords": site_coords,
    "site_indices": site_indices,
    "offset": offset,
}

voltage_regions = list(poisson_system.regions.voltage.tag_points.keys())

_, linear_terms = linear_Hamiltonian(
    poisson_system,
    poisson_params,
    trijunction,
    f_params,
    voltage_regions,
)

zero_potential = dict(
    zip(
        ta.array(site_coords[:, [0, 1]] - offset),
        np.zeros(len(site_coords)),
    )
)

def base_ham(parameters):
    parameters.update(potential=zero_potential)
    return trijunction.hamiltonian_submatrix(
        sparse=True, params=f_params(**parameters)
    )


base_ham, linear_terms = build_matrices(config)