import sys, os
import numpy as np
import tinyarray as ta
import json
from toolz.functoolz import memoize

import codes.finite_system
import codes.parameters
from codes.tools import linear_Hamiltonian
from codes.discretize import discretize_heterostructure
from codes.gate_design import gate_coords

sys.path.append(os.path.realpath('./../spin-qubit/'))
from Hamiltonian import discrete_system_coordinates
from potential import linear_problem_instance


@memoize
def make_system():

    with open('/home/tinkerer/trijunction-design/codes/config.json', 'r') as f:
        config = json.load(f)

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

    return zero_potential, trijunction, f_params, linear_terms