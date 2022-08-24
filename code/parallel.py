## popular python packages
import numpy as np
from dask_quantumtinkerer import Cluster, cluster_options
import tinyarray as ta
import json, pickle
from itertools import product
from collections import OrderedDict

# packages used here
import kwant
import sys, os


from optimization import Optimize, hamiltonian, optimize_phase_fn, optimize_gate_fn
import parameters
from constants import scale, majorana_pair_indices
from utils import voltage_dict, eigsh, svd_transformation



# options = cluster_options()
# options.worker_cores = 2
# options.worker_memory = 10
# options.extra_path = "/home/srangaswamykup/trijunction_design/code/"
# cluster_dashboard_link = (
#     "http://io.quantumtinkerer.tudelft.nl/user/srangaswamykup/proxy/"
# )


def parameter_tuning(newconfig):


    with open("/home/srangaswamykup/trijunction_design/code/config.json", "r") as outfile:
        config = json.load(outfile)

    optimize = Optimize(
        config, poisson_system=[], linear_problem=[], boundaries=[], scale=scale
    )

    thickness, gap = newconfig

    change_config = [
        {"device": {"thickness": {"dielectric": thickness}}},
        {"gate": {"channel_width": gap}},
    ]
    
    try:
        _, boundaries, poisson_system, linear_problem = optimize.changeconfig(change_config)
        return 'Success'
    
    except AssertionError:
        return 'ERROR'

    pairs = ["right-top", "left-top", "left-right"]
    voltages = OrderedDict()
    initial_condition = OrderedDict()
    for i, pair in enumerate(pairs):
        initial = [-1.5e-3, -1.5e-3, -1.5e-3, 3e-3]
        initial[i] = -3.5e-3
        voltages[pair] = voltage_dict(initial, True)
        initial_condition[pair] = initial.copy()

    poisson_params = {
        "poisson_system": poisson_system,
        "linear_problem": linear_problem,
        "site_coords": optimize.site_coords,
        "site_indices": optimize.site_indices,
        "offset": optimize.offset,
    }
    params = parameters.junction_parameters(m_nw=parameters.bands[0] * np.ones(3))

    kwant_params = {
        "kwant_sys": optimize.trijunction,
        "kwant_params_fn": optimize.f_params,
        "general_params": params,
        "linear_terms": optimize.optimizer_args["linear_terms"],
    }

    zero_potential = dict(
        zip(
            ta.array(optimize.site_coords[:, [0, 1]] - optimize.offset),
            np.zeros(len(optimize.site_coords)),
        )
    )

    kwant_params["general_params"].update(potential=zero_potential)

    intermediate_couplings = []
    

    iteration = 0
    tol = 1e-1
    max_tol = 1e-2

    voltages = [voltages[pair] for pair in [pairs[0], pairs[2]]]

    del initial_condition["left-top"]

    while np.any(tol > max_tol):

        optimal_phases = optimize_phase_fn(
            voltages, [pairs[0], pairs[2]], kwant_params, 10
        )

        # A = datetime.datetime.now()
        optimal_voltages = optimize_gate_fn(
            [pairs[0], pairs[2]],
            initial_condition,
            optimal_phases,
            optimize.optimizer_args,
            optimize.topological_gap,
        )

        voltages = [
            voltage_dict(optimal_voltages[pair].x, True)
            for pair in [pairs[0], pairs[2]]
        ]
        for pair in [pairs[0], pairs[2]]:
            initial_condition[pair] = optimal_voltages[pair].x

        couplings = {}
        for voltage, pair in zip(voltages, [pairs[0], pairs[2]]):
            _, coupling, _ = wave_functions_coupling(
                pair,
                optimal_phases[pair],
                voltage,
                optimize.trijunction,
                kwant_params["general_params"],
                optimize.f_params,
                optimize.optimizer_args["linear_terms"],
                optimize.mlwf,
                optimize.topological_gap,
            )
            couplings[pair] = coupling

        intermediate_couplings.append(np.array(list(couplings.values())))

        if iteration > 1:
            tol = np.diff(intermediate_couplings[-2:], axis=0)

        iteration += 1

    return intermediate_couplings, voltages, optimal_phases


# if __name__ == "__main__":

#     dielectric_thickness = [0.5, 1, 1.5, 2.0, 2.5, 3.0]
#     gate_separation = [5, 7, 9, 11, 13, 15]

#     SAVE_AT = "/home/tinkerer/trijunction_design/data/results/"

#     sequence = np.array(list(product(dielectric_thickness, gate_separation)))[:2]

#     with Cluster(options) as cluster:
#         cluster.scale(n=len(sequence))
#         client = cluster.get_client()
#         results = []
#         futures = client.map(parameter_tuning, sequence)
#         for future in futures:
#             if future.status == "error":
#                 print("Error!")
#                 results.append("E")
#             else:
#                 results.append(future.result())
#                 filename = SAVE_AT + str(thickness) + "_" + str(gap) + "_" + ".pkl"
#                 with open(filename, "wb") as outfile:
#                     pickle.dump(results[-1], outfile)


def wave_functions_coupling(
    pair,
    optimal_phase,
    voltages,
    kwant_system,
    kwant_params,
    kwant_params_fn,
    linear_terms,
    mlwf,
    energy_scale,
):

    kwant_params.update(optimal_phase)

    if not isinstance(voltages, dict):
        voltages = voltage_dict(voltages, True)

    params = {**kwant_params, **linear_terms}

    numerical_hamiltonian = hamiltonian(
        kwant_system, voltages, kwant_params_fn, **params
    )

    energies, wave_functions = eigsh(
        numerical_hamiltonian.tocsc(), 6, sigma=0, return_eigenvectors=True
    )

    pair_indices = majorana_pair_indices[pair].copy()
    pair_indices.append(list(set(range(3)) - set(pair_indices))[0])
    shuffle = pair_indices + [-3, -2, -1]
    desired_order = np.array(list(range(2, 5)) + list(range(2)) + [5])[shuffle]

    reference_wave_functions = mlwf[desired_order]

    transformed_hamiltonian = (
        svd_transformation(energies, wave_functions, reference_wave_functions)
        / energy_scale
    )

    coupled = np.abs(transformed_hamiltonian[0, 1])
    uncoupled = np.abs([transformed_hamiltonian[1, 2], transformed_hamiltonian[2, 3]])

    return wave_functions, coupled, uncoupled
