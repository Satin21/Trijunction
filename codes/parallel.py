import tinyarray as ta
import numpy as np
import matplotlib.pyplot as plt
import importlib
import json
import pickle
import kwant

from codes.gate_design import gate_coords
from codes.constants import scale, bands, topological_gap, sides
import codes.trijunction as trijunction
from codes.optimization import loss, shape_loss, wavefunction_loss
import codes.parameters as parameters
from codes.tools import hamiltonian
from codes.utils import eigsh, svd_transformation, dict_update, dep_acc_index
from codes.utils import order_wavefunctions
from codes.optimization import majorana_loss

from scipy.optimize import minimize, minimize_scalar


def optimize_phase_voltage(argv, config=None):
    filepath = "/home/srangaswamykup/trijunction-design/codes/"
    with open(filepath + "config.json", "r") as f:
        config = json.load(f)

    if config == None and len(argv) == 6:
        identifier, thickness, channel_width, angle, gap, pair = argv

        change_config = [
            {"device": {"thickness": {"dielectric": thickness}}},
            {"gate": {"channel_width": channel_width, "angle": angle, "gap": gap}},
        ]

        for local_config in change_config:
            config = dict_update(config, local_config)

    elif config == None and len(argv) == 2:
        identifier, pair = argv
        with open(filepath + "config.json", "r") as f:
            config = json.load(f)
    elif len(argv) == 3:
        identifier, pair, change_config = argv
        for local_config in change_config:
            config = dict_update(config, local_config)

    print(config)

    system = trijunction.Trijunction(config, optimize_phase_pairs=[])

    phase, voltage, coupling = {}, {}, {}

    filepath = "/home/srangaswamykup/trijunction-design/data/"

    fig, ax = plt.subplots(ncols=1, figsize=(6, 4))

    params = parameters.junction_parameters()
    params.update(potential=system.flat_potential())

    index = system.indices.copy()

    # remove 50% of the points from the channel to be depleted that is closest to the center.
    depleted_channel = list(set(sides) - set(pair.split("-")))[0]
    depleted_indices = index[depleted_channel]
    index[depleted_channel] = depleted_indices[: int(len(depleted_indices) * 50 / 100)]

    params["dep_acc_index"] = index

    args = (
        pair.split("-"),
        (system.base_ham, system.linear_terms),
        params["dep_acc_index"],
    )

    initial_condition = (-3e-3, -3e-3, -3e-3, 3e-3)

    sol1 = minimize(
        codes.optimization.shape_loss,
        x0=initial_condition,
        args=args,
        method="trust-constr",
        options={"initial_tr_radius": 1e-3},
    )

    print(f"sol1:{sol1.x}")

    ci, weights = 50, [1, 1, 1e1]
    args = (
        (
            system.base_ham,
            params,
            system.linear_terms,
            system.f_params,
            system.mlwf[order_wavefunctions(pair)],
        ),
        (pair.split("-"), ci, weights),
    )

    sol2 = minimize(
        codes.optimization.wavefunction_loss,
        x0=sol1.x,
        args=args,
        method="trust-constr",
        options={
            "initial_tr_radius": 1e-3,
            "verbose": 2,
        },
    )

    initial_condition = parameters.voltage_dict(sol2.x)
    print(f"sol2:{sol2.x}")

    params.update(parameters.voltage_dict(sol2.x))

    args = (
        pair,
        params,
        (
            system.trijunction,
            system.linear_terms,
            system.f_params,
            system.mlwf[order_wavefunctions(pair)],
        ),
    )

    sol3 = minimize_scalar(
        codes.optimization.loss, args=args, method="bounded", bounds=(0, 2)
    )

    phase = sol3.x * np.pi

    params.update(parameters.phase_pairs(pair, phase))

    base_ham = system.trijunction.hamiltonian_submatrix(
        sparse=True, params=system.f_params(**params)
    )

    args = (
        pair,
        params,
        (
            base_ham,
            system.linear_terms,
            system.f_params,
            system.mlwf[order_wavefunctions(pair)],
        ),
    )

    sol4 = minimize(
        codes.optimization.loss,
        x0=sol2.x,
        args=args,
        method="trust-constr",
        options={
            "initial_tr_radius": 1e-3,
            "verbose": 2,
        },
    )

    voltages = parameters.voltage_dict(sol4.x)

    params.update(voltages)

    base_ham = system.trijunction.hamiltonian_submatrix(
        sparse=True, params=system.f_params(**params)
    )

    linear_ham, full_ham = hamiltonian(base_ham, system.linear_terms, **params)

    evals, evecs = eigsh(full_ham, k=6, sigma=0, return_eigenvectors=True)

    desired, undesired = majorana_loss(
        evals, evecs, system.mlwf[order_wavefunctions(pair)]
    )

    couplings = {"desired": desired, "undesired": undesired}

    wfv = system.densityoperator(evecs[:, 0])

    # kwant.plotter.map(system.trijunction, lambda i: step_potential[i], ax=ax[0])
    kwant.plotter.density(system.trijunction, wfv, ax=ax)
    ax.set_title(np.round(desired / topological_gap, 3))

    plt.savefig(filepath + "wf_" + str(identifier) + ".pdf", format="pdf")

    if config == None:
        result = {
            "thickness": thickness,
            "channel_width": channel_width,
            "gap": gap,
            "angle": angle,
            "phase": phases,
            "voltages": voltages,
            "couplings": couplings,
            "pair": pair,
        }

    else:
        result = {
            "phase": phases,
            "voltages": voltages,
            "couplings": couplings,
            "pair": pair,
        }

    json.dump(result, open(filepath + "_" + pair + str(identifier) + ".json", "w"))

    return phases, voltages
