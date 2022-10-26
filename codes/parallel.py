import tinyarray as ta
import numpy as np
import matplotlib.pyplot as plt
import importlib
import json
import kwant

from codes.gate_design import gate_coords
from codes.constants import scale, bands, topological_gap
import codes.trijunction as trijunction
from codes.optimization import loss, shape_loss, wavefunction_loss
import codes.parameters as parameters
from codes.tools import hamiltonian
from codes.utils import eigsh, svd_transformation, dict_update, dep_acc_index
from codes.utils import order_wavefunctions, ratio_Gaussian_curvature

from scipy.optimize import minimize, minimize_scalar


def optimize_phase_voltage(pair):

    with open("/home/srangaswamykup/trijunction-design/codes/config.json", "r") as f:
        config = json.load(f)

    change_config = [
        {"device": {"thickness": {"dielectric": 1.0}}},
        {"gate": {"channel_width": 13.0, "angle": np.pi / 6, "gap": 2}},
    ]

    for local_config in change_config:
        config = dict_update(config, local_config)

    system = trijunction.Trijunction(config, optimize_phase_pairs=[])

    params = parameters.junction_parameters()
    params.update(potential=system.flat_potential())
    params["dep_acc_index"] = system.indices

    args = (
        pair.split("-"),
        (system.base_ham, system.linear_terms, system.densityoperator),
        system.indices,
    )

    initial_condition = (-3e-3, -3e-3, -3e-3, 3e-3)

    sol1 = minimize(
        shape_loss,
        x0=initial_condition,
        args=args,
        method="trust-constr",
        options={"initial_tr_radius": 1e-3},
    )

    ci, wf_amp = 50, 1e-4
    args = (
        (
            system.base_ham,
            params,
            system.linear_terms,
            system.f_params,
            system.densityoperator,
        ),
        (pair.split("-"), system.indices, (ci, wf_amp)),
    )

    sol2 = minimize(
        wavefunction_loss,
        x0=sol1.x,
        args=args,
        method="trust-constr",
        options={"initial_tr_radius": 1e-3},
    )

    params.update(parameters.voltage_dict(sol2.x))

    args = (
        pair,
        params,
        (
            system.trijunction,
            system.linear_terms,
            system.f_params,
            system.densityoperator,
            system.mlwf[order_wavefunctions(pair)],
        ),
    )

    sol3 = minimize_scalar(loss, args=args, method="bounded", bounds=(0, 2))

    phase = sol3.x * np.pi

    phase_dict = json.load(open("optimal_phase.json", "r"))
    json.dump(phase, open("optimal_phase.json", "w"))

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
            system.densityoperator,
            system.mlwf[order_wavefunctions(pair)],
        ),
    )

    sol4 = minimize(
        loss,
        x0=sol2.x,
        args=args,
        method="trust-constr",
        options={"initial_tr_radius": 1e-3},
    )

    voltages = parameters.voltage_dict(sol4.x)

    volt_dict = json.load(open("optimal_voltage.json", "r"))
    json.dump(voltages, open("optimal_voltage.json", "w"))

    return phase, voltage
