import tinyarray as ta
import numpy as np
import matplotlib.pyplot as plt
import importlib
import json
import pickle
import kwant
import os

from codes.gate_design import gate_coords
from codes.constants import scale, bands, topological_gap, sides
import codes.trijunction as trijunction
from codes.optimization import loss, shape_loss, wavefunction_loss, density
import codes.parameters as parameters
from codes.tools import hamiltonian
from codes.utils import eigsh, svd_transformation, dict_update, dep_acc_index
from codes.utils import order_wavefunctions
from codes.optimization import majorana_loss

from scipy.optimize import minimize, minimize_scalar


def optimize_phase_voltage(argv, config=None):

    dirname = os.path.dirname(__file__)
    
    with open(dirname + "/" + "config.json", "r") as f:
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
        with open(dirname + "config.json", "r") as f:
            config = json.load(f)
    elif len(argv) == 3:
        identifier, pair, change_config = argv
        for local_config in change_config:
            config = dict_update(config, local_config)

    filepath = os.path.realpath(os.path.join(dirname, '../data/'))
    
    system = trijunction.Trijunction(config, optimize_phase_pairs=[])
    

    phase, voltage, coupling = {}, {}, {}


    params = parameters.junction_parameters()
    params.update(potential=system.flat_potential())
    
    index = system.indices.copy()

    # remove 50% of the points from the channel to be depleted that is closest to the center.
    depleted_channel = list(set(sides)-set(pair.split('-')))[0]
    depleted_indices = index[depleted_channel]
    index[depleted_channel]  = depleted_indices[:int(len(depleted_indices)*50/100)]
    
    params['dep_acc_index'] = index
    
    initial_condition = (-3e-3, -3e-3, -3e-3, 10e-3)
    
    args = (pair.split('-'),
            (system.base_ham, system.linear_terms),
            params['dep_acc_index'], 
            )

    sol1 = minimize(shape_loss, 
             x0=initial_condition, 
             args=args, 
             method='trust-constr', 
             options={'initial_tr_radius':1e-3}
            )


    weights = [1, 1, 1e1]
    args = (
        pair.split('-'),
        (system.base_ham, 
         system.linear_terms,
         system.mlwf[order_wavefunctions(pair)]),
        params['dep_acc_index'],
        weights
    )
    
    
    sol2 = minimize(wavefunction_loss, 
         x0=sol1.x, 
         args=args, 
         method='trust-constr',
         options={
             'initial_tr_radius': 1e-3
         }
        )
        
    initial_condition = sol2.x

    params.update(parameters.voltage_dict(initial_conditions[pair]))
    
    args = (pair, 
            params, 
            (system.trijunction, 
             system.linear_terms, 
             system.f_params,
             system.mlwf[order_wavefunctions(pair)]
            )
           )

    sol3 = minimize_scalar(loss,
                            args=args, 
                            method='bounded',
                            bounds=(0,2),
                          )

    
    phase = sol3.x
    
    params.update(parameters.phase_pairs(pair, phase*np.pi))
    
    base_ham = system.trijunction.hamiltonian_submatrix(
        sparse=True, params=system.f_params(**params)
    )

    args = (pair, 
            params, 
            (base_ham, system.linear_terms, 
             None,
             system.mlwf[order_wavefunctions(pair)]
            ),
           )
    
    sol4 = minimize(loss, 
             x0=initial_conditions[pair].x, 
             args=args, 
             method='trust-constr', 
             options={
                 'initial_tr_radius':1e-3,
             }
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

    wfv = density(evecs[:, 0])

    fig, ax = plt.subplots(ncols=1, figsize=(6, 4))
    kwant.plotter.density(system.trijunction, wfv, ax=ax)
    ax.set_title(np.round(desired / topological_gap, 3))

    plt.savefig(filepath + "/" + pair + "_" + "wf_" + str(identifier) + ".pdf", format="pdf")

    if config == None:
        result = {
            "thickness": thickness,
            "channel_width": channel_width,
            "gap": gap,
            "angle": angle,
            "phase": phase,
            "voltages": voltages,
            "couplings": couplings,
            "pair": pair,
        }

    else:
        result = {
            "initial_condition": initial_condition,
            "phase": phase,
            "voltages": voltages,
            "couplings": couplings,
            "sol1": sol1,
            "sol2": sol2,
            "sol3": sol3,
            "sol4": sol4
        }

    pickle.dump(result, 
                open(filepath + "/" + pair + "_" + str(identifier) + ".pkl", "w"), 
               protocol=pickle.HIGHEST_PROTOCOL)

    return phase, voltages
