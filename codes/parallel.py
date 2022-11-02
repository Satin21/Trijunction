import tinyarray as ta
import numpy as np
import matplotlib.pyplot as plt
import importlib
import json
import pickle
import kwant

from codes.gate_design import gate_coords
from codes.constants import scale, bands, topological_gap
import codes.trijunction as trijunction
from codes.optimization import loss, shape_loss, wavefunction_loss
import codes.parameters as parameters
from codes.tools import hamiltonian
from codes.utils import eigsh, svd_transformation, dict_update, dep_acc_index
from codes.utils import order_wavefunctions, ratio_Gaussian_curvature
from codes.optimization import majorana_loss

from scipy.optimize import minimize, minimize_scalar


def optimize_phase_voltage(
    argv
):
    identifier, thickness, channel_width, angle, gap = argv
    pairs=['left-right', 'left-top', 'right-top']

    filepath = "/home/srangaswamykup/trijunction-design/codes/"
    
    with open(filepath + "config.json", "r") as f:
        config = json.load(f)

    change_config = [
        {"device": {"thickness": {"dielectric": thickness}}},
        {"gate": {"channel_width": channel_width, "angle": angle, "gap": gap}},
    ]

    for local_config in change_config:
        config = dict_update(config, local_config)

    system = trijunction.Trijunction(config, optimize_phase_pairs=[])
    
    phases, voltages, couplings = {}, {}, {}
    
    filepath = "/home/srangaswamykup/trijunction-design/data/"
    
    fig, ax = plt.subplots(ncols=3, figsize=(6, 4))
    
    for i, pair in enumerate(pairs):

        params = parameters.junction_parameters()
        params.update(potential=system.flat_potential())
        params['dep_acc_index'] = system.indices

        args = (pair.split('-'),
                (system.base_ham, system.linear_terms, system.densityoperator),
                params['dep_acc_index'], 
                )

        initial_condition = (-3e-3, -3e-3, -3e-3, 3e-3)

        sol1 = minimize(codes.optimization.shape_loss, 
                 x0=initial_condition, 
                 args=args, 
                 method='trust-constr', 
                 options={'initial_tr_radius':1e-3}
                )

        ci, wf_amp = 50, 1e-4
        args = ((system.base_ham, 
                params, 
                system.linear_terms, 
                system.f_params, 
                system.densityoperator,
                 system.mlwf[order_wavefunctions(pair)]),
                (pair.split('-'), params['dep_acc_index'], (ci, wf_amp))
               )

        sol2 = minimize(codes.optimization.wavefunction_loss, 
                 x0=sol1.x, 
                 args=args, 
                 method='trust-constr',
                 options={
                     'initial_tr_radius':1e-3,
                     # 'verbose':2,
                 }
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

        phases[pair] = sol3.x * np.pi


        params.update(parameters.phase_pairs(pair, phases[pair]))
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
            identifier
        )
        

        file = filepath + 'coupling' + str(identifier) + '.json'
        with open(file, 'w') as outfile:
            json.dump({}, outfile)

        sol4 = minimize(
            loss,
            x0=sol2.x,
            args=args,
            method="trust-constr",
            options={"initial_tr_radius": 1e-3},
        )
        
        file = filepath + 'coupling' + str(identifier) + '.json'
        with open(file, 'w') as outfile:
            json.dump({}, outfile)

        voltages[pair] = parameters.voltage_dict(sol4.x)
        
        params.update(voltages[pair])

        base_ham = system.trijunction.hamiltonian_submatrix(
            sparse=True, params=system.f_params(**params)
        )

        linear_ham, full_ham = hamiltonian(base_ham, 
                                           system.linear_terms, 
                                           **params
                                          )


        evals, evecs = eigsh(full_ham, k=6, sigma=0, return_eigenvectors=True)


        desired, undesired = majorana_loss(
            evals, evecs, system.mlwf[order_wavefunctions(pair)]
        )
        
        couplings[pair] = {'desired': desired, 
                           'undesired': undesired
                          }

        wfv = system.densityoperator(evecs[:, 0])

        # kwant.plotter.map(system.trijunction, lambda i: step_potential[i], ax=ax[0])
        kwant.plotter.density(system.trijunction, wfv, ax = ax[i]);
        ax[i].set_title(np.round(desired/topological_gap, 3))

    
    plt.savefig(filepath + 'wf_' + str(identifier) + '.pdf', format='pdf')
    
    result = {'thickness': thickness,
              'channel_width': channel_width,
              'gap': gap,
              'angle': angle,
              'phase': phases,
              'voltages':voltages,
              'couplings': couplings
              }
    
    json.dump(result, open(filepath + str(identifier) + '.json', "w"))
    
    return phases, voltages
