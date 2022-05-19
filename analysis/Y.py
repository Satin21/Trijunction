#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("/home/tinkerer/spin-qubit/codes/")
#sys.path.append("/home/tinkerer/Poisson/")

import numpy as np
from collections import OrderedDict
from ccode.plotting import plot_gates
from ccode.gates import rectangular_gate, half_disk_gate
from ccode.finite_system import finite_system
import ccode.parameters as pm
import ccode.solvers as sl
import ccode.tools as tl
import itertools as it
import dask.bag as db
from scipy import constants
from scipy.linalg import svd
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt
from dask_quantumtinkerer import Cluster, cluster_options

import poisson
import kwant
from kwant.operator import Density
from potential import gate_potential, linear_problem_instance
from plotting import plot_potential
from Hamiltonian import discrete_system_coordinates

from layout import (
    Layout,
    OverlappingGateLayer,
    PlanarGateLayer,
    SimpleChargeLayer,
    TwoDEGLayer,
)
from utility import wannier_1D_operator, wannier_basis

from itertools import product
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union



# In[2]:


a = 10e-9


# In[3]:


options = cluster_options()
options.worker_cores = 2
options.worker_memory = 2
options.extra_path = "/home/srangaswamykup/trijunction_design/"


# # Y-shaped geometry

# #### Parameters

# In[4]:


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


# In[5]:


# one only needs to define the arm length L, and the channel width
L = 41
width = 12
gap = 4

R = L/np.sqrt(2)


# In[6]:


R


# In[7]:


# Boundaries within Poisson region
zmin = -0.5
zmax = 0.5
xmax = R
xmin = -xmax
ymin = 0
ymax = R + L - width
total_width = 2*xmax
total_length = ymax


# #### Make gate polygons using Shapely

# In[8]:


Y = unary_union(
    (
        Polygon(half_disk_gate(R=R, npts=3)).difference(Polygon(half_disk_gate(R=R-width*np.sqrt(2), npts=3))),
        Polygon(rectangular_gate(center=(0,R+L/2-width), width=width, length=L))
    )
)

gates = Polygon(rectangular_gate(center=(0, (R+L-width)/2), length=R+L-width-1,width=2*R)).difference(Y)

aux_rectangle_1 = rectangular_gate(length=R+2*gap, width=R+gap, center=(R/2, R/2-width/2))
aux_rectangle_2 = rectangular_gate(length=R+2*gap, width=R+gap, center=(-R/2, R/2-width/2))


def gate_coords(obj, difference=None, common=None, gap=None):

    if type(common) == np.ndarray:
        return np.round(np.array(list(obj.intersection(Polygon(common)).exterior.coords)), 2)

    else:
        if gap is not None:
            return np.round(np.array(list(obj.difference(Polygon(difference).buffer(gap)).exterior.coords)), 2)
        else:
            return np.round(np.array(list(obj.difference(Polygon(difference)).exterior.coords)), 2)


gates_vertex = [gate_coords(gates[0], common = aux_rectangle_2), 
                gate_coords(gates[2], difference = aux_rectangle_1), 
                gate_coords(gates[2], difference = aux_rectangle_2), 
                gate_coords(gates[1], common = aux_rectangle_1), 
                gate_coords(gates[0], difference = aux_rectangle_2, gap = gap), 
                gate_coords(gates[1], difference = aux_rectangle_1, gap = gap)]

gates_name = ['left_1', 'left_2', 'right_1', 'right_2', 'top_1', 'top_2']


# #### Construct layout object for the Poisson solver

# In[9]:


# Solve for Poisson system

layout = Layout(total_width,
                total_length,
                grid_width_air=grid_spacing_air,
                margin=(50, 50, 50),
                shift=(0, total_length/2, 0))

layout.add_layer(
    SimpleChargeLayer(
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
        thickness_dielectric,
        permittivity_Al2O3,
        grid_spacing_dielectric,
        add_to_previous_layer = False,
        fix_overlap = False,
        z_bottom = None
    )
)

height += thickness_dielectric


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



# In[10]:


height += thickness_gates + 1
layout.add_layer(
    SimpleChargeLayer("Al2O3_2", thickness_dielectric, permittivity_Al2O3, grid_spacing_dielectric,
                     add_to_previous_layer = False,
                     z_bottom = height, fix_overlap = False)
)

height += thickness_dielectric + 1
thickness_accumulation_gate = 2
layout.add_layer(PlanarGateLayer("global_accumul", 
                                 thickness_accumulation_gate, 
                                 permittivity_metal, 
                                 grid_spacing_gate, 
                                 gate_coords = [],
                                 second_layer = [],
                                 fix_overlap = False,
                                 z_bottom = height
                                )
                )


# #### Build Poisson object

# In[11]:


get_ipython().run_cell_magic('time', '', '\npoisson_system = layout.build()\n')


# In[12]:


grid_points = poisson_system.grid.points
voltage_regions = poisson_system.regions.voltage.tag_points


# In[13]:


plt.figure(figsize = (8, 7))
for name, indices in voltage_regions.items():
    grid_to_plot = grid_points[indices][:, [0, 1]]
    plt.scatter(grid_to_plot[:, 0], grid_to_plot[:, 1], s = 0.5)
    
# for i in range(len(dep_indices)):
#     to_plot = grid_points[twodeg_grid[dep_indices[i]]]
#     print(to_plot)
#     plt.scatter(to_plot[0][0], to_plot[0][1]);


# #### LU decomposition of finite system

# In[14]:


get_ipython().run_cell_magic('time', '', '\nlinear_problem = linear_problem_instance(poisson_system)\n')


# In[15]:


site_coords, site_indices = discrete_system_coordinates(
    poisson_system, [('charge', 'twoDEG')], boundaries=None
)

crds = site_coords[:, [0, 1]]
grid_spacing = grid_spacing_twoDEG
offset = crds[0]%grid_spacing


# In[16]:


grid_points= poisson_system.grid.points


# In[17]:


charge_regions = poisson_system.regions.charge.tag_points


# In[18]:


voltage_regions = poisson_system.regions.voltage.tag_points
regions_to_add = []
for region in voltage_regions.keys():
    if region not in ['dirichlet_0', 'dirichlet_1', 'dirichlet_2', 'dirichlet_3']:
        regions_to_add.append(('voltage', region))
for region in charge_regions.keys():
    # if region not in ['air']:
    regions_to_add.append(('charge', region))


# #### Verify the potential

# In[19]:


voltage_regions = list(poisson_system.regions.voltage.tag_points.keys())


# In[20]:


depleted = -1e-3
acumulate = 4e-3

def voltage_dict(depleted, acumulate):
    voltages = {}

    voltages['left_1'] = depleted
    voltages['left_2'] = depleted

    voltages['right_1'] = depleted
    voltages['right_2'] = depleted

    voltages['top_1'] = depleted
    voltages['top_2'] = depleted

    voltages['global_accumul'] = acumulate

    return voltages


# In[21]:


voltages = voltage_dict(depleted, acumulate)


# In[22]:


get_ipython().run_cell_magic('time', '', '\ncharges = {}\nclean_potential = gate_potential(\n        poisson_system,\n        linear_problem,\n        site_coords[:, [0, 1]],\n        site_indices,\n        voltages,\n        charges,\n        offset=offset,\n        grid_spacing=a\n    )\n')


# In[23]:


coordinates = np.array(list(clean_potential.keys()))

x = coordinates[:, 0]
y = coordinates[:, 1]
width_plot = np.unique(x).shape[0]
X = x.reshape(width_plot, -1) / a
Y = y.reshape(width_plot, -1) / a
Z = np.round(np.array(list(clean_potential.values())).reshape(width_plot, -1) * -1, 4)


# In[24]:


plt.figure()
plt.imshow(np.rot90(Z), extent = (xmin, xmax, ymin, ymax), cmap = 'magma_r')
plt.colorbar();


# In[25]:


plt.imshow((np.rot90(Z[::-1,:])-np.rot90(Z)), extent = (xmin, xmax, ymin, ymax), cmap = 'magma_r')
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("V(x,y) - V(-x,y)")


# #### Build kwant object

# In[26]:


# Build kwant system

R_a = R*a
width_a = width*a
l = 150 * a
w = 7 * a
boundaries = [xmin, xmax, ymin, ymax, min(grid_points[:, 2]), max(grid_points[:, 2])]
boundaries = np.array(boundaries) * a

geometry ={'nw_l': l, 'nw_w': w,
           's_w': boundaries[1] - boundaries[0],
           's_l': boundaries[3] - boundaries[2],
           'centers':[[-R_a+width_a/np.sqrt(2)+0.25e-7, 0],
                      [-(-R_a+width_a/np.sqrt(2))-0.25e-7, 0], 
                      [0,boundaries[3]+l-a]
                     ]
          }


# In[27]:


geometry

import importlib
import ccode.parameters
importlib.reload(ccode.parameters)
# In[28]:


get_ipython().run_cell_magic('time', '', '\ntrijunction, f_params = finite_system(**geometry)\ntrijunction = trijunction.finalized()\n')


# In[29]:


fig, ax = plt.subplots()
kwant.plot(trijunction, ax=ax, lead_site_size = 4)
ax.set_ylim(-10*a, boundaries[3]+10*a)
grid_points = poisson_system.grid.points
voltage_regions = poisson_system.regions.voltage.tag_points

# for name, indices in voltage_regions.items():
#     if name not in ['global_accumul']:
#         grid_to_plot = a*grid_points[indices][:, [0, 1]]
#         ax.scatter(grid_to_plot[:, 0], grid_to_plot[:, 1], s = 0.5)

np.unique(grid_points[voltage_regions['global_accumul']][:, 1])
# In[30]:


import pickle


# In[31]:


with open("/home/tinkerer/trijunction-design/data/optimal_phase.pkl", 'rb') as infile:
    optimal_phase = pickle.load(infile)


# In[32]:


optimal_phase = optimal_phase * np.pi


# #### Optimal phases

# In[33]:


voltages = {'left_1': -5e-3, 
            'left_2': -5e-3,
            'right_1': -5e-3,
            'right_2': -5e-3,
            'top_1' : -18e-3,
            'top_2' : -18e-3,
            'global_accumul': 35e-3}


# In[34]:


potential = gate_potential(
        poisson_system,
        linear_problem,
        site_coords[:, [0, 1]],
        site_indices,
        voltages,
        charges,
        offset=offset[[0, 1]],
        grid_spacing=a
    )


# In[35]:


mu = pm.bands[0]
params = pm.junction_parameters(m_nw=[mu, mu, mu], m_qd=0)


# In[36]:


params.update(potential=clean_potential)


# In[2804]:


solver = sl.general_solver(geometries=[[trijunction, f_params]],
                           n=20,
                           base_parameters=params,
                           eigenvecs=False)


def wrap(arg):
    return solver(*arg)


# In[2806]:


args = list(it.product([0],pm.phase_params()))
args_db = db.from_sequence(args)


# In[2807]:


get_ipython().run_cell_magic('time', '', 'with Cluster(options) as cluster:\n    cluster.scale(10)\n    client = cluster.get_client()\n    print("http://io.quantumtinkerer.tudelft.nl/user/srangaswamykup/proxy/"+cluster.dashboard_link[17:])\n    result = args_db.map(wrap).compute()\n')


# In[2808]:


spectra, couplings, wfs, peaks, widths = tl.coupling_data(result, n=20)


# In[2809]:


phis = np.linspace(0, 2, 100)
max_phis = phis[np.array(peaks).flatten()]


# In[2810]:


max_phis

optimal_phase = np.zeros((3, ))
# In[2812]:


optimal_phase[0] = max_phis[0] 


# #### Optimize gate voltages

# In[37]:


from scipy import sparse


# In[38]:


poisson_params = {
    'poisson_system': poisson_system,
    'linear_problem': linear_problem,
    'site_coords': site_coords,
    'site_indices': site_indices,
}    

crds = site_coords
grid_spacing = grid_spacing_twoDEG
offset = crds[0]%grid_spacing

kwant_params = {
    'offset': offset,
    'grid_spacing': a,
    'finite_system_object': trijunction,
    'finite_system_params_object': f_params,
    
}


# In[39]:


def spectrum_f_voltages(
    kwant_params,
    general_params,
    potential,
    no_of_eigenvalues = 6,
    return_eigenvecs = False,
    phis = [0, 0],    
):
    kp = kwant_params
    
    general_params.update(potential=potential)
    

    general_params['phi1'] = phis[0]
    general_params['phi2'] = phis[1]
    
    ham_mat = kp['finite_system_object'].hamiltonian_submatrix(
        sparse=True, params=kp['finite_system_params_object'](**general_params))
    
    if return_eigenvecs:
        evals, evecs = sl.sort_eigen(sparse.linalg.eigsh(ham_mat.tocsc(), k=no_of_eigenvalues, sigma = 0))
        
        return evals, evecs
    else:
        evals = np.sort(
            sparse.linalg.eigsh(ham_mat.tocsc(), k = no_of_eigenvalues, sigma = 0,
                                return_eigenvectors = return_eigenvecs)
        )
        evecs = []
        
        return evals


# In[40]:


voltage_dict = {'left' : -0.0026, 
                'right' : -0.0028, 
                'top' : -0.0024, 
                'back' : 0.00045}


# In[41]:


for i in range(1, 3):
    voltages['left_'+str(i)] = voltage_dict['left']
    voltages['right_'+str(i)] = voltage_dict['right']
    voltages['top_'+str(i)] = voltage_dict['top']
    
voltages['global_accumul'] = voltage_dict['back']


# In[42]:


centers = np.array(geometry['centers']) / a
nw_l = geometry['nw_l'] / a
nw_w = geometry['nw_w'] / a

depletion_regions  = {'left': np.array(((centers[0][0] - nw_w /2, centers[0][1]),
                              (centers[0][0] + nw_w /2, centers[0][1])), dtype = 'float32'),
                      'right': np.array(((centers[1][0] - nw_w /2, centers[1][1]),
                               (centers[1][0] + nw_w /2, centers[1][1])), dtype = 'float32'),
                     'top':   np.array(((centers[2][0] - nw_w /2, centers[2][1] - nw_l),
                              (centers[2][0] + nw_w /2, centers[2][1] - nw_l)), dtype = 'float32')}


# In[149]:


grid = poisson_system.grid.points.astype('float32')


# In[150]:


voltage_regions = poisson_system.regions.voltage.tag_points


# In[151]:


twodeg_grid = site_indices
twodeg_grid_monolayer = twodeg_grid[grid[twodeg_grid][:, 2] == np.unique(grid[twodeg_grid][:, 2])[0]]


# In[152]:


def depletion_points(grid, monolayer, boundary):
    Y = np.argmin(np.abs((grid[monolayer][:, 1] - boundary[0, 1])))
    X = grid[monolayer][:, 1] == grid[monolayer[Y]][1]
    point_indices = np.multiply(grid[monolayer[X]][:, 0] > boundary[0, 0], 
                                grid[monolayer[X]][:, 0] < boundary[1, 0])
    
    return monolayer[X][point_indices]


# In[153]:


dep_indices = []
acc_indices = []
for gate in ['left_1', 'left_2', 'right_1', 'right_2', 'top_1', 'top_2']:
    indices = voltage_regions[gate]
    if gate in ['top_1', 'top_2']:
        x, y = grid_points[indices][::-1][0, 0], grid_points[indices][::-1][0, 1]
    else:
        x, y = grid_points[indices][0, 0], grid_points[indices][0, 1]
    points = grid_points[twodeg_grid][:, [0, 1]]
    points[:, 0] -= x
    points[:, 1] -= y
    dep_indices.append(np.array([np.argmin(np.sum(np.abs(points), axis = 1))]))

for gate in ['top']:
    indices = depletion_points(grid, twodeg_grid_monolayer, depletion_regions[gate])
    dep_indices.append(np.array([np.hstack(list(map(lambda x: np.argwhere(twodeg_grid == x), 
                                                    indices)))[0][0]]))

for gate in ['left', 'right']:
    indices = depletion_points(grid, twodeg_grid_monolayer, depletion_regions[gate])
    acc_indices.append(np.array([np.hstack(list(map(lambda x: np.argwhere(twodeg_grid == x), 
                                                    indices)))[0][0]]))


# In[154]:


def left(site):
    x, y = site.pos
    return x < 0
leftop = Density(trijunction, onsite = np.eye(4), where = left, sum = True)

def right(site):
    x, y = site.pos
    return x > 0
rightop = Density(trijunction, onsite = np.eye(4), where = right, sum = True)

def top(site):
    x, y = site.pos
    return y > 0
topop = Density(trijunction, onsite = np.eye(4), where = top, sum = True)


# In[160]:


from utility import gather_data


# In[361]:


def gate_tuner(x, *argv):
    
    voltages = {}
    
    for i in range(1, 3):
        voltages['left_'+str(i)] = x[0]
        voltages['right_'+str(i)] = x[1]
        voltages['top_'+str(i)] = x[2]
    
    voltages['global_accumul'] = x[3]
    
    poisson_params, kwant_params, general_params,  = argv[:3]
    dep_points, acc_points, coupled_pair, uncoupled_pairs  = argv[3:7]
    base_hamiltonian, linear_hamiltonian = argv[7:9]
    X_operator, Y_operator = argv[9:11]
    
    
    pp = poisson_params
    kp = kwant_params
    
    charges = {}
    potential = gate_potential(
            pp['poisson_system'],
            pp['linear_problem'],
            pp['site_coords'],
            pp['site_indices'],
            voltages,
            charges,
            offset = kp['offset'],
            grid_spacing = kp['grid_spacing']
        )
    
    potential.update((x, y*-1) for x, y in potential.items())
    
    
    potential_array = np.array(list(potential.values()))

    dep_potential = np.array([potential_array[points][0] for points in dep_points])
    acc_potential = np.array([potential_array[points][0] for points in acc_points])
    
    
    cost = []
    # if np.any(dep_potential < general_params['mus_nw'][0] + 1e-3):
    #     indices = np.argwhere(dep_potential < general_params['mus_nw'][0])
    barrier_cost = (general_params['mus_nw'][0] + 1e-3) - dep_potential
    cost.append(sum(np.abs(barrier_cost)))
    if np.any(acc_potential > general_params['mus_nw'][0]):
        indices = np.argwhere(acc_potential > general_params['mus_nw'][0])
        accumulation_cost = acc_potential[indices] - general_params['mus_nw'][0]
        cost.append(sum(accumulation_cost))
    else:
        cost.append(np.abs(acc_potential[1]- acc_potential[0]))
        
    
    summed_ham = sum(
        [
            linear_hamiltonian[key] * voltages[key]
            for key, value in linear_hamiltonian.items()
        ]
    )
    
    tight_binding_hamiltonian = base_hamiltonian + summed_ham


    eigval, eigvec = sl.sort_eigen(sparse.linalg.eigsh(tight_binding_hamiltonian.tocsc(), 
                                                       k=12, sigma = 0))
    
    eigval = eigval
    eigvec = eigvec
    
    density = kwant.operator.Density(kp['finite_system_object'], np.eye(4))
    
#     basis_sequence = []
#     for vec in eigvec[:3]:
#         basis_sequence.append(np.argmax([op(vec) for op in [leftop, 
#                                                             rightop, 
#                                                             topop]]))

#     basis_sequence = np.hstack(basis_sequence)

    # eigval = eigval[:3][basis_sequence]
    # eigvec[:3] = np.squeeze(eigvec[:3][basis_sequence])
    
    projected_X_operator = wannier_1D_operator(X_operator, 
                                           eigvec[3:9].T)

    projected_Y_operator = wannier_1D_operator(Y_operator, 
                                           eigvec[3:9].T)
    
    w_basis = wannier_basis([projected_X_operator, 
                         projected_Y_operator])

    mlwf = w_basis.T @ eigvec[3:9]
    
    fig, ax = plt.subplots(2, 6, figsize = (10, 5), sharey= True)
    for i in range(6):
        kwant.plotter.density(trijunction, density(mlwf[i]), ax = ax[0][i]);
    for i in range(6):
        kwant.plotter.density(trijunction, density(eigvec[3:9][i]), ax = ax[1][i]);
    
    filepath = '/home/tinkerer/trijunction-design/data/optimization/'
    seed = gather_data(filepath)

    if ".ipynb_checkpoints" in seed:
        os.system("rm -rf .ipynb_checkpoints")

    if len(seed):
        file_name = filepath + "plt_" + str(max(seed) + 1) + "_.png"
    else:
        file_name = filepath + "plt_" + str(0) + "_.png"

    plt.savefig(file_name, format="png", bbox_inches="tight", pad_inches=0.0)

    plt.close()

    if coupled_pair != None:

        S = eigvec[3:9] @ mlwf.T.conj()
        U, _, Vh = svd(S)
        S_prime = U @ Vh
        eff_Hamiltonian = S_prime.T.conj() @ np.diag(eigval[3:9]) @ S_prime
        
        
        coupled_cost = -1 * np.abs(eff_Hamiltonian[coupled_pair[0], coupled_pair[1]])
        uncoupled_cost = np.zeros((2, ))
        
        for i, pair in enumerate(uncoupled_pairs): 
            uncoupled_cost[i] = np.abs(eff_Hamiltonian[pair[0], pair[1]])
        
        total_coupling_cost = coupled_cost + sum(uncoupled_cost)
        
        print(coupled_cost)
        
        if np.abs(total_coupling_cost) > 1e-9:
            cost = [total_coupling_cost]
        else:
            cost.append(total_coupling_cost)
    
    print(cost)
    
    return sum(cost) # -1 to maximize the couplings
    


# In[362]:


mu = pm.bands[0]

params = pm.junction_parameters(m_nw=[mu, mu, mu], m_qd=0)
# In[363]:


coupled_pairs = [2, 3] # 0: left, 1: right, 2: top
uncoupled_pairs = [[2, 4], [3, 4]]
args = (poisson_params, kwant_params, params, 
        dep_indices, acc_indices, coupled_pairs, uncoupled_pairs,
        base_hamiltonian, linear_ham,
        X_operator, Y_operator)
initial_condition = list(voltages.values())


# In[364]:


initial_condition

from utility import gather_data
# In[366]:


get_ipython().run_cell_magic('time', '', "\nsol1 = minimize(\n            gate_tuner,\n            initial_condition,\n            args=args,\n    # ftol = 1e-6,\n    # verbose = 2\n    method = 'trust-constr',\n            options = {'disp': True, 'verbose': 2, 'maxiter': 20, \n                       'initial_tr_radius': 2e-4,\n                       'gtol': 1e-3}\n        )\n")


# In[2819]:


x = sol1.x
for i in range(1, 3):
    voltages['left_'+str(i)] = x[0] 
    voltages['right_'+str(i)] = x[1]
    voltages['top_'+str(i)] = x[2]

voltages['global_accumul'] = x[3]


# In[350]:


voltages = {'left_1': -0.003681622975055661,
 'left_2': -0.008681622975055661,
 'right_1': -0.008681622975055661,
 'right_2': -0.003681622975055661,
 'top_1': -0.0037495890626959454,
 'top_2': -0.0037495890626959454,
 'global_accumul': -6.98461859286267e-05}

voltages['left_2'] -= 0.005
voltages['right_1'] = voltages['left_2']
voltages['right_2'] = voltages['left_1']
# In[458]:


get_ipython().run_cell_magic('time', '', "\ncharges = {}\npotential = gate_potential(\n        poisson_system,\n        linear_problem,\n        site_coords[:, [0, 1]],\n        site_indices,\n        voltages,\n        charges,\n        offset = offset[[0, 1]],\n        grid_spacing = kwant_params['grid_spacing']\n    )\n\npotential.update((x, y*-1) for x, y in potential.items())\n\nparams.update(potential=potential)\n")


# In[459]:


f_mu = f_params(**params)['mu']


# In[460]:


def plot_f_mu(i):
    x, y = trijunction.sites[i].pos
    return f_mu(x, y)


# In[461]:


kwant_sites = np.array(list(site.pos for site in trijunction.sites))


# In[462]:


to_plot = [plot_f_mu(i) for i in range(len(kwant_sites))]


# In[463]:


grid_points = poisson_system.grid.points
voltage_regions = poisson_system.regions.voltage.tag_points


# In[464]:


pm.bands[0]


# In[465]:


fig, ax = plt.subplots(1, 1, figsize = (7, 5))
cax = ax.scatter(kwant_sites[:, 0 ], kwant_sites[:, 1], 
                 c = np.array(to_plot), cmap = 'inferno_r', 
                );
ax.set_ylim(-0.5e-6, 1.0e-6);
plt.colorbar(cax);
# plt.axis('equal')


# In[257]:


get_ipython().run_cell_magic('time', '', '\nbase_hamiltonian, linear_ham = linear_Hamiltonian(poisson_params, \n                                                  kwant_params, \n                                                  params, \n                                                  list(voltage_regions.keys()),\n                                                  phis = [optimal_phase[0], np.pi])\n')


# In[371]:


voltages


# In[434]:


voltages['left_2'] -= 0.001
voltages['right_1'] = voltages['left_2']


# In[435]:


voltages


# In[436]:


summed_ham = sum(
    [
        linear_ham[key] * voltages[key]
        for key, value in linear_ham.items()
    ]
)

tight_binding_hamiltonian = base_hamiltonian + summed_ham


eigval, eigvec = sl.sort_eigen(sparse.linalg.eigsh(tight_binding_hamiltonian.tocsc(), 
                                                   k=12, sigma = 0))


# In[437]:


eigval = eigval
eigvec = eigvec

basis_sequence = []
for vec in eigvec[:3]:
    basis_sequence.append(np.argmax([op(vec) for op in [leftop, 
                                                        rightop, 
                                                        topop]]))

basis_sequence = np.hstack(basis_sequence)

eigval[:3] = eigval[:3][basis_sequence]
eigvec[:3] = np.squeeze(eigvec[:3][basis_sequence])density = Density(trijunction, np.eye(4))
# In[457]:


fig, ax = plt.subplots(1, len(eigvec[3:9]), figsize = (5, 5), sharey = True)
for i in range(len(eigvec[3:9])):
    cax = kwant.plotter.density(trijunction, density(eigvec[3:9][i]), ax = ax[i]);


# #### Wannierization

# In[442]:


from utility import wannier_effective_Hamiltonian


# In[443]:


X_operator = Density(
            trijunction, onsite=lambda site: np.eye(4) * site.pos[0], sum=True
        )

Y_operator = Density(
            trijunction, onsite=lambda site: np.eye(4) * site.pos[1], sum=True
        )


# In[444]:


projected_X_operator = wannier_1D_operator(X_operator, 
                                           eigvec[3:9].T)

projected_Y_operator = wannier_1D_operator(Y_operator, 
                                           eigvec[3:9].T)

import scipy
# In[446]:


w_basis = wannier_basis([projected_X_operator, 
                         projected_Y_operator])

wannier_eigvec = w_basis.T @ eigvec[3:9]


# In[448]:


fig, ax = plt.subplots(1, 6, figsize = (5, 5), sharey = True)
for i in range(6):
    cax = kwant.plotter.density(trijunction, density(wannier_eigvec[i]), ax = ax[i]);


# ### Effective Majorana basis

# In[449]:


S = eigvec[3:9] @ wannier_eigvec.T.conj()
U, _, Vh = svd(S)
S_prime = U @ Vh
transformation = S_prime.T.conj() @ np.diag(eigval[3:9]) @ S_prime


# In[455]:


Matrix(np.round(transformation/params['Delta'], 10))


# #### Linear part of the Hamiltonian

# In[56]:


from tqdm import tqdm


# In[53]:


def linear_Hamiltonian(
    poisson_params,
    kwant_params,
    general_params,
    gates,
    phis = [0.0, 0.0]
):

    ## non-linear part of the Hamiltonian
    
    voltages = {}
    
    for gate in gates: voltages[gate] = 0.0
    
    kp = kwant_params
    pp = poisson_params
    
    charges = {}
    potential = gate_potential(
            pp['poisson_system'],
            pp['linear_problem'],
            pp['site_coords'][:, [0, 1]],
            pp['site_indices'],
            voltages,
            charges,
            offset = kp['offset'][[0, 1]],
            grid_spacing = kp['grid_spacing']
        )
    
    
    
    general_params.update(potential=potential)
    general_params['phi1'] = phis[0]
    general_params['phi2'] = phis[1]
    
    bare_hamiltonian = kp['finite_system_object'].hamiltonian_submatrix(
        sparse=True, params=kp['finite_system_params_object'](**general_params))
    
    
    hamiltonian_V = {}
    for gate in tqdm(gates):
        

        voltages_t = dict.fromkeys(voltages, 0.0)

        voltages_t[gate] = 1.0

        potential = gate_potential(
            pp['poisson_system'],
            pp['linear_problem'],
            pp['site_coords'][:, [0, 1]],
            pp['site_indices'],
            voltages_t,
            charges,
            offset = kp['offset'][[0, 1]],
            grid_spacing = kp['grid_spacing']
        )
    

        general_params.update(potential=potential)

        hamiltonian = kp['finite_system_object'].hamiltonian_submatrix(
            sparse=True, params=kp['finite_system_params_object'](**general_params))


        hamiltonian_V[gate] = hamiltonian - bare_hamiltonian

    return bare_hamiltonian, hamiltonian_V


# In[503]:


summed_ham = sum(
        [
            linear_ham[key] * voltages[key]
            for key, value in linear_ham.items()
        ]
    )


# In[504]:


pp = poisson_params
kp = kwant_params
potential = gate_potential(
    pp['poisson_system'],
    pp['linear_problem'],
    pp['site_coords'][:, [0, 1]],
    pp['site_indices'],
    voltages,
    charges,
    offset = kp['offset'][[0, 1]],
    grid_spacing = kp['grid_spacing']
)


params.update(potential=potential)

test_hamiltonian = kp['finite_system_object'].hamiltonian_submatrix(
    sparse=True, params=kp['finite_system_params_object'](**params))


# In[578]:


potential = gate_potential(
    pp['poisson_system'],
    pp['linear_problem'],
    pp['site_coords'],
    pp['site_indices'],
    voltages,
    charges,
    offset = kp['offset'],
    grid_spacing = kp['grid_spacing']
)


# In[510]:


potential = gate_potential(
    pp['poisson_system'],
    pp['linear_problem'],
    pp['site_coords'][:, [0, 1]],
    pp['site_indices'],
    voltages,
    charges,
    offset = kp['offset'][[0, 1]],
    grid_spacing = kp['grid_spacing']
)


# In[519]:


np.unique(grid_points[poisson_system.regions.charge.tag_points['twoDEG']][:, 2])


# In[575]:


site_tuples = map(tuple, site_coords[:, [0, 1]])

len(list(site_tuples))
# In[576]:


from potential import _linear_solution
solution = _linear_solution(poisson_system, linear_problem, voltages, {})

potential = solution[site_indices]


# In[577]:


potential_dict = {}
for i, site_tuple in enumerate(site_tuples):
    if grid_spacing == 1:
        potential_dict[ta.array((site_tuple-offset))] = potential[i]
    else:
        potential_dict[ta.array((site_tuple-offset)*kp['grid_spacing'])] = potential[i]


# In[565]:


len(potential_dict)


# In[ ]:




