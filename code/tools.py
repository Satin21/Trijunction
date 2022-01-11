import kwant
import kwant.continuum
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import scipy.sparse.linalg as sla
from scipy.constants import electron_mass, hbar
import matplotlib.animation as animation
import sys

# data extracted from sub_bands.ipynb
# Zeeman fields for the topological transition at each sub-band


# Bottom of each transverse band
bands = [
    0.0023960204649275973,
    0.009605416498312178,
    0.020395040147213304,
    0.03312226901926766,
    0.045849497891322026,
    0.056639121540223145,
    0.06384851757360771
]
b = 0.001
phi12 = 1.23232323*np.pi
phi13 = 0.02020202*np.pi
phi23 = 1.97979798*np.pi


def finite_coupling_parameters(index):
    mu = bands[index]
    params_12 = {'mus_nw': np.array([mu, mu, -2]), 'phi1': phi12}
    params_13 = {'mus_nw': np.array([mu, -2, mu]), 'phi2': phi13}
    params_23 = {'mus_nw': np.array([-2, mu, mu]), 'phi2': phi23}
    params = [params_12, params_13, params_23]
    return params

# helper functions
def lead_parameters(m_nw, m_qd, B):
    a = 10E-9
    t = hbar**2/(2*0.023*electron_mass)*(6.24E18)
    alpha = 0.4E-10
    Delta = 5E-3
    parameters = dict(mu_nw_1=m_nw[0],
                      mu_nw_2=m_nw[1],
                      mu_nw_3=m_nw[2],
                      mu_qd=m_qd,
                      t=t,
                      Delta=Delta,
                      alpha=alpha,
                      B_x=B,
                      phi=0,
                      a=a)
    return parameters


def junction_parameters(m_nw, m_qd, bx=b):
    """
    Typical parameters
    """

    a = 10E-9
    t = hbar**2/(2*0.023*electron_mass)*(6.24E18)
    alpha = 0.3E-10
    Delta = 5E-4
    parameters = dict(mus_nw=m_nw,
                      mu_qd=m_qd,
                      t=t,
                      Delta=Delta,
                      alpha=alpha,
                      B_x=bx,
                      phi1=0,
                      phi2=0,
                      a=a)
    return parameters


def solver(geometries, n, key, eigenvecs=False):

    def eigensystem_sla(geometry_index, mu, extra_params):

        params = junction_parameters(m_nw=np.array([-2, -2, -2]), m_qd=0)
        params.update(extra_params)
        params[key] = mu

        system, params_func = geometries[geometry_index]

        ham_mat = system.hamiltonian_submatrix(sparse=True, params=params_func(**params))

        if not eigenvecs:
            evals = np.sort(sla.eigsh(ham_mat.tocsc(), k=n, sigma=0, return_eigenvectors=False))
            evecs = []
        else:
            evals, evecs = sort_eigen(sla.eigsh(ham_mat.tocsc(), k=n, sigma=0))

        return evals, evecs

    return eigensystem_sla


def sort_eigen(ev):
    """
    Sort eigenvectors and eigenvalues using numpy methods.
    """
    evals, evecs = ev
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    return evals, evecs.T


def check_level_smooth(level):
    
    diffs = np.abs(np.diff(level))
    lowest_value = np.abs(np.min(level))
    condition = diffs > 0.5e-4
    
    indices = np.where(condition)[0]
    if indices.shape != 0:
        level[indices] = lowest_value

    return level

def check_level_smooth_v2(level):
    
    lowest_value = np.abs(np.min(level))
    indices = np.where(np.abs(level) > 2e-4)[0]
    if indices.shape != 0:
        level[indices] = lowest_value

    return level


def find_resonances(energies, n, n_peaks, i=1, sign=-1, prominence=0):
    levels = energies.T
    ground_state = check_level_smooth(check_level_smooth_v2(levels[n//2 + i]))
    if prominence > 0:
        crossings, _ = find_peaks(sign*np.abs(ground_state), prominence=prominence)
    else:
        crossings, _ = find_peaks(sign*np.abs(ground_state))

    return crossings[:n_peaks], ground_state


def extract_peaks(geometries_peaks, geometries_couplings, geometries_widths):
    peaks = []
    widths = []
    n_geometries = geometries_couplings.shape[0]
    for pair in range(3):
        peaks_pair = geometries_peaks[:, pair]
        couplings_pair = geometries_couplings[:, pair]
        widths_pair = geometries_widths[:, pair]
        n_peaks = peaks_pair.shape[1]

        geometry_peaks_pair = []
        geometry_widths_pair = []

        for i in range(n_geometries):
            geometry_peaks_pair.append(couplings_pair[i][peaks_pair[i]])
            geometry_widths_pair.append(widths_pair[i][:n_peaks])

        geometry_peaks_pair = np.array(geometry_peaks_pair).T
        peaks.append(geometry_peaks_pair)
        
        geometry_widths_pair = np.array(geometry_widths_pair).T
        widths.append(geometry_widths_pair)

    return np.array(peaks), np.array(widths)


def separate_data_wires(data):
    """
    data is formatted such that the first dimension corresponds to each of
    three possible tunnel configurations.
    This functions separates each case in a different array.
    """
    len_data = int(len(data)/3)
    w12 = []
    w13 = []
    w23 = []
    i = 0
    for dat in data:
        if i == 3:
            i = 0
        if i == 0:
            w13.append(dat)
        elif i == 1:
            w12.append(dat)
        elif i == 2:
            w23.append(dat)
        i += 1
    return w13, w12, w23


def separate_energies_wfs(data):
    """
    data is formated such that the first index represents a parameter value,
    and the second index separates energies and wavefunctions.
    """
    ens = []
    wfs = []
    for dat in data:
        ens.append(dat[0])
        wfs.append(dat[1])
    return np.array(ens), np.array(wfs)


def spectra_data(data):
    """
    Extract spectras for each connection in the trijunction.
    """
    dat_13, dat_12, dat_23 = separate_data_wires(data)
    en_13, wfs_13 = separate_energies_wfs(dat_13)
    en_12, wfs_12 = separate_energies_wfs(dat_12)
    en_23, wfs_23 = separate_energies_wfs(dat_23)
    energies = [en_13, en_12, en_23]
    wfs = [wfs_13, wfs_12, wfs_23]
    return energies, wfs


def coupling_data(data, n_peaks=-1, n=6, i=1, sign=-1, eigenvecs=False, prominence=0):
    """
    Extract data associated to the first excited level, i.e. coupled majoranas,
    for each connection in the trijunction.
    """
    dat_13, dat_12, dat_23 = separate_data_wires(data)
    en_13, wfs_13 = separate_energies_wfs(dat_13)
    en_12, wfs_12 = separate_energies_wfs(dat_12)
    en_23, wfs_23 = separate_energies_wfs(dat_23)
    peaks_13, coupling_13 = find_resonances(en_13, n=n, n_peaks=n_peaks, i=i, sign=sign, prominence=prominence)
    peaks_12, coupling_12 = find_resonances(en_12, n=n, n_peaks=n_peaks, i=i, sign=sign, prominence=prominence)
    peaks_23, coupling_23 = find_resonances(en_23, n=n, n_peaks=n_peaks, i=i, sign=sign, prominence=prominence)
    couplings = [coupling_13, coupling_12, coupling_23]
    if not eigenvecs:
        wfs = []
    else:
        wfs = [wfs_13[:, n//2+1], wfs_12[:, n//2+1], wfs_23[:, n//2+1]]
    peaks = [peaks_13, peaks_12, peaks_23]
    return couplings, wfs, peaks


def separate_data_geometries(data, n_geometries):
    separated_data = []
    step = int(len(data)/n_geometries)
    for i in range(n_geometries):
        separated_data.append(data[i*step:(i+1)*step])
    return separated_data


def first_minimum(coupling, first_index):
    differences = np.abs(np.diff(coupling[:first_index]))
    index = np.where(differences > 1e-7)[0]
    if index.size == 0:
        index = np.where(differences > 1e-10)[0]
        if index.size == 0:
            return 0
    return index[0]


def average_energy_levels(mus, result, prominence=0):
    geometry_data = []
    ens, _, peaks = coupling_data(result, prominence=prominence)

    for k in range(3):
        single_average = []
        single_mus = []
        single_widths = []
        first_index = first_minimum(ens[k], peaks[k][0])
        single_peaks = np.hstack([first_index, peaks[k], -1])
        indices_intervals = zip(single_peaks, single_peaks[1:])

        for i, j in indices_intervals:
            energy_section = np.abs(ens[k])[i:j]
            ediff = np.abs(mus[j]-mus[i])
            average = np.sum(energy_section)/np.abs(ediff)
            single_average.append(average)
            single_mus.append(mus[i]+ediff/2)
            single_widths.append(ediff)
        minimal_spacing = min(single_widths)
        geometry_data.append([single_mus, single_average, single_widths, minimal_spacing])

    return np.array(geometry_data)