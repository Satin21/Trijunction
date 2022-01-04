import kwant
import kwant.continuum
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import scipy.sparse.linalg as sla
from scipy.constants import electron_mass, hbar
import matplotlib.animation as animation

# data extracted from sub_bands.ipynb
# Zeeman fields for the topological transition at each sub-band


# Bottom of each transverse band
bands = [0.0023960204649275973,
 0.009605416498312178,
 0.020395040147213304,
 0.03312226901926766,
 0.045849497891322026,
 0.056639121540223145,
 0.06384851757360771]

b = 0.0106

def finite_coupling_parameters(index):
    mu = bands[index]
    params1 = junction_parameters(m_nw=np.array([mu, mu, -2]), m_qd=0, bx=b)
    params1.update(phi1=1.23232323*np.pi)
    params2 = junction_parameters(m_nw=np.array([mu, -2, mu]), m_qd=0, bx=b)
    params2.update(phi2=0.02020202*np.pi)
    params3 = junction_parameters(m_nw=np.array([-2, mu, mu]), m_qd=0, bx=b)
    params3.update(phi2=1.97979798*np.pi)
    params = [params1, params2, params3]
    return params

def infinite_coupling_parameters(index):
    mu = bands[index]
    b = fields_majorana[index]
    params1 = lead_parameters(m_nw=np.array([mu, mu, -2]), m_qd=0, bx=b)
    params1.update(phi1=1.23232323*np.pi)
    params2 = lead_parameters(m_nw=np.array([mu, -2, mu]), m_qd=0, bx=b)
    params2.update(phi2=0.02020202*np.pi)
    params3 = lead_parameters(m_nw=np.array([-2, mu, mu]), m_qd=0, bx=b)
    params3.update(phi2=1.97979798*np.pi)
    params = [params1, params2, params3]
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


def junction_parameters(m_nw, m_qd, bx):
    """
    Typical parameters
    """

    a = 10E-9
    t = hbar**2/(2*0.023*electron_mass)*(6.24E18)
    alpha = 0.4E-10
    Delta = 0.01
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


def solver(n, key):

    def eigensystem_sla(tj_system, mu, params):

        system, params_func = tj_system
        params[key] = mu

        ham_mat = system.hamiltonian_submatrix(sparse=True, params=params_func(**params))
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


def find_resonances(energies, n, i=1, sign=-1):
    levels = energies.T
    ground_state = levels[n//2 + i]
    crossings, _ = find_peaks(sign*np.abs(ground_state))
    return crossings, ground_state


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


def coupling_data(data, n=20, i=1, sign=-1):
    """
    Extract data associated to the first excited level, i.e. coupled majoranas,
    for each connection in the trijunction.
    """
    dat_13, dat_12, dat_23 = separate_data_wires(data)
    en_13, wfs_13 = separate_energies_wfs(dat_13)
    en_12, wfs_12 = separate_energies_wfs(dat_12)
    en_23, wfs_23 = separate_energies_wfs(dat_23)
    peaks_13, coupling_13 = find_resonances(en_13, n=n, i=i, sign=sign)
    peaks_12, coupling_12 = find_resonances(en_12, n=n, i=i, sign=sign)
    peaks_23, coupling_23 = find_resonances(en_23, n=n, i=i, sign=sign)
    couplings = [coupling_13, coupling_12, coupling_23]
    wfs = [wfs_13[:, n//2+1], wfs_12[:, n//2+1], wfs_23[:, n//2+1]]
    peaks = [peaks_13, peaks_12, peaks_23]
    return couplings, wfs, peaks


def separate_data_geometries(data, n_geometries):
    separated_data = []
    step = int(len(data)/n_geometries)
    for i in range(n_geometries):
        separated_data.append(data[i*step:(i+1)*step])
    return separated_data


def average_energy_levels(mus, result, normalized=True):
    geometry_data = []
    ens, _, peaks = coupling_data(result, n=6)

    for k in range(3):
        single_average = []
        single_mus = []
        single_widths = []
        single_peaks = np.hstack([0, peaks[k], -1])
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