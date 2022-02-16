import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import matplotlib.animation as animation
import sys
import tinyarray as ta


def get_potential(potential):
    def f(x, y):
        return potential[ta.array([x, y])]
    return f

def find_cuts(potentials, cut=10e-9, scale=1):

    flatten_potentials = np.array(list(potentials.values()))
    potential_cuts = []

    for element in flatten_potentials:
        coordinates = np.array(list(element.keys()))
        values = np.array(list(element.values()))

        x = scale*coordinates[:, 0]
        y = scale*coordinates[:, 1]
        width = np.unique(x).shape[0]
        X = x.reshape(width, -1)
        Y = y.reshape(width, -1)
        Z = sum([values]).reshape(width, -1)

        poten = Z.T[np.argwhere(Y[0] == cut)][0][0]
        potential_cuts.append(poten)

    return X[:, 0], potential_cuts


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


def find_resonances(energies, n_peaks, n=6, i=1, sign=1, **kwargs):
    """
    Extract peaks from np.abs(lowest) mode in energies.
    Cy choosing 'sign' we extract either peaks or dips.
    Parameters:
    -----------
    """
    levels = energies.T
    ground_state = check_level_smooth(check_level_smooth_v2(levels[n//2 + i]))
    crossings, _ = find_peaks(sign*np.abs(ground_state), **kwargs)

    return crossings[:n_peaks], ground_state


def coupling_data(data, n_peaks=-1, n=6, i=1, eigenvecs=False, **kwargs):
    """
    Extract data associated to the first excited level, i.e. coupled majoranas,
    for each connection in the trijunction.
    """
    dat_13, dat_12, dat_23 = separate_data_wires(data)

    en_13, wfs_13 = separate_energies_wfs(dat_13)
    en_12, wfs_12 = separate_energies_wfs(dat_12)
    en_23, wfs_23 = separate_energies_wfs(dat_23)

    peaks_13, coupling_13 = find_resonances(en_13, n_peaks=n_peaks, i=i, **kwargs)
    peaks_12, coupling_12 = find_resonances(en_12, n_peaks=n_peaks, i=i, **kwargs)
    peaks_23, coupling_23 = find_resonances(en_23, n_peaks=n_peaks, i=i, **kwargs)
    couplings = [coupling_13, coupling_12, coupling_23]

    peaks_13 = check_peaks(peaks_13, coupling_13, n_peaks=n_peaks)
    peaks_12 = check_peaks(peaks_12, coupling_12, n_peaks=n_peaks)
    peaks_23 = check_peaks(peaks_23, coupling_23, n_peaks=n_peaks)
    peaks = [peaks_13, peaks_12, peaks_23]

    if not eigenvecs:
        wfs = []
    else:
        wfs = [wfs_13[:, n//2+1], wfs_12[:, n//2+1], wfs_23[:, n//2+1]]

    return couplings, wfs, peaks


def check_peaks(peaks, coupling, n_peaks):
    """
    Check if the number of peaks is the correct, otherwise return random positions.
    This happens when the coupling is negligible.
    """

    if len(peaks) == n_peaks:
        peaks = peaks
    #elif 0 < len(peaks) < n_peaks:
     #   n_missing = n_peaks - len(peaks)
       # peaks = np.append(peaks, np.zeros(n_missing, dtype=int))
    elif len(peaks) == 0:
        peaks = np.arange(n_peaks, dtype=int)

    return peaks


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