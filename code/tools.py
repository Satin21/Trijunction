import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import matplotlib.animation as animation
import sys


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