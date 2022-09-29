import numpy as np
from scipy.constants import electron_mass, hbar
from collections.abc import Mapping
from codes.constants import bands, voltage_keys
from collections import OrderedDict


def voltage_dict(x, dirichlet=True):
    """Return dictionary of gate voltages
    x: list
    voltages

    dirichlet: bool
    Whether to add dirichlet gates
    """

    voltages = {key: x[index] for key, index in voltage_keys.items()}

    if dirichlet:
        for i in range(6):
            voltages["dirichlet_" + str(i)] = 0.0

    return voltages


def pair_voltages(initial=(-1.5e-3, -1.5e-3, -1.5e-3, 8e-3), depleted=-3.5e-3):
    """ """
    pairs = ["right-top", "left-top", "left-right"]
    voltages = OrderedDict()
    initial_condition = OrderedDict()

    for i, pair in enumerate(pairs):
        initial_copy = np.array(np.copy(initial))
        initial_copy[i] = depleted
        voltages[pair] = voltage_dict(initial_copy, True)
        initial_condition[pair] = initial_copy

    return voltages


def junction_parameters(m_nw=[bands[0]] * 3, bx=0.001):
    """
    Typical parameters
    """

    a = 10e-9
    t = hbar**2 / (2 * 0.023 * electron_mass) * (6.24e18)
    alpha = 0.3e-10
    Delta = 5e-4
    parameters = dict(
        mus_nw=m_nw,
        t=t,
        Delta=Delta,
        alpha=alpha,
        B_x=bx,
        phi1=0,
        phi2=0,
        sigma=0,
        a=a,
    )
    return parameters


def phase_pairs(pair, phi):
    if pair == "right-top":
        return {"phi2": phi, "phi1": 0}
    if pair == "left-top":
        return {"phi2": phi, "phi1": 0}
    if pair == "left-right":
        return {"phi1": phi, "phi2": 0}


def dict_update(d, u):
    # https://stackoverflow.com/a/3233356
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
