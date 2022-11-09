import numpy as np
from scipy.constants import electron_mass, hbar
from collections.abc import Mapping
from codes.constants import bands, voltage_keys, pairs
from collections import OrderedDict
from copy import copy


def voltage_dict(x, dirichlet=True):
    """Return dictionary of gate voltages
    x: list
    voltages

    dirichlet: bool
    Condition to add dirichlet gates
    """

    voltages = {key: x[index] for key, index in voltage_keys.items()}

    if dirichlet:
        for i in range(6):
            voltages["dirichlet_" + str(i)] = 0.0

    return voltages


def pair_voltages(initial=(-1.5e-3, -1.5e-3, -1.5e-3, 8e-3), depleted=-3.5e-3):
    """ 
    Returns voltage dictionary for every pair 
    
    Parameters:
    ----------
    initial: tuple or list with 4 elements
    
    depleted: float
    Value for the gate associated with the region to be depleted
    
    """
    pairs = ["right-top", "left-top", "left-right"]
    voltages = OrderedDict()
    initial_condition = OrderedDict()

    for i, pair in enumerate(pairs):
        initial_copy = np.array(np.copy(initial))
        initial_copy[i] = depleted
        voltages[pair] = voltage_dict(initial_copy, True)
        initial_condition[pair] = initial_copy

    return voltages


def voltage_initial_conditions(guess=(-3e-3, -3e-3, -3e-3, 10e-3)):
    """
    Find initial condition for the voltages based on the soft-threshold.
    """
    initial_conditions = {}
    for i, pair in enumerate(pairs):
        x = list(copy(guess))
        x[i] = -10e-3
        initial_conditions[pair] = x
    return initial_conditions


def junction_parameters(m_nw=[bands[0]] * 3, bx=0.001):
    """
    Typical parameters
    """

    a = 10e-9
    t = hbar**2 / (2 * 0.023 * electron_mass) * (6.24e18)
    alpha = 3e-3*a
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
    """
    Update parent dictionary with many child branches inside
    
    d: dict
    Parent dictionary
    u: dict
    Child dictionary 
    """
    # https://stackoverflow.com/a/3233356
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
