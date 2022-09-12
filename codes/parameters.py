import numpy as np
from scipy.constants import electron_mass, hbar

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
    0.06384851757360771,
]

b = 0.001

phi12 = 1.23232323 * np.pi
phi13 = 0.02020202 * np.pi
phi23 = 1.97979798 * np.pi
phis = np.array([1.23232323, 0.02020202, 1.97979798]) * np.pi


def pairs_parameters(index, phis1=[0, 0, 0], phis2=[0, 0, 0]):
    mu = bands[index]
    params_LR = {
        "mus_nw": np.array([mu, mu, -2]),
        "phi1": phis1[0],
        "phi2": phis2[0],
        "pair": "LR",
    }
    params_LC = {
        "mus_nw": np.array([mu, -2, mu]),
        "phi1": phis1[1],
        "phi2": phis2[1],
        "pair": "CR",
    }
    params_CR = {
        "mus_nw": np.array([-2, mu, mu]),
        "phi2": phis1[2],
        "phi1": phis2[2],
        "pair": "LC",
    }
    params = [params_LR, params_LC, params_CR]
    return params


def phase(pair, phis=phis):

    if pair == 0:
        key_phi = "phi1"
    else:
        key_phi = "phi2"

    extra_params = {key_phi: phis[pair]}
    return extra_params


def phase_params(band_index=0, n=100):
    """ """
    phases = np.linspace(0, 2 * np.pi, n)
    params = []

    for phase in phases:

        phis1 = [phase, 0, 0]
        phis2 = [0, phase, phase]
        wires = pairs_parameters(band_index, phis1=phis1, phis2=phis2)
        for wire in wires:
            params.append(wire)

    return params


def single_parameter(key, vals, max_phis, offset):
    """
    Make an array of parameters for three pairs of nanowires with the phase
    differences tunned to a given value.
    Parameters:
    -----------
        key: str to be varied
        vals: np.array with values to be changed
        max_phis: np.array with three pase differences for each nanowire pair

    Returns:
    --------
        params: array of parameters
    """
    params = []
    wires = pairs_parameters(index=0, phis=np.pi * max_phis)
    for val in vals:
        for wire in wires:
            dic = {key: val, "offset": offset}
            params.append(wire | dic)
    return params


def junction_parameters(m_nw, bx=b):
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
