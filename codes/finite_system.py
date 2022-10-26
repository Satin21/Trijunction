import kwant
import kwant.continuum
import numpy as np
import tinyarray as ta

from codes.constants import scale

rounding_limit = 3

hamiltonian = """( t * (k_x**2 + k_y**2 ) - mu(x,y) )* kron(sigma_0, sigma_z)
+ alpha * k_x * kron(sigma_y, sigma_z)
- alpha * k_y * kron(sigma_x, sigma_z)
+ Delta_re(x,y) * kron(sigma_0, sigma_x)
+ Delta_im(x,y) * kron(sigma_0, sigma_y)
+ B_x * kron(sigma_y, sigma_0)"""


def finite_system(**geometry):
    """
    Create a kwant builder that describes three wires connected by a cavity as defined in geometry.
    The builder is filled with a discretized continuum hamiltonian.

    Parameters:
    -----------
        geometry: dictionary containing overall system geometry.

    Returns:
    --------
        trijunction: kwant.FiniteSystem
        f_params: function
    """

    l = geometry["nw_l"]
    w = geometry["nw_w"]
    L = geometry["s_l"]
    W = geometry["s_w"]
    centers = geometry["centers"]

    def wires(mu=np.ones(3) > 0):
        """
        Function that span the wires region with mu.
        """

        def wires_shape(x, y):
            for i in range(3):
                x0, y0 = centers[i]

                if (
                    x0 - w / 2 - (scale * geometry["grid_spacing"] / 2)
                    <= x
                    <= x0 + w / 2 + (scale * geometry["grid_spacing"] / 2)
                ):
                    if y0 - l <= y <= y0:
                        return mu[i]
            return 0

        return wires_shape

    def fill_system(mus_nw, scattering_potential={}):
        """
        Make a spatillay dependent function that has flat potential in the wires region,
        and the potential in the cavity is taken as an input parameter `shape`.
        """

        def system(x, y):

            if -W / 2 < x < W / 2 and 0 <= y <= L:
                f = get_potential(scattering_potential)
            else:
                f = wires(mu=mus_nw)
            return f(x, y)

        return system

    def f_params(**params):

        mus_nw = params.pop("mus_nw")
        Delta = params.pop("Delta")
        phi1 = params.pop("phi1")
        phi2 = params.pop("phi2")
        potential = params.pop("potential")
        f_chemical_potential = fill_system(
            scattering_potential=potential, mus_nw=mus_nw
        )
        f_Delta_re = wires(mu=Delta * np.array([1, np.cos(phi1), np.cos(phi2)]))
        f_Delta_im = wires(mu=Delta * np.array([0, np.sin(phi1), np.sin(phi2)]))

        params.update(mu=f_chemical_potential)
        params.update(Delta_re=f_Delta_re)
        params.update(Delta_im=f_Delta_im)

        return params

    def make_junction(**geometry):
        """Create finalized Builder of a rectangle filled with template"""

        def rectangle(site):
            x, y = site.pos
            if (-W / 2 < x < W / 2 and 0 <= y <= L) or wires()(x, y):
                return True

        junction = kwant.Builder()

        template = kwant.continuum.discretize(
            hamiltonian, grid=scale * geometry["grid_spacing"]
        )

        junction.fill(template, shape=rectangle, start=[0, 0])
        return junction

    trijunction = make_junction(**geometry)

    return trijunction, f_params


def kwantsystem(config, boundaries, nw_centers, scale=1e-8):

    """

    Builds a tight binding Kwant system with scattering region and nw leads



    """

    a = scale
    l = config["kwant"]["nwl"]
    w = config["kwant"]["nww"]
    shift = np.array([0, l])

    geometry = {
        "grid_spacing": config["device"]["grid_spacing"]["twoDEG"],
        "nw_l": l * a,
        "nw_w": w * a,
        "s_w": (boundaries["xmax"] - boundaries["xmin"]) * a,
        "s_l": (boundaries["ymax"] - boundaries["ymin"]) * a,
        "centers": [
            nw_centers["left"] * a,
            nw_centers["right"] * a,
            nw_centers["top"] * a + shift * a,  # shift top center to the far end
        ],
    }

    ## Discretized Kwant system

    trijunction, f_params = finite_system(**geometry)
    trijunction = trijunction.finalized()

    trijunction = trijunction
    f_params = f_params

    return geometry, trijunction, f_params


def get_potential(potential):
    def f(x, y):
        return potential[ta.array(np.round(np.array([x, y]) / scale, rounding_limit))]

    return f
