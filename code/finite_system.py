import kwant
import kwant.continuum
import numpy as np

def finite_system(**geometry):
    """
    Create a kwant builder that describes three wires connected by a cavity as defined in geometry.
    The builder is filled with a discretized continuum hamiltonian.

    Parameters:
    -----------
        geometry: dict

    Returns:
    --------
        trijunction: kwant.FiniteSystem
        f_params: function
    """

    hamiltonian = """( t * (k_x**2 + k_y**2 ) - mu(x,y) )* kron(sigma_0, sigma_z)
    + alpha * k_x * kron(sigma_y, sigma_z)
    - alpha * k_y * kron(sigma_x, sigma_z)
    + Delta_re(x,y) * kron(sigma_0, sigma_x)
    + Delta_im(x,y) * kron(sigma_0, sigma_y)
    + B_x * kron(sigma_y, sigma_0)"""
    template = kwant.continuum.discretize(hamiltonian, grid=geometry['a'])

    def wires(mu=np.ones(3) > 0, **geometry):

        L = geometry['L']
        W = geometry['W']

        l = geometry['l']
        w = geometry['w']

        def wires_shape(x, y):
            center = W/4  # center of the wires in the lower region
            for i in range(2):
                if -w/2 - center*(-1)**i < x < w/2 - center*(-1)**i:
                    if -l <= y < 0:
                        return mu[i]
            i+=1
            if L <= y < L+l:
                if -w/2 <= x < w/2:
                    return mu[i]

        return wires_shape

    def scatter(mu=True, **geometry):

        L = geometry['L'] + 3*geometry['a']  # offset added to connect top wire smoothly
        W = geometry['W']
        angle = geometry['angle']

        def scatter_shape(x, y):
            if -W/2 <= x <= W/2:
                if angle <= np.pi/4:
                    if np.tan(angle)*x <= -(y-L) and np.tan(angle)*x >= (y-L):
                        return mu
                elif angle > np.pi/4:
                    if np.tan(angle)*(x-W/2) <= -y and np.tan(angle)*(x+W/2) >= y:
                        return mu

        return scatter_shape

    def builder_shape(**geometry):
        """Return a function used to create builder as TJ shape."""

        L = geometry['L']

        def system(site):
            x, y = site. pos
            if 0 <= y < L:
                f = scatter(**geometry)
            else:
                f = wires(**geometry)

            return f(x, y)

        return system

    def fill_system(mu_qd, mus_nw):

        L = geometry['L']

        def system(x, y):
            if 0 <= y < L:
                f = scatter(**geometry, mu=mu_qd)
            else:
                f = wires(**geometry, mu=mus_nw)

            return f(x, y)

        return system

    def f_params(**params):

        mus_nw = params.pop('mus_nw')
        mu_qd = params.pop('mu_qd')
        Delta = params.pop('Delta')
        phi1 = params.pop('phi1')
        phi2 = params.pop('phi2')

        f_chemical_potential = fill_system(mu_qd=mu_qd, mus_nw=mus_nw)
        f_Delta_re = fill_system(mu_qd=0, mus_nw=Delta * np.array([1, np.cos(phi1), np.cos(phi2)]))
        f_Delta_im = fill_system(mu_qd=0, mus_nw=Delta * np.array([0, np.sin(phi1), np.sin(phi2)]))

        params.update(mu=f_chemical_potential)
        params.update(Delta_re=f_Delta_re)
        params.update(Delta_im=f_Delta_im)

        return params

    def make_junction(**geometry):
        """Create finalized Builder of a rectangle filled with template"""
        junction = kwant.Builder()
        junction.fill(
            template,
            shape=builder_shape(**geometry),
            start=[0, 0]
        )
        return junction

    trijunction = make_junction(**geometry).finalized()

    return trijunction, f_params