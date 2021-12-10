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

    A = geometry['A']
    a = geometry['a']

    side = geometry['side']
    shape = geometry['shape']

    l = geometry['l']
    w = geometry['w']

    if shape == 'circle':
        R = 50*a  # outer radius
        L = R  # relabel L
        r = np.sqrt(np.abs(A/np.pi - R**2))
        connection = 0
        center = R - (R-r)/2
        start = [R-a, 0]

    elif shape == 'triangle':
        angle = geometry['angle']
        L = np.sqrt(A*np.tan(angle))

        if side == 'up':
            connection = np.tan(angle)*(w/2)
        elif side == 'down':
            connection = 0

        center = np.abs((L/np.tan(angle))/2)  # center of the wires in the lower region

        start = [0,0]

    elif shape == 'rectangle':
        L = np.sqrt(A)/2
        W = 2*L
        center = W/4
        connection = 0
        start = [0, 0]
    #W = np.sqrt(A/np.tan(angle))


    hamiltonian = """( t * (k_x**2 + k_y**2 ) - mu(x,y) )* kron(sigma_0, sigma_z)
    + alpha * k_x * kron(sigma_y, sigma_z)
    - alpha * k_y * kron(sigma_x, sigma_z)
    + Delta_re(x,y) * kron(sigma_0, sigma_x)
    + Delta_im(x,y) * kron(sigma_0, sigma_y)
    + B_x * kron(sigma_y, sigma_0)"""
    template = kwant.continuum.discretize(hamiltonian, grid=geometry['a'])

    def wires(mu=np.ones(3) > 0, **geometry):

        def wires_shape(x, y):

            for i in range(2):
                if -w/2 - center*(-1)**i < x < w/2 - center*(-1)**i:
                    if -l <= y <= 0:
                        return mu[i]
            i+=1
            # last wire can be at top or bot side
            if side == 'up':
                if L-connection <= y < L+l-connection:
                    if -w/2 <= x < w/2:
                        return mu[i]

            elif side == 'down':
                if -l <= y < 0:
                    if -w/2 <= x < w/2:
                        return mu[i]

        return wires_shape

    def triangle(mu=True, **geometry):
        # triangle shape
        def triangle_shape(x, y):
            if np.tan(angle)*x <= -(y-L) and np.tan(angle)*x >= (y-L):
                return mu

        return triangle_shape

    def circle(mu=True, **geometry):
        # circle shape
        def circle_shape(x, y):
            if x**2 + y**2 <= R**2 and x**2 + y**2 >= r**2:
                return mu
        return circle_shape

    def rectangle(mu=True, **geometry):
        def rectangle_shape(x, y):
            if -W/2 <= x <= W/2:
                return mu
        return rectangle_shape

    def builder_shape(**geometry):
        """Return a function used to create builder as TJ shape."""

        def system(site):
            x, y = site. pos
            if 0 <= y < L-connection:
                f = scatter(**geometry)
            else:
                f = wires(**geometry)

            return f(x, y)

        return system

    def scatter(mu=True, **geometry):
        if shape == 'circle':
            return circle(mu, **geometry)
        elif shape == 'triangle':
            return triangle(mu, **geometry)
        elif shape == 'rectangle':
            return rectangle(mu, **geometry)

    def fill_system(mu_qd, mus_nw):

        def system(x, y):
            if 0 <= y < L-connection:
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
            start=start
        )
        return junction

    trijunction = make_junction(**geometry)

    return trijunction, f_params