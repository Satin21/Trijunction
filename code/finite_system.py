import kwant
import kwant.continuum
import numpy as np

a = 10e-9

hamiltonian = """( t * (k_x**2 + k_y**2 ) - mu(x,y) )* kron(sigma_0, sigma_z)
+ alpha * k_x * kron(sigma_y, sigma_z)
- alpha * k_y * kron(sigma_x, sigma_z)
+ Delta_re(x,y) * kron(sigma_0, sigma_x)
+ Delta_im(x,y) * kron(sigma_0, sigma_y)
+ B_x * kron(sigma_y, sigma_0)"""

template = kwant.continuum.discretize(hamiltonian, grid=a)


def finite_system(potential, **geometry):
    """
    Create a kwant builder that describes three wires connected by a cavity as defined in geometry.
    The builder is filled with a discretized continuum hamiltonian.

    Parameters:
    -----------
        shape: function that defines the potential in the cavity.
        geometry: dictionary containing overall system geometry.

    Returns:
    --------
        trijunction: kwant.FiniteSystem
        f_params: function
    """

    l = geometry['l']
    w = geometry['w']
    L = geometry['L']
    W = geometry['W']
    centers = geometry['centers']

    def wires(mu=np.ones(3) > 0):
        """
        Function that span the wires region with mu.
        """
        def wires_shape(x, y):

            for i in range(3):
                x0, y0 = centers[i]
                if -w/2 - x0 < x < w/2 - x0:
                    if y0 - l <= y <= y0:
                        return mu[i]
            return 0

        return wires_shape


    def fill_system(mu_qd, mus_nw, pair, offset):
        """
        Make a spatillay dependent function that has flat potential in the wires region,
        and the potential in the cavity is taken as an input parameter `shape`.
        """
        def system(x, y):

            if (-W/2 < x < W/2 and 0 <= y < L):
                f = side_geometry(potential(mu_qd), pair, offset)
            else:
                f = wires(mu=mus_nw)
            return f(x,y)

        return system

    
    def side_geometry(potential, pair, offset):

        def f(x, y):
            if pair=='LR' or (pair=='LC' and x<offset) or (pair=='CR' and x>-offset):
                return potential(x, y)
            else:
                return -2
        return f

    def f_params(**params):

        mus_nw = params.pop('mus_nw')
        mu_qd = params.pop('mu_qd')
        Delta = params.pop('Delta')
        phi1 = params.pop('phi1')
        phi2 = params.pop('phi2')
        pair = params.pop('pair')
        offset = params.pop('offset')
        f_chemical_potential = fill_system(mu_qd=mu_qd, mus_nw=mus_nw, pair=pair, offset=offset)
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
            if (-W/2 < x < W/2 and 0 <= y < L) or wires()(x, y):
                return True
        
        junction = kwant.Builder()
        junction.fill(
            template,
            shape=rectangle,
            start=[0,0]
        )
        return junction

    trijunction = make_junction(**geometry)

    return trijunction, f_params