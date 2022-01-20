import kwant
import kwant.continuum
import numpy as np

def finite_system(**geometry):
    """
    Create a kwant builder that describes three wires connected by a cavity as defined in geometry.
    The builder is filled with a discretized continuum hamiltonian.

    Parameters:
    -----------
        geometry: dictionary
            required parameters:
                a: float
                    grid spacing
                side: string
                    position of the middle wire up or down
                shape: string
                    shape of the cavity region circle, triangle, rectangle
                l: float
                    wire length
                w: float
                    wire width

            geometry dependent parameters:
                circle:
                    R: float
                        outer radius
                    r: float
                        inner radius
                triangle:

    Returns:
    --------
        trijunction: kwant.FiniteSystem
        f_params: function
    """

    a = geometry['a']

    side = geometry['side']
    shape = geometry['shape']

    l = geometry['l']
    w = geometry['w']

    if shape == 'ring':
        R = geometry['R']  # outer radius
        r = geometry['r']  # inner radius
        L = R  # relabel L
        connection = 0
        center = R - (R-r)/2  # wires centered in the middle of the ring
        centers = [center, -center]
        start = [R-a, 0]

    elif shape == 'triangle':
        A = geometry['A']  # triangles defined by the total area
        angle = geometry['angle']
        L = np.sqrt(A*np.tan(angle))

        if side == 'up':
            connection = np.tan(angle)*(w/2)
        elif side == 'down':
            connection = 0

        centers = geometry['centers']

        start = [0,0]
        
    elif shape == 'v':
        A = geometry['A']  # triangles defined by the total area
        angle = geometry['angle']

        L = np.sqrt(A*np.tan(angle))
        W = np.sqrt(np.abs(A/np.tan(angle)))
        
        w_v = geometry['w_v']
        difference = W - w_v
        smaller_area = np.tan(angle)*difference**2
        l_w = np.sqrt(smaller_area*np.tan(angle))

        center = W - w_v/2
        centers = [center, -center]
        connection = np.tan(angle)*(w/2)
        start = [0, L]

    elif shape == 'rectangle':
        L = geometry['L']
        W = geometry['W']
        centers = geometry['centers']
        connection = 0
        start = [0, 0]

    hamiltonian = """( t * (k_x**2 + k_y**2 ) - mu(x,y) )* kron(sigma_0, sigma_z)
    + alpha * k_x * kron(sigma_y, sigma_z)
    - alpha * k_y * kron(sigma_x, sigma_z)
    + Delta_re(x,y) * kron(sigma_0, sigma_x)
    + Delta_im(x,y) * kron(sigma_0, sigma_y)
    + B_x * kron(sigma_y, sigma_0)"""
    template = kwant.continuum.discretize(hamiltonian, grid=a)

    def wires(mu=np.ones(3) > 0):

        def wires_shape(x, y):

            for i in range(2):
                if -w/2 - centers[i] < x < w/2 - centers[i]:
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

    def triangle(mu):
        # triangle shape
        def triangle_shape(x, y):
            if np.tan(angle)*x <= -(y-L) and np.tan(angle)*x >= (y-L):
                return mu

        return triangle_shape

    def v(mu):
        def v_shape(x, y):
            if np.tan(angle)*x <= -(y-L) and np.tan(angle)*x >= (y-L):
                if not (np.tan(angle)*x <= -(y-l_w) and np.tan(angle)*x >= (y-l_w)):
                    return mu
        return v_shape

    def ring(mu):
        # circle shape
        def circle_shape(x, y):
            if x**2 + y**2 <= R**2 and x**2 + y**2 >= r**2:
                return mu
        return circle_shape

    def rectangle(mu):
        def rectangle_shape(x, y):
            if -W/2 <= x <= W/2:
                return mu
        return rectangle_shape

    def builder_shape(**geometry):
        """Return a function used to create builder as TJ shape."""

        def system(site):
            x, y = site. pos
            if 0 <= y < L-connection:
                f = scatter()
            else:
                f = wires()

            return f(x, y)

        return system

    def scatter(mu=True):
        if shape == 'ring':
            return ring(mu)
        elif shape == 'triangle':
            return triangle(mu)
        elif shape == 'v':
            return v(mu)
        elif shape == 'rectangle':
            return rectangle(mu)

    def fill_system(mu_qd, mus_nw, sigma=0):

        def system(x, y):
            if 0 <= y < L-connection:
                f = scatter(mu=mu_qd)
                noise = np.random.normal(0, sigma)
            else:
                f = wires(mu=mus_nw)
                noise = 0
            return f(x, y) + noise

        return system

    def f_params(**params):

        mus_nw = params.pop('mus_nw')
        mu_qd = params.pop('mu_qd')
        Delta = params.pop('Delta')
        phi1 = params.pop('phi1')
        phi2 = params.pop('phi2')
        sigma = params.pop('sigma')
        f_chemical_potential = fill_system(mu_qd=mu_qd, mus_nw=mus_nw, sigma=sigma)
        f_Delta_re = fill_system(mu_qd=0, mus_nw=Delta * np.array([1, np.cos(phi1), np.cos(phi2)]))
        f_Delta_im = fill_system(mu_qd=0, mus_nw=Delta * np.array([0, np.sin(phi1), np.sin(phi2)]))

        params.update(mu=f_chemical_potential)
        params.update(Delta_re=f_Delta_re)
        params.update(Delta_im=f_Delta_im)

        return params

    def f_mu_potential(potential, mus_nw):
        def shape(x, y):
            if 0 <= y < L:
                f = potential
            else:
                f = wires(mu=mus_nw)
            return f(x, y)
        return shape

    def f_params_potential(potential, params):
        mus_nw = params['mus_nw']
        f = f_params(**params)
        f.update(mu=f_mu_potential(potential=potential, mus_nw=mus_nw))
        return f

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

    return trijunction, f_params, f_params_potential


def circular_junction(**geometry):

    R = geometry['R']
    L = geometry['L']
    w = geometry['w']

    hamiltonian = """( t * (k_x**2 + k_y**2 ) - mu(x,y) )* kron(sigma_0, sigma_z)
    + alpha * k_x * kron(sigma_y, sigma_z)
    - alpha * k_y * kron(sigma_x, sigma_z)
    + Delta_re(x,y) * kron(sigma_0, sigma_x)
    + Delta_im(x,y) * kron(sigma_0, sigma_y)
    + B_x * kron(sigma_y, sigma_0)"""

    a = 10e-9
    template = kwant.continuum.discretize(hamiltonian, grid=a)

    lower_wires_center = R*np.array([np.cos(-np.pi/6), np.sin(-np.pi/6)])
    x_c, y_c = lower_wires_center

    def circle_shape(mu_qd=True, mus_nw=[True, True, True]):
        def shape(x, y):
            if x**2 + y**2 <= R**2:
                return mu_qd
            elif -L <= y - y_c <= 0 and -w/2 <= x - x_c + w/2 < w/2:
                return mus_nw[0]
            elif -L <= y - y_c <= 0 and -w/2 <= x + x_c - w/2 < w/2:
                return mus_nw[1]
            elif L >= y >= R and -w/2 <= x <= w/2:
                return mus_nw[2]
        return shape

    def builder_shape(**geometry):
        """Return a function used to create builder as TJ shape."""

        def system(site):
            x, y = site. pos
            f = circle_shape()
            return f(x, y)

        return system

    def fill_system(mu_qd, mus_nw):
        def filled(x, y):
            f = circle_shape(mu_qd=mu_qd, mus_nw=mus_nw)
            return f(x, y)
        return filled

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

    def make_junction():
        """Create finalized Builder of a rectangle filled with template"""
        junction = kwant.Builder()
        junction.fill(
            template,
            shape=builder_shape(),
            start=[0, 0]
        )
        return junction

    return make_junction().finalized(), f_params


def inverted_triangle_junction(**geometry):
    
    area = geometry['area']
    angle = geometry['angle']
    L = geometry['L']
    w = geometry['w']
    side = geometry['side']

    hamiltonian = """( t * (k_x**2 + k_y**2 ) - mu(x,y) )* kron(sigma_0, sigma_z)
    + alpha * k_x * kron(sigma_y, sigma_z)
    - alpha * k_y * kron(sigma_x, sigma_z)
    + Delta_re(x,y) * kron(sigma_0, sigma_x)
    + Delta_im(x,y) * kron(sigma_0, sigma_y)
    + B_x * kron(sigma_y, sigma_0)"""

    a = 10e-9
    template = kwant.continuum.discretize(hamiltonian, grid=a)

    triangle_length = np.sqrt(area*np.tan(angle))
    x_c = np.abs((triangle_length/np.tan(angle))*0.5)
    y_wire_end = triangle_length - np.abs(np.tan(angle)*(x_c - w))
    y_c = triangle_length - np.tan(angle)*x_c 
    connection = np.abs(y_c - y_wire_end)
    y_c = y_c - connection
    y_connection = np.tan(angle)*(w/2)
    

    def triangle_shape(mu_qd=True, mus_nw=[True, True, True]):
        def shape(x, y):
            if y >= 0 and np.tan(angle)*x <= -(y-triangle_length) and np.tan(angle)*x >= (y-triangle_length):
                return mu_qd
            elif 0 <= y - y_c <= L and -w/2 <= x - x_c < w/2:
                return mus_nw[0]
            elif 0 <= y - y_c <= L and -w/2 <= x + x_c < w/2:
                return mus_nw[1]
            elif side =='up' and 0 >= y >= -L and -w/2 <= x <= w/2:
                return mus_nw[2]
            elif side =='down' and 0 <= y - triangle_length + y_connection + w/2 <= L and -w/2 <= x <= w/2:
                return mus_nw[2]
        return shape

    def builder_shape(**geometry):
        """Return a function used to create builder as TJ shape."""

        def system(site):
            x, y = site. pos
            f = triangle_shape()
            return f(x, y)

        return system

    def fill_system(mu_qd, mus_nw):
        def filled(x, y):
            f = triangle_shape(mu_qd=mu_qd, mus_nw=mus_nw)
            return f(x, y)
        return filled

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

    def make_junction():
        """Create finalized Builder of a rectangle filled with template"""
        junction = kwant.Builder()
        junction.fill(
            template,
            shape=builder_shape(),
            start=[0, 0]
        )
        return junction

    return make_junction().finalized(), f_params