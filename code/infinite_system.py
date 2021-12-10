import numpy as np
import kwant
import kwant.continuum


def infinite_system(flag_leads, **geometry):
    """
    Create a kwant builder that describes a scattering region connected to three topological leads.
    The builder is filled with a discretized continuum hamiltonian.

    Parameters:
    -----------
        geometry: dict

    Returns:
    --------
        trijunction: kwant.InfiniteSystem
        f_params: function
    """

    def lead_hamiltonian(mu_wire): 
        h = """
        ( t * (k_x**2 + k_y**2) - """+mu_wire+""" )* kron(sigma_0, sigma_z)
        + alpha * k_x * kron(sigma_y, sigma_z)
        - alpha * k_y * kron(sigma_x, sigma_z)
        + Delta_re * kron(sigma_0, sigma_x)
        + Delta_im * kron(sigma_0, sigma_y)
        + B_x * kron(sigma_y, sigma_0)
        """
        return h

    scattering_hamiltonian = """
    ( t * (k_x**2 + k_y**2 ) - mu_qd )* kron(sigma_0, sigma_z)
    + alpha * k_x * kron(sigma_y, sigma_z)
    - alpha * k_y * kron(sigma_x, sigma_z)
    + B_x * kron(sigma_y, sigma_0)
    """

    a = geometry['a']
    template_scattering = kwant.continuum.discretize(scattering_hamiltonian, grid=a)
    
    A = geometry['A']
    angle = geometry['angle']
    L = np.sqrt(A*np.tan(angle))
    w = geometry['w']
    connection = np.tan(angle)*(w/2)
    center = np.abs((L/np.tan(angle))/2)  # center of the wires in the lower region
    start = [0,0]

    def builder_shape(**geometry):
        """Return a function used to create builder as TJ shape."""

        def system(site):
            x, y = site.pos
            f = triangle()
            return f(x, y)

        return system
    
    def triangle(mu=True, **geometry):
        # triangle shape
        def triangle_shape(x, y):
            if np.tan(angle)*x <= -(y-L) and np.tan(angle)*x >= (y-L):
                if 0 <= y <= L-connection:
                    return mu

        return triangle_shape

    def make_scattering_region(**geometry):
        """Create finalized Builder of a rectangle filled with template"""
        junction = kwant.Builder()
        junction.fill(
            template_scattering,
            shape=builder_shape(**geometry),
            start=start
        )
        return junction

    def make_leads(**geometry):

        lead_top = kwant.Builder(symmetry=kwant.TranslationalSymmetry([0, a]))
        lead_bot_left = kwant.Builder(symmetry=kwant.TranslationalSymmetry([0, -a]))
        lead_bot_right = kwant.Builder(symmetry=kwant.TranslationalSymmetry([0, -a]))
        
        template_lead_1 = kwant.continuum.discretize(lead_hamiltonian('mu_nw_1'), grid=a)
        template_lead_2 = kwant.continuum.discretize(lead_hamiltonian('mu_nw_2'), grid=a)
        template_lead_3 = kwant.continuum.discretize(lead_hamiltonian('mu_nw_3'), grid=a)

        lead_top.fill(template_lead_3, shape=lambda site: -w/2 <= site.pos[0] < w/2, start=[0]);
        lead_bot_right.fill(template_lead_2, shape=lambda site: -w/2 + center <= site.pos[0]  < w/2+center, start=[center]);
        lead_bot_left.fill(template_lead_1, shape=lambda site: -w/2 - center <= site.pos[0]  < w/2-center, start=[-center]);

        return lead_bot_left, lead_bot_right, lead_top

    def f_params(**params):
        """Convert the raw parameters into position-dependent functions."""
        phi = params['phi']
        Delta = params['Delta']
        params.update(Delta_im=Delta*np.sin(phi))
        params.update(Delta_re=Delta*np.cos(phi))
        return params

    cavity = make_scattering_region(**geometry)
    leads = make_leads(**geometry)
    i = 0
    for lead in flag_leads:
        if lead:
            cavity.attach_lead(leads[i]);
        i += 1
    cavity = cavity.finalized()

    return cavity, f_params


def infinite_system_square(**geometry):
    """
    Create a kwant builder that describes a scattering region connected to three topological leads.
    The builder is filled with a discretized continuum hamiltonian.

    Parameters:
    -----------
        geometry: dict

    Returns:
    --------
        trijunction: kwant.InfiniteSystem
        f_params: function
    """

    def lead_hamiltonian(mu_wire): 
        h = """
        ( t * (k_x**2 + k_y**2) - """+mu_wire+""" )* kron(sigma_0, sigma_z)
        + alpha * k_x * kron(sigma_y, sigma_z)
        - alpha * k_y * kron(sigma_x, sigma_z)
        + Delta_re * kron(sigma_0, sigma_x)
        + Delta_im * kron(sigma_0, sigma_y)
        + B_x * kron(sigma_y, sigma_0)
        """
        return h

    scattering_hamiltonian = """
    ( t * (k_x**2 + k_y**2 ) - mu_qd )* kron(sigma_0, sigma_z)
    + alpha * k_x * kron(sigma_y, sigma_z)
    - alpha * k_y * kron(sigma_x, sigma_z)
    + B_x * kron(sigma_y, sigma_0)
    """

    a = geometry['a']
    template_scattering = kwant.continuum.discretize(scattering_hamiltonian, grid=a)
    
    A = geometry['A']
    angle = geometry['angle']
    L = np.sqrt(A*np.tan(angle))
    w = geometry['w']
    connection = np.tan(angle)*(w/2)
    center = np.abs((L/np.tan(angle))/2)  # center of the wires in the lower region
    start = [0,0]

    def builder_shape(**geometry):
        """Return a function used to create builder as TJ shape."""

        def system(site):
            x, y = site.pos
            f = triangle()
            return f(x, y)

        return system
    
    def triangle(mu=True, **geometry):
        # triangle shape
        def triangle_shape(x, y):
            if np.tan(angle)*x <= -(y-L) and np.tan(angle)*x >= (y-L):
                if 0 <= y <= L-connection:
                    return mu

        return triangle_shape

    def make_scattering_region(**geometry):
        """Create finalized Builder of a rectangle filled with template"""
        junction = kwant.Builder()
        junction.fill(
            template_scattering,
            shape=builder_shape(**geometry),
            start=start
        )
        return junction

    cavity = make_scattering_region(**geometry)
    cavity = cavity.finalized()

    return cavity