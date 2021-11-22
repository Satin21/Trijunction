import numpy as np
import kwant
import kwant.continuum


def infinite_system(**geometry):
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

    lead_hamiltonian = """
    ( t(x,y)*(k_x**2 + k_y**2 ) - mu_nw )* kron(sigma_0, sigma_z)
    + alpha * k_x * kron(sigma_y, sigma_z)
    - alpha * k_y * kron(sigma_x, sigma_z)
    + Delta_re * kron(sigma_0, sigma_x)
    + Delta_im * kron(sigma_0, sigma_y)
    + B_x * kron(sigma_x, sigma_0)
    """

    scattering_hamiltonian = """
    ( t * (k_x**2 + k_y**2 ) - mu_qd )* kron(sigma_0, sigma_z)
    + alpha * k_x * kron(sigma_y, sigma_z)
    - alpha * k_y * kron(sigma_x, sigma_z)
    + B_x * kron(sigma_y, sigma_0)
    """

    a = geometry['a']
    template_lead = kwant.continuum.discretize(lead_hamiltonian, grid=a)
    template_scattering = kwant.continuum.discretize(scattering_hamiltonian, grid=a)

    def builder_shape(**geometry):
        """Return a function used to create builder as TJ shape."""

        L = geometry['L']
        W = geometry['W']
        angle = geometry['angle']

        def system_shape(x, y):
            if -W/2 <= x <= W/2 and 0 <= y < L:
                if angle <= np.pi/4:
                    if np.tan(angle)*x <= -(y-L) and np.tan(angle)*x >= (y-L):
                        return True
                elif angle > np.pi/4:
                    if np.tan(angle)*(x-W/2) <= -y and np.tan(angle)*(x+W/2) >= y:
                        return True

        def system(site):
            x, y = site. pos
            return system_shape(x, y)

        return system

    def make_scattering_region(**geometry):
        """Create finalized Builder of a rectangle filled with template"""
        junction = kwant.Builder()
        junction.fill(
            template_scattering,
            shape=builder_shape(**geometry),
            start=[0, 0]
        )
        return junction

    def make_leads(**geometry):

        w = geometry['w']
        W = geometry['W']

        center = W/4
        lead_top = kwant.Builder(symmetry=kwant.TranslationalSymmetry([0, a]))
        lead_bot_left = kwant.Builder(symmetry=kwant.TranslationalSymmetry([0, -a]))
        lead_bot_right = kwant.Builder(symmetry=kwant.TranslationalSymmetry([0, -a]))

        lead_top.fill(template_lead, shape=lambda site: -w/2 <= site.pos[0] < w/2, start=[0]);
        lead_bot_right.fill(template_lead, shape=lambda site: -w/2 + center <= site.pos[0]  < w/2+center, start=[center]);
        lead_bot_left.fill(template_lead, shape=lambda site: -w/2 - center <= site.pos[0]  < w/2-center, start=[-center]);

        return lead_top, lead_bot_right, lead_bot_left

    def f_params(**params):
        """Convert the raw parameters into position-dependent functions."""
        phi = params['phi']
        Delta = params['Delta']
        params.update(Delta_im=Delta*np.sin(phi))
        params.update(Delta_re=Delta*np.cos(phi))
        return params

    cavity = make_scattering_region(**geometry)
    leads = make_leads(**geometry)
    for lead in leads:
        cavity.attach_lead(lead);
    cavity = cavity.finalized()
    return cavity, f_params
