#### Soft-threshold for the wavefunction curvature


If $\frac{K_{min}}{K_{max}}$ > -0.5, add $\frac{K_{min}}{K_{max}}$ to the loss. 

where $K_{max}$ and $K_{min}$ are the maximum and minimum Gaussian curvature of the wavefunction probalility $|\psi(x)|^2$. -0.5 is the soft-threshold. Ideally when the threshold is zero, there are no saddle points, as the determinant of the hessian matrix is all positive. We cannot have the threshold to be positive or zero, because there are many points at which the curvature is zero (those are optimal points). Hence we use a softer-threshold.
This makes the minimal Gaussian curvature flat, thereby avoiding many saddle points. This way the number of peaks in the wavefunction can be reduced to a single peak. 

We can obtain min-max ratio of Gaussian curvature of any Kwant wavefunction as follows
```python

step = system.geometry['grid_spacing']*scale

xmin, xmax, ymin, ymax = np.array(list(system.boundaries.values()))*scale
bounds = lambda site: (site.pos[0] > xmin 
                    and site.pos[0] < xmax
                    and site.pos[1] > ymin
                    and site.pos[1] < ymax)
scattering_region_density = kwant.operator.Density(system.trijunction, where=bounds)

wavefunction_density = scattering_region_density(wave_functions[:, 0])

scattering_sites = kwant_sites[[bounds(site) for site in system.trijunction.sites]]
x, y = scattering_sites[:, 0], scattering_sites[:, 1]
nx, ny = len(np.unique(x)), len(np.unique(y))

interpolated_wavefunction = wavefunction_density.reshape((nx, -1))

from codes.utils import ratio_Gaussian_curvature
ratio = ratio_Gaussian_curvature(interpolated_wavefunction, step)

```