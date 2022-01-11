import kwant
import matplotlib.pyplot as plt
import numpy as np

def plot_couplings(n_geometries, geometries, mus_qd_units, geometries_peaks, geometries_couplings, title, units):
    fig, axes = plt.subplots(ncols=int(n_geometries/2), nrows=4, figsize=(18, 12))
    fig.tight_layout(h_pad=4, w_pad=2)
    geometry = 0
    pair = 0
    label = r'$E_{left-right}$'
    color = 'blue'
    for ax in axes.flatten():
        if geometry == n_geometries:
            geometry = 0
            pair = 1
            label = r'$E_{left-center}$'
            color = 'green'
        ax_title = title+str(np.round(geometries[geometry], 1))+units
        peaks = geometries_peaks[geometry][pair]
        couplings = 1e6*geometries_couplings[geometry][pair]
        ax.plot(mus_qd_units, couplings, color=color, label=label)
        ax.vlines(x=mus_qd_units[peaks], ymin=-1, ymax=200, linewidth=0.5, color='gray')
        ax.set_ylim(0, 170)
        ax.set_title(ax_title)
        ax.set_xlabel(r'$\mu_{cavity} [meV]$')
        ax.set_ylabel(r'E [$\mu$eV]')
        ax.legend(fontsize=14, loc='upper left')
        geometry += 1
    plt.show()


def plot_average_couplings(n_geometries, geometries, geometries_averages, title, units):
    fig, axes = plt.subplots(ncols=int(n_geometries/2), nrows=4, figsize=(18, 12))
    fig.tight_layout(h_pad=4, w_pad=2)
    geometry = 0
    pair = 0
    label = r'$E_{left-right}$'
    color = 'blue'
    for ax in axes.flatten():
        if geometry == n_geometries:
            geometry = 0
            pair = 1
            label = r'$E_{left-center}$'
            color = 'green'
        ax_title = title+str(np.round(geometries[geometry], 2))+units
        average_data = geometries_averages[geometry][pair]
        ax.bar(x=1e3*np.array(average_data[0]),
                     height=1e2*np.array(average_data[1]),
                     width=1e3*np.array(average_data[2]),
                     edgecolor='black', color=color, label=label)
        ax.set_ylim(0, 200)
        ax.set_title(ax_title)
        ax.set_xlabel(r'$\mu_{cavity} [meV]$')
        ax.set_ylabel(r'E [$\mu$eV]')
        ax.legend(fontsize=14, loc='upper left')
        geometry += 1
    plt.show()


def wfs_animation(junction, wfs, couplings, mus, ylims, xlims):

    density = kwant.operator.Density(junction, np.eye(4))
    labels = [r'$E_{12}$', r'$E_{13}$', r'$E_{23}$']

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
    plt.close()

    wf1 = kwant.plotter.density(junction, density(wfs[0][0]), ax=ax[0])
    wf2 = kwant.plotter.density(junction, density(wfs[1][0]), ax=ax[1])

    for i in range(0, 2):
        ax[i].set_ylim(ylims[0], ylims[1])
        ax[i].set_xlim(xlims[0], xlims[1])
        ax[i].set_title(labels[i], fontsize=15)

    def animate(i):
        wf1 = kwant.plotter.density(junction, density(wfs[0][i]), ax=ax[0])
        wf2 = kwant.plotter.density(junction, density(wfs[1][i]), ax=ax[1])

        for j in range(2):
            ax[j].set_ylim(ylims[0], ylims[1])
            ax[j].set_xlim(xlims[0], xlims[1])
            lab = labels[j]+'='+str(np.round(couplings[j][i]*1000, 3))+'meV '
            lab += r'$\mu_{qd}$='+str(np.round(mus[i]*1000, 3))+'meV'
            ax[j].set_title(lab, fontsize=15)

        return wf1, wf2,

    anim = animation.FuncAnimation(fig, animate, frames=len(wfs[0]), interval=150)

    return anim


def potential_animation(trijunction, potentials):

    fig, ax = plt.subplots()
    plt.close()
    f_mu = f_params_potential(potential=get_potential(potentials[0]), params=params)['mu']
    f_mu_wrap = wrap_function(f_mu)
    pot_plot = kwant.plotter.map(trijunction, f_mu_wrap, cmap='inferno', colorbar=True, ax=ax);

    def animate(i):
        f_mu = f_params_potential(potential=get_potential(potentials[i]), params=params)['mu']
        f_mu_wrap = wrap_function(f_mu)
        pot_plot = kwant.plotter.map(trijunction, f_mu_wrap, cmap='inferno', colorbar=True, ax=ax);

        return pot_plot

    anim = animation.FuncAnimation(fig, animate, frames=len(potentials), interval=150)

    return anim


def wrap_function(function_on_trijunction, trijunction):
    def f_mu_wrap(i):
        x, y = trijunction.sites[i].pos
        return function_on_trijunction(x, y)
    return f_mu_wrap


