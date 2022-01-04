import kwant
import matplotlib.pyplot as plt
import numpy as np


def plot_average_couplings(tjs, average_data, title, n_geometries, subtitles, xlims, ylims, ylim):
    xmin, xmax = xlims
    ymin, ymax = ylims
    plt.close()
    fig, ax = plt.subplots(ncols=4, nrows=n_geometries, figsize=(30, 7*n_geometries))
    fig.suptitle(title, fontsize=22)
    labels = [r'$E_{12}$', r'$E_{13}$', r'$E_{23}$']
    colors = ['lightblue', 'orange', 'green']
    g = 0
    for i in range(n_geometries):
        for j in range(4):
            if j == 0:
                kwant.plot(tjs[i][0], ax=ax[i][j])
                ax[i][j].set_xlim(xmin, xmax)
                ax[i][j].set_ylim(ymin, ymax)
                ax[i][j].set_title(subtitles[i])
            if j > 0:
                w = j-1
                mus = 1000*np.array(average_data[i][w][0])
                bars = 100*np.array(average_data[i][w][1])
                widths = 1000*np.array(average_data[i][w][2])
                min_widht = np.round(1000*average_data[i][w][3], 3)
                average_peak_width = np.round(np.mean(widths), 3)
                average_peak_height = np.round(np.mean(bars), 3)
                ax[i][j].bar(mus, bars, width=widths, label=labels[w], edgecolor='black', color=colors[w])
                ax[i][j].set_ylim(0, ylim)
                title_subplots = r'Av. height: '+str(average_peak_height)+'[$\mu$eV]\n Av. width: '+str(average_peak_width)+'[meV], \n Min. width = '+str(min_widht)+'[meV]'
                ax[i][j].set_title(title_subplots)
                ax[i][j].legend()
            if j == 1:
                ax[i][j].set_ylabel(r'$E[\mu$eV]')
            if i == n_geometries-1 and j > 0:
                ax[i][j].set_xlabel(r'$\mu_{qd}$[meV]')

        g += 1
    fig.tight_layout(pad=4, h_pad=3, w_pad=3)
    plt.savefig('../data/average_'+title.replace(' ', '_')+'.svg')


def plot_couplings(tjs, mus, full_data, title, n_geometries, subtitles, xlims, ylims, ylim):
    xmin, xmax = xlims
    ymin, ymax = ylims
    plt.close()
    fig, ax = plt.subplots(ncols=4, nrows=n_geometries, figsize=(30, 8*n_geometries))
    fig.suptitle(title, fontsize=22)
    labels = [r'$E_{12}$', r'$E_{13}$', r'$E_{23}$']
    colors = ['lightblue', 'orange', 'green']
    g = 0
    for i in range(n_geometries):
        for j in range(4):
            if j == 0:
                kwant.plot(tjs[i][0], ax=ax[i][j])
                ax[i][j].set_xlim(xmin, xmax)
                ax[i][j].set_ylim(ymin, ymax)
                ax[i][j].set_title(subtitles[i])
            if j > 0:
                w = j-1
                coupling = 1e6*np.array(full_data[i][w])
                average_coupling = np.round(np.mean(coupling), 3)
                ax[i][j].plot(1000*mus, coupling, label=labels[w], color=colors[w])
                ax[i][j].set_ylim(0, ylim)
                title_subplots = r'Average coupling: '+str(average_coupling)+'[$\mu$eV]'
                ax[i][j].set_title(title_subplots)
                ax[i][j].legend()
            if j == 1:
                ax[i][j].set_ylabel(r'$E[\mu$eV]')
            if i == n_geometries-1 and j > 0:
                ax[i][j].set_xlabel(r'$\mu_{qd}$[eV]')  
        g += 1
    fig.tight_layout(pad=4, h_pad=3, w_pad=3)
    plt.savefig('../data/full_'+title.replace(' ', '_')+'.svg')


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


