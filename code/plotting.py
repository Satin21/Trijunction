import kwant
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from tools import find_cuts


def plot_potential_at_barriers(
    volts, potentials, mu, color_label, title="Potential cut along barriers"
):
    yint = 1e3 * volts
    x, potential_cuts_bot = find_cuts(potentials, cut=10e-9)
    potential_cuts_bot = np.array(potential_cuts_bot)
    xs = np.array([x for i in range(len(potential_cuts_bot))])

    fig, ax = plt.subplots(figsize=(10, 4))
    lc1 = multiline(
        1e9 * xs, -1e3 * (potential_cuts_bot - mu), yint, cmap="viridis", lw=0.3, ax=ax
    )

    axcb = fig.colorbar(lc1)
    axcb.set_label(color_label, fontsize=13)
    ax.set_title(title)
    ax.set_ylabel(r"$V-\mu_0$ [mV]")
    ax.set_xlabel(r"x [nm]")
    plt.show()


def plot_gates(gates, scale=10, address="gates.pdf", **kwargs):
    fig, ax = plt.subplots(**kwargs)

    types_of_gates = list(gates.keys())
    colors = ["b", "g", "r"]
    labels = ["Plunger gate", "Screen gates", "Barrier gates"]

    i = 0
    for key in types_of_gates:

        for _, gate in gates[key].items():
            gate = scale * gate
            ax.plot(gate[:, 0], gate[:, 1], c=colors[i], label=labels[i], linewidth=2)
            ax.plot(
                [gate[0, 0], gate[-1, 0]],
                [gate[0, 1], gate[-1, 1]],
                c=colors[i],
                linewidth=2,
            )
        i += 1

    ax.set_ylabel("y [nm]")
    ax.set_xlabel("x [nm]")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left")
    plt.savefig(address, bbox_inches="tight")

    return fig


def plot_couplings(
    n_geometries,
    geometries,
    mus_qd_units,
    geometries_peaks,
    geometries_couplings,
    title,
    units,
    n_cols,
    n_rows,
    figsize=(18, 12),
    ylim=150,
):
    """ """
    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=figsize)
    fig.tight_layout(h_pad=4, w_pad=2)
    geometry = 0
    pair = 0
    label = r"$E_{left-right}$"
    color = "blue"
    for ax in axes.flatten():
        if geometry == n_geometries:
            geometry = 0
            pair = 1
            label = r"$E_{left-center}$"
            color = "green"
        ax_title = title + str(np.round(geometries[geometry], 3)) + units
        peaks = geometries_peaks[geometry][pair]
        couplings = 1e6 * geometries_couplings[geometry][pair]
        ax.plot(mus_qd_units, couplings, color=color, label=label)
        ax.vlines(x=mus_qd_units[peaks], ymin=-1, ymax=200, linewidth=0.5, color="gray")
        ax.set_ylim(0, ylim)
        ax.set_title(ax_title)
        ax.set_xlabel(r"$\mu_{cavity} [meV]$")
        ax.set_ylabel(r"E [$\mu$eV]")
        geometry += 1
    plt.show()


def plot_average_couplings(n_geometries, geometries, geometries_averages, title, units):
    fig, axes = plt.subplots(ncols=int(n_geometries / 2), nrows=4, figsize=(18, 12))
    fig.tight_layout(h_pad=4, w_pad=2)
    geometry = 0
    pair = 0
    label = r"$E_{left-right}$"
    color = "blue"
    for ax in axes.flatten():
        if geometry == n_geometries:
            geometry = 0
            pair = 1
            label = r"$E_{left-center}$"
            color = "green"
        ax_title = title + str(np.round(geometries[geometry], 2)) + units
        average_data = geometries_averages[geometry][pair]
        ax.bar(
            x=1e3 * np.array(average_data[0]),
            height=1e2 * np.array(average_data[1]),
            width=1e3 * np.array(average_data[2]),
            edgecolor="black",
            color=color,
            label=label,
        )
        ax.set_ylim(0, 200)
        ax.set_title(ax_title)
        ax.set_xlabel(r"$\mu_{cavity} [meV]$")
        ax.set_ylabel(r"E [$\mu$eV]")
        ax.legend(fontsize=14, loc="upper left")
        geometry += 1
    plt.show()


def wfs_animation(junction, wfs, couplings, mus, ylims, xlims):

    density = kwant.operator.Density(junction, np.eye(4))
    labels = [r"$E_{12}$", r"$E_{13}$", r"$E_{23}$"]

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
            lab = labels[j] + "=" + str(np.round(couplings[j][i] * 1000, 3)) + "meV "
            lab += r"$\mu_{qd}$=" + str(np.round(mus[i] * 1000, 3)) + "meV"
            ax[j].set_title(lab, fontsize=15)

        return (
            wf1,
            wf2,
        )

    anim = animation.FuncAnimation(fig, animate, frames=len(wfs[0]), interval=150)

    return anim


def potential_animation(trijunction, potentials):

    fig, ax = plt.subplots()
    plt.close()
    f_mu = f_params_potential(potential=get_potential(potentials[0]), params=params)[
        "mu"
    ]
    f_mu_wrap = wrap_function(f_mu)
    pot_plot = kwant.plotter.map(
        trijunction, f_mu_wrap, cmap="inferno", colorbar=True, ax=ax
    )

    def animate(i):
        f_mu = f_params_potential(
            potential=get_potential(potentials[i]), params=params
        )["mu"]
        f_mu_wrap = wrap_function(f_mu)
        pot_plot = kwant.plotter.map(
            trijunction, f_mu_wrap, cmap="inferno", colorbar=True, ax=ax
        )

        return pot_plot

    anim = animation.FuncAnimation(fig, animate, frames=len(potentials), interval=150)

    return anim


def wrap_function(function_on_trijunction, trijunction):
    def f_mu_wrap(i):
        x, y = trijunction.sites[i].pos
        return function_on_trijunction(x, y)

    return f_mu_wrap


def multiline(xs, ys, c, ax=None, **kwargs):
    """
    From: https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap/38219022
    Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def plot_potential_lines(
    figsize=(15, 5), potential=[], cmap="gist_heat_r", scale=1, **kwargs
):

    total_potential = []
    for element in potential:
        coordinates = np.array(list(element.keys()))
        values = np.array(list(element.values()))
        total_potential.append(values)

    x = scale * coordinates[:, 0]
    y = scale * coordinates[:, 1]
    width = np.unique(x).shape[0]
    X = x.reshape(width, -1)
    Y = y.reshape(width, -1)
    Z = sum(total_potential).reshape(width, -1)

    fig, ax = plt.subplots(figsize=figsize)
    cont = ax.contour(X, Y, 1e3 * Z, cmap=cmap, levels=100, **kwargs)
    cbar = fig.colorbar(cont, ax=ax, format="%.2f")
    cbar.set_label(r"$V_{eqipot}$ [mV]")
    ax.set_xlabel("X [nm]")
    ax.set_ylabel("Y [nm]")
