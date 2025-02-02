import mdtraj as md
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


def calc_contact_angle(trj, z_surf, z_max, r_range, n_bins,
        trim_z, trim_r, rho_cutoff, direction):
    """
    Calculate the contact angle of a droplet

    Arguments
    ---------
    trj : mdtraj.Trajectory
        Trajectory, sliced to include only the droplet
    z_surf : float
        z coordinate defining the edge of the surface
    z_max : flaot
        Maximum and minimum values of `z` to consider in fit
    r_range : tuple
        Minimum and maximum values of `r` to consider in fit
    n_bins : int
        Number of bins to use in histgram
    trim_z : float
        Height off surface to ignore in fit
    trim_r : float
        Distance from center of droplet to ignore in fit
    rho_cutoff : float
        Atomic number density (in number/nm^3) that specifies the edge of
        the droplet. For most systems, the bulk atomistic density is an
        appropriate value.
    r0 : float
        Initial guess of droplet size. If None, will be set to half the width
        of `r_range`.
    direction : str
        Direction to consider in fitting the edge of the droplet. Must be one
        of `top`, `bottom`, or `both`.

    Returns
    -------
    droplet : dict
        theta : float
            Contact angle of droplet in degrees
        heatmap : np.ndarray
            Density of fluid in droplet as a function of `r` and `z`
        r_edges : np.ndarray
            Bins in the `r` dimension of the density heatmap
        z_edges : np.ndarray
            Bins in the `z` dimension of the density heatmap
        fit_r   : np.ndarray
            Fitted values in the `r` dimension
        fit_z   : np.ndarray
            Fitted values in the `z` dimension
    """

    # Transform coordinates from `x,y,z` to `r,z`
    com = md.compute_center_of_mass(trj)
    _r = np.transpose(np.sqrt(np.square(np.transpose(trj.xyz[:, :, 0])-com[:, 0]) +
                              np.square(np.transpose(trj.xyz[:, :, 1])-com[:, 1])))
    _z = trj.xyz[:, :, 2]-z_surf

    r = np.array(_r).flatten()
    z = np.array(_z).flatten()

    # Generate 2-D histogram of droplet
    H, r_edges, z_edges = np.histogram2d(r, z, bins=(np.linspace(r_range[0], r_range[1], n_bins),
                                                     np.linspace(0, z_max, n_bins)))
    H = H.T
    dz = z_max/n_bins
    dr = (r_range[1]-r_range[0])/n_bins
    H = np.divide(H, np.pi * dz * dr)
    H = np.divide(H, 2 * r_edges[1:] + dr)
    H = np.divide(H, len(trj))

    fit_r, fit_z = _find_edge(H, r_edges, z_edges, direction, rho_cutoff)

    vals = np.intersect1d(np.where(fit_z > trim_z),
            np.where(fit_r > trim_r))
    if vals.size == 0:
        raise ValueError('Malformed droplet, all atoms are below `z_trim`')
    fit_r = fit_r[vals]
    fit_z = fit_z[vals]

    h = np.max(fit_z)

    # Calculate contact angle from fit of sphere to droplet
    sol = least_squares(_fitting_func, r_range[1], args=(fit_r, fit_z, h))
    R = sol.x
    theta = np.arccos((R-h)/R) * 180 / np.pi

    drop = dict()
    drop['theta'] = theta
    drop['heatmap'] = H
    drop['r_edges'] = r_edges
    drop['z_edges'] = z_edges
    drop['fit_r'] = fit_r
    drop['fit_z'] = fit_z

    return drop


def _find_edge(H, r_edges, z_edges, direction, rho_cutoff):
    """Find the edge of a droplet"""
    if direction not in ['top', 'side', 'both']:
        raise ValueError("`direction` must be one of "
                         "`top`, `side`, or `both`")

    fit_r = []
    fit_z = []

    if direction in ['top', 'both']:
        for i, r in enumerate(r_edges[:-1]):
            if i < 1:
                continue
            else:
                for j, z in reversed(list(enumerate(z_edges[:-1]))):
                    freq = H[j, i]
                    if freq > rho_cutoff:
                        fit_r.append(r)
                        fit_z.append(z)
                        break

    if direction in ['side', 'both']:
        for j, z in enumerate(z_edges[:-1]):
            if j < 1:
                continue
            else:
                for i, r in reversed(list(enumerate(r_edges[:-1]))):
                    freq = H[j, i]
                    if freq > rho_cutoff:
                        fit_r.append(r)
                        fit_z.append(z)
                        break

    fit_r = np.array(fit_r)
    fit_z = np.array(fit_z)

    return fit_r, fit_z


def _fitting_func(r, fit_r, fit_z, h):
    return np.sqrt(np.square(fit_r) + np.square(fit_z + r - h)) - r


def plot_heatmap(droplet, xlim, ylim, fit=False):
    """
    Plot the heat map of a droplet

    Arguments
    ---------
    droplet : dict
        theta : float
            Contact angle of droplet in degrees
        heatmap : np.ndarray
            Density of fluid in droplet as a function of `r` and `z`
        r_edges : np.ndarray
            Bins in the `r` dimension of the density heatmap
        z_edges : np.ndarray
            Bins in the `z` dimension of the density heatmap
        fit_r   : np.ndarray
            Fitted values in the `r` dimension
        fit_z   : np.ndarray
            Fitted values in the `z` dimension
    x_lim   : tuple
        Minimum and maximum values of x-axis
    y_lim   : tuple
        Minimum and maximum values of y-axis    
    fit     : bool
        Option to plot fitted line on heatmap
    
    Returns
    -------
    fig     : matplotlib.figure 
        Matplotlib figure object
    axs     : matplotlib.axes.Axes
        Matplotlib axes object
    """

    X, Y = np.meshgrid(droplet['r_edges'], droplet['z_edges'])
    fig, axs = plt.subplots()
    axs.pcolormesh(X, Y, droplet['heatmap'], cmap='Blues')
    if fit:
        fig, axs = _plot_fit_line(fig, axs, droplet['fit_r'], droplet['fit_z'])

    axs.set_xlim(xlim)
    axs.set_ylim(ylim)
    axs.set_xlabel('r')
    axs.set_ylabel('z')
    return [fig, axs]


def _plot_fit_line(fig, axs, fit_r, fit_z):
    axs.plot(fit_r, fit_z, c='k')
    return [fig, axs]


def plot_r_density(droplet, xlim):
    """
    Plot density vs r of a droplet

    Arguments
    ---------
    droplet : dict
        theta : float
            Contact angle of droplet in degrees
        heatmap : np.ndarray
            Density of fluid in droplet as a function of `r` and `z`
        r_edges : np.ndarray
            Bins in the `r` dimension of the density heatmap
        z_edges : np.ndarray
            Bins in the `z` dimension of the density heatmap
        fit_r   : np.ndarray
            Fitted values in the `r` dimension
        fit_z   : np.ndarray
            Fitted values in the `z` dimension
    x_lim   : tuple
        Minimum and maximum values of x-axis
    
    Returns
    -------
    fig     : matplotlib.figure 
        Matplotlib figure object
    axs     : matplotlib.axes.Axes
        Matplotlib axes object
    """

    fig, axs = plt.subplots()
    axs.plot(droplet['r_edges'][:-1], np.sum(droplet['heatmap'],axis=0))

    axs.set_xlim(xlim)
    axs.set_xlabel('r')
    axs.set_ylabel('density')
    return [fig,axs]


def plot_z_density(droplet, xlim):
    """
    Plot density vs z of a droplet

    Arguments
    ---------
    droplet : dict
        theta : float
            Contact angle of droplet in degrees
        heatmap : np.ndarray
            Density of fluid in droplet as a function of `r` and `z`
        r_edges : np.ndarray
            Bins in the `r` dimension of the density heatmap
        z_edges : np.ndarray
            Bins in the `z` dimension of the density heatmap
        fit_r   : np.ndarray
            Fitted values in the `r` dimension
        fit_z   : np.ndarray
            Fitted values in the `z` dimension
    x_lim   : tuple
        Minimum and maximum values of x-axis
    
    Returns
    -------
    fig     : matplotlib.figure 
        Matplotlib figure object
    axs     : matplotlib.axes.Axes
        Matplotlib axes object
    """

    fig, axs = plt.subplots()
    axs.plot(droplet['z_edges'][:-1], np.sum(droplet['heatmap'],axis=1))

    axs.set_xlim(xlim)
    axs.set_xlabel('z')
    axs.set_ylabel('density')
    return [fig,axs]

