import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def calc_contact_angle(trj, z_surf, z_max, r_range, n_bins, trim_z, rho_cutoff):
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
    rho_cutoff : float
        Density (in number/nm^3) that specifies the edge of the droplet. For
        most systems, the bulk density is an appropriate value.
    r0 : float
        Initial guess of droplet size. If None, will be set to half the width
        of `r_range`.

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
    """

    # Transform coordinates from `x,y,z` to `r,z`
    com = md.compute_center_of_mass(trj)
    _r = np.transpose(np.sqrt(np.square(np.transpose(trj.xyz[:, :, 0])-com[:, 0]) +
                              np.square(np.transpose(trj.xyz[:, :, 1])-com[:, 1])))
    _z = trj.xyz[:,:,2]-z_surf

    r = np.array(_r).flatten()
    z = np.array(_z).flatten()

    # Generate 2-D histogram of droplet
    H, r_edges, z_edges = np.histogram2d(r, z, bins=(np.linspace(r_range[0], r_range[1], n_bins),
                                                     np.linspace(0, z_max, n_bins)))
    H=H.T
    dz = z_max/n_bins
    dr = (r_range[1]-r_range[0])/n_bins
    H=np.divide(H, np.pi * dz * dr )
    H=np.divide(H, 2*r_edges[1:] + dr)
    H=np.divide(H, len(trj))

    fit_x, fit_y = _find_edge(H, r_edges, z_edges, direction)

    vals = np.where(fit_y > trim_z)
    fit_x = fit_x[vals]
    fit_y = fit_y[vals]
    


    h = np.max(fit_y)

    # Calculate contact angle from fit of sphere to droplet
    sol = least_squares(_fitting_func, r_range[1], args=(fit_x, fit_y, h))
    R = sol.x
    theta = np.arccos((R-h)/R) * 180 / np.pi

    drop = dict()
    drop['theta'] = theta
    drop['heatmap'] = H
    drop['r_edges'] = r_edges
    drop['z_edges'] = z_edges

    return drop

def _find_edge(H, r_edges, z_edges, direction):
    """Find the edge of a droplet"""
    if direction not in ['top', 'side', 'both']:
        raise ValueError("`direction` must be one of "
                         "`top`, `side`, or `both`")

    fit_x = []
    fit_y = []

    if direction in ['top', 'both']:
        for i, r in enumerate(r_edges[:-1]):
            if i < 1:
                continue
            else:
                for j, z in reversed(list(enumerate(z_edges[:-1]))):
                    freq = H[j, i]
                    if freq > rho_cutoff:
                        fit_x.append(r)
                        fit_y.append(z)
                        break

    if direction in ['side', 'both']:
        for j, z in enumerate(z_edges[:-1]):
            if j < 1:
                continue
            else:
                for i, r in reversed(list(enumerate(r_edges[:-1]))):
                    freq = H[j, i]
                    if freq > rho_cutoff:
                        fit_x.append(r)
                        fit_y.append(z)
                        break

    fit_x = np.array(fit_x)
    fit_y = np.array(fit_y)

    return fit_x, fit_y

def _fitting_func(r, fit_x, fit_y, h):
    return np.sqrt(np.square(fit_x) + np.square(fit_y + r - h)) - r
