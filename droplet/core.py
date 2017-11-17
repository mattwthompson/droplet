import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def calc_contact_angle():
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
    r0 : float
        Initial guess of droplet size. If None, will be set to half the width
        of `r_range`.

    Returns
    -------
    theta : float
        contact angle of droplet
    """

    # Transform coordinates from `x,y,z` to `r,z`
    com = md.compute_center_of_mass(traj)
    _r = np.transpose(np.sqrt(np.square(np.transpose(traj.xyz[:, :, 0])-com[:, 0]) +
                              np.square(np.transpose(traj.xyz[:, :, 1])-com[:, 1])))
    _z = traj.xyz[:,:,2]-z_surf

    r = np.array(_r).flatten()
    z = np.array(_z).flatten()

    # Generate 2-D histogram of droplet
    H, r_edges, z_edges = np.histogram2d(r, z, bins=(np.linspace(0, z_max, n_bins),
                                                     np.linspace(r_range[0], r_range[1], n_bins)))
    H=H.T
    H=np.divide(H,r_edges[1:])

    # Find coordinates of droplet edge
    fit_x = np.zeros(n_bins)
    fit_y = np.zeros(n_bins)

    for i, r in enumerate(r_edges[:-1]):
        if i < 1 :
            continue
        else:
            for j, z in reversed(list(enumerate(z_edges[:-1]))):
                freq=H[j, i]
                if freq > 1000 :
                    fit_x[i] = r
                    fit_y[j] = z
                    break
    fit_x = np.trim_zeros(fit_x)
    fit_y = np.trim_zeros(fit_y)

    vals = np.where(fit_y > trim_z)
    fit_x = fit_x[vals]
    fit_y = fit_y[vals]

    h = np.max(fit_y)

    # Calculate contact angle from fit of sphere to droplet
    sol = least_squares(r_error, r_range[1], args=(fit_x, fit_y, h))
    R = sol.x
    theta = np.arccos((R-h)/R)
 

def _fitting_func(r, fit_x, fit_y, h):
    return np.sqrt(np.square(fit_x) + np.square(fit_y + r - h)) - r
