import numpy as np
import scipy
from .misc import *

g = 9.81
omega = 7.292115e-5  # (1/s) (Groten, 2004).

def compute_f(lat):
    """
    Computes the Coriolis parameter.

    Parameters:
        lat : float
            Latitude in degrees.

    Returns:
        float
            The Coriolis parameter.
    """
     
    f = 2 * omega * np.sin(np.radians(lat))
    return np.asanyarray(f)
    
def compute_ocean_diagnostics_from_eta(eta, x, y, proj='lonlat', derivative='dxdy', lat=None, n=9, parallel=True, axis=(-2, -1), verbose=True, kernel = 'circular'):
    """
    This function calculates geostrophic and cyclo-geostrophic currents, strain rate, 
    and relative vorticity based on the provided sea surface height data 'eta'. The calculations 
    can be performed using different derivative methods and may include parallel processing. 

    Parameters:
        eta : array_like
            Sea surface heights.
        x : array_like
            X-coordinates (longitude or Cartesian).
        y : array_like
            Y-coordinates (latitude or Cartesian).
        proj : str, optional
            Projection type; can be 'lonlat' for geographic coordinates or 'xy' for Cartesian coordinates (default is 'lonlat').
        derivative : str, optional
            Method for calculating derivatives; can be 'dxdy' (using first derivatives) or 'fit' (using fitting derivatives) (default is 'dxdy').
        lat : array_like, optional
            Latitude values, required if proj is 'xy'.
        n : int, optional
            Size of the stencil used for derivative calculations (default is 9).
        parallel : bool, optional
            If True, uses parallel processing for efficiency (default is True).
        axis : tuple of int, optional
            Axes along which to compute derivatives (default is (-2, -1)).
        verbose : bool, optional
            If True, enables verbose output for tracking progress (default is True).

    Returns:
        dict
            A dictionary containing:
                - 'ug': Geostrophic eastward current
                - 'vg': Geostrophic northward current
                - 'ucg': Cyclo-geostrophic eastward current
                - 'vcg': Cyclo-geostrophic northward current
                - 'S': Strain rate
                - 'zeta': Relative vorticity
    """
    
    # Initialize diagnostics dictionary
    diag = {}

    # Ensure input arrays are numpy arrays
    x, y = np.asanyarray(x), np.asanyarray(y)

    # Compute Coriolis parameter based on projection
    if proj == 'lonlat':
        f = compute_f(y)
    elif proj == 'xy' and lat is not None:
        f = compute_f(lat)
    else:
        print('Invalid "proj" argument (must be "lonlat" or "xy") or "lat" is missing.')
        return
    
    # Compute first and second derivatives
    if derivative == 'dxdy':
        # Compute first derivatives by finite differences
        dy, dx = first_derivative(eta, n=n, axis=axis)
        dyy, dyx = first_derivative(dy, n=n, axis=axis)
        dxy, dxx = first_derivative(dx, n=n, axis=axis)
    elif derivative == 'fit':
        # Fit derivatives using the fitting kernel method
        dxx, dxy, dx, dyy, dy = fit_derivatives(eta, n=n, parallel=parallel, order=2, verbose=verbose, kernel = kernel)
        
    # Ensure the Coriolis parameter has the correct shape
    if f.shape != dx.shape:
        f = (np.ones((len(x), len(y))) * f).T

    # Compute scale factors
    e1, e2 = scale_factor(x, y, proj=proj)
    
    # Normalize derivatives by scale factors
    dxx, dxy, dx, dyy, dy = (
        dxx / (e1 ** 2), 
        dxy / (e1 * e2), 
        dx / np.abs(e1), 
        dyy / (e2 ** 2), 
        dy / np.abs(e2)
    )

    # Re-adjust the Coriolis parameter shape if necessary
    if f.shape != dxx.shape:
        f = (np.ones((len(x), len(y))) * f).T

    ### Calculate geostrophic currents and Cyclo-geostrophic currents (first approximation, see Tranchant et al., 2024)
    ug, vg = (-g / f * dy), (g / f * dx)  # Geostrophic currents
    ucg, vcg = (
        (-g / f * dy) + (g ** 2 / f ** 3) * (dy * dxx - dx * dxy), 
        (g / f * dx) + (g ** 2 / f ** 3) * (dy * dxy - dx * dyy)
    )  # Cyclo-geostrophic currents (first approximation)

    # Rotate currents along x and y axis
    if x.ndim > 1:
        theta = compute_angle(x, y, proj=proj)
        if eta.ndim == 3:
            # Rotate currents for 3D eta
            ug, vg = np.swapaxes([rotate(_u, _v, theta) for _u, _v in zip(ug, vg)], 0, 1)
            ucg, vcg = np.swapaxes([rotate(_u, _v, theta) for _u, _v in zip(ucg, vcg)], 0, 1)
        elif eta.ndim == 2:
            # Rotate currents for 2D eta
            ug, vg = rotate(ug, vg, theta)
            ucg, vcg = rotate(ucg, vcg, theta)
    # Store currents in diagnostics dictionary
    diag['ug'] = ug
    diag['vg'] = vg
    diag['ucg'] = ucg
    diag['vcg'] = vcg   

    ### Calculate strain rate
    Sn = 2 * (g / f) * dxy
    Ss = (g / f) * (dxx - dyy)
    S = np.sqrt(Sn ** 2 + Ss ** 2) / np.abs(f)

    diag['S'] = S

    ### Calculate relative vorticity
    zeta = ((g / f * dxx) + (g / f * dyy)) / f
    diag['zeta'] = zeta

    return diag