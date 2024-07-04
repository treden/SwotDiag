import numpy as np
import scipy
from .misc import *

g = 9.81
omega = 7.292115e-5  # (1/s)   (Groten, 2004).

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
    
def compute_geostrophic_velocities(eta, x, y, proj = 'lonlat', derivative = 'dxdy', lat = None, n = 9, parallel = True, axis = (-2,-1), verbose = True):

    """
    Computes the geostrophic velocities.

    Parameters:
        eta : array_like
            A N-dimensional array representing the height field.
        x : array_like
            The x-coordinates of the grid.
        y : array_like
            The y-coordinates of the grid.
        proj : str, optional
            Projection type ('lonlat' or 'xy', default is 'lonlat').
        derivative : str, optional
            Method for computing derivatives ('dxdy' or 'fit', default is 'dxdy').
        lat : float, optional
            Latitude for computing the Coriolis parameter (default is None).
        n : int, optional
            Number of points in the stencil or in the kernel (default is 9).
        parallel : bool, optional
            If True, computes in parallel (default is True).
        axis : int or tuple of ints, optional
            Axis or axes along which to compute the derivative. Default is (-2,-1).

    Returns:
        tuple of ndarrays
            The geostrophic velocities in the x and y directions.
    """

    x, y = np.asanyarray(x), np.asanyarray(y)

    axis = tuple(np.hstack([axis]))
        
    if proj == 'lonlat':
        f = compute_f(y)
    elif (proj == 'xy')&(not isinstance(lat,type(None))):
        f = compute_f(lat)
    else:
        print('Bad "proj" argument (must be "lonlat" or "xy") or bad "lat" argument')
        return
        
    # dyeta, dxeta = np.gradient(eta, axis = (-2,-1))
    if derivative == 'dxdy':
        dyeta, dxeta = first_derivative(eta, n=n, axis = axis)
    elif derivative == 'fit':
        fit = fit_derivatives(eta, n = n, parallel = parallel, order = 1, verbose = verbose)
        dxeta, dyeta = fit[0], fit[1]

    if f.shape != dxeta.shape:
        f = (np.ones((len(x),len(y)))*f).T

    e1, e2 = scale_factor(x, y, proj = proj)
    ug = (-g/f*dyeta)/np.abs(e2)
    vg = (g/f*dxeta)/np.abs(e1)

    if x.ndim > 1:
        theta = compute_angle(x, y, proj = proj)
        if eta.ndim == 3:
            ug, vg = np.swapaxes([rotate(_ug , _vg , theta) for _ug, _vg in zip(ug, vg)], 0, 1)
        elif eta.ndim == 2:
            ug, vg = rotate(ug , vg , theta)

    return ug, vg

def compute_relative_vorticity_from_eta(eta, x, y, proj = 'lonlat', derivative = 'dxdy', lat = None, n = 9, parallel = True, axis = (-2,-1), verbose = True):
    """
    Computes the relative vorticity from the surface height.

    Parameters:
        eta : array_like
            A N-dimensional array representing the height field.
        x : array_like
            The x-coordinates of the grid.
        y : array_like
            The y-coordinates of the grid.
        proj : str, optional
            Projection type ('lonlat' or 'xy', default is 'lonlat').
        derivative : str, optional
            Method for computing derivatives ('dxdy' or 'fit', default is 'dxdy').
        lat : float, optional
            Latitude for computing the Coriolis parameter (default is None).
        n_stencil : int, optional
            Number of points in the stencil (default is 9).

    Returns:
        ndarray
            The relative vorticity.
    """

    x, y = np.asanyarray(x), np.asanyarray(y)

    if proj == 'lonlat':
        f = compute_f(y)
    elif (proj == 'xy')&(not isinstance(lat,type(None))):
        f = compute_f(lat)
    else:
        print('Bad "proj" argument (must be "lonlat" or "xy") or bad "lat" argument')
        return
        
    if derivative == 'dxdy':
        dyeta, dxeta = first_derivative(eta, n=n, axis = axis)
        dyyeta = first_derivative(dyeta, n=n, axis = axis)[0]
        dxxeta = first_derivative(dxeta, n=n, axis = axis)[1]

    elif derivative == 'fit':
        fit = fit_derivatives(eta, n = n, parallel = parallel, order = 1, verbose = verbose)
        dxeta, dyeta = fit[0], fit[1]

        dxxeta = fit_derivatives(dxeta, n = n, parallel = parallel, order = 1, verbose = verbose)[0]
        dyyeta = fit_derivatives(dyeta, n = n, parallel = parallel, order = 1, verbose = verbose)[1]
    

    # if derivative == 'dxdy':


    #     dxxeta = second_derivative(eta, n = n, axis = axis)[1]
    #     dyyeta = second_derivative(eta, n = n, axis = axis)[0]
    # elif derivative == 'fit':
        
    #     # fit =  fit_derivatives(eta, n = n, parallel = parallel, order = 2, verbose = verbose)
    #     # dxxeta, dyyeta, dxdyeta = fit[0], fit[3], fit[1]

    if f.shape != dxxeta.shape:
        f = (np.ones((len(x),len(y)))*f).T
        
    e1, e2 = scale_factor(x, y, proj = proj)
    zeta = ((g/f*dxxeta)/e1**2 + (g/f*dyyeta)/e2**2)/f

    return zeta

def compute_relative_vorticity_from_uv(u, v, x, y, proj = 'lonlat', derivative = 'dxdy', lat = None, n = 9, parallel = True, axis = (-2,-1), verbose = True):

    x, y = np.asanyarray(x), np.asanyarray(y)

    if proj == 'lonlat':
        f = compute_f(y)
    elif (proj == 'xy')&(not isinstance(lat,type(None))):
        f = compute_f(lat)
    else:
        print('Bad "proj" argument (must be "lonlat" or "xy") or bad "lat" argument')
        return
        
    if derivative == 'dxdy':
        dxv = first_derivative(v, n = n, axis = axis)[1]
        dyu = first_derivative(u, n = n, axis = axis)[0]
    elif derivative == 'fit':
        dxu, dyu = fit_derivatives(u, n = n, parallel = parallel, order = 1, verbose = verbose)
        dxv, dyv = fit_derivatives(v, n = n, parallel = parallel, order = 1, verbose = verbose)

    if f.shape != dxv.shape:
        f = (np.ones((len(x),len(y)))*f).T
        
    e1, e2 = scale_factor(x, y, proj = proj)
    zeta = (dxv/e1 - dyu/e2)/f
    
    return zeta

def compute_strain_rate_from_eta(eta, x, y, proj = 'lonlat', derivative = 'dxdy', lat = None, n_stencil = 9, parallel = True, axis = (-2,-1)):

    """
    Computes the strain rate from the surface height.

    Parameters:
        eta : array_like
            A N-dimensional array representing the height field.
        x : array_like
            The x-coordinates of the grid.
        y : array_like
            The y-coordinates of the grid.
        proj : str, optional
            Projection type ('lonlat' or 'xy', default is 'lonlat').
        derivative : str, optional
            Method for computing derivatives ('dxdy' or 'fit', default is 'dxdy').
        lat : float, optional
            Latitude for computing the Coriolis parameter (default is None).
        n_stencil : int, optional
            Number of points in the stencil (default is 9).

    Returns:
        ndarray
            The strain rate.
    """

    x, y = np.asanyarray(x), np.asanyarray(y)

    if proj == 'lonlat':
        f = compute_f(y.mean())
    elif (proj == 'xy')&(not isinstance(lat,type(None))):
        f = compute_f(lat)
    else:
        print('Bad "proj" argument (must be "lonlat" or "xy") or bad "lat" argument')
        return
        
    if derivative == 'dxdy':
        dxxeta = second_derivative(eta, n = n_stencil, axis = axis)[1]
        dyyeta = second_derivative(eta, n = n_stencil, axis = axis)[0]
        dxdyeta = first_derivative(first_derivative(eta, n = n_stencil, axis = axis)[0], n = n_stencil, axis = axis)[1]

    elif derivative == 'fit':
        fit =  fit_derivatives(eta, n = n_stencil, parallel = parallel, order = 2)
        dxxeta, dyyeta, dxdyeta = fit[0], fit[3], fit[1]
        
    e1, e2 = scale_factor(x, y, proj = proj)
    
    Sn = 2*(g/f)*dxdyeta/(e1*e2)
    Ss = (g/f)*(dxxeta/(e1**2) - dyyeta/(e2**2))
    
    S = np.sqrt(Sn**2 + Ss**2)/np.abs(f)

    return S

def compute_EKE(ug, vg, ug_mean, vg_mean):
    
    """
    Computes the eddy kinetic energy.

    Parameters:
        ug : array_like
            A N-dimensional array representing the zonal velocity anomaly.
        vg : array_like
            A N-dimensional array representing the meridional velocity anomaly.
        ug_mean : array_like
            A N-dimensional array representing the mean zonal velocity.
        vg_mean : array_like
            A N-dimensional array representing the mean meridional velocity.

    Returns:
        ndarray
            The eddy kinetic energy.
    """

    return 0.5*((ug - ug_mean)**2 + (vg - vg_mean)**2)
