import numpy as np
import pandas as pd

earthrad = 6371229     # mean earth radius (m)
deg2rad = np.pi / 180.

def first_derivative(E, n = 9, axis = (-2,-1)):

    """
    Returns the first derivative using an n-point stencil method, from Arbic et al. (2012), doi:10.1029/2011JC007367.

    Parameters:
        E : array_like
            A N-dimensional array.
        n : int, optional
            Number of points in the stencil (default is 9).
        axis : int or tuple of ints, optional
            Axis or axes along which to compute the derivative. Default is (-2,-1).

    Returns:
        ndarray
            The first derivative along the specified axis or axes.
    """

    E = np.asanyarray(E)
    axis = tuple(np.hstack([axis]))
    D = []

    if n == 3:
        D = np.gradient(E, axis = axis)

    elif n == 5:
        for ax in axis:
            e = E.swapaxes(ax, 0)
            de = np.zeros(np.shape(e))
            de[2:-2] = np.array(-e[4:] + 8*e[3:-1] - 8*e[1:-3] + e[:-4])/12
            
            de[:2] = first_derivative(e[:10], axis = 0, n = n-2)[:2]
            de[-2:] = first_derivative(e[-10:], axis = 0, n = n-2)[-2:] 
                        
            D.append(de.swapaxes(0, ax))

    elif n == 7:
        
        for ax in axis:
            e = E.swapaxes(ax, 0)
            de = np.zeros(np.shape(e))
            de[3:-3] = np.array(e[6:] - 9*e[5:-1] +45*e[4:-2] - 45*e[2:-4] + 9*e[1:-5] -e[:-6])/60
            

            de[:3] = first_derivative(e[:10], axis = 0, n = n-2)[:3]
            de[-3:] = first_derivative(e[-10:], axis = 0, n = n-2)[-3:] 
                        
            D.append(de.swapaxes(0, ax))

    elif n == 9:
        for ax in axis:
            e = E.swapaxes(ax, 0)
            de = np.zeros(np.shape(e))
            de[4:-4] = np.array(-3*e[8:]+32*e[7:-1]-168*e[6:-2] + 672*e[5:-3] - 672*e[3:-5]+ 168*e[2:-6] -32*e[1:-7] + 3*e[:-8])/840
            
            de[:4] = first_derivative(e[:10], axis = 0, n = n-2)[:4]
            de[-4:] = first_derivative(e[-10:], axis = 0, n = n-2)[-4:]

            D.append(de.swapaxes(0, ax))
            
    if len(D) == 1:
        D = D[0]
    return np.array(D)

def second_derivative(E, n = 9, axis = (-2,-1)):

    """
    Returns the second derivative using an n-point stencil method.

    Parameters:
        E : array_like
            A N-dimensional array.
        n : int, optional
            Number of points in the stencil (default is 9).
        axis : int or tuple of ints, optional
            Axis or axes along which to compute the derivative. Default is (-2,-1).

    Returns:
        ndarray
            The second derivative along the specified axis or axes.
    """

    E = np.asanyarray(E)
    axis = tuple(np.hstack([axis]))
    D = []
    
    if n == 3:
        for ax in axis:
            e = E.swapaxes(ax, 0)
            de = np.ones(np.shape(e))*np.nan
            de[1:-1] = np.array(e[2:] - 2*e[1:-1] + e[:-2])
            
            D.append(de.swapaxes(0, ax))
    
    elif n == 5:
        for ax in axis:
            e = E.swapaxes(ax, 0)
            de = np.zeros(np.shape(e))
            de[2:-2] = np.array(-e[4:] + 16*e[3:-1] - 30*e[2:-2] + 16*e[1:-3] - e[:-4])/12
            
            de[:2] = second_derivative(e[:10], axis = 0, n = n-2)[:2]
            de[-2:] = second_derivative(e[-10:], axis = 0, n = n-2)[-2:] 
                        
            D.append(de.swapaxes(0, ax))
    
    elif n == 7:
        
        for ax in axis:
            e = E.swapaxes(ax, 0)
            de = np.zeros(np.shape(e))
            de[3:-3] = np.array(2*e[6:] - 27*e[5:-1] +270*e[4:-2] - 490*e[3:-3] + 270*e[2:-4] - 27*e[1:-5] +2*e[:-6])/180
            
            de[:3] = second_derivative(e[:10], axis = 0, n = n-2)[:3]
            de[-3:] = second_derivative(e[-10:], axis = 0, n = n-2)[-3:] 
                        
            D.append(de.swapaxes(0, ax))
    
    elif n == 9:
        for ax in axis:
            e = E.swapaxes(ax, 0)
            de = np.zeros(np.shape(e))
            de[4:-4] = np.array(-9*e[8:]+128*e[7:-1]-1008*e[6:-2] + 8064*e[5:-3] - 14350*e[4:-4] +8064*e[3:-5]- 1008*e[2:-6] +128*e[1:-7] - 9*e[:-8])/5040

            de[:4] = second_derivative(e[:10], axis = 0, n = n-2)[:4]
            de[-4:] = second_derivative(e[-10:], axis = 0, n = n-2)[-4:]
    
            D.append(de.swapaxes(0, ax))
            
    if len(D) == 1:
        D = D[0]
    return D

def generate_polynomial_terms(x, y=None, order=2, x_order=None, y_order=None, return_str=False):
    """
    Generate polynomial terms for 1D or 2D data with different orders for x and y.

    Parameters:
        x : array_like
            The x-coordinates of the data points.
        y : array_like, optional
            The y-coordinates of the data points (default is None for 1D data).
        order : int, optional
            The overall order of the polynomial (default is 2).
        x_order : int or None, optional
            The order of the polynomial in the x direction (default is None).
            If None, it will be set to `order`.
        y_order : int or None, optional
            The order of the polynomial in the y direction (default is None).
            If None, it will be set to `order`.
        return_str : bool, optional
            If True, returns the terms as string expressions (default is False).

    Returns:
        list
            List of polynomial terms.
    """
    if isinstance(x_order, type(None)) or isinstance(y_order, type(None)):
        x_order = order
        y_order = order
    
    terms = []
    if y is None:  # 1D data
        max_order = max(x_order, y_order)
        for i in range(x_order, -1, -1):
            if i <= max_order:
                if return_str:
                    terms.append(f'x**{i}')
                else:
                    terms.append(x**i)
    else:  # 2D data
        max_x_order = max(x_order, y_order)
        max_y_order = max(x_order, y_order)

        for i in range(x_order, -1, -1):
            for j in range(y_order, -1, -1):
                if i + j <= max_x_order and i + j <= max_y_order:
                    if return_str:
                        terms.append(f'x**{i} * y**{j}')
                    else:
                        terms.append(x**i * y**j)

    return terms

def fit_surface(data, x=None, y=None, return_coef=False, proj='lonlat', order=1, x_order=None, y_order=None, direction='xy'):
    """
    Fits a surface to the given data.

    Parameters:
        data : array_like
            The data to fit the surface to.
        x : array_like, optional
            The x-coordinates of the data points.
        y : array_like, optional
            The y-coordinates of the data points.
        return_coef : bool, optional
            If True, returns the coefficients of the fitted surface (default is False).
        proj : str, optional
            Projection type ('lonlat' or 'xy', default is 'lonlat').
        order : int, optional
            The order of the surface to fit (1 for linear, 2 for quadratic, default is 1).
        x_order : int or None, optional
            The order of the polynomial in the x direction (default is None).
            If None, it will be set to `order`.
        y_order : int or None, optional
            The order of the polynomial in the y direction (default is None).
            If None, it will be set to `order`.
        direction : str, optional
            The direction for fitting ('xy' for both x and y, 'x' for only x, 'y' for only y, default is 'xy').
            Note that if x_order = 2 and y_order = 0 (for example), it is equivalent to order = 2 and direction = 'x'

    Returns:
        ndarray
            The fitted surface.
    """
    import scipy.linalg

    if isinstance(x_order, type(None)) or isinstance(y_order, type(None)):
        x_order = order
        y_order = order

    # Convert data to numpy array
    data = np.asarray(data)

    # If x and y are not provided, create them
    if x is None:
        x, y = np.meshgrid(np.arange(0, data.shape[-1], 1), np.arange(0, data.shape[-2], 1))

    # Flatten x and y
    x = np.ravel(x)
    y = np.ravel(y)
    x_orig, y_orig = x, y

    # Reshape data and handle NaN values
    if data.ndim > 2:
        b = data.reshape(data.shape[0], -1)
        surface = np.zeros(data.shape)
    else:
        b = np.ravel(data)
        ind_nan = np.isnan(b)
        b = b[~ind_nan]
        x = x[~ind_nan]
        y = y[~ind_nan]
        surface = np.zeros(np.hstack((1, data.shape)))

    # Set direction variable based on input
    if direction == 'x':
        a = x
        a_orig = x_orig
    elif direction == 'y':
        a = y
        a_orig = y_orig

    # Generate polynomial terms
    if direction == 'xy':
        A = np.column_stack(generate_polynomial_terms(x, y, order, x_order, y_order))
        A_orig = np.column_stack(generate_polynomial_terms(x_orig, y_orig, order, x_order, y_order))
    else:
        A = np.column_stack(generate_polynomial_terms(a, order, x_order, y_order))
        A_orig = np.column_stack(generate_polynomial_terms(a_orig, order, x_order, y_order))

    # Perform least squares regression
    fit, _, _, _ = scipy.linalg.lstsq(A, b.T)

    # Return coefficients if requested
    if return_coef:
        return fit
        
    # Reconstruct the surface using the original x and y arrays
    surface = np.dot(A_orig, fit).reshape(surface.shape)

    return surface.squeeze()


def fit_derivatives(E, n = 9, order = 1, parallel = True, verbose = True):

    """
    Fits the slope and curvature (first and second derivatives) from a surface.

    Parameters:
        E : array_like
            A N-dimensional array.
        n : int, optional
            Number of points in the stencil (default is 9).
        parallel : bool, optional
            If True, computes in parallel (default is True).

    Returns:
        tuple of ndarrays
            The fitted slope in the x and y directions.
    """

    if parallel:
        return fit_derivatives_parallel(E, n, order, verbose)
        
    E = np.asanyarray(E)
    m1,m2 = np.shape(E)
    
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(range(m1))
    else:
        iterator = range(m1)
        
    if order == 1:
        a1, a2 = np.ones((m1, m2))*np.nan, np.ones((m1, m2))*np.nan
        for j in iterator:
            for i in range(m2):
                subset = E[j-int(n/2):j+int(np.ceil(n/2)), i-int(n/2):i+int(np.ceil(n/2))]
                if (not np.isnan(subset).any())&(subset.size == n**2):
                    fit = fit_surface(subset, return_coef=True, order = order)
                    a1[j, i], a2[j, i] = fit[0], fit[1]
        return a1,a2
    
    elif order == 2:
        a1, a2, a3, a4, a5 = np.ones((m1, m2))*np.nan, np.ones((m1, m2))*np.nan, np.ones((m1, m2))*np.nan, np.ones((m1, m2))*np.nan, np.ones((m1, m2))*np.nan
        for j in iterator:
            for i in range(m2):
                subset = E[j-int(n/2):j+int(np.ceil(n/2)), i-int(n/2):i+int(np.ceil(n/2))]
                if (not np.isnan(subset).any())&(subset.size == n**2):
                    fit = fit_surface(subset, return_coef=True, order = order)
                    a1[j, i], a2[j, i], a3[j, i], a4[j, i], a5[j, i] = fit[0], fit[1], fit[2], fit[3],fit[4]
        return a1,a2,a3,a4,a5

def fit_derivatives_parallel(E, n=9, order = 1, verbose = True):
    
    from joblib import Parallel, delayed

    E = np.asanyarray(E)
    m1, m2 = np.shape(E)
    
    def process_subset(j, i):
        subset = E[j-int(n/2):j+int(np.ceil(n/2)), i-int(n/2):i+int(np.ceil(n/2))]
        if (np.count_nonzero(np.isnan(subset))<(n**2)/2.5) and (subset.size >= (n**2)):
        # if (subset.size == n**2):
            fit = fit_surface(subset, return_coef=True, order = order)
            return fit
        else:
            if order == 1:
                return np.nan, np.nan
            elif order == 2:
                return np.nan, np.nan, np.nan, np.nan, np.nan
                
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(range(m1))
    else:
        iterator = range(m1)
                
    results = Parallel(n_jobs=-1)(delayed(process_subset)(j, i) for j in iterator for i in range(m2))
    
    a1 = np.array([result[0] for result in results]).reshape((m1, m2))
    a2 = np.array([result[1] for result in results]).reshape((m1, m2))

    if order == 2:
        a3 = np.array([result[2] for result in results]).reshape((m1, m2))
        a4 = np.array([result[3] for result in results]).reshape((m1, m2))
        a5 = np.array([result[4] for result in results]).reshape((m1, m2))
        return a1, a2, a3, a4, a5
    else:
        return a1, a2

def distance(lon1, lat1, lon2, lat2):
    
   lat1 = deg2rad*(lat1)
   lon1 = deg2rad*(lon1)
   lat2 = deg2rad*(lat2)
   lon2 = deg2rad*(lon2)

   d = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)* np.cos(lat2) * np.sin((lon2 - lon1)/2)**2

   return 2 * earthrad * np.arcsin(np.sqrt(d))

def compute_dxdy(x, y, proj = 'lonlat'):
    x, y = np.asanyarray(x), np.asanyarray(y)
    
    if x.ndim ==1:
        x, y = np.meshgrid(x, y)

    djx, dix = np.gradient(x)
    djy, diy = np.gradient(y)
    
    if proj == 'lonlat':
        dix = dix*111320*np.cos(deg2rad*y)
        djx = djx*111320*np.cos(deg2rad*y)
        diy = diy*110574
        djy = djy*110574

    else:
        print('Proj must be "lonlat" or "xy"')
        return

    return dix, djx, diy, djy

def compute_angle(x, y, proj = 'lonlat'):
    
    dix, djx, diy, djy = compute_dxdy(x, y, proj = proj)
    return np.angle((dix) + 1j*(diy))

def rotate(u, v, theta):
    new_u = (u*np.cos(theta) - v*np.sin(theta))
    new_v = (u*np.sin(theta) + v*np.cos(theta))
    return new_u, new_v

def scale_factor(x, y, proj ='lonlat'):

    x, y = np.asanyarray(x), np.asanyarray(y)
    
    if x.ndim ==1:
        x, y = np.meshgrid(x, y)

    djx, dix = np.gradient(x,edge_order = 2)
    djy, diy = np.gradient(y,edge_order = 2)
    
    if proj == 'lonlat':
        e1 = 2 * earthrad * np.arcsin(np.sqrt(np.sin(deg2rad*diy/2)**2 + np.cos(deg2rad*(y - diy/2))*np.cos(deg2rad*(y + diy/2)) * np.sin(deg2rad*(dix)/2)**2))
        e2 = 2 * earthrad * np.arcsin(np.sqrt(np.sin(deg2rad*djy/2)**2 + np.cos(deg2rad*(y - djy/2))*np.cos(deg2rad*(y + djy/2)) * np.sin(deg2rad*(djx)/2)**2))

    elif proj == 'xy':
        e1 = dix
        e2 = djy

    else:
        print('Proj must be "lonlat" or "xy"')
        return
    
    e1, e2 = e1*np.sign(dix), e2*np.sign(djy)

    return e1, e2

def build_regridder(input, output, method='bilinear'):
    """
    Builds a regridder object using xESMF (xarray extension for geospatial data) to interpolate data
    from a source grid to a target grid.
    
    Parameters:
    - input (xarray.Dataset): The dataset providing the source grid.
    - output (xarray.Dataset): The dataset providing the target grid.
    - method (str, optional): The method used for regridding. Default is 'bilinear'.
    
    Returns:
    - regridder (xesmf.Regridder): The regridder object configured for regridding data from the source grid to the target grid.
    """
    
    import xesmf as xe

    # Rename longitude and latitude to lon and lat if they exist in the input and output datasets
    if 'longitude' in input.coords:
        input = input.rename(longitude='lon', latitude='lat')
    if 'longitude' in output.coords:
        output = output.rename(longitude='lon', latitude='lat')

    # Create the regridder object
    regridder = xe.Regridder(input, output, method=method, unmapped_to_nan=True)
    return regridder

def regrid(input, regridder):
    """
    Regrids the input field using the provided regridder object.
    
    Parameters:
    - input (xarray.DataArray): The field to be regridded.
    - regridder (xesmf.Regridder): The regridder object configured for regridding from a specific source grid to target grid.
    
    Returns:
    - interpolated_field (xarray.DataArray): The regridded field.
    """
    # Use the regridder object to interpolate the input field
    interpolated_field = regridder(input)
    return interpolated_field.rename(lon = 'longitude', lat = 'latitude')


def interp_xyt(ds, var, x_to=None, y_to=None, t_to=None, ds_dims = {'time' : 'time', 'latitude': 'latitude', 'longitude': 'longitude'}):

    from scipy.interpolate import RegularGridInterpolator, interp1d
    
    if ('time' in ds_dims.keys()) & ('longitude' in ds_dims.keys()):
        X, Y, T = ds[ds_dims['longitude']].values,ds[ds_dims['latitude']].values,ds[ds_dims['time']].values
        T, t_to = pd.DatetimeIndex(T).to_julian_date(), pd.DatetimeIndex(t_to).to_julian_date()
        var = ds[var].transpose(ds_dims['time'], ds_dims['latitude'], ds_dims['longitude']).values
        interp = RegularGridInterpolator((T, Y, X), var)(np.array([t_to, y_to, x_to]).T)
        
    elif ('time' in ds_dims.keys()) & ('longitude' not in ds_dims.keys()):
        T = ds[ds_dims['time']].values
        T, t_to = pd.DatetimeIndex(T).to_julian_date(), pd.DatetimeIndex(t_to).to_julian_date()
        var = ds[var].values
        interp = interp1d(T, var)(t_to)
        
    elif ('time' not in ds_dims.keys()) & ('longitude' in ds_dims.keys()):
        X, Y = ds[ds_dims['longitude']].values,ds[ds_dims['latitude']].values
        var = ds[var].transpose(ds_dims['latitude'], ds_dims['longitude']).values
        interp = RegularGridInterpolator((Y, X), var)(np.array([y_to, x_to]).T)

    ds.close()
    del ds 
    return interp
