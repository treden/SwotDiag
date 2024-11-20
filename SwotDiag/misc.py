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

def fit_surface(data, x=None, y=None, return_coef=False, proj='lonlat', order=1, x_order=None, y_order=None, direction='xy', kernel = 'circular'):
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
        # x, y = np.meshgrid(np.arange(0, data.shape[-1], 1), np.arange(0, data.shape[-2], 1))
        x, y = np.meshgrid(np.arange(-int(data.shape[-1]/2), int(data.shape[-1]/2) + 1, 1), np.arange(-int(data.shape[-2]/2), int(data.shape[-2]/2) + 1, 1))

    n = np.min(data.shape)

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

    if kernel == 'circular':
        center_x, center_y = 0, 0  # Assuming the center is at (0, 0) for meshgrid centered at origin
        distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        within_radius = distances <= n/2#int(n/2)  +1
        x = x[within_radius]
        y = y[within_radius]
        b = b[within_radius]

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
    surface = np.dot(A_orig, fit)
    # surface = surface

    if kernel == 'circular':
        center_x, center_y = 0, 0  # Assuming the center is at (0, 0) for meshgrid centered at origin
        distances = np.sqrt((x_orig - center_x) ** 2 + (y_orig - center_y) ** 2)
        within_radius = distances <= n/2 # int(n/2) +1
        surface[~within_radius] = np.nan

    return surface.reshape(data.shape).squeeze()

def fit_derivatives(E, n=9, order=1, parallel=True, verbose=True, kernel = 'circular', return_ssh = False, min_valid_points = 0.5):

    """
    Calculate the first and second-order spatial derivatives of a 2D surface 
    represented by the array E. This is achieved by applying a fitting kernel 
    on subsets of n points. The function supports parallel processing making it 
    particularly efficient for large datasets, such as those derived from 
    SWOT sea surface height (SSH) observations.

    Parameters:
        E : array_like
            A 2D (or N-dimensional) array representing the surface data 
            from which derivatives will be calculated.
        n : int, optional
            The size (number of points) of the kernel used for fitting. It must be an 
            odd integer. If an even integer is provided, it will be 
            incremented by 1. Default is 9.
        order : int, optional
            The order of derivatives to compute. Can be either 1 (first order) 
            or 2 (second order). Default is 1.
        parallel : bool, optional
            If True, computes in parallel using joblib. Default is True.
        verbose : bool, optional
            If True, displays a progress bar using tqdm. Default is True.

    Returns:
        tuple of ndarrays
            The fitted coefficients. For first-order fits, returns two 
            arrays representing the slope in the x and y directions. 
            For second-order fits, returns five arrays representing the 
            coefficients for x², xy, x, y², and y.

    Notes:
        If `n` is not an odd number, it will be incremented to the nearest odd number.
        If parallel processing or verbose mode cannot be enabled due to import errors, 
        the function will revert to sequential processing without progress bars.
    """
    
    # Check if n is odd, and increment if necessary
    if n % 2 != 1:
        print(f'n = {n} is not an odd number, increasing the kernel length to {n + 1}')
        n += 1

    E = np.asanyarray(E)
    m1, m2 = np.shape(E)

    def process_subset(j, i):
        """Process a subset of the input array to fit the surface."""
        subset = E[j-int(n/2):j+int(np.ceil(n/2)), i-int(n/2):i+int(np.ceil(n/2))]
        if ((n**2 - np.count_nonzero(np.isnan(subset))) > (n**2)*min_valid_points) and subset.size == (n**2):
            fit = fit_surface(subset, return_coef=True, order=order, kernel = kernel)
            return fit
        else:
            if order == 1:
                return np.nan, np.nan, np.nan
            elif order == 2:
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Attempt to import tqdm for progress tracking
    try:
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(range(m1), desc=f"Fitting derivatives on {m1*m2} points ({n}*{n} points kernel)")
        else:
            iterator = range(m1)
    except ImportError:
        print("TQDM library not available. Running without progress bar.")
        verbose = False
        iterator = range(m1)

    # Attempt to import joblib for parallel processing
    try:
        if parallel:
            from joblib import Parallel, delayed                
            results = Parallel(n_jobs=-1)(delayed(process_subset)(j, i) for j in iterator for i in range(m2))
        else:
            results = []
            for j in iterator:
                for i in range(m2):
                    results.append(process_subset(j, i))
    except ImportError:
        print("Joblib library not available. Running in sequential mode.")
        parallel = False
        results = []
        for j in iterator:
            for i in range(m2):
                results.append(process_subset(j, i))

    # Reshape results into appropriate arrays


    a1 = np.array([result[0] for result in results]).reshape((m1, m2))
    a2 = np.array([result[1] for result in results]).reshape((m1, m2))
    a3 = np.array([result[2] for result in results]).reshape((m1, m2))

    if order == 2:
        a4 = np.array([result[3] for result in results]).reshape((m1, m2))
        a5 = np.array([result[4] for result in results]).reshape((m1, m2))
        a6 = np.array([result[5] for result in results]).reshape((m1, m2))

        if return_ssh:
            return a1*2, a2, a3, a4*2, a5, a6
        else:
            return a1*2, a2, a3, a4*2, a5
    else:
        if return_ssh:
            return a1, a2, a3
        else:
            return a1, a2

def distance(lon1, lat1, lon2, lat2):
   
   #### Compute the distance on a sphere using Haversine formula

   #### TODO : Integrate ellipsoidal distance calculations 
    
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

        # TODO 

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
