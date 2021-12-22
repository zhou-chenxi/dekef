import numpy as np
from dekef.unnormalized_density import *
from dekef.check import *


def metric_corr(data, test_data, kernel_function, base_density, coef, true_density, basis_type, grid_points=None):
    
    """
    Computes the cosine value of the angle and the Pearson's correlation between the density estimates and
    the true density values at test_data as a way of measuring the quality of the estimated density function.
    Let p_0 be the true density function and hat{p} be the estimated density function.
    The cosine value of the angle is computed as
      E (p_0 (X) hat{p} (X)) / sqrt{E (p_0 ^ 2 (X)) E (hat{p} ^ 2 (X))},
    where the expectations is estimated using test_data.
    The correlation is the sample Pearson's correlation coefficient.
    
    Parameters
    ----------
    data : numpy.ndarray
        The array of observations with which the density function is estimated.
    
    test_data : numpy.ndarray
        The array of data from the true density function using which the quality of the estimated density function is
        to be assessed.
    
    kernel_function : kernel_function object
        The kernel function used to estimate the probability density function.
        __type__ must be 'kernel_function'.
        
    base_density : base_density object
        The base density function used to estimate the probability density function.
        __type__ must be 'base_density'.
    
    coef : numpy.ndarray
        The array of coefficients of basis functions in the natural parameter in the estimated density function.
    
    true_density : true_density object
        The true density function from which data are drawn.
    
    basis_type : str
        The type of the basis functions in the natural parameter. Must be one of
            - 'gubasis', the basis functions being the kernel functions centered at data,
            - 'smbasis', the basis functions being the same as those in the score matching density estimator, i.e.,
                         a linear combination of the first two partial derivatives of the kernel functions centered
                         at data,
            - 'grid_points', the basis functions being the kernel functions centered at a set of
                             pre-specified grid points.
    
    grid_points : numpy.ndarray, optional
        The set of grid points at which the kernel functions are centered.
        Only need to supple when basis_type is 'grid_points'. Default is None.
    
    Returns
    -------
    float, float
        The floating point numbers of the cosine value of the angle and the correlation
        between the density estimates and the true density values at test_data.

    """

    check_kernelfunction(kernel_function)
    check_basedensity(base_density)

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    if len(test_data.shape) == 1:
        test_data = test_data.reshape(-1, 1)
    
    if len(grid_points.shape) == 1:
        grid_points = grid_points.reshape(-1, 1)
    
    coef = coef.reshape(-1, 1)
        
    N, d = data.shape
    n, d1 = test_data.shape
    
    if d != d1:
        raise ValueError("The dimensionality of data does not match that of new_data.")
    
    # density estimate
    unnorm = UnnormalizedDensity(
        data=data,
        kernel_function=kernel_function,
        base_density=base_density,
        coef=coef,
        basis_type=basis_type,
        grid_points=grid_points
    )
    
    if basis_type == 'gubasis':
        
        # using Gu's basis
        den_est = unnorm.density_eval_gubasis(test_data)
        
    elif basis_type == 'smbasis':
        
        # using score matching basis
        den_est = unnorm.density_eval_smbasis(test_data)
    
    elif basis_type == 'grid_points':
    
        # using grid points basis
        den_est = unnorm.density_eval_grid_points(test_data)
        
    else:
        
        raise ValueError(f"basis_type must be one of 'gubasis', 'smbasis', and 'grid_points', but got {basis_type}.")
    
    true_den = true_density.density_eval(test_data)
    
    # Numerator
    num = np.sum(true_den * den_est)

    # Denominator
    var1 = np.sum(true_den ** 2)
    var2 = np.sum(den_est ** 2)
    deno = np.sqrt(var1 * var2)

    return num / deno, np.corrcoef(true_den, den_est)[0, 1]
