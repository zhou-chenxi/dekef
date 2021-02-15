import numpy as np
from denest_kernelexpofam.unnormalized_density import *


def metric_corr(data, test_data, kernel_function, base_density, coef, true_density):
    
    """
    Computes the value of the cosine angle and correlation between the estimated density values at test_data and
    the true density values at test_data as a way of measuring the quality of the estimated density function.
    Let p_0 be the true density function and hat{p} be the estimated density function.
    The value of the cosine angle is computed as
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
        Must be instantiated from the classes with __type__ being 'kernel_function'.
        
    base_density : base_density object
        The base density function used to estimate the probability density function.
        Must be instantiated from the classes with __type__ being 'base_density'.
    
    coef : numpy.ndarray
        The array of coefficients for basis functions in the natural parameter in the estimated density function.
    
    true_density : true_density object
        The true density function from which data are drawn.
        Must instantiated from the classes in true_density.
    
    Returns
    -------
    float, float
        The floating point numbers of the cosine angle and correlation
        between the estimated density values at test_data and the true density values at test_data.

    """
    
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    if len(test_data.shape) == 1:
        test_data = test_data.reshape(-1, 1)
    
    coef = coef.reshape(-1, 1)
        
    N, d = data.shape
    n, d1 = data.shape
    
    if d != d1:
        raise ValueError("The dimensionality of data does not match that of new_data.")
    
    # density estimate
    unnorm = UnnormalizedDensity(
        data=data,
        kernel_function=kernel_function,
        base_density=base_density,
        coef=coef)
    
    if coef.shape[0] == N:
        
        # using Gu's basis
        den_est = unnorm.density_eval_gubasis(test_data)
        
    elif coef.shape[0] == N * d + 1:
        
        # using score matching basis
        den_est = unnorm.density_eval_smbasis(test_data)
        
    else:
        
        raise ValueError(("The length of coef is not correct and matches neither Gu's basis functions "
                         "nor score matching basis functions."))
    
    true_den = true_density.density_eval(test_data)
    
    # Numerator
    num = np.sum(true_den * den_est)

    # Denomenator
    var1 = np.sum(true_den ** 2)
    var2 = np.sum(den_est ** 2)
    deno = np.sqrt(var1 * var2)

    return num / deno, np.corrcoef(true_den, den_est)[0, 1]
