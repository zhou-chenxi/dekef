import numpy as np
from dekef.check import *


def scorematching_loss_function(data, new_data, coef, kernel_function, base_density):
    
    """
    Evaluate the score matching loss function on new_data, 
    where the density function is estimated using data.
    
    Parameters
    ----------
    data : numpy.ndarray
        The array of observations whose density function is to be estimated.
        
    new_data : ndarray 
        The array of data at which the score matching loss function is to be evaluated.
        
    coef : ndarray
        The array of coefficients at which the score matching loss function is to be evaluated.
        Must be of shape (data.shape[0] * data.shape[1] + 1, 1).
        
    kernel_function : kernel_function object
        The kernel function used to estimate the probability density function.
        __type__ must be 'kernel_function'.
        
    base_density : base_density object
        The base density function used to estimate the probability density function.
        __type__ must be 'base_density'.

    Returns
    -------
    float
        The value of the score matching loss function evaluated on new_data.
        
    """
    
    check_kernelfunction(kernel_function)
    check_basedensity(base_density)
    
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
        
    if len(new_data.shape) == 1:
        new_data = new_data.reshape(-1, 1)
    
    N, d = data.shape
    n = new_data.shape[0]
    
    if len(coef.shape) == 1:
        coef = coef.reshape(-1, 1)
    
    if len(coef) != N * d + 1: 
        raise ValueError("The shape of coef should be ({true_shape}, 1), but got {got_shape}".format(
            true_shape=N * d + 1, got_shape=coef.shape))
    
    # compute matrices involving the partial derivatives of the kernel function 
    A11 = kernel_function.partial_kernel_matrix_11(new_data=new_data)
    A12 = kernel_function.partial_kernel_matrix_12(new_data=new_data)
    A21 = kernel_function.partial_kernel_matrix_21(new_data=new_data)
    A22 = kernel_function.partial_kernel_matrix_22(new_data=new_data)
    
    # partial derivatives of log base density at data 
    baseden_partial = np.zeros(data.shape, dtype=np.float64)
    for u in range(data.shape[1]): 
        baseden_partial[:, u] = base_density.logbaseden_deriv1(new_data=data, j=u).flatten()
    baseden_partial = baseden_partial.flatten()
    
    # form tilde A1
    zx11 = -np.sum(np.matmul(np.diag(baseden_partial), A11), axis=0, keepdims=True) / N
    zx12 = -np.sum(A21, axis=0, keepdims=True) / N
    zx1 = zx11 + zx12
    tildeA1 = np.vstack((A11, zx1))
    
    # form tilde A2
    zx21 = -np.sum(np.matmul(np.diag(baseden_partial), A12), axis=0, keepdims=True) / N
    zx22 = -np.sum(A22, axis=0, keepdims=True) / N
    zx2 = zx21 + zx22
    tildeA2 = np.vstack((A12, zx2))
    
    # partial derivatives of log base density at new_data 
    baseden_partial_new = np.zeros(new_data.shape, dtype=np.float64)
    for u in range(new_data.shape[1]): 
        baseden_partial_new[:, u] = base_density.logbaseden_deriv1(new_data=new_data, j=u).flatten()
    baseden_partial_new = baseden_partial_new.flatten()
    
    result1 = (0.5 * np.sum(np.matmul(coef.T, tildeA1) ** 2) / n).flatten()
    result2 = np.sum(np.matmul(coef.T, tildeA2), axis=1) / n
    result3 = (np.matmul(np.matmul(coef.T, tildeA1), baseden_partial_new.reshape(-1, 1)) / n).flatten()
    
    result = (result1 + result2 + result3)[0]
    
    return result
