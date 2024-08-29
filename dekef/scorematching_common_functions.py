import numpy as np
from dekef.check import *


def kernel_partial10_hatz(data, new_data, kernel_function, base_density):
    
    """
    Returns an array of shape (data.shape[0] * data.shape[1] + 1, new_data.shape[0]) with
    - the ((i-1)*d+u, j)-th entry being partial_u k (X_i, Y_j), and
    - the (nd + 1, j)-th entry being the inner product between hat{z} and k (Y_j, .),
    for all i = 1, ..., data.shape[0], u = 1, ..., data.shape[1], and j = 1, ..., new_data.shape[0],
    where X_i is the i-th row of data and Y_j is the j-th row of new_data.
    This array is used to evaluate the natural parameter at new_data.
    
    Parameters
    ----------
    data : numpy.ndarry
        The array of observations whose density function is to be estimated.
    
    new_data : numpy.ndarray
        The array of points at which the natural parameter is to be evaluated.
        
    kernel_function : kernel_function object
        The kernel function used to estimate the probability density function.
        __type__ must be 'kernel_function'.
        
    base_density : base_density object
        The base density function used to estimate the probability density function.
        __type__ must be 'base_density'.
        
    Returns
    -------
    numpy.ndarray
        An array of shape (data.shape[0] * data.shape[1] + 1, new_data.shape[0]).
        See the description above for details.
        
    """
    
    check_kernelfunction(kernel_function)
    check_basedensity(base_density)
    
    # ------------------------------------------------------------------------------
    # evaluate hat{z}
    # partial derivatives of kernel function 
    kernel_partial_10 = kernel_function.partial_kernel_matrix_10(new_data=new_data)
    kernel_partial_20 = kernel_function.partial_kernel_matrix_20(new_data=new_data)
    
    # partial derivatives of log base density 
    baseden_partial = np.zeros(data.shape, dtype=np.float64)
    for u in range(data.shape[1]): 
        baseden_partial[:, u] = base_density.logbaseden_deriv1(new_data=data, j=u).flatten()
    
    baseden_partial = baseden_partial.flatten()
    
    hatz1 = -np.sum(kernel_partial_10 * baseden_partial[:, np.newaxis], axis=0).reshape(1, -1) / data.shape[0]
    hatz2 = -np.sum(kernel_partial_20, axis=0).reshape(1, -1) / data.shape[0]
    hatz = hatz1 + hatz2
    
    # the upper nd * new_data.shape[0] matrix corresponds to \partial_1 k(data, new_data)
    # the last row corresponds to \hat{z}(new_data)
    output = np.vstack([kernel_partial_10, hatz])
    
    return output
    

def sq_rkhs_norm_matrix(data, kernel_function, base_density):
    
    """
    Returns the Gram matrix of shape (data.shape[0] * data.shape[1] + 1, data.shape[0] * data.shape[0] + 1) to
    compute the squared RKHS norm of the natural parameter. It can be decomposed as the following blocked matrix
    
        A    |            h
        ---------------------------------
        h^T | ||z||_{mathcal{H}}^2
    
    where
    - A is a matrix of shape (data.shape[0] * data.shape[1], data.shape[0] * data.shape[1]) with
    the ((i-1)*d+u, (i-1)*d+u)-th entry being partial_u partial_{v+d} k (X_i, X_j),
    - h is a matrix of shape (data.shape[0] * data.shape[1], 1) with the ((i-1)*d+u, 1)-th entry being the inner
      product between hat{z} and partial_u k (X_i, .),
    - ||hat{z}||_{mathcal{H}}^2 is the squared RKHS norm of hat{z}.
    
    Parameters
    ----------
    data : numpy.ndarry
        The array of observations whose density function is to be estimated.
        
    kernel_function : kernel_function object
        The kernel function used to estimate the probability density function.
        __type__ must be 'kernel_function'.
        
    base_density : base_density object
        The base density function used to estimate the probability density function.
        __type__ must be 'base_density'.
        
    Returns
    -------
    numpy.ndarray
        An array of shape (data.shape[0] * data.shape[1] + 1, data.shape[0] * data.shape[0] + 1).
        See the description above for details.
    
    """
    
    check_kernelfunction(kernel_function)
    check_basedensity(base_density)
    
    # entries are \partial_u \partial_v k(X_i, X_j)
    kernel_partial_11 = kernel_function.partial_kernel_matrix_11(new_data=data)
    
    # entries are <\hat{z}, \partial_u k(X_i, \cdot)>
    kernel_partial_12 = kernel_function.partial_kernel_matrix_12(new_data=data)
    baseden_partial = np.zeros(data.shape, dtype=np.float64)
    for u in range(data.shape[1]): 
        baseden_partial[:, u] = base_density.logbaseden_deriv1(new_data=data, j=u).flatten()
    
    baseden_partial = baseden_partial.reshape(-1, 1)

    h1 = np.sum(kernel_partial_12, axis=1).reshape(-1, 1) / data.shape[0]
    h2 = np.sum(np.matmul(kernel_partial_11, baseden_partial), axis=1).reshape(-1, 1) / data.shape[0]
    h = -(h1 + h2)  # the negative sign comes from the \hat{z} part
    
    # |\hat{z}|^2
    z_norm1 = np.matmul(baseden_partial.T, np.matmul(kernel_partial_11, baseden_partial))
    z_norm2 = np.sum(baseden_partial * kernel_partial_12)
    kernel_partial_21 = kernel_function.partial_kernel_matrix_21(new_data=data)
    z_norm3 = np.sum(np.matmul(kernel_partial_21, baseden_partial))
    z_norm4 = np.sum(kernel_function.partial_kernel_matrix_22(new_data=data))
    z_norm = (z_norm1 + z_norm2 + z_norm3 + z_norm4) / data.shape[0] ** 2
    
    output = np.vstack([np.hstack([kernel_partial_11, h]), np.hstack([h.T, z_norm])])
    
    return output


def vector_h(data, kernel_function, base_density):
    
    """
    Returns the vector whose (i-1)*d+u element is the inner product between hat{z} and partial_u k (X_i, .) and
    is equal to
    - (1 / N) sum_{j=1}^N sum_{v=1}^d (partial_u partial_{v+d}^2 k (X_i, X_j) +
                                            partial_v log mu (X_j) partial_u partial_{v+d} k (X_i, X_j)),
    where N = data.shape[0], d = data.shape[1], mu is the base density function, and X_i is the i-th row of data.
                                            
    Parameters
    ----------
    data : numpy.ndarry
        The array of observations whose density function is to be estimated.
        
    kernel_function : kernel_function object
        The kernel function used to estimate the probability density function.
        __type__ must be 'kernel_function'.
        
    base_density : base_density object
        The base density function used to estimate the probability density function.
        __type__ must be 'base_density'.
        
    Returns
    -------
    numpy.ndarray
        An array of shape (data.shape[0] * data.shape[1], 1).
        See the description above for details.
    
    """
    
    check_kernelfunction(kernel_function)
    check_basedensity(base_density)
    
    # entries are \partial_u \partial_v k(X_i, X_j)
    kernel_partial_11 = kernel_function.partial_kernel_matrix_11(new_data=data)
    
    # entries are <\hat{z}, \partial_u k(X_i, \cdot)>
    kernel_partial_12 = kernel_function.partial_kernel_matrix_12(new_data=data)
    baseden_partial = np.zeros(data.shape, dtype=np.float64)
    
    for u in range(data.shape[1]): 
        baseden_partial[:, u] = base_density.logbaseden_deriv1(new_data=data, j=u).flatten()
    
    baseden_partial = baseden_partial.reshape(-1, 1)

    h1 = np.sum(kernel_partial_12, axis=1).reshape(-1, 1) / data.shape[0]
    h2 = np.sum(np.matmul(kernel_partial_11, baseden_partial), axis=1).reshape(-1, 1) / data.shape[0]
    h = -(h1 + h2)  # the negative sign comes from the \hat{z} part
    
    return h 
