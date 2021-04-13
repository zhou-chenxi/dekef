import numpy as np
from dekef.kernel_function import *
from dekef.base_density import *


def scorematching_finexpfam_gubasis_coef(data, kernel_function, base_density):
	
	"""
	Returns the solution to minimizing the score matching loss function
	in finite-dimensional exponential family whose canonical statistics are
	kernel functions with one argument evaluated at data.

	Parameters
	----------
	data : numpy.ndarray
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
		An array of coefficients for the natural parameter in the score matching density estimate.

	"""
	
	N, d = data.shape
	n_basis = kernel_function.data.shape[0]
	
	# compute DT(X_i) DT(X_i)^\top
	DT = kernel_function.partial_kernel_matrix_01(data)
	dt_prod_term = sum([np.matmul(DT[:, i].reshape(n_basis, d),
								  DT[:, i].reshape(n_basis, d).T) for i in range(N)])
	
	# compute the matrix G, which involves the second derivative
	matG = kernel_function.partial_kernel_matrix_02(data)
	sum_matG = np.sum(sum([matG[:, i].reshape(n_basis, d) for i in range(N)]), axis=1, keepdims=True)
	
	# compute DT and grad of log mu
	# compute the gradient of log mu at data
	# each row corresponds to one data point
	grad_logmu = np.array([base_density.logbaseden_deriv1(data, j).flatten() for j in range(d)]).T
	dt_baseden_term = sum([np.matmul(DT[:, i].reshape(n_basis, d), grad_logmu[[i]].T) for i in range(N)])
	
	b_term = sum_matG + dt_baseden_term
	
	coef = -np.linalg.lstsq(dt_prod_term, b_term, rcond=None)[0]
	
	return coef
