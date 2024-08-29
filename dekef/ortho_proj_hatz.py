import numpy as np
from dekef.check import *


def hatz(data, new_data, kernel_function, base_density):
	
	"""
	Evaluates hat{z} at new_data, where
	hat{z} = (1 / N) sum_{j=1}^N sum_{v=1}^d (partial_v^2 k (X_j, .) + partial_v log mu (X_j) partial_v^2 k (X_j, .)),
	N = data.shape[0], d = data.shape[1], mu is the base density, X_j is the j-th row of data.
	
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
		A 1-dimensional array of hat{z} evaluated at new_data.
	
	"""
	
	check_kernelfunction(kernel_function)
	check_basedensity(base_density)
	
	if len(data.shape) == 1:
		data = data.reshape(-1, 1)
	
	if len(new_data.shape) == 1:
		new_data = new_data.reshape(-1, 1)
	
	if data.shape[1] != new_data.shape[1]:
		raise ValueError('The dimensionality of data and that of new_data do not match.')
	
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
	output = hatz1 + hatz2
	
	return output


def innerp_hatz_partial_kernel(data, kernel_function, base_density):
	
	"""
	Evaluates the inner product between hat{z} and partial_u k (X_i, .), which is equal to
	(1 / N) sum_{j=1}^N sum_{v=1}^d (partial_u partial_v^2 k (X_i, X_j) +
	partial_v log mu (X_j) partial_u partial_v k (X_i, X_j)).
	Here,
	hat{z} = (1 / N) sum_{j=1}^N sum_{v=1}^d (partial_v^2 k (X_j, .) + partial_v log mu (X_j) partial_v k (X_j, .)),
	where N = data.shape[0], d = data.shape[1], mu is the base density, X_j is the j-th row of data.
	
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
		An array of shape (data.shape[0] * data.shape[1],) whose ((i-1)*d+u)-th element is
		the inner product between hat{z} and partial_u k (X_i, .)
		for i = 1, ..., data.shape[0], u = 1, ...,, data.shape[1].
		
	"""
	
	check_kernelfunction(kernel_function)
	check_basedensity(base_density)
	
	if len(data.shape) == 1:
		data = data.reshape(-1, 1)
	
	kernel_partial_11 = kernel_function.partial_kernel_matrix_11(new_data=data)
	kernel_partial_12 = kernel_function.partial_kernel_matrix_12(new_data=data)
	baseden_partial = np.zeros(data.shape, dtype=np.float32)
	for u in range(data.shape[1]):
		baseden_partial[:, u] = base_density.logbaseden_deriv1(new_data=data, j=u).flatten()
	baseden_partial = baseden_partial.reshape(-1, 1)
	
	h1 = np.sum(kernel_partial_12, axis=1).reshape(-1, 1) / data.shape[0]
	h2 = np.sum(np.matmul(kernel_partial_11, baseden_partial), axis=1).reshape(-1, 1) / data.shape[0]
	h = -(h1 + h2)  # the negative sign comes from the hat{z} part
	
	return h.flatten()


def hatz_projection_C(data, new_data, kernel_function, base_density):
	
	"""
	Evaluates the orthogonal projection of hat{z} onto the range of hat{C} at new_data, where
	hat{z} = (1 / N) sum_{j=1}^N sum_{v=1}^d (partial_v^2 k (X_j, .) + partial_v log mu (X_j) partial_v k (X_j, .)),
	hat{C} = (1 / N) sum_{j=1}^N sum_{v=1}^d (partial_v k (X_j, .) otimes partial_v k (X_j, .)),
	N = data.shape[0], d = data.shape[1], mu is the base density, and X_j is the j-th row of data.
	
	To compute the orthogonal projection of hat{z} onto the range of hat{C}, we solve
	the following minimization problem
	minimize_{a_1, ..., a_{N*d}} ||hat{z} - sum_{j=1}^N sum_{v=1}^d a_{(j-1)*d+v} partial_v k (X_j, .)||_{mathcal{H}}^2.
	
	Parameters
	----------
	data : numpy.ndarray
		The array of observations whose density function is to be estimated.
		
	new_data : numpy.ndarray
		The array of data at which the orthogonal projection of hat{z} onto the range of hat{C} is to be evaluated.
	
	kernel_function : kernel_function object
		The kernel function used to estimate the probability density function.
		__type__ must be 'kernel_function'.
	
	base_density : base_density object
		The base density function used to estimate the probability density function.
		__type__ must be 'base_density'.
	
	Returns
	-------
	numpy.ndarray
		An array of shape (new_data.shape[0],) of the orthogonal projection of hat{z}
		onto the range of hat{C} evaluated at new_data.
	
	"""
	
	check_kernelfunction(kernel_function)
	check_basedensity(base_density)
	
	if len(data.shape) == 1:
		data = data.reshape(-1, 1)
	
	if len(new_data.shape) == 1:
		new_data = new_data.reshape(-1, 1)
	
	if data.shape[1] != new_data.shape[1]:
		raise ValueError('The dimensionality of data and that of new_data do not match.')
	
	kernel_partial_10_newdata = kernel_function.partial_kernel_matrix_10(new_data=new_data)
	
	# compute the coefficients of partial_u k(X_i, .)
	h_vec = innerp_hatz_partial_kernel(
		data=data,
		kernel_function=kernel_function,
		base_density=base_density).reshape(-1, 1)
	
	Gmat = kernel_function.partial_kernel_matrix_11(new_data=data)
	coef_vec = np.linalg.lstsq(Gmat, h_vec, rcond=None)[0]
	# np.linalg.solve(Gmat + 1e-10 * np.eye(data.shape[0] * data.shape[1]), h_vec)
	
	output = np.matmul(kernel_partial_10_newdata.T, coef_vec).flatten()
	
	return output


def hatz_projection_C_perp(data, new_data, kernel_function, base_density):
	
	"""
	Evaluates the orthogonal projection of hat{z} onto the null space of hat{C} at new_data, where
	hat{z} = (1 / N) sum_{j=1}^N sum_{v=1}^d (partial_v^2 k (X_j, .) + partial_v log mu (X_j) partial_v k (X_j, .)),
	hat{C} = (1 / N) sum_{j=1}^N sum_{v=1}^d (partial_v k (X_j, .) otimes partial_v k (X_j, .)),
	N = data.shape[0], d = data.shape[1], mu is the base density, and X_j is the j-th row of data.
	
	Parameters
	----------
	data : numpy.ndarray
		The array of observations whose density function is to be estimated.

	new_data : numpy.ndarray
		The array of data at which the orthogonal projection of hat{z} onto the orthogonal complement of
		the range of hat{C} is to be evaluated.
		
	kernel_function : kernel_function object
		The kernel function used to estimate the probability density function.
		Must be instantiated from the classes with __type__ being 'kernel_function'.

	base_density : base_density object
		The base density function used to estimate the probability density function.
		Must be instantiated from the classes with __type__ being 'base_density'.

	Returns
	-------
	numpy.ndarray
		An array of shape (new_data.shape[0],) of values of the orthogonal projection of hat{z}
		onto the null space of hat{C} at new_data.

	"""
	
	check_kernelfunction(kernel_function)
	check_basedensity(base_density)
	
	# compute hat{z}
	hatz_newdata = hatz(
		data=data,
		new_data=new_data,
		kernel_function=kernel_function,
		base_density=base_density)
	
	# compute the projection of \hat{z} onto Range(\hat{C})
	projC = hatz_projection_C(
		data=data,
		new_data=new_data,
		kernel_function=kernel_function,
		base_density=base_density)
	
	output = hatz_newdata - projC
	
	return output
