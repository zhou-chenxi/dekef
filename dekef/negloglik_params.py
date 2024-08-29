def batch_montecarlo_params(mc_batch_size=1000, mc_tol=1e-2):
	
	"""
	Returns a dictionary of parameters for the batch Monte Carlo method
	in approximating the partition function and the gradient of the log-partition function.

	Parameters
	----------
	mc_batch_size : int
		The batch size in the batch Monte Carlo method; default is 1000.

	mc_tol : float
		The floating point number below which sampling in the batch Monte Carlo is terminated; default is 1e-2.

	Returns
	-------
	dict
		The dictionary containing both supplied parameters.

	"""
	
	mc_batch_size = int(mc_batch_size)
	
	output = {"mc_batch_size": mc_batch_size,
			  "mc_tol": mc_tol}
	
	return output


def negloglik_optalgo_params(start_pt, step_size=0.01, max_iter=1e2, rel_tol=1e-5, abs_tol=1e-5):
	
	"""
	Returns a dictionary of parameters used in minimizing the (penalized) negative log-likelihood loss function
	by using the gradient descent algorithm.

	Parameters
	----------
	start_pt : numpy.ndarray
		The starting point of the gradient descent algorithm to minimize
		the penalized negative log-likelihood loss function.

	step_size : float
		The step size used in the gradient descent algorithm; default is 0.01.

	max_iter : int
		The maximal number of iterations in the gradient descent algorithm; default is 100.

	rel_tol : float
		The relative tolerance parameter to terminate the gradient descent algorithm in minimizing
		the penalized negative log-likelihood loss function; default is 1e-5.
	
	abs_tol : float
		The absolute tolerance parameter to terminate the gradient descent algorithm in minimizing
		the penalized negative log-likelihood loss function; default is 1e-5.

	Returns
	-------
	dict
		The dictionary containing all supplied parameters.

	"""
	
	max_iter = int(max_iter)
	assert rel_tol > 0.
	assert abs_tol > 0.
	
	output = {'start_pt': start_pt,
			  'step_size': step_size,
			  'max_iter': max_iter,
			  'rel_tol': rel_tol,
			  'abs_tol': abs_tol}
	
	return output
