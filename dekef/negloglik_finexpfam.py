from dekef.check import *
from dekef.kernel_function import *
from dekef.negloglik_gubasis import negloglik_gubasis_grad_logpar_batchmc_se, negloglik_gubasis_grad_logpar_batchmc


def negloglik_finexpfam_gubasis_coef(data, kernel_function, base_density, optalgo_params, batchmc_params,
									 batch_mc=True, batch_mc_se=False, print_error=True):
	
	"""
	Returns the solution to minimizing the negative log-likelihood loss function
	in finite-dimensional exponential family whose canonical statistics are
	kernel functions with one argument evaluated at data.
	The underlying minimization algorithm is the gradient descent algorithm.

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

	optalgo_params : dict
		The dictionary of parameters to control the gradient descent algorithm.
		Must be returned from the function negloglik_penalized_optalgoparams.

	batchmc_params : dict
		The dictionary of parameters to control the batch Monte Carlo method
		to approximate the log-partition function and its gradient.
		Must be returned from the function batch_montecarlo_params.

	batch_mc : bool, optional
		Whether to use the batch Monte Carlo method with the termination criterion
		being the relative difference of two consecutive approximations; default is True.

	batch_mc_se : bool, optional
		Whether to use the batch Monte Carlo method with the termination criterion
		being the standard deviation of approximations; default is False.

	print_error : bool, optional
		Whether to print the error of the gradient descent algorithm at each iteration; default is True.

	Returns
	-------
	numpy.ndarray
		An array of coefficients for the natural parameter in the negative log-likelihood density estimate.

	"""
	
	check_kernelfunction(kernel_function)
	check_basedensity(base_density)
	
	if len(data.shape) == 1:
		data = data.reshape(-1, 1)
	
	N, d = data.shape
	
	# parameters associated with gradient descent algorithm
	start_pt = optalgo_params["start_pt"]
	step_size = optalgo_params["step_size"]
	max_iter = optalgo_params["max_iter"]
	rel_tol = optalgo_params["rel_tol"]
	
	if not type(step_size) == float:
		raise TypeError(("The type of step_size in optalgo_params should be float, but got {}".format(
			type(step_size))))
	
	if step_size <= 0.:
		raise ValueError("The step_size in optalgo_params must be strictly positive, but got {}.".format(
			step_size))
	
	if len(start_pt) != N:
		raise ValueError(("The supplied start_pt in optalgo_params is not correct. "
						  "The expected length of start_pt is {exp_len}, but got {act_len}.").format(
			exp_len=N, act_len=len(start_pt)))
	
	# parameters associated with batch Monte Carlo estimation
	mc_batch_size = batchmc_params["mc_batch_size"]
	mc_tol = batchmc_params["mc_tol"]
	
	# the gradient of the loss function is
	# nabla L (alpha) = nabla A (alpha) - (1 / n) gram_matrix boldone_n
	# the gradient descent update is
	# new_iter = current_iter - step_size * nabla L (alpha)
	
	current_iter = start_pt.reshape(-1, 1)
	
	# compute the gradient of the log-partition function at current_iter
	if batch_mc:
		
		mc_output1, mc_output2 = negloglik_gubasis_grad_logpar_batchmc(
			data=data,
			kernel_function=kernel_function,
			base_density=base_density,
			coef=current_iter,
			batch_size=mc_batch_size,
			tol_param=mc_tol,
			normalizing_const_only=False,
			print_error=False)
		
		grad_logpar = mc_output2.reshape(-1, 1)
	
	elif batch_mc_se:
		
		mc_output1, mc_output2 = negloglik_gubasis_grad_logpar_batchmc_se(
			data=data,
			kernel_function=kernel_function,
			base_density=base_density,
			coef=current_iter,
			batch_size=mc_batch_size,
			tol_param=mc_tol,
			normalizing_const_only=False,
			print_error=False)
		
		grad_logpar = mc_output2.reshape(-1, 1)
	
	else:
		raise NotImplementedError(("In order to approximate the gradient of the log-partition function, "
								   "exactly one of 'batch_mc' and 'batch_mc_se' must be set True."))
	
	# form the Gram matrix
	gram = kernel_function.kernel_gram_matrix(data)
	grad_term2 = gram.mean(axis=1, keepdims=True)
	
	# compute the gradient of the loss function at current_iter
	current_grad = grad_logpar - grad_term2
	
	# compute the updated iter
	new_iter = current_iter - step_size * current_grad
	
	# compute the error of the first update
	grad0_norm = np.linalg.norm(current_grad, 2)
	error = grad0_norm / grad0_norm
	# np.linalg.norm(new_iter - current_iter, 2) / (np.linalg.norm(current_iter, 2) + 1e-1)
	
	iter_num = 1
	
	if print_error:
		print("Iter = {iter_num}, GradNorm = {gradnorm}, Relative Error = {error}".format(
			iter_num=iter_num, gradnorm=grad0_norm, error=error))
	
	while error > rel_tol and iter_num < max_iter:
		
		current_iter = new_iter
		
		# compute the gradient at current_iter
		if batch_mc:
			
			mc_output1, mc_output2 = negloglik_gubasis_grad_logpar_batchmc(
				data=data,
				kernel_function=kernel_function,
				base_density=base_density,
				coef=current_iter,
				batch_size=mc_batch_size,
				tol_param=mc_tol,
				normalizing_const_only=False,
				print_error=False)
			
			grad_logpar = mc_output2.reshape(-1, 1)
		
		elif batch_mc_se:
			
			mc_output1, mc_output2 = negloglik_gubasis_grad_logpar_batchmc_se(
				data=data,
				kernel_function=kernel_function,
				base_density=base_density,
				coef=current_iter,
				batch_size=mc_batch_size,
				tol_param=mc_tol,
				normalizing_const_only=False,
				print_error=False)
			
			grad_logpar = mc_output2.reshape(-1, 1)
		
		else:
			
			raise NotImplementedError(("In order to approximate the gradient of the log-partition function, "
									   "exactly one of 'batch_mc' and 'batch_mc_se' must be set True."))
		
		# compute the gradient of the loss function
		current_grad = grad_logpar - grad_term2
		
		# compute the updated iter
		new_iter = current_iter - step_size * current_grad
		
		# compute the error of the first update
		grad_new_norm = np.linalg.norm(current_grad, 2)
		error = grad_new_norm / grad0_norm
		# np.linalg.norm(new_iter - current_iter, 2) / (np.linalg.norm(current_iter, 2) + 1e-1)
		
		iter_num += 1
		
		if print_error:
			print("Iter = {iter_num}, GradNorm = {gradnorm}, Relative Error = {error}".format(
				iter_num=iter_num, gradnorm=grad_new_norm, error=error))
	
	coefficients = new_iter
	
	return coefficients
