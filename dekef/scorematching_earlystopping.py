from dekef.kernel_function import *
from dekef.scorematching_common_functions import *
from dekef.scorematching_loss_function import *


def scorematching_earlystopping_arbistep_update(data, kernel_function, base_density, iter_num, step_size,
												threshold=1e-8):
	"""
	Computes the coefficients of basis functions in the early stopping score matching density estimation,
	assuming the starting point of the gradient descent algorithm is the zero function.
	
	Parameters
	----------
	data : numpy.ndarray
		The array of observations whose density function is to be estimated.
	
	kernel_function : kernel_function object
		The kernel function used to estimate the probability density function.
		Must be instantiated from the classes with __type__ being 'kernel_function'.
	
	base_density : base_density object
		The base density function used to estimate the probability density function.
		Must be instantiated from the classes with __type__ being 'base_density'.
	
	iter_num : int, float
		The number of iterations in the gradient descent algorithm.
	
	step_size : float
		The constant step size used in the gradient descent algorithm.
	
	threshold : float, optional
		The threshold below which the number is regarded as zero; default is 1e-8.
		Used in arithmetics and operations of eigenvalues to ensure numerical stability.
		
	Returns
	-------
	numpy.ndarray
		An array of coefficients for the early stopping score matching density estimate.
	
	"""
	
	N, d = data.shape
	
	if iter_num == 0:
		return np.zeros((N * d + 1,), dtype=np.float64).reshape(-1, 1)
	else:
		iter_num = iter_num - 1
		
		# form the C matrix and the h vector
		c_mat = kernel_function.partial_kernel_matrix_11(data)
		h_vec = vector_h(
			data=data,
			kernel_function=kernel_function,
			base_density=base_density)
		
		ei_values, ei_vector = np.linalg.eigh(c_mat)
		ei_values1 = step_size * ei_values / N
		
		tilde_lamb = ((- ei_values1 ** (-2) * (1 - ei_values1) * (1 - (1 - ei_values1) ** iter_num) +
					   iter_num / ei_values1) * (ei_values >= threshold) +
					  (iter_num * (iter_num + 1.) / 2.) * (ei_values < threshold))
		
		coef1 = - step_size ** 2 / N * np.matmul(np.matmul(np.matmul(ei_vector, np.diag(tilde_lamb)),
														   ei_vector.T), h_vec)
		
		coef = np.vstack([coef1, (iter_num + 1) * step_size])
		
		return coef


def scorematching_earlystopping_optiter_new(data, kernel_function, base_density, iternum_cand, k_folds, step_size,
											save_dir, save_info=False, threshold=1e-8):
	
	"""
	Selects the optimal number of iterations in the early stopping score matching density estimation
	using k-fold cross validation and computes the coefficient vector at this optimal number of iterations.
	
	Parameters
	----------
	data : numpy.ndarray
		The array of observations whose density function is to be estimated.
	
	kernel_function : kernel_function object
		The kernel function used to estimate the probability density function.
		Must be instantiated from the classes with __type__ being 'kernel_function'.
	
	base_density : base_density object
		The base density function used to estimate the probability density function.
		Must be instantiated from the classes with __type__ being 'base_density'.
	
	iternum_cand : list or 1-dimensional numpy.ndarray
		The list of numbers of iterations candidates.
	
	k_folds : int
		The number of folds for cross validation.
	
	step_size : float
		The constant step size used in the gradient descent algorithm.
	
	save_dir : str
		The directory path to which the estimation information is saved; only works when save_info is True.

	save_info : bool, optional
		Whether to save the estimation information, including the values of score matching loss function of
		each fold and the coefficient vector at the optimal number of iterations, to a local file;
		default is False.
	
	threshold : float, optional
		The threshold below which the number is regarded as zero; default is 1e-8.
		Used in arithmetics and operations of eigenvalues to ensure numerical stability.

	Returns
	-------
	dict
		A dictionary containing opt_lambda, the optimal number of iterations, and
		opt_coef, the coefficient vector at the optimal number of iterations.
	
	"""
	
	if len(data.shape) == 1:
		data = data.reshape(-1, 1)
	
	N, d = data.shape
	
	# check the non-negativity of iternum_cand
	iternum_cand = np.array(iternum_cand).flatten()
	if np.any(iternum_cand < 0.):
		raise ValueError("There exists at least one element in iternum_cand whose value is negative. Please modify.")
	
	n_iter = len(iternum_cand)
	
	folds_i = np.random.randint(low=0, high=k_folds, size=N)
	
	sm_scores = np.zeros((n_iter,), dtype=np.float64)
	
	if save_info:
		f_log = open('%s/log.txt' % save_dir, 'w')
	
	for j in range(n_iter):
		
		# initialize the sm score
		score = 0
		numiter_val = iternum_cand[j]
		
		print("Iteration Number " + str(j) + ": " + str(numiter_val))
		
		if save_info:
			f_log.write('lambda: %.8f, ' % numiter_val)
		
		for i in range(k_folds):
			# data split
			train_data = data[folds_i != i, ]
			test_data = data[folds_i == i, ]
			
			if kernel_function.kernel_type == 'gaussian_poly2':
				
				kernel_function_sub = GaussianPoly2(
					data=train_data,
					r1=kernel_function.r1,
					r2=kernel_function.r2,
					c=kernel_function.c,
					bw=kernel_function.bw)
			
			elif kernel_function.kernel_type == 'rationalquad_poly2':
				
				kernel_function_sub = RationalQuadPoly2(
					data=train_data,
					r1=kernel_function.r1,
					r2=kernel_function.r2,
					c=kernel_function.c,
					bw=kernel_function.bw)
			
			else:
				
				raise NotImplementedError(("The kernel function should be one of 'gaussian_poly2' and "
										   "'rationalquad_poly2'."))
			
			# compute the coefficient vector for the given lambda
			coef_vec = scorematching_earlystopping_arbistep_update(
				data=train_data,
				kernel_function=kernel_function_sub,
				base_density=base_density,
				iter_num=numiter_val,
				step_size=step_size,
				threshold=threshold
			)
			
			score += scorematching_loss_function(
				data=train_data,
				new_data=test_data,
				coef=coef_vec,
				kernel_function=kernel_function_sub,
				base_density=base_density)
		
		sm_scores[j, ] = score / k_folds
		if save_info:
			f_log.write('score: %.8f\n' % sm_scores[j, ])
	
	# find the optimal regularization parameter
	cv_result = {np.round(x, 5): np.round(y, 10) for x, y in zip(iternum_cand, sm_scores)}
	print("The cross validation scores are:\n" + str(cv_result))
	
	opt_iter = iternum_cand[np.argmin(sm_scores)]
	
	# compute the coefficient vector at the optimal regularization parameter
	opt_coef = scorematching_earlystopping_arbistep_update(
		data=data,
		kernel_function=kernel_function,
		base_density=base_density,
		iter_num=opt_iter,
		step_size=step_size,
		threshold=threshold
	)
	
	if save_info:
		f_log.close()
	
	if save_info:
		f_optcoef = open('%s/scorematching_optcoef.npy' % save_dir, 'wb')
		np.save(f_optcoef, opt_coef)
		f_optcoef.close()
	
	output = {"opt_iter": opt_iter,
			  "opt_coef": opt_coef}
	
	return output
