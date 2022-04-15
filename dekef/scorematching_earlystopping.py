from dekef.kernel_function import *
from dekef.scorematching_common_functions import *
from dekef.scorematching_loss_function import *
from dekef.check import *


class ScoreMatchingEarlyStopping:
	
	"""
	A class to estimate the probability density function by minimizing the score matching loss function
	using the gradient descent algorithm.
	
	...
	
	Attributes
	---------
	data : numpy.ndarray
		The array of observations whose density function is to be estimated.
	
	base_density : base_density object
		The base density function used to estimate the probability density function.
	
	kernel_function : kernel_function object
		The kernel function used to estimate the probability density function.
	
	Methods
	-------
	coef(data, iter_num, step_size, threshold=1e-8)
		Computes the coefficients of basis functions in the early stopping score matching density estimate,
		assuming the starting point of the gradient descent algorithm is the zero function.
	
	optiter(iternum_cand, k_folds, step_size, save_dir, save_info=False, threshold=1e-8)
		Selects the optimal number of iterations in the early stopping score matching density estimation
		using k-fold cross validation and computes the coefficient vector of basis functions
		at this optimal number of iterations.
		
	"""
	
	def __init__(self, data, base_density,
				 kernel_type='gaussian_poly2', kernel_r1=1.0, kernel_r2=0., kernel_c=0., kernel_bw=1.0):
		
		"""
		data : numpy.ndarray
			The array of observations whose density function is to be estimated.
		
		kernel_function : kernel_function object
			The kernel function used to estimate the probability density function.
			__type__ must be 'kernel_function'.
		
		base_density : base_density object
			The base density function used to estimate the probability density function.
			__type__ must be 'base_density'.
		
		kernel_type : str, optional
			The type of the kernel function used to estimate the probability density function;
			must be one of 'gaussian_poly2' and 'rationalquad_poly2'; default is 'gaussian_poly2'.
		
		kernel_r1 : float, optional
			The multiplicative constant associated with the Gaussian kernel function or the rational quadratic kernel
			function, depending on kernel_type; default is 1.
		
		kernel_r2 : float, optional
			The multiplicative constant associated with the polynomial kernel function of degree 2; default is 0.
		
		kernel_c : float, optional
			The non-homogenous additive constant in the polynomial kernel function of degree 2; default is 0.
		
		kernel_bw : float, optional
			The bandwidth parameter in the Gaussian kernel function or the rational quadratic kernel function,
			depending on kernel_type; default is 1.
			
		"""
		
		check_basedensity(base_density)
		if len(data.shape) == 1:
			data = data.reshape(-1, 1)
		
		self.data = data
		self.base_density = base_density
		
		if kernel_type == 'gaussian_poly2':
			
			kernel_function = GaussianPoly2(
				data=data,
				r1=kernel_r1,
				r2=kernel_r2,
				c=kernel_c,
				bw=kernel_bw
			)
			
			self.kernel_function = kernel_function
		
		elif kernel_type == 'rationalquad_poly2':
			
			kernel_function = RationalQuadPoly2(
				data=data,
				r1=kernel_r1,
				r2=kernel_r2,
				c=kernel_c,
				bw=kernel_bw
			)
			
			self.kernel_function = kernel_function
		
		else:
			
			raise ValueError(f"kernel_type must be one of 'gaussian_poly2' and 'rationalquad_poly2, "
							 f"but got{kernel_type}'.")
	
	def coef(self, data, iter_num, step_size, threshold=1e-8):
		
		"""
		Computes the coefficients of basis functions in the early stopping score matching density estimate,
		assuming the starting point of the gradient descent algorithm is the zero function.
		
		Parameters
		----------
		data : numpy.ndarray
			The data used to estimate the probability density function.
			This data may be different from self.data, especially in applying the cross validation.
		
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
			An array of coefficients of basis functions in the early stopping score matching density estimate.
			
		"""
		
		if len(data.shape) == 1:
			data = data.reshape(-1, 1)
		
		if data.shape[1] != self.data.shape[1]:
			raise ValueError('The dimensionality of data and that of self.data do not match.')
		
		N, d = data.shape
		
		if iter_num == 0:
			return np.zeros((N * d + 1, ), dtype=np.float64).reshape(-1, 1)
		else:
			iter_num = iter_num - 1
			
			if self.kernel_function.kernel_type == 'gaussian_poly2':
				
				kernel_function_sub = GaussianPoly2(
					data=data,
					r1=self.kernel_function.r1,
					r2=self.kernel_function.r2,
					c=self.kernel_function.c,
					bw=self.kernel_function.bw
				)
			
			elif self.kernel_function.kernel_type == 'rationalquad_poly2':
				
				kernel_function_sub = RationalQuadPoly2(
					data=data,
					r1=self.kernel_function.r1,
					r2=self.kernel_function.r2,
					c=self.kernel_function.c,
					bw=self.kernel_function.bw
				)
			
			# form the C matrix and the h vector
			c_mat = kernel_function_sub.partial_kernel_matrix_11(data)
			h_vec = vector_h(
				data=data,
				kernel_function=kernel_function_sub,
				base_density=self.base_density)
			
			ei_values, ei_vector = np.linalg.eigh(c_mat)
			ei_values1 = step_size * ei_values / N
			
			tilde_lamb = ((- ei_values1 ** (-2) * (1 - ei_values1) * (1 - (1 - ei_values1) ** iter_num) +
						   iter_num / ei_values1) * (ei_values >= threshold) +
						  (iter_num * (iter_num + 1.) / 2.) * (ei_values < threshold))
			
			coef1 = - step_size ** 2 / N * np.matmul(np.matmul(np.matmul(ei_vector, np.diag(tilde_lamb)),
															   ei_vector.T), h_vec)
			
			coef = np.vstack([coef1, (iter_num + 1) * step_size])
			
			return coef

	def optiter(self, iternum_cand, k_folds, step_size, save_dir, save_info=False, threshold=1e-8):
		
		"""
		Selects the optimal number of iterations in the early stopping score matching density estimation
		using k-fold cross validation and computes the coefficient vector of basis functions
		at this optimal number of iterations.
		
		Parameters
		----------
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
			A dictionary containing opt_iter, the optimal number of iterations, and
			opt_coef, the coefficient vector of basis functions in the early stopping score matching density estimate
			at the optimal number of iterations.
		
		"""
		
		N, d = self.data.shape
		
		# check the non-negativity of iternum_cand
		iternum_cand = np.array(iternum_cand).flatten()
		if np.any(iternum_cand < 0.):
			raise ValueError("There exists at least one element in iternum_cand whose value is negative. "
							 "Please modify.")
		
		n_iter = len(iternum_cand)
		
		folds_i = np.random.randint(low=0, high=k_folds, size=N)
		
		sm_scores = np.zeros((n_iter, ), dtype=np.float64)
		
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
				train_data = self.data[folds_i != i, ]
				test_data = self.data[folds_i == i, ]
				
				if self.kernel_function.kernel_type == 'gaussian_poly2':
					
					kernel_function_sub = GaussianPoly2(
						data=train_data,
						r1=self.kernel_function.r1,
						r2=self.kernel_function.r2,
						c=self.kernel_function.c,
						bw=self.kernel_function.bw)
				
				elif self.kernel_function.kernel_type == 'rationalquad_poly2':
					
					kernel_function_sub = RationalQuadPoly2(
						data=train_data,
						r1=self.kernel_function.r1,
						r2=self.kernel_function.r2,
						c=self.kernel_function.c,
						bw=self.kernel_function.bw)
				
				# compute the coefficient vector for the given lambda
				coef_vec = self.coef(
					data=train_data,
					iter_num=numiter_val,
					step_size=step_size,
					threshold=threshold
				)
				
				score += scorematching_loss_function(
					data=train_data,
					new_data=test_data,
					coef=coef_vec,
					kernel_function=kernel_function_sub,
					base_density=self.base_density)
			
			sm_scores[j, ] = score / k_folds
			if save_info:
				f_log.write('score: %.8f\n' % sm_scores[j, ])
		
		# find the optimal regularization parameter
		cv_result = {np.round(x, 5): np.round(y, 10) for x, y in zip(iternum_cand, sm_scores)}
		print("The cross validation scores are:\n" + str(cv_result))
		
		opt_iter = iternum_cand[np.argmin(sm_scores)]
		
		print("=" * 50)
		print("The optimal number of iterations is {}.".format(opt_iter))
		print("=" * 50 + "\nFinal run with the optimal number of iterations.")
		
		# compute the coefficient vector at the optimal regularization parameter
		opt_coef = self.coef(
			data=self.data,
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
		
		output = {"opt_iter": opt_iter, "opt_coef": opt_coef}
		
		return output
