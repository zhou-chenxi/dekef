import warnings
from dekef.check import *
from dekef.kernel_function import *
from dekef.negloglik_params import *


class NegLogLikGuBasis:
    
    """
    A class to estimate the probability density function by minimizing
    the penalized negative log-likelihood loss function
    in a kernel exponential family, where the natural parameter is
    a linear combination of kernel functions centered at data.

    ...

    Attributes
    ----------
    data : numpy.ndarray
        The array of observations whose density function is to be estimated.
        
    base_density : base_density object
        The base density function used to estimate the probability density function.

    kernel_function : kernel_function object
        The kernel function used to estimate the probability density function.

    Methods
    -------
    grad_logpar_batchmc(coef, batch_size, tol_param, compute_grad=True, print_error=False)
        Approximates the partition function and the gradient of the log-partition function at coef
        with the kernel functions centered at self.data.
        The approximation method used is the batch Monte Carlo method.
        Terminate the sampling process until the relative difference of
        two consecutive approximations is less than tol_param.

    grad_logpar_batchmc_se(coef, batch_size, tol_param, compute_grad=True, print_error=False)
        Approximates the partition function and the gradient of the log-partition function at coef
        with the kernel functions centered at self.data.
        The approximation method used is the batch Monte Carlo method.
        Terminate the sampling process until the standard deviation of the approximations is less than tol_param.

    coef(lambda_param, optalgo_params, batchmc_params, batch_mc=True, print_error=True)
        Returns the solution that minimizes the penalized negative log-likelihood loss function
        with the basis functions centered at self.data.

    eval_loss_function(new_data, coef, batchmc_params, batch_mc=True)
        Evaluates the negative log-likelihood loss function evaluated at coef and on new_data, i.e.,
        A (f) - (1 / n) sum_{j=1}^n f (Y_j), where the natural parameter f is sum_{i=1}^N coef[i] k(X_i, .),
        X_1, ..., X_N are self.data, and Y_1, ..., Y_n are new_data.

    penalized_optlambda(lambda_cand, k_folds, print_error, optalgo_params,
                        batchmc_params, save_dir, save_info=False, batch_mc=True)
        Selects the optimal penalty parameter in the penalized negative log-likelihood density estimation
        using k-fold cross validation and computes the coefficient vector at this optimal penalty parameter.
        The basis functions of the natural parameter are the kernel functions centered at self.data.

    """
    
    def __init__(self, data, base_density, kernel_type='gaussian_poly2',
                 kernel_r1=1.0, kernel_r2=0., kernel_c=0., kernel_bw=1.0):
        
        """
        Parameters
        ----------
        data : numpy.ndarray
            The array of observations whose density function is to be estimated.

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
        
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        check_basedensity(base_density)
        
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

    def grad_logpar_batchmc(self, coef, batch_size, tol_param, compute_grad=True, print_error=False):
        
        """
        Approximates the partition function and the gradient of the log-partition function at coef
        using the basis functions centered at self.data.
        The approximation method used is the batch Monte Carlo method.
        Terminate the sampling process until the relative difference of
        two consecutive approximations is less than tol_param.
    
        Let Y_1, ..., Y_M be random samples from the base density.
        The partition function evaluated at coef is approximated by
        (1 / M) sum_{j=1}^M exp ( sum_{i=1}^n coef[i] k (X_i, Y_j) ),
        and the gradient of the log-partition function evaluated at coef is approximated by
        (1 / M) sum_{j=1}^M k (X_l, Y_j) exp ( sum_{i=1}^N coef[i] k (X_i, Y_j) - A (coef)), for all l = 1, ..., n,
        where A(coef) is the log-partition function at coef, and X_l is the l-th observation.
        
        Parameters
        ----------
        coef : numpy.ndarray
            The array of coefficients at which the partition function and
            the gradient of the log-partition function are approximated.
    
        batch_size : int
            The batch size in the batch Monte Carlo method.
    
        tol_param : float
            The floating point number below which sampling in the batch Monte Carlo is terminated.
            The smaller the tol_param is, the more accurate the approximations are.
        
        compute_grad : bool, optional
            Whether to approximate the gradient of the log-partition function; default is True.
    
        print_error : bool, optional
            Whether to print the error in the batch Monte Carlo method; default is False.
    
        Returns
        -------
        float
            The approximation of the partition function evaluated at coef.
        
        numpy.ndarray
            The approximation of the gradient of the log-partition function evaluated at coef;
            only returns when compute_grad is True.
            
        References
        ----------
        Gu, Chong, and Chunfu Qiu. 1993. “Smoothing Spline Density Estimation: Theory.”
            Annals of Statistics 21 (1): 217–34.
        Gu, Chong. 1993. “Smoothing Spline Density Estimation: A Dimensionless Automatic Algorithm.”
            Journal of the American Statistical Association 88 (422): 495–504.
    
        """
        
        N, d = self.data.shape
        coef_len = N
    
        if len(coef) != coef_len:
            raise ValueError("The length of coef is incorrect, which should be {l}.".format(l=coef_len))
        
        ###########################################################################
        # estimate the normalizing constant
        # first drawing
        mc_samples1 = self.base_density.sample(batch_size)
        mc_kernel_matrix1 = self.kernel_function.kernel_gram_matrix(mc_samples1)
        unnorm_density_part1 = np.exp(np.matmul(mc_kernel_matrix1.T, coef))
        norm_const1 = np.mean(unnorm_density_part1)
        
        # second drawing
        mc_samples2 = self.base_density.sample(batch_size)
        mc_kernel_matrix2 = self.kernel_function.kernel_gram_matrix(mc_samples2)
        unnorm_density_part2 = np.exp(np.matmul(mc_kernel_matrix2.T, coef))
        norm_const2 = np.mean(unnorm_density_part2)
        
        norm_est_old = norm_const1
        norm_est_new = (norm_const1 + norm_const2) / 2
        
        error_norm = np.abs(norm_est_old - norm_est_new) / norm_est_old
        
        if print_error:
            print('normalizing constant error = {error:.7f}'.format(error=error_norm))
        
        batch_cnt = 2
        
        while error_norm > tol_param:
            
            norm_est_old = norm_est_new
            
            # another draw
            mc_samples = self.base_density.sample(batch_size)
            mc_kernel_matrix = self.kernel_function.kernel_gram_matrix(mc_samples)
            unnorm_density_part = np.exp(np.matmul(mc_kernel_matrix.T, coef))
            norm_const2 = np.mean(unnorm_density_part)
            
            # update the Monte Carlo estimation
            norm_est_new = (norm_est_old * batch_cnt + norm_const2) / (batch_cnt + 1)
    
            batch_cnt += 1
            
            error_norm = np.abs(norm_est_old - norm_est_new) / norm_est_old
            
            if print_error:
                print('normalizing constant error = {error:.7f}'.format(error=error_norm))
        
        normalizing_const = norm_est_new
        
        if compute_grad:
            
            if print_error:
                print("#" * 45 + "\nEstimating the gradient of the log-partition now.")
            
            mc_samples1 = self.base_density.sample(batch_size)
            mc_kernel_matrix1 = self.kernel_function.kernel_gram_matrix(mc_samples1)
            density_part1 = np.exp(np.matmul(mc_kernel_matrix1.T, coef).flatten()) / normalizing_const
            exp_est1 = np.array([np.mean(mc_kernel_matrix1[l1, :] * density_part1)
                                 for l1 in range(N)]).astype(np.float64).reshape(1, -1)[0]
            
            mc_samples2 = self.base_density.sample(batch_size)
            mc_kernel_matrix2 = self.kernel_function.kernel_gram_matrix(mc_samples2)
            density_part2 = np.exp(np.matmul(mc_kernel_matrix2.T, coef).flatten()) / normalizing_const
            exp_est2 = np.array([np.mean(mc_kernel_matrix2[l1, :] * density_part2)
                                 for l1 in range(N)]).astype(np.float64).reshape(1, -1)[0]
            
            grad_est_old = exp_est1
            grad_est_new = (exp_est1 + exp_est2) / 2
            
            error_grad = np.linalg.norm(grad_est_old - grad_est_new, 2) / (np.linalg.norm(grad_est_old, 2) * N)
            
            if print_error:
                print('gradient error = {error:.7f}'.format(error=error_grad))
            
            batch_cnt = 2
            
            while error_grad > tol_param:
            
                grad_est_old = grad_est_new
    
                # another draw
                mc_samples = self.base_density.sample(batch_size)
                mc_kernel_matrix = self.kernel_function.kernel_gram_matrix(mc_samples)
                density_part = np.exp(np.matmul(mc_kernel_matrix.T, coef).flatten()) / normalizing_const
                exp_est2 = np.array([np.mean(mc_kernel_matrix[l1, :] * density_part)
                                     for l1 in range(N)]).astype(np.float64).reshape(1, -1)[0]
                
                grad_est_new = (grad_est_old * batch_cnt + exp_est2) / (batch_cnt + 1)
    
                batch_cnt += 1
    
                error_grad = np.linalg.norm(grad_est_old - grad_est_new, 2) / (np.linalg.norm(grad_est_old, 2) * N)
    
                if print_error:
                    print('gradient error = {error:.7f}'.format(error=error_grad))
        
        if not compute_grad:
            return norm_est_new
        else:
            return norm_est_new, grad_est_new
    
    def grad_logpar_batchmc_se(self, coef, batch_size, tol_param, compute_grad=True, print_error=False):
        
        """
        Approximates the partition function and the gradient of the log-partition function at coef
        using the basis functions centered at self.data.
        The approximation method used is the batch Monte Carlo method.
        Terminate the sampling process until
        the standard deviation of the approximations is less than tol_param.
    
        Let Y_1, ..., Y_M be random samples from the base density.
        The partition function evaluated at coef is approximated by
        (1 / M) sum_{j=1}^M exp ( sum_{i=1}^n coef[i] k (X_i, Y_j) ),
        and the gradient of the log-partition function evaluated at coef is approximated by
        (1 / M) sum_{j=1}^M k (X_l, Y_j) exp ( sum_{i=1}^N coef[i] k (X_i, Y_j) - A(coef)), for all l = 1, ..., n,
        where A(coef) is the log-partition function, and X_l is the l-th observation.
        
        Parameters
        ----------
        coef : numpy.ndarray
            The array of coefficients at which the partition function and
            the gradient of the log-partition function are approximated.
    
        batch_size : int
            The batch size in the batch Monte Carlo method.
    
        tol_param : float
            The floating point number below which sampling in the batch Monte Carlo is terminated.
            The smaller the tol_param is, the more accurate the approximations are.
        
        compute_grad : bool, optional
            Whether to approximate the gradient of the log-partition function; default is True.
    
        print_error : bool, optional
            Whether to print the error in the batch Monte Carlo method; default is False.
    
        Returns
        -------
        float
            The approximation of the partition function evaluated at coef.
        
        numpy.ndarray
            The approximation of the gradient of the log-partition function evaluated at coef;
            only returns when compute_grad is True.
            
        References
        ----------
        Gu, Chong, and Chunfu Qiu. 1993. “Smoothing Spline Density Estimation: Theory.”
            Annals of Statistics 21 (1): 217–34.
        Gu, Chong. 1993. “Smoothing Spline Density Estimation: A Dimensionless Automatic Algorithm.”
            Journal of the American Statistical Association 88 (422): 495–504.
    
        """
            
        N, d = self.data.shape
        
        ###########################################################################
        # estimate the normalizing constant
        # first drawing
        mc_samples = self.base_density.sample(batch_size)
        mc_kernel_matrix = self.kernel_function.kernel_gram_matrix(mc_samples)
        unnorm_density_part = np.exp(np.matmul(mc_kernel_matrix.T, coef))
        avg_norm_const = np.mean(unnorm_density_part)
        sq_norm_const = np.sum(unnorm_density_part ** 2)
        
        error_norm = np.sqrt(sq_norm_const / batch_size - avg_norm_const ** 2) / np.sqrt(batch_size)
        
        if print_error:
            print('normalizing constant error = {error:.7f}'.format(error=error_norm))
        
        batch_cnt = 1
        
        while error_norm > tol_param:
            
            # another draw
            mc_samples = self.base_density.sample(batch_size)
            mc_kernel_matrix = self.kernel_function.kernel_gram_matrix(mc_samples)
            unnorm_density_part = np.exp(np.matmul(mc_kernel_matrix.T, coef))
            avg_norm_const2 = np.mean(unnorm_density_part)
            sq_norm_const += np.sum(unnorm_density_part ** 2)
            
            # update Monte Carlo estimation
            avg_norm_const = (avg_norm_const * batch_cnt + avg_norm_const2) / (batch_cnt + 1)
            
            error_norm = (np.sqrt(sq_norm_const / (batch_size * (batch_cnt + 1)) - avg_norm_const ** 2) /
                          np.sqrt(batch_size * (batch_cnt + 1)))
        
            batch_cnt += 1
            
            if print_error:
                print('normalizing constant error = {error:.7f}'.format(error=error_norm))
        
        normalizing_const = avg_norm_const
        print(batch_cnt)
        
        if compute_grad:
            
            print("#" * 45 + "\nApproximating the gradient of the log-partition now.")
            
            mc_samples = self.base_density.sample(batch_size)
            mc_kernel_matrix = self.kernel_function.kernel_gram_matrix(mc_samples)
            density_part = np.exp(np.matmul(mc_kernel_matrix.T, coef).flatten()) / normalizing_const
            grad_est = (np.array([np.mean(mc_kernel_matrix[l1, :] * density_part)
                                  for l1 in range(N)]).astype(np.float64).reshape(1, -1)[0])
            sq_grad_est = (np.array([(mc_kernel_matrix[l1, :] * density_part) ** 2
                                     for l1 in range(N)]).astype(np.float64).reshape(1, -1)[0])
            
            error_grad = np.sqrt(np.sum(np.mean(sq_grad_est, axis=0) - grad_est ** 2)) / np.sqrt(batch_size)
            
            if print_error:
                print('gradient error = {error:.7f}'.format(error=error_grad))
            
            batch_cnt = 1
            
            while error_grad > tol_param:
                
                # another draw
                mc_samples = self.base_density.sample(batch_size)
                mc_kernel_matrix = self.kernel_function.kernel_gram_matrix(mc_samples)
                density_part = np.exp(np.matmul(mc_kernel_matrix.T, coef).flatten()) / normalizing_const
                grad_est2 = np.array([np.mean(mc_kernel_matrix[l1, :] * density_part)
                                      for l1 in range(N)]).astype(np.float64).reshape(1, -1)[0]
                sq_grad_est += (np.array([(mc_kernel_matrix[l1, :] * density_part) ** 2
                                          for l1 in range(N)]).astype(np.float64).reshape(1, -1)[0])
                
                grad_est = (grad_est * batch_cnt + grad_est2) / (batch_cnt + 1)
                
                error_grad = (np.sqrt(np.sum(np.mean(sq_grad_est, axis=0) / (batch_cnt + 1) - grad_est ** 2)) /
                              np.sqrt(batch_size * (batch_cnt + 1)))
                
                batch_cnt += 1
                
                if print_error:
                    print('gradient error = {error:.7f}'.format(error=error_grad))
                    
            print(batch_cnt)
        
        if not compute_grad:
            return normalizing_const
        else:
            return normalizing_const, grad_est
    
    def coef(self, data, lambda_param, optalgo_params, batchmc_params, batch_mc=True, print_error=True):
        
        """
        Returns the solution that minimizes the penalized negative log-likelihood loss function
        using the basis functions centered at self.data.
        The underlying minimization algorithm is the gradient descent algorithm.
    
        Parameters
        ----------
        data : numpy.ndarray
            The data used to estimate the probability density function.
            This data may be different from self.data, especially in applying the cross validation.
            
        lambda_param : float
            The penalty parameter. Must be non-negative.
    
        optalgo_params : dict
            The dictionary of parameters to control the gradient descent algorithm.
            Must be returned from the function negloglik_optalgo_params.
    
        batchmc_params : dict
            The dictionary of parameters to control the batch Monte Carlo method
            to approximate the partition function and the gradient of the log-partition function.
            Must be returned from the function batch_montecarlo_params.
    
        batch_mc : bool, optional
            Whether to use the batch Monte Carlo method with the termination criterion
            being the relative difference of two consecutive approximations; default is True.
            If it is False, the batch Monte Carlo method with the termination criterion
            being the standard deviation of the approximations will be used.
    
        print_error : bool, optional
            Whether to print the error of the gradient descent algorithm at each iteration; default is True.
    
        Returns
        -------
        numpy.ndarray
            An array of coefficients for the natural parameter
            in the penalized negative log-likelihood density estimate.
    
        References
        ----------
        Gu, Chong, and Chunfu Qiu. 1993. “Smoothing Spline Density Estimation: Theory.”
            Annals of Statistics 21 (1): 217–34.
        Gu, Chong. 1993. “Smoothing Spline Density Estimation: A Dimensionless Automatic Algorithm.”
            Journal of the American Statistical Association 88 (422): 495–504.
        
        """
    
        if lambda_param < 0.:
            raise ValueError("The lambda_param cannot be negative.")
        
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        if data.shape[1] != self.data.shape[1]:
            raise ValueError('The dimensionality of data provided and that of self.data do not match.')
        
        N, d = self.data.shape
    
        # parameters associated with gradient descent algorithm
        start_pt = optalgo_params['start_pt']
        step_size = optalgo_params['step_size']
        max_iter = optalgo_params['max_iter']
        rel_tol = optalgo_params['rel_tol']
        abs_tol = optalgo_params['abs_tol']

        if not isinstance(step_size, float):
            raise TypeError(("The type of step_size in optalgo_params should be float, "
                             "but got {}".format(type(step_size))))
    
        if step_size <= 0.:
            raise ValueError("The step_size in optalgo_params must be strictly positive, but got {}.".format(step_size))
        
        if len(start_pt) != N:
            raise ValueError(("The supplied start_pt in optalgo_params is not correct. "
                              "The expected length of start_pt is {exp_len}, but got {act_len}.").format(
                exp_len=N, act_len=len(start_pt)))
        
        # parameters associated with batch Monte Carlo estimation
        mc_batch_size = batchmc_params['mc_batch_size']
        mc_tol = batchmc_params['mc_tol']
    
        # the gradient of the loss function is
        # nabla L (alpha) = nabla A (alpha) - (1 / n) gram_matrix boldone_n + lambda_param * gram_matrix * alpha
        # the gradient descent update is
        # new_iter = current_iter - step_size * nabla L (alpha)
    
        # form the Gram matrix
        gram = self.kernel_function.kernel_gram_matrix(self.data)
        data_gram = self.kernel_function.kernel_gram_matrix(data)
        grad_term2 = data_gram.mean(axis=1, keepdims=True)
    
        current_iter = start_pt.reshape(-1, 1)
    
        # compute the gradient of the log-partition function at current_iter
        if batch_mc:
            
            mc_output1, mc_output2 = self.grad_logpar_batchmc(
                coef=current_iter,
                batch_size=mc_batch_size,
                tol_param=mc_tol,
                compute_grad=True,
                print_error=False)
            
            grad_logpar = mc_output2.reshape(-1, 1)
        
        else:
            
            mc_output1, mc_output2 = self.grad_logpar_batchmc_se(
                coef=current_iter,
                batch_size=mc_batch_size,
                tol_param=mc_tol,
                compute_grad=True,
                print_error=False)
            
            grad_logpar = mc_output2.reshape(-1, 1)
        
        # compute the gradient of the loss function at current_iter
        current_grad = grad_logpar - grad_term2 + lambda_param * np.matmul(gram, current_iter)
        
        # compute the updated iter
        new_iter = current_iter - step_size * current_grad
        
        # compute the error of the first update
        grad0_norm = np.linalg.norm(current_grad, 2)
        grad_new_norm = grad0_norm
        error = grad0_norm / grad0_norm
        # np.linalg.norm(new_iter - current_iter, 2) / (np.linalg.norm(current_iter, 2) + 1e-1)
    
        iter_num = 1
    
        if print_error:
            print("Iter = {iter_num}, GradNorm = {gradnorm}, Relative Error = {error}".format(
                iter_num=iter_num, gradnorm=grad0_norm, error=error))
    
        while error > rel_tol and grad_new_norm > abs_tol and iter_num < max_iter:
            
            current_iter = new_iter
            
            # compute the gradient at current_iter
            if batch_mc:
                
                mc_output1, mc_output2 = self.grad_logpar_batchmc(
                    coef=current_iter,
                    batch_size=mc_batch_size,
                    tol_param=mc_tol,
                    compute_grad=True,
                    print_error=False)
                
                grad_logpar = mc_output2.reshape(-1, 1)
            
            else:
        
                mc_output1, mc_output2 = self.grad_logpar_batchmc_se(
                    coef=current_iter,
                    batch_size=mc_batch_size,
                    tol_param=mc_tol,
                    compute_grad=True,
                    print_error=False)
                
                grad_logpar = mc_output2.reshape(-1, 1)
    
            # compute the gradient of the loss function
            current_grad = grad_logpar - grad_term2 + lambda_param * np.matmul(gram, current_iter)
    
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
    
    def eval_loss_function(self, new_data, coef, batchmc_params, batch_mc=True):
        
        """
        Evaluates the negative log-likelihood loss function evaluated at coef and on new_data, i.e.,
        A (f) - (1 / n) sum_{j=1}^n f (Y_j),
        where the natural parameter f is determined by data and coef and is equal to sum_{i=1}^N coef[i] phi_i,
        phi_1, ..., phi_N are the kernel functions centered at self.data, and Y_1, ..., Y_n are new_data.
        
        Parameters
        ----------
        new_data : numpy.ndarray
            The array of data at which the negative log-likelihood loss function is to be evaluated.
            
        coef : numpy.ndarray
            The array of coefficients at which the negative log-likelihood loss function is to be evaluated.
            Must be of shape (data.shape[0] * data.shape[1] + 1, 1).
            
        batchmc_params : dict
            The dictionary of parameters to control the batch Monte Carlo method
            to approximate the partition function and the gradient of the log-partition function.
            Must be returned from the function batch_montecarlo_params.
        
        batch_mc : bool, optional
            Whether to use the batch Monte Carlo method with the termination criterion
            being the relative difference of two consecutive approximations; default is True.
            If it is False, the batch Monte Carlo method with the termination criterion
            being the standard deviation of the approximations will be used.
        
        Returns
        -------
        float
            The value of the negative log-likelihood loss function evaluated at coef and on new_data.
            
        References
        ----------
        Gu, Chong, and Chunfu Qiu. 1993. “Smoothing Spline Density Estimation: Theory.”
            Annals of Statistics 21 (1): 217–34.
        Gu, Chong. 1993. “Smoothing Spline Density Estimation: A Dimensionless Automatic Algorithm.”
            Journal of the American Statistical Association 88 (422): 495–504.
        
        """
    
        if len(new_data.shape) == 1:
            new_data = new_data.reshape(-1, 1)
        
        if self.data.shape[1] != new_data.shape[1]:
            raise ValueError('The dimensionality of new_data and that of self.data do not match.')
        
        N, d = self.data.shape
        coef = coef.reshape(-1, 1)
        if coef.shape[0] != N:
            raise ValueError(("The supplied coef is not correct. "
                              "The expected length of coef is {exp_len}, but got {act_len}.").format(
                exp_len=N, act_len=len(coef)))
        
        mc_batch_size = batchmc_params["mc_batch_size"]
        mc_tol = batchmc_params["mc_tol"]
    
        # compute A(f)
        if batch_mc:
        
            mc_output1 = self.grad_logpar_batchmc(
                coef=coef,
                batch_size=mc_batch_size,
                tol_param=mc_tol,
                compute_grad=False,
                print_error=False)
    
        else:
        
            mc_output1 = self.grad_logpar_batchmc_se(
                coef=coef,
                batch_size=mc_batch_size,
                tol_param=mc_tol,
                compute_grad=False,
                print_error=False)
    
        norm_const = mc_output1
        Af = np.log(norm_const)
        
        # compute (1 / n) sum_{j=1}^n f (Y_j), where Y_j is the j-th row of new_data
        kernel_mat_new = self.kernel_function.kernel_gram_matrix(new_data)
        avg_fx = np.mean(np.matmul(kernel_mat_new.T, coef))
        
        loss_val = Af - avg_fx
        
        return loss_val
    
    def penalized_optlambda(self, lambda_cand, k_folds, print_error, optalgo_params, batchmc_params,
                            save_dir, save_info=False, batch_mc=True):
        
        """
        Selects the optimal penalty parameter in the penalized negative log-likelihood density estimation
        using k-fold cross validation and computes the coefficient vector at this optimal penalty parameter.
        The basis functions of the natural parameter are the kernel functions centered at self.data.
        
        Parameters
        ----------
        lambda_cand : list or 1-dimensional numpy.ndarray
            The list of penalty parameter candidates. Each of them must be non-negative.
        
        k_folds : int
            The number of folds for cross validation.
        
        print_error : bool
            Whether to print the error of the gradient descent algorithm at each iteration.
        
        optalgo_params : dict
            The dictionary of parameters to control the gradient descent algorithm.
            Must be returned from the function negloglik_penalized_optalgoparams.
            
        batchmc_params : dict
            The dictionary of parameters to control the batch Monte Carlo method
            to approximate the partition function and the gradient of the log-partition function.
            Must be returned from the function batch_montecarlo_params.
        
        save_dir : str
            The directory path to which the estimation information is saved;
            only works when save_info is True.
        
        save_info : bool, optional
            Whether to save the estimation information, including the values of negative log-likelihood
            loss function of each fold and the coefficient vector at the optimal penalty parameter, to a local file;
            default is False.
        
        batch_mc : bool, optional
            Whether to use the batch Monte Carlo method with the termination criterion
            being the relative difference of two consecutive approximations; default is True.
            If it is False, the batch Monte Carlo method with the termination criterion
            being the standard deviation of the approximations will be used.
        
        Returns
        -------
        dict
            A dictionary containing opt_lambda, the optimal penalty parameter, and
            opt_coef, the coefficient vector at the optimal penalty parameter.
            
        References
        ----------
        Gu, Chong, and Chunfu Qiu. 1993. “Smoothing Spline Density Estimation: Theory.”
            Annals of Statistics 21 (1): 217–34.
        Gu, Chong. 1993. “Smoothing Spline Density Estimation: A Dimensionless Automatic Algorithm.”
            Journal of the American Statistical Association 88 (422): 495–504.
        
        """
        
        N, d = self.data.shape
        
        # check the non-negativity of lambda_cand
        lambda_cand = np.array(lambda_cand).flatten()
        if np.any(lambda_cand < 0.):
            raise ValueError("There exists at least one element in lambda_cand whose value is negative. Please modify.")
    
        n_lambda = len(lambda_cand)
        
        # check the step size
        step_size = optalgo_params['step_size']
        if isinstance(step_size, float):
        
            warn_msg = ("The step_size in optalgo_params is a float, and will be used in computing "
                        "density estimates for all {} different lambda values in lambda_cand."
                        "It is better to supply a list or numpy.ndarray for step_size.").format(n_lambda)
        
            print(warn_msg)
            
            step_size = np.array([step_size] * n_lambda)
            
        elif isinstance(step_size, list):
            
            step_size = np.array(step_size)
        
        if len(step_size) != n_lambda:
            raise ValueError("The length of step_size in optalgo_params is not the same as that of lambda_cand.")
        
        folds_i = np.random.randint(low=0, high=k_folds, size=N)
    
        nll_scores = np.zeros((n_lambda,), dtype=np.float64)
        
        if save_info:
            f_log = open('%s/log.txt' % save_dir, 'w')
    
        for j in range(n_lambda):
    
            # initialize the loss score
            score = 0.
            lambda_param = lambda_cand[j]
            
            print("Lambda " + str(j) + ": " + str(lambda_param))
            
            if save_info:
                f_log.write('lambda: %.8f, ' % lambda_param)
    
            for i in range(k_folds):
                # data split
                train_data = self.data[folds_i != i, ]
                test_data = self.data[folds_i == i, ]
    
                # compute the coefficient vector for the given lambda
                train_algo_control = negloglik_optalgo_params(
                    start_pt=np.zeros((N, 1), dtype=np.float64),
                    step_size=float(step_size[j]),
                    max_iter=optalgo_params["max_iter"],
                    rel_tol=optalgo_params["rel_tol"])
                
                coef = self.coef(
                    data=train_data,
                    lambda_param=lambda_param,
                    optalgo_params=train_algo_control,
                    batchmc_params=batchmc_params,
                    batch_mc=batch_mc,
                    print_error=print_error)
    
                score += self.eval_loss_function(
                    new_data=test_data,
                    coef=coef,
                    batchmc_params=batchmc_params,
                    batch_mc=batch_mc)
                
            nll_scores[j, ] = score / k_folds
            if save_info:
                f_log.write('score: %.8f\n' % nll_scores[j, ])
        
        if save_info:
            f_log.close()
    
        cv_result = {np.round(x, 5): np.round(y, 10) for x, y in zip(lambda_cand, nll_scores)}
        print("The cross validation scores are:\n" + str(cv_result))
        
        # find the optimal penalty parameter
        opt_lambda = lambda_cand[np.argmin(nll_scores)]
        print("=" * 50)
        print("The optimal penalty parameter is {}.".format(opt_lambda))
        print("=" * 50 + "\nFinal run with the optimal lambda.")
    
        # compute the coefficient vector at the optimal penalty parameter
        optalgo_params['step_size'] = float(step_size[np.argmin(nll_scores)])
        opt_coef = self.coef(
            data=self.data,
            lambda_param=opt_lambda,
            optalgo_params=optalgo_params,
            batch_mc=batch_mc,
            batchmc_params=batchmc_params,
            print_error=print_error)
        
        if save_info:
            f_optcoef = open('%s/negloglik_optcoef.npy' % save_dir, 'wb')
            np.save(f_optcoef, opt_coef)
            f_optcoef.close()
    
        output = {"opt_lambda": opt_lambda,
                  "opt_coef": opt_coef}
    
        return output
