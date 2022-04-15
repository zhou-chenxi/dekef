from dekef.kernel_function import *
from dekef.base_density import *
from dekef.ortho_proj_hatz import hatz
from dekef.check import *


class ScoreMatchingGridPointsEarlyStopping:
    
    """
    A class to estimate the probability density function by minimizing the score matching loss function
    using the gradient descent algorithm.
    
    ...
    
    Attributes
    ---------
    data : numpy.ndarray
        The array of observations whose density function is to be estimated.
    
    grid_points : numpy.ndarray
        The array at which the kernel functions are centered.
    
    base_density : base_density object
        The base density function used to estimate the probability density function.
    
    kernel_function_data : kernel_function object
        The kernel function centered at self.data.
    
    kernel_function_gridpoints : kernel_function object
        The kernel function centered at self.grid_points.
    
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
    
    def __init__(self, data, grid_points, base_density,
                 kernel_type='gaussian_poly2', kernel_r1=1.0, kernel_r2=0., kernel_c=0., kernel_bw=1.0):
        
        """
        data : numpy.ndarray
            The array of observations whose density function is to be estimated.
        
        grid_points : numpy.ndarray
            The array at which the kernel functions are centered.
        
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
            
        if len(grid_points.shape) == 1:
            grid_points = grid_points.reshape(-1, 1)
        
        if grid_points.shape[1] != data.shape[1]:
            raise ValueError('The dimensionality of grid_points and that of data do not match.')
        
        self.data = data
        self.grid_points = grid_points
        self.base_density = base_density
        
        if kernel_type == 'gaussian_poly2':
            
            kernel_function_gridpoints = GaussianPoly2(
                data=grid_points,
                r1=kernel_r1,
                r2=kernel_r2,
                c=kernel_c,
                bw=kernel_bw
            )
            
            self.kernel_function_gridpoints = kernel_function_gridpoints
            
            kernel_function_data = GaussianPoly2(
                data=data,
                r1=kernel_r1,
                r2=kernel_r2,
                c=kernel_c,
                bw=kernel_bw
            )
            
            self.kernel_function_data = kernel_function_data
        
        elif kernel_type == 'rationalquad_poly2':
            
            kernel_function_gridpoints = RationalQuadPoly2(
                data=grid_points,
                r1=kernel_r1,
                r2=kernel_r2,
                c=kernel_c,
                bw=kernel_bw
            )
            
            self.kernel_function_gridpoints = kernel_function_gridpoints
            
            kernel_function_data = RationalQuadPoly2(
                data=data,
                r1=kernel_r1,
                r2=kernel_r2,
                c=kernel_c,
                bw=kernel_bw
            )
            
            self.kernel_function_data = kernel_function_data
        
        else:
            
            raise ValueError(f"kernel_type must be one of 'gaussian_poly2' and 'rationalquad_poly2, "
                             f"but got {kernel_type}'.")
    
    def coef(self, data, iter_num, step_size, threshold=1e-8):
        
        """
        Computes the coefficients of basis functions in the early stopping score matching density estimate,
        assuming the starting point of the gradient descent algorithm is the zero function.
        
        Parameters
        ----------
        data : numpy.ndarray
            The array of observations whose density function is to be estimated.
            This data may be different from self.data, especially in applying the cross validation.
        
        iter_num : int
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

        assert isinstance(iter_num, int), 'iter_num must be an integer.'
        assert step_size > 0, 'step_size must be strictly positive.'

        n, d = data.shape
        
        if self.kernel_function_gridpoints.kernel_type == 'gaussian_poly2':
            
            kernel_function_sub = GaussianPoly2(
                data=data,
                r1=self.kernel_function_gridpoints.r1,
                r2=self.kernel_function_gridpoints.r2,
                c=self.kernel_function_gridpoints.c,
                bw=self.kernel_function_gridpoints.bw
            )
            
        elif self.kernel_function_gridpoints.kernel_type == 'rationalquad_poly2':
            
            kernel_function_sub = RationalQuadPoly2(
                data=data,
                r1=self.kernel_function_gridpoints.r1,
                r2=self.kernel_function_gridpoints.r2,
                c=self.kernel_function_gridpoints.c,
                bw=self.kernel_function_gridpoints.bw
            )
            
        matg = kernel_function_sub.partial_kernel_matrix_10(self.grid_points).T
        
        eq_lhs = matg @ matg.T / n
        
        eq_rhs = hatz(
            data=data,
            new_data=self.grid_points,
            kernel_function=kernel_function_sub,
            base_density=self.base_density
        ).reshape(-1, 1)
        
        ei_values, ei_vector = np.linalg.eigh(eq_lhs)
        
        tilde_lamb = (
                ((1. - (1. - step_size * ei_values) ** iter_num) / ei_values / step_size) * (ei_values >= threshold) +
                (iter_num) * (ei_values < threshold)
        )
        
        coef1 = ei_vector @ np.diag(tilde_lamb) @ ei_vector.T @ eq_rhs
        output = step_size * coef1
        
        return output
    
    def optiternum(self, iter_num_cand, step_size, k_folds, save_dir, save_info=False, threshold=1e-8):
        
        """
        Selects the optimal number of iterations in the early stopping score matching density estimation
        using k-fold cross validation and computes the coefficient vector of basis functions
        at this optimal number of iterations.
        
        Parameters
        ----------
        iter_num_cand : list or 1-dimensional numpy.ndarray
            The list of numbers of iterations candidates.
        
        step_size : float
            The constant step size used in the gradient descent algorithm.

        k_folds : int
            The number of folds for cross validation.
        
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
        
        # check the non-negativity of iter_num_cand
        iter_num_cand = np.array(iter_num_cand).flatten()
        n_iternum = len(iter_num_cand)
        
        folds_i = np.random.randint(low=0, high=k_folds, size=N)
        sm_scores = np.zeros((n_iternum,), dtype=np.float64)
        
        if save_info:
            f_log = open('%s/log.txt' % save_dir, 'w')
        
        for j in range(n_iternum):
            
            # initialize the sm score
            score = 0
            iter_num = int(iter_num_cand[j])
            print("Iter num " + str(j) + ": " + str(iter_num))
            
            if save_info:
                f_log.write('iteration number: %d ' % iter_num)
            
            for i in range(k_folds):
                
                # data split
                train_data = self.data[folds_i != i,]
                test_data = self.data[folds_i == i,]
                
                if self.kernel_function_gridpoints.kernel_type == 'gaussian_poly2':
                    
                    kernel_function_test = GaussianPoly2(
                        data=test_data,
                        r1=self.kernel_function_gridpoints.r1,
                        r2=self.kernel_function_gridpoints.r2,
                        c=self.kernel_function_gridpoints.c,
                        bw=self.kernel_function_gridpoints.bw
                    )
                
                elif self.kernel_function_gridpoints.kernel_type == 'rationalquad_poly2':
                    
                    kernel_function_test = RationalQuadPoly2(
                        data=test_data,
                        r1=self.kernel_function_gridpoints.r1,
                        r2=self.kernel_function_gridpoints.r2,
                        c=self.kernel_function_gridpoints.c,
                        bw=self.kernel_function_gridpoints.bw
                    )
                
                # compute the coefficient vector for the given iter_num
                coef_vec = self.coef(
                    data=train_data,
                    iter_num=iter_num,
                    step_size=step_size,
                    threshold=threshold
                )
                
                # evaluate the score matching loss function
                matg = kernel_function_test.partial_kernel_matrix_10(self.grid_points).T
                
                t_vec = hatz(
                    data=test_data,
                    new_data=self.grid_points,
                    kernel_function=kernel_function_test,
                    base_density=self.base_density
                ).reshape(-1, 1)
                
                loss1 = coef_vec.T @ (matg @ matg.T / len(test_data)) @ coef_vec / 2. - coef_vec.T @ t_vec
                
                score += loss1.item()
                
            sm_scores[j,] = score / k_folds
            if save_info:
                f_log.write('score: %.8f\n' % sm_scores[j,])
                
        cv_result = {np.round(x, 5): np.round(y, 10) for x, y in zip(iter_num_cand, sm_scores)}
        print("The cross validation scores are:\n" + str(cv_result))
        
        # find the optimal regularization parameter
        opt_iternum = iter_num_cand[np.argmin(sm_scores)]
        
        print("=" * 50)
        print("The optimal number of iteration is {}.".format(opt_iternum))
        print("=" * 50 + "\nFinal run with the optimal number of iteration.")
        
        opt_coef = self.coef(
            data=self.data,
            iter_num=int(opt_iternum),
            step_size=step_size,
            threshold=threshold
        )
        
        if save_info:
            f_log.close()
        
        if save_info:
            f_optcoef = open('%s/scorematching_optcoef.npy' % save_dir, 'wb')
            np.save(f_optcoef, opt_coef)
            f_optcoef.close()
        
        output = {"opt_iter": opt_iternum, "opt_coef": opt_coef}
        
        return output
