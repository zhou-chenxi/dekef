from dekef.kernel_function import *
from dekef.scorematching_common_functions import *
from dekef.scorematching_loss_function import *


def scorematching_penalized_coef(data, kernel_function, base_density, lambda_param):
    
    """
    Computes the coefficients of basis functions in the penalized score matching density estimation.

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
    
    lambda_param : float
        The penalty parameter. Must be non-negative.

    Returns
    -------
    numpy.ndarray
        An array of coefficients for the penalized score matching density estimate.
    
    References
    ----------
    Sriperumbudur, Bharath, Kenji Fukumizu, Arthur Gretton, Aapo Hyvärinen, and Revant Kumar. 2017. “Density Estimation
        in Infinite Dimensional Exponential Families.” Journal of Machine Learning Research: JMLR 18 (57): 1–59.
    
    """

    # check that lambda must be non-negative
    if lambda_param < 0.:
        raise ValueError("The lambda_val cannot be negative.")

    N, d = data.shape

    # form the C matrix and the h vector
    c_mat = kernel_function.partial_kernel_matrix_11(data)
    h_vec = vector_h(data=data, kernel_function=kernel_function, base_density=base_density)
    
    # form the system of equations to solve 
    eq_left = c_mat + N * lambda_param * np.diag(np.ones((N * d,), dtype=np.float64))
    eq_right = - h_vec / lambda_param
    
    # solve the linear system 
    coef1 = np.linalg.lstsq(eq_left, eq_right, rcond=None)[0].reshape(-1, 1)
    
    # combine the coefficients 
    output = np.vstack((coef1, 1 / lambda_param))

    return output


def scorematching_penalized_optlambda(data, kernel_function, base_density, lambda_cand, k_folds,
                                      save_dir, save_info=False):
    
    """
    Selects the optimal penalty parameter in the penalized score matching density estimation
    using k-fold cross validation and computes the coefficient vector at this optimal penalty parameter.
    
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

    lambda_cand : list or 1-dimensional numpy.ndarray
        The list of penalty parameter candidates. Each of them must be non-negative.

    k_folds : int
        The number of folds for cross validation.

    save_dir : str
        The directory path to which the estimation information is saved; only works when save_info is True.

    save_info : bool, optional
        Whether to save the estimation information, including the values of score matching loss function of
        each fold and the coefficient vector at the optimal penalty parameter, to a local file;
        default is False.

    Returns
    -------
    dict
        A dictionary containing opt_lambda, the optimal penalty parameter, and
        opt_coef, the coefficient vector at the optimal penalty parameter.
    
    References
    ----------
    Sriperumbudur, Bharath, Kenji Fukumizu, Arthur Gretton, Aapo Hyvärinen, and Revant Kumar. 2017. “Density Estimation
        in Infinite Dimensional Exponential Families.” Journal of Machine Learning Research: JMLR 18 (57): 1–59.
    
    """

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    N, d = data.shape

    # check the non-negativity of lambda_cand
    lambda_cand = np.array(lambda_cand).flatten()
    if np.any(lambda_cand < 0.):
        raise ValueError("There exists at least one element in lambda_cand whose value is negative. Please modify.")

    n_lambda = len(lambda_cand)
    
    folds_i = np.random.randint(low=0, high=k_folds, size=N)

    sm_scores = np.zeros((n_lambda,), dtype=np.float64)
    
    if save_info:
        f_log = open('%s/log.txt' % save_dir, 'w')
    
    for j in range(n_lambda):
        
        # initialize the sm score
        score = 0
        lambda_param = lambda_cand[j]
        
        print("Lambda " + str(j) + ": " + str(lambda_param))
        
        if save_info:
            f_log.write('lambda: %.8f, ' % lambda_param)

        for i in range(k_folds):
            # data split
            train_data = data[folds_i != i, ]
            test_data = data[folds_i == i, ]
            
            if kernel_function.kernel_type == 'gaussian_poly2':
                
                kernel_function_sub = GaussianPoly2(data=train_data,
                                                    r1=kernel_function.r1,
                                                    r2=kernel_function.r2,
                                                    c=kernel_function.c,
                                                    bw=kernel_function.bw)
                
            elif kernel_function.kernel_type == 'rationalquad_poly2':
                
                kernel_function_sub = RationalQuadPoly2(data=train_data,
                                                        r1=kernel_function.r1,
                                                        r2=kernel_function.r2,
                                                        c=kernel_function.c,
                                                        bw=kernel_function.bw)
                
            else:
    
                raise NotImplementedError(("The kernel function should be one of 'gaussian_poly2' and "
                                           "'rationalquad_poly2'."))
            
            # compute the coefficient vector for the given lambda
            coef_vec = scorematching_penalized_coef(
                data=train_data,
                kernel_function=kernel_function_sub,
                base_density=base_density,
                lambda_param=lambda_param)
            
            score += scorematching_loss_function(
                data=train_data,
                new_data=test_data,
                coef=coef_vec,
                kernel_function=kernel_function_sub,
                base_density=base_density)

        sm_scores[j, ] = score / k_folds
        if save_info:
            f_log.write('score: %.8f\n' % sm_scores[j, ])

    cv_result = {np.round(x, 5): np.round(y, 10) for x, y in zip(lambda_cand, sm_scores)}
    print("The cross validation scores are:\n" + str(cv_result))
    
    # find the optimal regularization parameter
    opt_lambda = lambda_cand[np.argmin(sm_scores)]
    
    opt_coef = scorematching_penalized_coef(
        data=data,
        kernel_function=kernel_function,
        base_density=base_density,
        lambda_param=opt_lambda)
    
    if save_info:
        f_log.close()
    
    if save_info:
        f_optcoef = open('%s/scorematching_optcoef.npy' % save_dir, 'wb')
        np.save(f_optcoef, opt_coef)
        f_optcoef.close()
    
    output = {"opt_lambda": opt_lambda,
              "opt_coef": opt_coef}

    return output
