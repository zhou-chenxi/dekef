from dekef.scorematching_common_functions import *
from dekef.check import *


class UnnormalizedDensity:
    
    """
    A class for computing un-normalized density estimate.

    ...

    Attributes
    ----------
    data : numpy.ndarray
        The array of observations whose probability density function is estimated.
    
    kernel_function : kernel_function object
        The kernel function used to estimate the probability density function.
        __type__ must be 'kernel_function'.
        
    base_density : base_density object
        The base density function used to estimate the probability density function.
        __type__ must be 'base_density'.
    
    coef : numpy.ndarray
        The array of coefficients for basis functions in the natural parameter in the estimated density function.
        
    dim : int
        The dimensionality of the data, i.e., the number of columns of the data.
    
    basis_type : str
        The type of the basis functions in the natural parameter. Must be one of
            - 'gubasis', the basis functions being the kernel functions centered at data,
            - 'smbasis', the basis functions being the same as those in the score matching density estimator, i.e.,
                         a linear combination of the first two partial derivatives of the kernel functions centered
                         at data,
            - 'grid_points', the basis functions being the kernel functions centered at a set of
                             pre-specified grid points.
    
    grid_points : numpy.ndarray, optional
        The set of grid points at which the kernel functions are centered.
        Only need to supple when basis_type is 'grid_points'. Default is None.
    
    Methods
    -------
    density_eval_gubasis(new_data)
        Evaluates the un-normalized density estimate at new data using the basis functions from
        Gu and Qiu (1993) and Gu (1993). That is, the basis functions of the natural parameter are
        k (X_1, .), ..., k (X_n, .).
    
    natural_param_eval_smbasis(new_data)
        Evaluates the natural parameter at new data using the same basis functions as those of
        the score matching density estimator.
    
    density_eval_smbasis(new_data)
        Evaluates the un-normalized density estimate at new data using the basis functions of
        the score matching density estimator.
    
    density_eval_grid_points(new_data)
        Evaluates the un-normalized density estimate at new data using the basis functions in the natural parameter
        being the kernel functions centered at self.grid_points.
        
    density_eval_gubasis_1d(x)
        Evaluates the un-normalized density estimate at a 1-dimensional data point x.
        The basis functions of the natural parameter are those from Gu and Qiu (1993) and Gu (1993) and
        are k (X_1, .), ..., k (X_n, .).

    density_eval_gubasis_2d(x0, x1)
        Evaluates the un-normalized density estimate at a 2-dimensional data point (x0, x1),
        where x0 and x1 are the two coordinates, respectively. The basis functions of the natural parameter
        are those from Gu and Qiu (1993) and Gu (1993) and are k (X_1, .), ..., k (X_n, .).

    density_eval_smbasis_1d(x)
        Evaluates the un-normalized density estimate at a 1-dimensional data point x using
        the same basis functions as those in the score matching density estimators.
    
    density_eval_smbasis_2d(x0, x1)
        Evaluates the un-normalized density estimate at a 2-dimensional data point (x0, x1),
        where x0 and x1 are the two coordinates, respectively, using the same basis functions as
        those in the score matching density estimators.
    
    density_eval_grid_points_1d(x)
        Evaluates the un-normalized density estimate at a 1-dimensional data point x.
        The basis functions of the natural parameter are the kernel functions centered at self.grid_points.
    
    density_eval_grid_points_2d(x0, x1)
        Evaluates the un-normalized density estimate at a 2-dimensional data point (x0, x1),
        where x0 and x1 are the two coordinates, respectively. The basis functions of the natural parameter
        are the kernel functions centered at self.grid_points.
        
    References
    ----------
    Gu, Chong, and Chunfu Qiu. 1993. “Smoothing Spline Density Estimation: Theory.” Annals of Statistics 21 (1):
        217–34.
    Gu, Chong. 1993. “Smoothing Spline Density Estimation: A Dimensionless Automatic Algorithm.” Journal of the
        American Statistical Association 88 (422): 495–504.
    Sriperumbudur, Bharath, Kenji Fukumizu, Arthur Gretton, Aapo Hyvärinen, and Revant Kumar. 2017.
        “Density Estimation in Infinite Dimensional Exponential Families.” Journal of Machine Learning Research:
        JMLR 18 (57): 1–59.
        
    """
    
    def __init__(self, data, kernel_function, base_density, coef, basis_type, grid_points=None):
        
        """
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
    
        coef : numpy.ndarray
            The array of coefficients for the natural parameter in the density estimate.
            
        basis_type : str
            The type of the basis functions in the natural parameter. Must be one of
                - 'gubasis', the basis functions being the kernel functions centered at data,
                - 'smbasis', the basis functions being the same as those in the score matching density estimator, i.e.,
                             a linear combination of the first two partial derivatives of the kernel functions centered
                             at data,
                - 'grid_points', the basis functions being the kernel functions centered at a set of
                                 pre-specified grid points.
        
        grid_points : numpy.ndarray, optional
            The set of grid points at which the kernel functions are centered.
            Only need to supple when basis_type is 'grid_points'. Default is None.
            
        """

        check_kernelfunction(kernel_function)
        check_basedensity(base_density)

        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        self.data = data
        self.kernel_function = kernel_function
        self.base_density = base_density
        self.coef = coef.reshape(-1, 1)
        
        self.dim = data.shape[1]
        
        if basis_type not in ['gubasis', 'smbasis', 'grid_points']:
            raise ValueError("basis_type must be one of 'gubasis', 'smbasis' and 'grid_points'.")

        self.basis_type = basis_type
        
        if basis_type == 'gubasis' or basis_type == 'smbasis':
            # check the data stored in kernel_function is data
            test = np.allclose(self.kernel_function.data, self.data)
            if not test:
                raise ValueError('Please check the compatibility of data and kernel_function.data.')
        
        if basis_type == 'grid_points':
            if grid_points is None:
                raise ValueError("The basis_type is 'grid_points'. grid_points cannot be None.")
            else:
                
                if len(grid_points.shape) == 1:
                    grid_points = grid_points.reshape(-1, 1)
                if grid_points.shape[1] != self.data.shape[1]:
                    raise ValueError('The dimensionality of grid_points and that of data do not match.')
                self.grid_points = grid_points
                
                test = np.allclose(self.kernel_function.data, self.grid_points)
                if not test:
                    raise ValueError('Please check the compatibility of grid_points and kernel_function.data.')
                    
    def density_eval_gubasis(self, new_data):
        
        """
        Evaluates the un-normalized density estimate at new data using the basis functions in the natural parameter
        being the kernel functions centered at self.data.
        
        Parameters
        ----------
        new_data : numpy.ndarray
            The array of data at which the un-normalized density estimate is to be evaluated.
        
        Returns
        -------
        numpy.ndarray
            The 1-dimensional array of the un-normalized density estimates at new_data.
            
        References
        ----------
        Gu, Chong, and Chunfu Qiu. 1993. “Smoothing Spline Density Estimation: Theory.” Annals of Statistics 21 (1):
            217–34.
        Gu, Chong. 1993. “Smoothing Spline Density Estimation: A Dimensionless Automatic Algorithm.” Journal of the
            American Statistical Association 88 (422): 495–504.
        
        """
        
        if self.basis_type != 'gubasis':
            raise ValueError(f'The self.basis_type is {self.basis_type}, density_eval_gubasis cannot be used.')
        
        if len(self.coef) != self.data.shape[0]: 
            raise ValueError(("The length of coef is incorrect. "
                              "If you want to use Gu's basis functions, the length of the coef should be "
                              "the same as the number of data, which is {ll}").format(ll=self.data.shape[0]))
        
        if len(new_data.shape) == 1:
            new_data = new_data.reshape(-1, 1)
            
        n, d = new_data.shape
        
        if d != self.dim: 
            raise ValueError("The dimensionality of new_data does not match that of data.")
        
        baseden_part = self.base_density.baseden_eval(new_data).flatten()
        exp_part = np.exp(np.matmul(self.kernel_function.kernel_gram_matrix(new_data).T, self.coef)).flatten()
        
        output = baseden_part * exp_part
        
        return output
    
    def natural_param_eval_smbasis(self, new_data):

        """
        Evaluates the natural parameter at new data using the same basis functions as those in
        the score matching density estimator.
        
        Parameters
        ----------
        new_data : numpy.ndarray
            The array of data at which the natural parameter is to be evaluated.
        
        Returns
        -------
        numpy.ndarray
            The 1-dimensional array of the natural parameter estimates at new_data.
        
        References
        ----------
        Sriperumbudur, Bharath, Kenji Fukumizu, Arthur Gretton, Aapo Hyvärinen, and Revant Kumar. 2017.
            “Density Estimation in Infinite Dimensional Exponential Families.” Journal of Machine Learning Research:
            JMLR 18 (57): 1–59.
        
        """

        if self.basis_type != 'smbasis':
            raise ValueError(f'The self.basis_type is {self.basis_type}, natual_param_eval_smbasis cannot be used.')
        
        if len(self.coef) != self.data.shape[1] * self.data.shape[0] + 1:
            raise ValueError(("The length of coef is incorrect. "
                              "If you want to use score matching basis functions, the length of the coef should be "
                              "the same as the number of data, which is {ll}").format(
                ll=self.data.shape[0] * self.data.shape[1] + 1))

        if len(new_data.shape) == 1:
            new_data = new_data.reshape(-1, 1)
            
        if self.dim != new_data.shape[1]: 
            raise ValueError("The dimensionality of new_data and that of data do not match. ")
        
        f_matrix = kernel_partial10_hatz(data=self.data,
                                         new_data=new_data,
                                         kernel_function=self.kernel_function,
                                         base_density=self.base_density)
        
        output = np.matmul(f_matrix.T, self.coef).flatten()
        
        return output
    
    def density_eval_smbasis(self, new_data):
        
        """
        Evaluates the un-normalized density estimate at new data using the same basis functions as those in
        the score matching density estimator.
        
        Parameters
        ----------
        new_data : numpy.ndarray
            The array of data at which the un-normalized density estimate is to be evaluated.
        
        Returns
        -------
        numpy.ndarray
            The 1-dimensional array of the un-normalized density estimates at new_data.
            
        References
        ----------
        Sriperumbudur, Bharath, Kenji Fukumizu, Arthur Gretton, Aapo Hyvärinen, and Revant Kumar. 2017.
            “Density Estimation in Infinite Dimensional Exponential Families.” Journal of Machine Learning Research:
            JMLR 18 (57): 1–59.
            
        """

        if self.basis_type != 'smbasis':
            raise ValueError(f'The self.basis_type is {self.basis_type}, natual_param_eval_smbasis cannot be used.')
        
        if len(self.coef) != self.data.shape[1] * self.data.shape[0] + 1:
            raise ValueError(("The length of coef is incorrect. "
                              "If you want to use score matching basis functions, the length of the coef should be "
                              "the same as the number of data, which is {ll}").format(
                ll=self.data.shape[0] * self.data.shape[1] + 1))
        
        if len(new_data.shape) == 1:
            new_data = new_data.reshape(-1, 1)
            
        n, d = new_data.shape

        if d != self.dim:
            raise ValueError("The dimensionality of new_data does not match that of data.")

        base_part = self.base_density.baseden_eval(new_data=new_data).flatten()
        f_part = self.natural_param_eval_smbasis(new_data=new_data)
        output = base_part * np.exp(f_part)
        
        return output

    def density_eval_grid_points(self, new_data):
    
        """
        Evaluates the un-normalized density estimate at new data using the basis functions in the natural parameter
        being the kernel functions centered at self.grid_points.

        Parameters
        ----------
        new_data : numpy.ndarray
            The array of data at which the un-normalized density estimate is to be evaluated.

        Returns
        -------
        numpy.ndarray
            The 1-dimensional array of the un-normalized density estimates at new_data.
        """
    
        if self.basis_type != 'grid_points':
            raise ValueError(f'The self.basis_type is {self.basis_type}, density_eval_grid_points cannot be used.')
    
        if len(self.coef) != self.grid_points.shape[0]:
            raise ValueError(("The length of coef is incorrect. "
                              "If you want to use basis functions centered at self.grid_points, "
                              "the length of the coef should be the same as the number of self.grid_points, "
                              "which is {ll}").format(ll=self.data.shape[0]))
    
        if len(new_data.shape) == 1:
            new_data = new_data.reshape(-1, 1)
            
        n, d = new_data.shape

        if d != self.dim:
            raise ValueError("The dimensionality of new_data does not match that of data.")
 
        baseden_part = self.base_density.baseden_eval(new_data).flatten()
        
        exp_part = np.exp(np.matmul(self.kernel_function.kernel_gram_matrix(new_data).T, self.coef)).flatten()
        
        output = baseden_part * exp_part
    
        return output
    
    def density_eval_gubasis_1d(self, x):
        
        """
        Evaluates the un-normalized density estimate at a 1-dimensional data point x.
        The basis functions of the natural parameter are those from Gu and Qiu (1993) and Gu (1993) and
        are k (X_1, .), ..., k (X_n, .).

        Parameters
        ----------
        x : float
            A floating point number at which the un-normalized density estimate is to be evaluated.

        Returns
        -------
        float
            A floating point number of the un-normalized density estimate at x.
        
        References
        ----------
        Gu, Chong, and Chunfu Qiu. 1993. “Smoothing Spline Density Estimation: Theory.” Annals of Statistics 21 (1):
            217–34.
        Gu, Chong. 1993. “Smoothing Spline Density Estimation: A Dimensionless Automatic Algorithm.” Journal of the
            American Statistical Association 88 (422): 495–504.
            
        """

        if self.basis_type != 'gubasis':
            raise ValueError(f'The self.basis_type is {self.basis_type}, density_eval_gubasis_1d cannot be used.')
        
        if self.dim != 1:
            raise ValueError('In order to use density_eval_gubasis_1d, '
                             f'the dimensionality of self.data must be 1-dimensional, but is {self.dim}.')
        n_obs = self.data.shape[0]
    
        den = (self.base_density.baseden_eval_1d(x) *
               np.exp(np.sum([self.coef[i] * self.kernel_function.kernel_x_1d(self.data[i, ])(x)
                              for i in range(n_obs)])))
    
        return den

    def density_eval_gubasis_2d(self, x0, x1):
        
        """
        Evaluates the un-normalized density estimate at a 2-dimensional data point (x0, x1),
        where x0 and x1 are the two coordinates, respectively. The basis functions of the natural parameter
        are those from Gu and Qiu (1993) and Gu (1993) and are k (X_1, .), ..., k (X_n, .).

        Parameters
        ----------
        x0, x1 : float
            Two floating point numbers forming the coordinates of a 2-dimensional data point at which
            un-normalized density estimate is to be evaluated.

        Returns
        -------
        float
            A floating point number of the un-normalized density estimate at (x0, x1).
        
        References
        ----------
        Gu, Chong, and Chunfu Qiu. 1993. “Smoothing Spline Density Estimation: Theory.” Annals of Statistics 21 (1):
            217–34.
        Gu, Chong. 1993. “Smoothing Spline Density Estimation: A Dimensionless Automatic Algorithm.” Journal of the
            American Statistical Association 88 (422): 495–504.
            
        """

        if self.basis_type != 'gubasis':
            raise ValueError(f'The self.basis_type is {self.basis_type}, density_eval_gubasis_2d cannot be used.')
        
        if self.dim != 2:
            raise ValueError('In order to use density_eval_gubasis_2d, '
                             f'the dimensionality of self.data must be 2-dimensional, but is {self.dim}.')
        
        n_obs = self.data.shape[0]
        
        den = (self.base_density.baseden_eval_2d(x0, x1) *
               np.exp(np.sum([self.coef[i] * self.kernel_function.kernel_x_2d(self.data[i, ])(x0, x1)
                              for i in range(n_obs)])))
    
        return den

    def density_eval_smbasis_1d(self, x):
        
        """
        Evaluates the un-normalized density estimate at a 1-dimensional data point x using
        the same basis functions as those in the score matching density estimators.

        Parameters
        ----------
        x : float
            A floating point number at which the un-normalized density estimate is to be evaluated.

        Returns
        -------
        float
            A floating point number of the un-normalized density estimate at x.
        
        References
        ----------
        Sriperumbudur, Bharath, Kenji Fukumizu, Arthur Gretton, Aapo Hyvärinen, and Revant Kumar. 2017.
            “Density Estimation in Infinite Dimensional Exponential Families.” Journal of Machine Learning Research:
            JMLR 18 (57): 1–59.
            
        """

        if self.basis_type != 'smbasis':
            raise ValueError(f'The self.basis_type is {self.basis_type}, density_eval_smbasis_1d cannot be used.')
        
        if self.dim != 1:
            raise ValueError('In order to use density_eval_smbasis_1d, '
                             f'the dimensionality of self.data must be 1-dimensional, but is {self.dim}.')
        
        n_obs = self.data.shape[0]
        n_basis = n_obs + 1
        
        # linear combination of first derivatives 
        fx1 = np.sum([self.coef[i] * self.kernel_function.kernel_x_1d_deriv1(self.data[i, ])(x) 
                      for i in range(n_basis - 1)])
        # xi part 
        hatz1 = np.sum([self.base_density.logbaseden_deriv1(new_data=self.data[i, ].reshape(1, 1), j=0) *
                        self.kernel_function.kernel_x_1d_deriv1(self.data[i, ])(x)
                        for i in range(n_obs)])
        hatz2 = np.sum([self.kernel_function.kernel_x_1d_deriv2(self.data[i, ])(x)
                        for i in range(n_obs)])
        hatz = -(hatz1 + hatz2) / n_obs
        
        output1 = self.base_density.baseden_eval_1d(x) * np.exp(hatz * self.coef[-1] + fx1)
        
        return output1
    
    def density_eval_smbasis_2d(self, x0, x1):
        
        """
        Evaluates the un-normalized density estimate at a 2-dimensional data point (x0, x1),
        where x0 and x1 are the two coordinates, respectively, using the same basis functions as
        those in the score matching density estimators.

        Parameters
        ----------
        x0, x1 : float
            Two floating point numbers forming the coordinates of a 2-dimensional data point at which
            un-normalized density estimate is to be evaluated.

        Returns
        -------
        float
            A floating point number of the un-normalized density estimate at (x0, x1).
            
        References
        ----------
        Sriperumbudur, Bharath, Kenji Fukumizu, Arthur Gretton, Aapo Hyvärinen, and Revant Kumar. 2017.
            “Density Estimation in Infinite Dimensional Exponential Families.” Journal of Machine Learning Research:
            JMLR 18 (57): 1–59.
            
        """

        if self.basis_type != 'smbasis':
            raise ValueError(f'The self.basis_type is {self.basis_type}, density_eval_smbasis_2d cannot be used.')

        if self.dim != 2:
            raise ValueError('In order to use density_eval_smbasis_2d, '
                             f'the dimensionality of self.data must be 2-dimensional, but is {self.dim}.')
        
        n_obs = self.data.shape[0]
        n_basis = 2 * n_obs + 1
        
        # linear combination of first derivatives 
        fx1 = np.sum([self.coef[i] * self.kernel_function.kernel_x_2d_deriv1_0(self.data[int(i / 2), ])(x0, x1) + 
                      self.coef[i + 1] * self.kernel_function.kernel_x_2d_deriv1_1(self.data[int(i / 2), ])(x0, x1) 
                      for i in range(n_basis - 1)[::2]])
        
        # xi part 
        hatz1 = np.sum([self.base_density.logbaseden_deriv1(new_data=self.data[i, ], j=0) *
                        self.kernel_function.kernel_x_2d_deriv1_0(self.data[i, ])(x0, x1) +
                        self.base_density.logbaseden_deriv1(new_data=self.data[i, ], j=1) *
                        self.kernel_function.kernel_x_2d_deriv1_1(self.data[i, ])(x0, x1)
                        for i in range(n_obs)])
        hatz2 = np.sum([self.kernel_function.kernel_x_2d_deriv2_0(self.data[i, ])(x0, x1) +
                        self.kernel_function.kernel_x_2d_deriv2_1(self.data[i, ])(x0, x1)
                        for i in range(n_obs)])
        hatz = -(hatz1 + hatz2) / n_obs
        
        output1 = self.base_density.baseden_eval_2d(x0, x1) * np.exp(hatz * self.coef[-1] + fx1)
        
        return output1

    def density_eval_grid_points_1d(self, x):
    
        """
        Evaluates the un-normalized density estimate at a 1-dimensional data point x.
        The basis functions of the natural parameter are the kernel functions centered at self.grid_points.

        Parameters
        ----------
        x : float
            A floating point number at which the un-normalized density estimate is to be evaluated.

        Returns
        -------
        float
            A floating point number of the un-normalized density estimate at x.
            
        """
    
        if self.basis_type != 'grid_points':
            raise ValueError(f'The self.basis_type is {self.basis_type}, density_eval_grid_points_1d cannot be used.')
    
        if self.dim != 1:
            raise ValueError('In order to use density_eval_grid_points_1d, '
                             f'the dimensionality of self.data must be 1-dimensional, but is {self.dim}.')
        n = self.grid_points.shape[0]
    
        den = (self.base_density.baseden_eval_1d(x) *
               np.exp(np.sum([self.coef[i] * self.kernel_function.kernel_x_1d(self.grid_points[i, ])(x)
                              for i in range(n)])))
    
        return den

    def density_eval_grid_points_2d(self, x0, x1):
    
        """
        Evaluates the un-normalized density estimate at a 2-dimensional data point (x0, x1),
        where x0 and x1 are the two coordinates, respectively. The basis functions of the natural parameter
        are the kernel functions centered at self.grid_points.

        Parameters
        ----------
        x0, x1 : float
            Two floating point numbers forming the coordinates of a 2-dimensional data point at which
            un-normalized density estimate is to be evaluated.

        Returns
        -------
        float
            A floating point number of the un-normalized density estimate at (x0, x1).
            
        """
    
        if self.basis_type != 'grid_points':
            raise ValueError(f'The self.basis_type is {self.basis_type}, density_eval_grid_points_2d cannot be used.')
    
        if self.dim != 2:
            raise ValueError('In order to use density_eval_grid_points_2d, '
                             f'the dimensionality of self.data must be 2-dimensional, but is {self.dim}.')
    
        n = self.grid_points.shape[0]
    
        den = (self.base_density.baseden_eval_2d(x0, x1) *
               np.exp(np.sum([self.coef[i] * self.kernel_function.kernel_x_2d(self.grid_points[i, ])(x0, x1)
                              for i in range(n)])))
    
        return den
