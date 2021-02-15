from denest_kernelexpofam.scorematching_common_functions import *


class UnnormalizedDensity:
    
    """
    A class for computing un-normalized density functions.

    ...

    Attributes
    ----------
    data : numpy.ndarray
        The array of observations with which the density function is estimated.
    
    kernel_function : kernel_function object
        The kernel function used to estimate the probability density function.
        Must be instantiated from the classes with __type__ being 'kernel_function'.
        
    base_density : base_density object
        The base density function used to estimate the probability density function.
        Must be instantiated from the classes with __type__ being 'base_density'.
    
    coef : numpy.ndarray
        The array of coefficients for basis functions in the natural parameter in the estimated density function.
        
    dim : int
        The dimensionality of the data, i.e., the number of columns of the data.
    
    Methods
    -------
    density_eval_gubasis(new_data)
        Evaluates the un-normalized density estimate at new data using the basis functions from
        Gu and Qiu (1993) and Gu (1993). That is, the basis functions of the natural parameter are
        k (X_1, .), ..., k (X_n, .).
    
    natural_param_eval_smbasis(new_data)
        Evaluates the natural parameter at new data using the basis functions of
        the score matching density estimator.
    
    density_eval_smbasis(new_data)
        Evaluates the un-normalized density estimate at new data using the basis functions of
        the score matching density estimator.
        
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
        the basis functions of the natural parameter in the score matching density estimators.
    
    density_eval_smbasis_2d(x0, x1)
        Evaluates the un-normalized density estimate at a 2-dimensional data point (x0, x1),
        where x0 and x1 are the two coordinates, respectively, using the basis functions of
        the natural parameter in the score matching density estimators.
    
    References
    ----------
    Gu, Chong, and Chunfu Qiu. 1993. “Smoothing Spline Density Estimation: Theory.” Annals of Statistics 21 (1):
        217–34.
    Gu, Chong. 1993. “Smoothing Spline Density Estimation: A Dimensionless Automatic Algorithm.” Journal of the
        American Statistical Association 88 (422): 495–504.
        
    """
    
    def __init__(self, data, kernel_function, base_density, coef):
        
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
            
        """
        
        self.data = data
        self.kernel_function = kernel_function
        self.base_density = base_density
        self.coef = coef.reshape(-1, 1)
        
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        self.dim = data.shape[1]
    
    def density_eval_gubasis(self, new_data):
        
        """
        Evaluates the un-normalized density estimate at new data using the basis functions from
        Gu and Qiu (1993) and Gu (1993). That is, the basis functions of the natural parameter are
        k (X_1, .), ..., k (X_n, .).
        
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
        
        if len(self.coef) != self.data.shape[0]: 
            raise ValueError(("The length of coef is incorrect. "
                              "If you want to use Gu's basis functions, the length of the coef should be "
                              "the same as the number of data, which is {ll}").format(ll=self.data.shape[0]))
        
        if len(new_data.shape) == 1:
            new_data = new_data.reshape(-1, 1)
            
        n, d = new_data.shape
        
        if d != self.dim: 
            raise ValueError("The dimensionality of new_data does not match that of data.")
        
        baseden_part = self.base_density.baseden_eval(new_data)
        exp_part = np.exp(self.kernel_function.kernel_gram_matrix(new_data).T, self.coef).flatten()
        
        output = baseden_part * exp_part
        
        return output
    
    def natural_param_eval_smbasis(self, new_data):

        """
        Evaluates the natural parameter at new data using the basis functions of
        the score matching density estimator.
        
        Parameters
        ----------
        new_data : numpy.ndarray
            The array of data at which the natural parameter is to be evaluated.
        
        Returns
        -------
        numpy.ndarray
            The 1-dimensional array of the natural parameter estimates at new_data.
        
        """
        
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
        Evaluates the un-normalized density estimate at new data using the basis functions of
        the score matching density estimator.
        
        Parameters
        ----------
        new_data : numpy.ndarray
            The array of data at which the un-normalized density estimate is to be evaluated.
        
        Returns
        -------
        numpy.ndarray
            The 1-dimensional array of the un-normalized density estimates at new_data.
            
        """
        
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
        
        n_obs = self.data.shape[0]
    
        den = (self.base_density.baseden_eval_2d(x0, x1) *
               np.exp(np.sum([self.coef[i] * self.kernel_function.kernel_x_2d(self.data[i, ])(x0, x1)
                              for i in range(n_obs)])))
    
        return den

    def density_eval_smbasis_1d(self, x):
        
        """
        Evaluates the un-normalized density estimate at a 1-dimensional data point x using
        the basis functions of the natural parameter in the score matching density estimators.

        Parameters
        ----------
        x : float
            A floating point number at which the un-normalized density estimate is to be evaluated.

        Returns
        -------
        float
            A floating point number of the un-normalized density estimate at x.
        
        """
        
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
        where x0 and x1 are the two coordinates, respectively, using the basis functions of
        the natural parameter in the score matching density estimators.

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
