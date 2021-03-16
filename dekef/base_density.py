import numpy as np
from scipy.stats import multivariate_normal, gamma, lognorm


class Baseden:
    
    """
    A parent class for the base density.
    
    __type__
        Set __type__ to be the base density.
    
    """
    
    def __init__(self):
        
        pass
    
    @staticmethod
    def __type__():
        
        return 'base_density'
    

class BasedenUniform(Baseden):

    """
    A class for the uniform base density function.

    ...

    Attributes
    ----------
    data : numpy.ndarray
        The array of observations whose density function is to be estimated.
        
    domain : numpy.ndarray
        A d by 2 array to represent the domain of the uniform base density function, where d is the
        dimensionality of data. The j-th element is an array of the minimum and
        the maximum of each dimension, for j = 0, ..., d-1.
        For example, if domain = [[0., 1.], [3., 4.]], there are two dimensions,
        and the first dimension is from 0. to 1., and the second dimension is from 3. to 4.
        
    dim : int
        The dimensionality of data and the underlying uniform base density function, that is, d.

    Methods
    -------
    sample(n_samples)
        Draws samples from uniform base distribution of sample size n_samples.

    baseden_eval(new_data)
        Evaluates the uniform base density function at new_data.

    logbaseden_deriv1(new_data, j)
        Evaluates the first partial derivative of the logarithm of the uniform base density function
        along the j-th coordinate at new_data.

    logbaseden_deriv2(new_data, j)
        Evaluates the second partial derivative of the logarithm of the uniform base density function
        along the j-th coordinate at new_data.

    baseden_eval_1d(x)
        Evaluates the uniform base density function at a 1-dimensional data point x.

    baseden_eval_2d(x0, x1)
        Evaluates the uniform base density function at a 2-dimensional data point (x0, x1),
        where x0 and x1 are the two coordinates, respectively.
    """

    def __init__(self, data, domain=None):
        """
        Parameters
        ----------
        data : numpy.ndarray
            The array of observations whose density function is to be estimated.

        domain : numpy.ndarray, optional
            The domain of the uniform base density function (default is None).
            If supplied, must be an numpy.ndarray of shape d by 2, where d is the number of columns of data,
            i.e., the dimensionality of data.
            The j-th element is an array of the minimum and the maximum of each dimension, for j = 0, ..., d-1.
            If not supplied, domain is computed from data. For each dimension, it is from the minimum of data of
            that dimension to the maximum of data of the same dimension.
        
        """

        super().__init__()

        if len(data.shape) == 1:
            # if data is just 1-dimensional, convert to n x 1 form
            data = data.reshape(-1, 1)

        if domain is None:
            # if the domain is not supplied
            domain_min = data.min(axis=0)
            domain_max = data.max(axis=0)
            domain = np.array([domain_min, domain_max]).T
        else:

            if not isinstance(domain, np.ndarray):
                # check the type of domain
                domain = np.array(domain)

            if domain.shape[1] != 2:
                # check the number of columns of domain
                raise ValueError("The domain should be of 2 columns, but got {ncol} columns.".format(
                    ncol=domain.shape[1]))

            if set([len(item) for item in domain]) != {2}:
                raise ValueError("Each component of domain must be of length 2.")

            if domain.shape[0] != data.shape[1]:
                # check domain nrow == data ncol
                raise ValueError(("There are {domain_coor} coordinates in domain, but {data_coor} "
                                  "coordinates in data.").format(
                    domain_coor=domain.shape[0],
                    data_coor=data.shape[1]))

            # check that the data are within the domain specified
            d_min = np.min(data, axis=0, keepdims=True)
            d_max = np.max(data, axis=0, keepdims=True)
            if np.any(d_min < domain[:, 0]) or np.any(d_max > domain[:, 1]):
                raise ValueError("The data does not lie in the domain specified. ")

        self.domain = np.array(domain)
        self.dim = data.shape[1]
        self.data = data

    def sample(self, n_samples):

        """
        Draws samples from the uniform distribution of sample size n_samples.

        Parameters
        ----------
        n_samples : int
            The number of new samples to be drawn.

        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, self.dim) of new samples drawn from the uniform distribution.
            
        """

        output = np.zeros((n_samples, self.dim), dtype=np.float32)

        for i in range(self.dim):
            output[:, i] = np.random.uniform(low=self.domain[i][0],
                                             high=self.domain[i][1],
                                             size=n_samples)

        return output

    def baseden_eval(self, new_data):

        """
        Evaluates the uniform base density function at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            The new data at which the uniform base density function is to be evaluated.

        Returns
        -------
        numpy.ndarray
            A (new_data.shape[0], 1) array of the uniform base density values at new_data.
            
        """

        if new_data.shape[1] != self.dim:
            raise ValueError("The dimensionality of new_data and that of self.data do not match. ")

        product = 1. / np.prod(self.domain[:, 1] - self.domain[:, 0])

        output = np.apply_along_axis(
            lambda x: 0. if np.prod(np.hstack([x - self.domain[:, 0] >= 0,
                                               x - self.domain[:, 1] <= 0])) == 0. else product,
            axis=1,
            arr=new_data)

        return output.reshape(-1, 1)

    def logbaseden_deriv1(self, new_data, j):

        """
        Evaluates the first partial derivative of the logarithm of the uniform base density function
        along the j-th coordinate at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            The new data at which the first partial derivative of the logarithm of the uniform base density function
            is to be evaluated.
        j : int
            The coordinate with respect to which the first partial derivative is taken. It should be between 0
            and self.dim-1.

        Returns
        -------
        numpy.ndarray
            A (new_data.shape[0], 1) array of the first partial derivative values of the logarithm of
            the uniform base density function along the j-th coordinate at new_data.
            
        """

        if new_data.shape[1] != self.dim:
            raise ValueError("The dimensionality of new_data and that of self.data do not match. ")

        if j not in range(self.dim):
            raise ValueError("The index j must be between 0 and {d}.".format(d=self.dim - 1))

        output = np.zeros((new_data.shape[0], 1), dtype=np.float64)

        return output

    def logbaseden_deriv2(self, new_data, j):

        """
        Evaluates the second partial derivative of the logarithm of the uniform base density function
        along the j-th coordinate at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            The new data at which the second partial derivative of the logarithm of the uniform base density function
            is to be evaluated.
        j : int
            The coordinate with respect to which the partial derivative is taken. It should be between 0
            and self.dim-1.

        Returns
        -------
        numpy.ndarray
            A (new_data.shape[0], 1) array of the second partial derivative values of the logarithm of
            the uniform base density function along the j-th coordinate at new_data.
            
        """

        if new_data.shape[1] != self.dim:
            raise ValueError("The dimensionality of new_data and that of self.data do not match. ")

        if j not in range(self.dim):
            raise ValueError("The index j must be between 0 and {d}.".format(d=self.dim - 1))

        output = np.zeros((new_data.shape[0], 1), dtype=np.float64)

        return output

    def baseden_eval_1d(self, x):

        """
        Evaluates the uniform base density function at a 1-dimensional data point x.

        Parameters
        ----------
        x : float
            A floating point number at which the uniform base density function is to be evaluated.

        Returns
        -------
        float
            A floating point number of the uniform base density value at x.
            
        """

        # check the dimensionality
        if self.dim != 1:
            raise ValueError("The dimensionality should be 1.")

        if x < self.domain[:, 0] or x > self.domain[:, 1]:
            output = 0.
        else:
            output = 1. / (self.domain[:, 1] - self.domain[:, 0])

        return output

    def baseden_eval_2d(self, x0, x1):

        """
        Evaluates the uniform base density function at a 2-dimensional data point (x0, x1),
        where x0 and x1 are the two coordinates, respectively.

        Parameters
        ----------
        x0, x1 : float
            Two floating point numbers forming the coordinates of a 2-dimensional data point at which
            the uniform base density function is to be evaluated.

        Returns
        -------
        float
            A floating point number of the uniform base density value at (x0, x1).
            
        """

        # check the dimensionality
        if self.dim != 2:
            raise ValueError("The dimensionality should be 2.")

        x = np.array([[x0, x1]])

        if np.any(x < self.domain[:, 0]) or np.any(x > self.domain[:, 1]):
            output = 0.
        else:
            output = 1. / np.prod(self.domain[:, 1] - self.domain[:, 0])

        return output


class BasedenExp(Baseden):

    """
    A class for the exponential base density function.

    ...

    Attributes
    ----------
    data : numpy.ndarray
        The array of observations whose density function is to be estimated.
        
    scale : float
        The scale parameter in the exponential distribution.
        
    domain : numpy.ndarray
        The list [[0., np.inf]] to represent the domain of the exponential base density function.
        
    dim : int
        The dimensionality of data and the underlying exponential base density function, that is, 1.

    Methods
    -------
    sample(n_samples)
        Draws samples from the exponential base distribution of sample size n_samples.

    baseden_eval(new_data)
        Evaluates the exponential base density function at new_data.

    logbaseden_deriv1(new_data, j)
        Evaluates the first derivative of the logarithm of the exponential base density function
        along the j-th coordinate at new_data.

    logbaseden_deriv2(new_data, j)
        Evaluates the second derivative of the logarithm of the exponential base density function
        along the j-th coordinate at new_data.

    baseden_eval_1d(x)
        Evaluates the exponential base density function at a 1-dimensional data point x.

    """

    def __init__(self, data, scale):

        """
        Parameters
        ----------
        data : numpy.ndarray
            The array of observations whose density function is to be estimated.

        scale : float
            The scale parameter in the exponential distribution. Must be strictly positive. Default is None,
            under which condition the scale parameter is estimated using the method of maximum likelihood.
            
        """

        super().__init__()

        self.data = np.array(data).reshape(-1, 1)

        if np.any(data < 0.):
            raise ValueError("Data contain negative values. Exponential base density function cannot be used.")
        if len(data.shape) != 1 and data.shape[1] != 1:
            raise ValueError("The data must be 1-dimensional. ")
        if scale <= 0:
            raise ValueError("The scale parameter must be strictly positive.")

        if scale is not None:
            self.scale = scale
        else:
            self.scale = np.mean(data)

        self.dim = 1
        self.domain = [[0., np.inf]]

    def sample(self, n_samples):

        """
        Draws samples from the exponential distribution of sample size n_samples.

        Parameters
        ----------
        n_samples : int
            The number of new samples to be drawn.

        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, 1) of new samples drawn from the exponential distribution.
            
        """

        output = np.random.exponential(scale=self.scale, size=n_samples).reshape(-1, 1)

        return output

    def baseden_eval(self, new_data):

        """
        Evaluates the exponential base density function at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            The new data at which the exponential base density function is to be evaluated.

        Returns
        -------
        numpy.ndarray
            A (new_data.shape[0], 1) array of the exponential base density values at new_data.
            
        """

        if len(new_data.shape) == 1:
            new_data = new_data.reshape(-1, 1)

        if np.any(new_data <= 0.):
            raise ValueError("The new_data contain non-positive values.")

        new_data = new_data.reshape(-1, 1)

        output = np.apply_along_axis(lambda x: 0. if x <= 0. else np.exp(- x / self.scale) / self.scale,
                                     axis=1, arr=new_data)

        return output.reshape(-1, 1)

    def logbaseden_deriv1(self, new_data, j=0):

        """
        Evaluates the first derivative of the logarithm of the exponential base density function
        along the j-th coordinate at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            The new data at which the first derivative of the logarithm of the exponential base density function
            is to be evaluated.
        j : int
            The coordinate with respect to which the first derivative is taken. Default is 0 and
            it should always be 0.

        Returns
        -------
        numpy.ndarray
            A (new_data.shape[0], 1) array of the first derivative values of the logarithm of
            the exponential base density function along the j-th coordinate at new_data.
            
        """

        if len(new_data.shape) == 1:
            new_data = new_data.reshape(-1, 1)

        if np.any(new_data <= 0.):
            raise ValueError("The new_data contain non-positive values.")

        if j != 0:
            raise ValueError(("Since the exponential base density function is defined on positive real numbers,"
                              " the value of j should be 0."))

        new_data = new_data.reshape(-1, 1)

        output = np.apply_along_axis(lambda x: 0. if x < 0. else -1. / self.scale,
                                     axis=1, arr=new_data)

        return output.reshape(-1, 1)

    @staticmethod
    def logbaseden_deriv2(new_data, j=0):
        
        """
        Evaluates the second derivative of the logarithm of the exponential base density function
        along the j-th coordinate at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            The new data at which the second derivative of the logarithm of the exponential base density function
            is to be evaluated.
        j : int
            The coordinate with respect to which the second derivative is taken. Default is 0 and
            it should always be 0.

        Returns
        -------
        numpy.ndarray
            A (new_data.shape[0], 1) array of the second derivative values of the logarithm of
            the exponential base density function along the j-th coordinate at new_data.
            
        """

        if len(new_data.shape) == 1:
            new_data = new_data.reshape(-1, 1)

        if np.any(new_data <= 0.):
            raise ValueError("The new_data contain non-positive values.")

        if j != 0:
            raise ValueError(("Since the exponential base density function is defined on positive real numbers,"
                              " the value of j should be 0."))

        output = np.zeros((new_data.shape[0], 1), dtype=np.float32)

        return output

    def baseden_eval_1d(self, x):

        """
        Evaluates the exponential base density function at a 1-dimensional data point x.

        Parameters
        ----------
        x : float
            A floating point number at which the exponential base density function is to be evaluated.

        Returns
        -------
        float
            A floating point number of the exponential base density value at x.
            
        """

        if x < 0.:
            output = 0.
        else:
            output = np.exp(- x / self.scale) / self.scale

        return output


class BasedenGamma(Baseden):

    """
    A class for the gamma base density function.

    ...

    Attributes
    ----------
    data : numpy.ndarray
        The array of observations whose density function is to be estimated.
        
    a : float
        The shape parameter in the gamma distribution.
        
    loc : float
        The location parameter in the gamma distribution. Equal to 0.
        
    scale : float
        The scale parameter in the gamma distribution.
        
    domain : numpy.ndarray
        The list [[0., np.inf]] to represent the domain of the gamma base density function.
        
    dim : int
        The dimensionality of data and the underlying gamma base density function, that is, 1.

    Methods
    -------
    sample(n_samples)
        Draws samples from the gamma base distribution of sample size n_samples.

    baseden_eval(new_data)
        Evaluates the gamma base density function at new_data.

    logbaseden_deriv1(new_data, j)
        Evaluates the first derivative of the logarithm of the gamma base density function
        along the j-th coordinate at new_data.

    logbaseden_deriv2(new_data, j)
        Evaluates the second derivative of the logarithm of the gamma base density function
        along the j-th coordinate at new_data.

    baseden_eval_1d(x)
        Evaluates the gamma base density function at a 1-dimensional data point x.

    """

    def __init__(self, data, a=None, scale=None):

        """
        Parameters
        ----------
        data : numpy.ndarray
            The array of observations whose density function is to be estimated.

        a : float
            The shape parameter in the gamma distribution. Must be strictly positive. Default is None,
            under which condition the shape parameter is estimated using the method of maximum likelihood.

        scale : float
            The scale parameter in the gamma distribution. Must be strictly positive. Default is None,
            under which condition the scale parameter is estimated using the method of maximum likelihood.
            
        """

        super().__init__()

        self.data = np.array(data).reshape(-1, 1)

        if np.any(data < 0.):
            raise ValueError("Data contain negative values. Gamma base density function cannot be used.")

        if len(data.shape) != 1 and data.shape[1] != 1:
            raise ValueError("The data must be 1-dimensional. ")

        if a is not None and a <= 0.:
            raise ValueError("The parameter a must be strictly positive.")

        if scale is not None and scale <= 0.:
            raise ValueError("The parameter scale must be strictly positive.")

        if a is not None and scale is not None:
            # neither of a and scale is None
            self.a = a
            self.scale = scale
        elif a is None and scale is not None:
            # a is None but scale is not
            # estimate the parameter using ML
            self.a, _, self.scale = gamma.fit(self.data, floc=0., fscale=scale)
        elif a is not None and scale is None:
            # a is not None but scale is
            # estimate the parameter using ML
            self.a, _, self.scale = gamma.fit(self.data, floc=0., fa=a)
        elif a is None and scale is None:
            # both a and scale are None
            # estimate the parameter using ML
            self.a, _, self.scale = gamma.fit(self.data, floc=0.)

        self.loc = 0.
        self.dim = 1
        self.domain = [[0., np.inf]]

    def sample(self, n_samples):

        """
        Draws samples from the gamma distribution of sample size n_samples.

        Parameters
        ----------
        n_samples : int
            The number of new samples to be drawn.

        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, 1) of new samples drawn from the gamma distribution.
            
        """

        output = gamma.rvs(a=self.a, loc=self.loc, scale=self.scale, size=n_samples).reshape(-1, 1)

        return output

    def baseden_eval(self, new_data):

        """
        Evaluates the gamma base density function at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            The new data at which the gamma base density function is to be evaluated.

        Returns
        -------
        numpy.ndarray
            A (new_data.shape[0], 1) array of the gamma base density values at new_data.
            
        """

        new_data = new_data.reshape(-1, 1)

        output = gamma.pdf(x=new_data, a=self.a, loc=self.loc, scale=self.scale)

        return output.reshape(-1, 1)

    def logbaseden_deriv1(self, new_data, j=0):

        """
        Evaluates the first derivative of the logarithm of the gamma base density function
        along the j-th coordinate at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            The new data at which the first derivative of the logarithm of the gamma base density function
            is to be evaluated.
        j : int
            The coordinate with respect to which the first derivative is taken. Default is 0 and
            it should always be 0.

        Returns
        -------
        numpy.ndarray
            A (new_data.shape[0], 1) array of the first derivative values of the logarithm of
            the gamma base density function along the j-th coordinate at new_data.
            
        """

        if len(new_data.shape) == 1:
            new_data = new_data.reshape(-1, 1)

        if np.any(new_data <= 0.):
            raise ValueError("The new_data contain non-positive values.")

        if j != 0:
            raise ValueError(("Since the gamma base density function is defined on positive real numbers,"
                              " the value of j should be 0."))

        new_data = new_data.reshape(-1, 1)

        output = np.apply_along_axis(lambda x: 0. if x <= 0. else (self.a - 1) / x - 1 / self.scale,
                                     axis=1, arr=new_data)

        return output.reshape(-1, 1)

    def logbaseden_deriv2(self, new_data, j=0):

        """
        Evaluates the second derivative of the logarithm of the gamma base density function
        along the j-th coordinate at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            The new data at which the second derivative of the logarithm of the gamma base density function
            is to be evaluated.
        j : int
            The coordinate with respect to which the second derivative is taken. Default is 0 and
            it should always be 0.

        Returns
        -------
        numpy.ndarray
            A (new_data.shape[0], 1) array of the second derivative values of the logarithm of
            the gamma base density function along the j-th coordinate at new_data.
            
        """

        if len(new_data.shape) == 1:
            new_data = new_data.reshape(-1, 1)

        if np.any(new_data <= 0.):
            raise ValueError("The new_data contain non-positive values.")

        if j != 0:
            raise ValueError(("Since the gamma base density function is defined on positive real numbers,"
                              " the value of j should be 0."))

        new_data = new_data.reshape(-1, 1)

        output = np.apply_along_axis(lambda x: 0. if x <= 0. else - (self.a - 1) / x ** 2,
                                     axis=1, arr=new_data)

        return output.reshape(-1, 1)

    def baseden_eval_1d(self, x):

        """
        Evaluates the gamma base density function at a 1-dimensional data point x.

        Parameters
        ----------
        x : float
            A floating point number at which the gamma base density function is to be evaluated.

        Returns
        -------
        float
            A floating point number of the gamma base density value at x.
            
        """

        output = gamma.pdf(x=x, a=self.a, loc=self.loc, scale=self.scale)

        return output


class BasedenNormal(Baseden):

    """
    A class for the multivariate normal base density function.

    ...

    Attributes
    ----------
    data : numpy.ndarray
        The array of observations whose density function is to be estimated.
        
    mean : numpy.ndarry
        The mean vector of the multivariate normal base density function.
        
    base_sd : float
        The standard deviation of the multivariate normal base density function.
        
    covmat : numpy.ndarray
        The covariance matrix of the multivariate normal base density function;
        the covariance matrix is base_sd ** 2 times an identity matrix.
        
    domain : numpy.ndarray
        A d by 2 array to represent the domain of the normal base density function
        with the j-th element being [-np.inf, np.inf].
        
    dim : int
        The dimensionality of data and the underlying multivariate normal base density function, that is, d.

    Methods
    -------
    sample(n_samples)
        Draws samples from the multivariate normal base distribution of sample size n_samples.

    baseden_eval(new_data)
        Evaluates the multivariate normal base density function at new_data.

    logbaseden_deriv1(new_data, j)
        Evaluates the first partial derivative of the logarithm of the multivariate normal base density function
        along the j-th coordinate at new_data.

    logbaseden_deriv2(new_data, j)
        Evaluates the second partial derivative of the logarithm of the multivariate normal base density function
        along the j-th coordinate at new_data.

    baseden_eval_1d(x)
        Evaluates the normal base density function at a 1-dimensional data point x.

    baseden_eval_2d(x0, x1)
        Evaluates the multivariate normal base density function at a 2-dimensional data point (x0, x1),
        where x0 and x1 are the two coordinates, respectively.
        
    """

    def __init__(self, data, mean, base_sd):

        """
        Parameters
        ----------
        data : numpy.ndarray
            The array of observations whose density function is to be estimated.

        mean : numpy.ndarray
            The mean vector of length data.shape[1].

        base_sd : float
            The standard deviation of the multivariate normal distribution. Must be strictly positive.

        """

        super().__init__()

        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        self.data = data
        self.dim = self.data.shape[1]

        self.mean = np.array(mean).flatten()

        # check the compatibility of data and mean
        if data.shape[1] != len(mean):
            raise ValueError("The dimensionality of data does not match that of the mean vector.")

        # check the positivity of base_sd
        if base_sd <= 0.:
            raise ValueError("The base_sd must be strictly positive.")

        self.base_sd = base_sd
        self.covmat = base_sd ** 2 * np.eye(self.dim, dtype=np.float64)
        self.domain = np.array([[-np.inf, np.inf]] * self.dim)

    def sample(self, n_samples):

        """
        Draws samples from the multivariate normal distribution of sample size n_samples.

        Parameters
        ----------
        n_samples : int
            The number of new samples to be drawn.

        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, self.dim) of new samples drawn from the multivariate normal distribution.
            
        """

        output = np.random.multivariate_normal(mean=self.mean, cov=self.covmat, size=n_samples)

        return output

    def baseden_eval(self, new_data):

        """
        Evaluates the multivariate normal base density function at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            The new data at which the multivariate normal base density function is to be evaluated.

        Returns
        -------
        numpy.ndarray
            A (new_data.shape[0], 1) array of the multivariate normal base density values at new_data.
            
        """

        output = multivariate_normal.pdf(new_data, mean=self.mean, cov=self.covmat)

        return output.reshape(-1, 1)

    def logbaseden_deriv1(self, new_data, j):

        """
        Evaluates the first partial derivative of the logarithm of the multivariate normal base density function
        along the j-th coordinate at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            The new data at which the first partial derivative of the logarithm of the multivariate normal
            base density function is to be evaluated.
        j : int
            The coordinate with respect to which the first partial derivative is taken. It should be between 0
            and self.dim-1.

        Returns
        -------
        numpy.ndarray
            A (new_data.shape[0], 1) array of the first partial derivative values of the logarithm of
            the multivariate normal base density function along the j-th coordinate at new_data.
            
        """

        if len(new_data.shape) == 1:
            new_data = new_data.reshape(-1, 1)

        if new_data.shape[1] != self.dim:
            raise ValueError("The dimensionality of data and that of new_data do not match. ")

        if j not in range(self.dim):
            raise ValueError("The index j must be between 0 and {d}.".format(d=self.dim - 1))

        output = - (new_data[:, j] - self.mean[j]) / self.base_sd ** 2

        return output.reshape(-1, 1)

    def logbaseden_deriv2(self, new_data, j):

        """
        Evaluates the second partial derivative of the logarithm of the multivariate normal base density function
        along the j-th coordinate at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            The new data at which the second partial derivative of the logarithm of the multivariate normal
            base density function is to be evaluated.
        j : int
            The coordinate with respect to which the second partial derivative is taken. It should be between 0
            and self.dim-1.

        Returns
        -------
        numpy.ndarray
            A (new_data.shape[0], 1) array of the second partial derivative values of the logarithm of
            the multivariate normal base density function along the j-th coordinate at new_data.
            
        """

        if len(new_data.shape) == 1:
            new_data = new_data.reshape(-1, 1)

        if new_data.shape[1] != self.dim:
            raise ValueError("The dimensionality of data and that of new_data do not match. ")

        if j not in range(self.dim):
            raise ValueError("The index j must be between 0 and {d}.".format(d=self.dim - 1))

        output = - 1 / self.base_sd ** 2 * np.ones((new_data.shape[0], 1), dtype=np.float32)

        return output

    def baseden_eval_1d(self, x):

        """
        Evaluates the normal base density function at a 1-dimensional data point x.

        Parameters
        ----------
        x : float
            A floating point number at which the normal base density function is to be evaluated.

        Returns
        -------
        float
            A floating point number of the normal base density value at x.
            
        """

        if self.dim != 1:
            raise ValueError(("In order to use this baseden_eval_1d function, " 
                              "the underlying dimensionality of the normal distribution must be 1."))

        # exponent part
        power_part = - (1 / 2) * (x - self.mean) ** 2 / self.base_sd ** 2
        
        output = (2 * np.pi * self.base_sd ** 2) ** (- 1 / 2) * np.exp(power_part)

        return output

    def baseden_eval_2d(self, x0, x1):

        """
        Evaluates the 2-dimensional normal base density function at a 2-dimensional data point (x0, x1),
        where x0 and x1 are the two coordinates, respectively.

        Parameters
        ----------
        x0, x1 : float
            Two floating point numbers forming the coordinates of a 2-dimensional data point at which
            the 2-dimensional normal base density function is to be evaluated.
            
        Returns
        -------
        float
            A floating point number of the 2-dimensional normal base density value at (x0, x1).
            
        """

        if self.dim != 2:
            raise ValueError(("In order to use this baseden_eval_2d function, " 
                              "the underlying dimensionality of the multivariate normal distribution must be 2."))

        # exponent part
        power_part = - (1 / 2) * ((x0 - self.mean[0]) ** 2 + (x1 - self.mean[1]) ** 2) / self.base_sd ** 2

        # determinant of covariance matrix
        det_covmat = np.linalg.det(self.covmat)

        output = (2 * np.pi) ** (- self.dim / 2) * det_covmat ** (- 1 / 2) * np.exp(power_part)

        return output

    def baseden_eval_3d(self, x0, x1, x2):

        """
        Evaluates the 3-dimensional normal base density function at a 3-dimensional data point (x0, x1, x2),
        where x0, x1 and x2 are the three coordinates, respectively.

        Parameters
        ----------
        x0, x1, x2 : float
            Three floating point numbers forming the coordinates of a 3-dimensional data point at which
            the multivariate normal base density function is to be evaluated.

        Returns
        -------
        float
            A floating point number of the 3-dimensional normal base density value at (x0, x1, x2).
            
        """

        if self.dim != 3:
            raise ValueError(("In order to use this baseden_eval_3d function, " 
                              "the underlying dimensionality of the multivariate normal distribution must be 3."))

        # exponent part
        power_part = - (1 / 2) * ((x0 - self.mean[0]) ** 2 + (x1 - self.mean[1]) ** 2 + (
                    x2 - self.mean[2]) ** 2) / self.base_sd ** 2

        # determinant of covariance matrix
        det_covmat = np.linalg.det(self.covmat)

        output = (2 * np.pi) ** (- self.dim / 2) * det_covmat ** (- 1 / 2) * np.exp(power_part)

        return output


class BasedenLognormal(Baseden):

    """
    A class for the log-normal base density function.

    ...

    Attributes
    ----------
    data : numpy.ndarray
        The array of observations whose density function is to be estimated.
        
    s : float
        The shape parameter in the log-normal base density function.
        Estimated from data using the method of maximum likelihood.
        
    loc : float
        The location parameter in the log-normal base density function. Equal to 0.
        
    scale : float
        The scale parameter in the log-normal base density function.
        Estimated from data using the method of maximum likelihood.
        
    domain : numpy.ndarray
        The list [[0., np.inf]] to represent the domain of the log-normal base density function.
        
    dim : int
        The dimensionality of data and the underlying log-normal base density function, that is, 1.

    Methods
    -------
    sample(n_samples)
        Draws samples from the log-normal base distribution of sample size n_samples.

    baseden_eval(new_data)
        Evaluates the log-normal base density function at new_data.

    logbaseden_deriv1(new_data, j)
        Evaluates the first derivative of the logarithm of the log-normal base density function
        along j-th coordinate at new_data.

    logbaseden_deriv2(new_data, j)
        Evaluates the second derivative of the logarithm of the log-normal base density function
        along j-th coordinate at new_data.

    baseden_eval_1d(x)
        Evaluates the log-normal base density function at a 1-dimensional data point x.

    """

    def __init__(self, data):

        """
        Parameters
        ----------
        data : numpy.ndarray
            The array of observations whose density function is to be estimated.
            
        """
        
        super().__init__()
        
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        if np.any(data <= 0.):
            raise ValueError("Data contain negative value. Log-normal base density function cannot be used.")

        self.data = data
        s, _, scale = lognorm.fit(self.data, floc=0.)
        self.s = s
        self.loc = 0.
        self.scale = scale

        self.dim = 1
        self.domain = [[0., np.inf]]

    def sample(self, n_samples):

        """
        Draws samples from the log-normal distribution of sample size n_samples.

        Parameters
        ----------
        n_samples : int
            The number of new samples to be drawn.

        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, 1) of new samples drawn from the log-normal distribution.
            
        """

        output = lognorm.rvs(s=self.s, loc=self.loc, scale=self.scale, size=n_samples).reshape(-1, 1)

        return output

    def baseden_eval(self, new_data):

        """
        Evaluates the log-normal base density function at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            The new data at which the log-normal base density function is to be evaluated.

        Returns
        -------
        numpy.ndarray
            A (new_data.shape[0], 1) array of the log-normal base density values at new_data.
            
        """

        output = lognorm.pdf(x=new_data, s=self.s, loc=self.loc, scale=self.scale)

        return output.reshape(-1, 1)

    def logbaseden_deriv1(self, new_data, j=0):

        """
        Evaluates the first derivative of the logarithm of the log-normal base density function
        along j-th coordinate at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            The new data at which the first derivative of the logarithm of the log-normal base density function
            is to be evaluated.
        j : int
            The coordinate with respect to which the first partial derivative is taken. Default is 0 and
            it should always be 0.

        Returns
        -------
        numpy.ndarray
            A (new_data.shape[0], 1) array of the first derivative values of the logarithm of
            the log-normal base density function along the j-th coordinate at new_data.
            
        """

        if len(new_data.shape) == 1:
            new_data = new_data.reshape(-1, 1)

        if np.any(new_data <= 0.):
            raise ValueError("The new_data contain non-positive values.")

        if j != 0:
            raise ValueError(("Since the log-normal base density function is defined on positive real numbers,"
                              " the value of j should be 0."))

        output = (- 1. / new_data - (np.log(new_data) - np.log(self.scale)) / (new_data * self.s ** 2)) * (
                    new_data > 0.)

        return output.reshape(-1, 1)

    def logbaseden_deriv2(self, new_data, j=0):

        """
        Evaluates the second derivative of the logarithm of the log-normal base density function
        along j-th coordinate at new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            The new data at which the second derivative of the logarithm of the log-normal base density function
            is to be evaluated.
        j : int
            The coordinate with respect to which the second partial derivative is taken. Default is 0 and
            it should always be 0.

        Returns
        -------
        numpy.ndarray
            A (new_data.shape[0], 1) array of the second derivative values of the logarithm of
            the log-normal base density function along the j-th coordinate at new_data.
            
        """

        if len(new_data.shape) == 1:
            new_data = new_data.reshape(-1, 1)

        if np.any(new_data <= 0.):
            raise ValueError("The new_data contain non-positive values.")

        if j != 0:
            raise ValueError(("Since the log-normal base density function is defined on positive real numbers,"
                              " the value of j should be 0."))

        output = (1 / new_data ** 2 - (1 - np.log(new_data) + np.log(self.scale)) / (new_data ** 2 * self.s ** 2)) * (
                    new_data > 0)

        return output.reshape(-1, 1)

    def baseden_eval_1d(self, x):

        """
        Evaluates the log-normal base density function at a 1-dimensional data point x.

        Parameters
        ----------
        x : float
            A floating point number at which the log-normal base density function is to be evaluated.

        Returns
        -------
        float
            A floating point number of the log-normal base density value at x.
            
        """

        output = lognorm.pdf(x=x, s=self.s, loc=self.loc, scale=self.scale)

        return output
