from scipy import integrate
from scipy import stats
from dekef.base_density import *


class NormalDensity:
    
    """
    A class of the multivariate normal distribution as the true distribution for the simulation purpose.
    
    ...

    Attributes
    ----------
    mean : numpy.ndarray
        The mean vector of the multivariate normal distribution.
    
    covmat : numpy.ndarray
        The covariance matrix of the multivariate normal distribution.
    
    type : str
        The name of the true distribution; i.e., "normal".

    Methods
    -------
    density_eval(x)
        Evaluates the density function of the multivariate normal distribution at x.
        
    sample(n_samples)
        Draws samples from the multivariate normal distribution of sample size n_samples.
    
    density_eval_1d(x)
        Evaluates the density function of the normal distribution at a 1-dimensional data point x.
    
    density_eval_2d(x0, x1)
        Evaluates the density function of the multivariate normal distribution at a 2-dimensional data point (x0, x1),
        where x0 and x1 are the two coordinates, respectively.
    
    density_eval_3d(x0, x1, x2)
        Evaluates the density function of the multivariate normal distribution at a 3-dimensional data point
        (x0, x1, x2), where x0, x1 and x2 are the three coordinates, respectively.
        
    """
    
    def __init__(self, mean, covmat):
        
        """
        Parameters
        ----------
        mean : numpy.ndarray
            The mean vector of the multivariate normal distribution.
    
        covmat : numpy.ndarray
            The covariance matrix of the multivariate normal distribution.
        
        """
        
        self.mean = mean
        self.covmat = covmat
        self.type = "normal"

    def density_eval(self, x):
        
        """
        Evaluates the density function of the multivariate normal distribution at x.
        
        Parameters
        ----------
        x : numpy.ndarray
            The array of data at which the density function of the multivariate normal distribution is to be evaluated.

        Returns
        -------
        numpy.ndarray
            The array of density values of the multivariate normal distribution at x.
        
        """

        den_vals = stats.multivariate_normal.pdf(x, mean=self.mean, cov=self.covmat)

        return den_vals

    def sample(self, n_samples):
        
        """
        Draws samples from the multivariate normal distribution of sample size n_samples.
        
        Parameters
        ----------
        n_samples : int
            The number of samples to be drawn.

        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, self.dim) of samples drawn from the multivariate normal distribution.
            
        """

        samples = np.random.multivariate_normal(mean=self.mean, cov=self.covmat, size=n_samples)

        return samples

    def density_eval_1d(self, x):
        
        """
        Evaluates the normal density function at a 1-dimensional data point x.

        Parameters
        ----------
        x : float
            A floating point number at which the normal density function is to be evaluated.

        Returns
        -------
        float
            The normal density value at x.
            
        """

        d = 1
        mean = self.mean[0]
        cov = self.covmat[0]

        const1 = (2 * np.pi) ** (-d/2) * cov ** (-1/2)
        exponent = -0.5 * (x - mean) ** 2 / cov

        output = const1 * np.exp(exponent)

        return output

    def density_eval_2d(self, x0, x1):
        
        """
        Evaluates the multivariate normal density function at a 2-dimensional data point (x0, x1),
        where x0 and x1 are the two coordinates, respectively.

        Parameters
        ----------
        x0, x1 : float
            Two floating point numbers forming the coordinates of a 2-dimensional data point at which
            the multivariate normal density function is to be evaluated.

        Returns
        -------
        float
            The multivariate normal density value at (x0, x1).
            
        """

        d = 2
        mean = self.mean.reshape(d, 1)
        cov = self.covmat
        invcov = np.linalg.inv(cov)

        if mean.shape[0] != d or cov.shape != (d, d):
            raise ValueError("The dimensionality should be 2.")

        const1 = (2 * np.pi) ** (-d/2) * np.linalg.det(cov) ** (-1/2)
        x_sub_mean = np.array([x0, x1]).reshape(d, 1) - mean

        exponent = -0.5 * np.matmul(np.matmul(x_sub_mean.T, invcov), x_sub_mean)

        output = (const1 * np.exp(exponent))[0][0]

        return output

    def density_eval_3d(self, x0, x1, x2):
        
        """
        Evaluates the multivariate normal density function at a 3-dimensional data point (x0, x1, x2),
        where x0, x1 and x2 are the three coordinates, respectively.

        Parameters
        ----------
        x0, x1, x2 : float
            Three floating point numbers forming the coordinates of a 3-dimensional data point at which
            the multivariate normal density function is to be evaluated.

        Returns
        -------
        float
            The multivariate normal density value at (x0, x1, x2).

        """

        d = 3
        mean = self.mean.reshape(d, 1)
        cov = self.covmat
        invcov = np.linalg.inv(cov)

        if mean.shape[0] != d or cov.shape != (d, d):
            raise ValueError("The dimensionality should be 3.")

        const1 = (2 * np.pi) ** (-d/2) * np.linalg.det(cov) ** (-1/2)
        x_sub_mean = np.array([x0, x1, x2]).reshape(d, 1) - mean

        exponent = -0.5 * np.matmul(np.matmul(x_sub_mean.T, invcov), x_sub_mean)

        output = (const1 * np.exp(exponent))[0][0]

        return output


class MixNormalDensity:
    
    """
    A class of the mixture of multivariate normal distributions as the true distribution for the simulation purpose.

    ...

    Attributes
    ----------
    mix_weights : list or numpy.ndarray
        The list or array of mixture weights. Each component must be strictly positive and
        all components must sum to 1.
    
    mean : numpy.ndarray
        The mean vectors of each multivariate normal distribution component.
    
    covmat : numpy.ndarray
        The covariance matrices of each multivariate normal distribution component.
    
    n_comps : int
        The number of mixture components in the mixture of multivariate normal distributions.
    
    type : str
        The name of the true distribution; i.e., "mix_normal".

    Methods
    -------
    density_eval(x)
        Evaluates the density function of the mixture of multivariate normal distributions at x.
    
    sample(n_samples)
        Draws samples from the mixture of multivariate normal distributions of sample size n_samples.
        
    density_eval_1d(x)
        Evaluates the density function of the mixture of normal distributions at a 1-dimensional data point x.
    
    density_eval_2d(x0, x1)
        Evaluates the density function of the mixture of multivariate normal distributions
        at a 2-dimensional data point (x0, x1), where x0 and x1 are the two coordinates, respectively.
    
    density_eval_3d(x0, x1, x2)
        Evaluates the density function of the mixture of multivariate normal distributions
        at a 3-dimensional data point (x0, x1, x2), where x0, x1 and x2 are the three coordinates, respectively.
        
    """
    
    def __init__(self, mix_weights, mean, covmat):

        """
        Parameters
        ----------
        mix_weights : list or 1-dimensional numpy.ndarray
            The mixture weights of each multivariate normal distribution component.
            Each component must be strictly positive, and all components must sum to 1.
            
        mean : numpy.ndarray
            The mean vectors of each multivariate normal distribution component.
    
        covmat : numpy.ndarray
            The covariance matrices of each multivariate normal distribution component.
        
        """

        ncomp_mixw = len(mix_weights)
        ncomp_mean = len(mean)
        ncomp_covm = len(covmat)

        # check mix_weights:
        check_pos = [mix_weights[i] > 0. for i in range(ncomp_mixw)]
        if False in check_pos:
            raise ValueError("mix_weights cannot contain non-positive components.")

        if np.sum(mix_weights) != 1.:
            raise ValueError("mix_weights does not sum to 1.")

        # check the equal components
        if not ncomp_mixw == ncomp_mean == ncomp_covm:
            raise ValueError("mix_weights, mean and covmat do not have the same length.")

        # check the equal dimensionality
        len_mean = [len(mean[i]) for i in range(ncomp_mean)]
        if len(set(len_mean)) != 1:
            raise ValueError("mean vectors do not have the same dimensionality.")

        len_covm = [covmat[i].shape for i in range(ncomp_covm)]
        if len(set(len_covm)) != 1:
            raise ValueError("covmat do not have the same dimensionality.")

        self.mix_weights = mix_weights
        self.mean = mean
        self.covmat = covmat
        self.n_comps = ncomp_mixw
        self.type = "mix_normal"

    def density_eval(self, x):
        
        """
        Evaluates the density function of the mixture of multivariate normal distributions at x.
        
        Parameters
        ----------
        x : numpy.ndarray
            The array of data at which the density function of the mixture of multivariate normal distributions
            is to be evaluated.

        Returns
        -------
        numpy.ndarray
            The density values of the mixture of multivariate normal distributions at x.
            
        """

        den_list = []

        for j in range(self.n_comps):
            den_list.append(stats.multivariate_normal.pdf(x, mean=self.mean[j], cov=self.covmat[j]))

        den_vals = np.array([self.mix_weights[j] * den_list[j] for j in range(self.n_comps)])

        return np.sum(den_vals, axis=0)
    
    def sample(self, n_samples):
        
        """
        Draws samples from the mixture of multivariate normal distributions of sample size n_samples.
        
        Parameters
        ----------
        n_samples : int
            The number of samples to be drawn.

        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, self.dim) of samples drawn
            from the mixture of multivariate normal distributions.
        
        """

        # which components?
        compt = np.random.choice(np.arange(self.n_comps), p=self.mix_weights, size=n_samples)

        samples = []
        for i in range(n_samples):
            samples.append(np.random.multivariate_normal(mean=self.mean[compt[i]],
                                                         cov=self.covmat[compt[i]],
                                                         size=1)[0])

        return np.array(samples)

    def density_eval_1d(self, x):
        
        """
        Evaluates the density function of the mixture of normal distributions
        at a 1-dimensional data point x.
        
        Parameters
        ----------
        x : float
            A floating point number at which the density function of the mixture of normal distributions
            is to be evaluated.

        Returns
        -------
        float
            The density value of the mixture of normal distribution at x.
            
        """

        d = 1

        output = 0.
        
        mean_dim_check = [len(mean_i) == 1 for mean_i in self.mean]
        cov_dim_check = [len(cov_i) == 1 for cov_i in self.covmat]
        if False in mean_dim_check.extend(cov_dim_check):
            raise ValueError(("The dimensionality is not correct. You can use density_eval_1d only "
                              "when each component of the mixed normal distribution is 1-dimensional."))
        
        for i in range(self.n_comps):

            mean = self.mean[i]
            cov = self.covmat[i]

            const1 = (2 * np.pi) ** (-d/2) * cov ** (-1/2)
            exponent = -0.5 * (x - mean) ** 2 / cov

            den_val_i = (const1 * np.exp(exponent))
            output += self.mix_weights[i] * den_val_i

        return output

    def density_eval_2d(self, x0, x1):

        """
        Evaluates the density function of the mixture of multivariate normal distributions
        at a 2-dimensional data point (x0, x1), where x0 and x1 are the two coordinates, respectively.

        Parameters
        ----------
        x0, x1 : float
            Two floating point numbers forming the coordinates of a 2-dimensional data point at which
            the density function of the mixture of multivariate normal distributions is to be evaluated.

        Returns
        -------
        float
            The density value of the mixture of multivariate normal distributions at (x0, x1).

        """

        mean_dim_check = [len(mean_i) == 2 for mean_i in self.mean]
        cov_dim_check = [cov_i.shape == (2, 2) for cov_i in self.covmat]
        
        if False in (mean_dim_check + cov_dim_check):
            raise ValueError(("The dimensionality is not correct. You can use density_eval_2d only "
                              "when each component of the mixed normal distribution is 2-dimensional."))
        
        d = 2
        output = 0.

        for i in range(self.n_comps):

            mean = self.mean[i].reshape(d, 1)
            cov = self.covmat[i]
            invcov = np.linalg.inv(cov)

            const1 = (2 * np.pi) ** (-d/2) * np.linalg.det(cov) ** (-1/2)
            x_sub_mean = np.array([x0, x1]).reshape(d, 1) - mean

            exponent = -0.5 * np.matmul(np.matmul(x_sub_mean.T, invcov), x_sub_mean)

            den_val_i = (const1 * np.exp(exponent))[0][0]

            output += self.mix_weights[i] * den_val_i

        return output

    def density_eval_3d(self, x0, x1, x2):
        
        """
        Evaluates the density function of the mixture of multivariate normal distributions
        at a 3-dimensional data point (x0, x1, x2), where x0, x1 and x2 are the three coordinates, respectively.

        Parameters
        ----------
        x0, x1, x2 : float
            Three floating point numbers forming the coordinates of a 3-dimensional data point at which
            the density function of the mixture of multivariate normal distributions is to be evaluated.

        Returns
        -------
        float
            The density value of the mixture of multivariate normal distributions at (x0, x1, x2).
            
        """

        mean_dim_check = [len(mean_i) == 3 for mean_i in self.mean]
        cov_dim_check = [cov_i.shape == (3, 3) for cov_i in self.covmat]

        if False in (mean_dim_check + cov_dim_check):
            raise ValueError(("The dimensionality is not correct. You can use density_eval_3d only "
                              "when each component of the mixed normal distribution is 3-dimensional."))

        d = 3
        output = 0.

        for i in range(self.n_comps):

            mean = self.mean[i].reshape(d, 1)
            cov = self.covmat[i]
            invcov = np.linalg.inv(cov)

            const1 = (2 * np.pi) ** (-d/2) * np.linalg.det(cov) ** (-1/2)
            x_sub_mean = np.array([x0, x1, x2]).reshape(d, 1) - mean
            print(x_sub_mean)

            exponent = -0.5 * np.matmul(np.matmul(x_sub_mean.T, invcov), x_sub_mean)
            print(exponent)

            den_val_i = (const1 * np.exp(exponent))[0][0]

            output += self.mix_weights[i] * den_val_i

        return output


class TwoMoonsDensity:
    
    """
    A class of the two moon distribution as the true distribution for the simulation purpose.
    Its density function is proportional to exp (- (1 / 2) ((sqrt{x0 ^ 2 + x1 ^ 2} - 2) / 0.4) ^ 2 +
    log (exp (- ((x1 - 2) / 0.6) ^ 2 / 2) + exp (- ((x1 + 2) / 0.6) ^ 2 / 2)), up to the normalizing constant.
    
    ...

    Attributes
    ----------
    domain : numpy.ndarray, optional
        The domain of the two moon distribution; default is [[-3., 3.], [-3., 3.]].
    
    type : str
        The name of the true distribution; i.e., "two_moons".

    Methods
    -------
    unnormalized_den_eval(x0, x1)
        Evaluates the un-normalized density function of the two moon distribution
        at a 2-dimensional data point (x0, x1), where x0 and x1 are the two coordinates, respectively.
        
    normalizing_const()
        Computes the normalizing constant of the density function of the two moon distribution over domain.
        
    density_eval(x)
        Evaluates the density function of the two moon distribution at x.
        
    sample(n_samples)
        Draws samples from the two moon distribution of sample size n_samples
        using the acceptance-rejection algorithm.
    
    """
    
    def __init__(self, domain=None):
        
        """
        Parameters
        ----------
        domain : numpy.ndarray, optional
            The domain of the two moon distribution; default is [[-3., 3.], [-3., 3.]].
            
        """

        if domain is None:
            domain = [[-3., 3.], [-3., 3.]]
        self.domain = domain
        self.type = "two_moons"
    
    def unnormalized_den_eval(self, x0, x1):

        """
        Evaluates the un-normalized density function of the two moon distribution
        at a 2-dimensional data point (x0, x1), where x0 and x1 are the two coordinates, respectively.
        
        Parameters
        ----------
        x0, x1 : float
            Two floating point numbers forming the coordinates of a 2-dimensional data point at which
            the density function of the two moon distribution is to be evaluated.

        Returns
        -------
        float
           The density value of the two moon distribution at (x0, x1).
        
        """

        # from Table 1 in Variational inference with normalizing flows by Danilo Jimenez Rezende and Shakir Mohamed
        # (https://arxiv.org/pdf/1505.05770.pdf)

        term1 = (np.sqrt(x0 ** 2 + x1 ** 2) - 2) / 0.4
        term2_expo1 = - 0.5 * ((x0 - 2) / 0.6) ** 2
        term2_expo2 = - 0.5 * ((x0 + 2) / 0.6) ** 2

        potential_part = 0.5 * term1 ** 2 - np.log(np.exp(term2_expo1) + np.exp(term2_expo2))

        output = np.exp(- potential_part)

        return output

    def normalizing_const(self):
        
        """
        Computes the normalizing constant of the density function of the two moon distribution over domain.
        
        Returns
        -------
        The normalizing constant of the density function of the two moon distribution over domain.
        
        """

        val, _ = integrate.nquad(self.unnormalized_den_eval, self.domain)

        return val

    def density_eval(self, x):
        
        """
        Evaluates the density function of the two moon distribution at x.
        
        Parameters
        ----------
        x : numpy.ndarray
            The array of data at which the density value of the two moon distribution is to be evaluated.
            Must be a 2-dimensional array with the number of columns being 2.

        Returns
        -------
        numpy.ndarray
            The array of density values of the two moon distribution at x.
        
        """

        assert x.shape[1] == 2, "The input x must have 2 columns. "

        val, _ = integrate.nquad(self.unnormalized_den_eval, self.domain)

        # U(x) part
        norm_x = np.apply_along_axis(np.linalg.norm, axis=1, arr=x)
        
        # first term
        fir_term = 0.5 * ((norm_x - 2) / 0.4) ** 2

        # second term
        exp1 = - 0.5 * ((x[:, 0] - 2) / 0.6) ** 2
        exp2 = - 0.5 * ((x[:, 0] + 2) / 0.6) ** 2
        sec_term = np.log(np.exp(exp1) + np.exp(exp2))
        u_term = fir_term - sec_term

        output = np.exp(- u_term) / val

        return output

    def sample(self, n_samples):
        
        """
        Draws samples from the two moon distribution of sample size n_samples
        using the acceptance-rejection algorithm.
        
        Parameters
        ----------
        n_samples : int
            The number of samples to be drawn.

        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, 2) of samples drawn from the two moon distribution.

        """

        # use acceptance-rejection sampling methods
        # http://www.columbia.edu/~ks20/4703-Sigman/4703-07-Notes-ARM.pdf
        # the reference density is uniform over domain
        cnt = 0
        trial_cnt = 0
        output = []
        # c = 36.  # c = sup (f(x) / g(x))

        while cnt < n_samples:
            ref_sample1 = np.random.uniform(low=self.domain[0][0],
                                            high=self.domain[0][1],
                                            size=1)[0]
            ref_sample2 = np.random.uniform(low=self.domain[1][0],
                                            high=self.domain[1][1],
                                            size=1)[0]
            u = np.random.rand(1)
            
            if u <= self.unnormalized_den_eval(ref_sample1, ref_sample2):

                output.append([ref_sample1, ref_sample2])
                cnt += 1
                
            trial_cnt += 1

        return np.array(output)
