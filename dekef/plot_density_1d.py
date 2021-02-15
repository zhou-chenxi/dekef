import matplotlib.pyplot as plt
from scipy import integrate
from denest_kernelexpofam.unnormalized_density import *
import sys


def plot_density_1d(data, kernel_function, base_density, coef, normalizing, method, x_label,
                    save_plot, save_dir, save_filename, plot_kwargs):

    """
    Makes the plot of the density estimate with the histogram over a bounded one-dimensional interval.
    
    Parameters
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
        
    normalizing : bool
        Whether to plot the normalized density estimate.
        
    method : str
        The density estimation method.
    
    x_label : str
        The label of the horizontal axis.
    
    save_plot : bool
        Whether to save the plot of the density estimate as a local file.
    
    save_dir : str
        The directory path to which the plot of the density estimate is saved;
        only works when save_plot is set to be True.
    
    save_filename : str
        The file name for the plot of the density estimate saved as a local file;
        only works when save_plot is set to be True.
    
    plot_kwargs : dict
        The dict containing parameters for plotting the density estimate, including
        x_limit : tuple
            The tuple to specify the domain of the plot of density estimate in the horizontal axis.
            Must be of length 2. Both components must be finite numbers.
            
        y_limit : tuple
            The tuple to specify the domain of the plot of density estimate in the vertical axis.
            Must be of length 2. Both components must be finite numbers.
        
        plot_pts_cnt : int
            The number of points to be evaluated along the plot_domain to make a plot of density estimate.
        
        figsize : typle
            The size of the plot of density estimate.
            
        den_color : str or tuple
            The color for plotting the density estimate; see details at
            https://matplotlib.org/3.1.0/tutorials/colors/colors.html.
        
        hist_color : str or tuple
            The color for plotting the histogram; see details at
            https://matplotlib.org/3.1.0/tutorials/colors/colors.html.
        
        bins : int or sequence or str
            The bins used for plotting the histogram; see details at
            https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.hist.html.
        
        hist_alpha : float
            Set the alpha value used for blending in plotting the histogram.
        
        font_size : int
            The font size in the plot.
        
    Returns
    -------
    dict
        A dictionary of x_vals, the values of the horizontal axis for plotting, and
        den_vals, the values of the vertial axis for plotting.
    
    """
    
    if len(data.shape) != 1:
        data = data.reshape(-1, 1)
    
    if data.shape[1] != 1:
        raise ValueError("The data should be of 1 column.")
    
    if len(plot_kwargs['x_limit']) != 2:
        raise ValueError("The length of x_limit in plot_kwargs must be 2.")
    
    if len(plot_kwargs['y_limit']) != 2:
        raise ValueError("The length of y_limit in plot_kwargs must be 2.")
    
    if np.inf in plot_kwargs['x_limit'] or -np.inf in plot_kwargs['x_limit']:
        raise ValueError("The 'x_limit' in plot_kwargs contains non-finite values.")
    
    if np.inf in plot_kwargs['y_limit'] or -np.inf in plot_kwargs['y_limit']:
        raise ValueError("The 'y_limit' in plot_kwargs contains non-finite values.")
    
    coef = coef.reshape(-1, 1)
    
    n_obs = data.shape[0]
    plot_domain = [[plot_kwargs['x_limit'][0], plot_kwargs['x_limit'][1]]]
    plot_pts_cnt = plot_kwargs['plot_pts_cnt']
    
    unnorm = UnnormalizedDensity(
        data=data,
        kernel_function=kernel_function,
        base_density=base_density,
        coef=coef)
    
    if coef.shape[0] == n_obs:
        
        # Gu's basis
        unnorm_fun = unnorm.density_eval_gubasis
        unnorm_fun_int = unnorm.density_eval_gubasis_1d
        
    elif coef.shape[0] == n_obs + 1:
        
        # score matching basis
        unnorm_fun = unnorm.density_eval_smbasis
        unnorm_fun_int = unnorm.density_eval_smbasis_1d

    else:
    
        raise ValueError(("The length of coef is not correct and matches neither Gu's basis functions "
                          "nor score matching basis functions."))
    
    x0_cand = np.linspace(plot_domain[0][0], plot_domain[0][1], num=plot_pts_cnt).reshape(-1, 1)
    plot_val = unnorm_fun(x0_cand)

    if normalizing:
        norm_const, _ = integrate.nquad(unnorm_fun_int, base_density.domain, opts={'limit': 100})
        plot_val /= norm_const
        
    fig = plt.figure(figsize=plot_kwargs['figsize'])
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])

    plt.plot(x0_cand, plot_val, plot_kwargs['den_color'])
    plt.hist(data.flatten(),
             color=plot_kwargs['hist_color'],
             bins=plot_kwargs['bins'],
             range=plot_kwargs['x_limit'],
             density=True,
             alpha=plot_kwargs['hist_alpha'])
    # plt.plot(data, [0.01] * len(data), '|', color = 'k')
    
    ax.set_title('Density Plot (' + method + ')')
    ax.set_xlabel(x_label, fontsize=plot_kwargs['fontsize'])
    ax.set_ylabel('density', fontsize=plot_kwargs['fontsize'])
    ax.set_xlim(plot_kwargs['x_limit'])
    ax.set_ylim(plot_kwargs['y_limit'])
    ax.tick_params(axis='both', labelsize=plot_kwargs['fontsize'])
    if save_plot: 
        plt.savefig(save_dir + save_filename + '.pdf')
    plt.show()
    
    return {"x_vals": x0_cand, "den_vals": plot_val}


def plot_density_1d_scorematchingbasis_updated(data, kernel_function, base_density, coef,
                                               normalizing, method, x_label, save_plot, save_dir, save_filename,
                                               batch_size, plot_kwargs):
    
    """
    Makes the plot of the density estimate with the histogram over a bounded one-dimensional interval.
    
    Parameters
    ----------
    data : numpy.ndarray
        The array of observations with which the density function is estimated.
    
    kernel_function : kernel_function object
        The kernel function used to estimate the probability density function.
        Must be instantiated from the classes in kernel_function.
        
    base_density : base_density object
        The base density function used to estimate the probability density function.
        Must be instantiated from the classes in base_density.
    
    coef : numpy.ndarray
        The array of coefficients for basis functions in the natural parameter in the estimated density function.
        
    normalizing : bool
        Whether to plot the normalized density estimate.
        
    method : str
        The density estimation method.
    
    x_label : str
        The label of the horizontal axis.
    
    save_plot : bool
        Whether to save the plot of the density estimate as a local file.
    
    save_dir : str
        The directory path to which the plot of the density estimate is saved;
        only works when save_plot is set to be True.
    
    save_filename : str
        The file name for the plot of the density estimate saved as a local file;
        only works when save_plot is set to be True.
    
    batch_size : int
        The batch size used in batch Monte Carlo to approximate the normalizing constant.
    
    plot_kwargs : dict
        The dict containing parameters for plotting the density estimate, including
        x_limit : tuple
            The tuple to specify the domain of the plot of density estimate in the horizontal axis.
            Must be of length 2. Both components must be finite numbers.
            
        y_limit : tuple
            The tuple to specify the domain of the plot of density estimate in the vertical axis.
            Must be of length 2. Both components must be finite numbers.
        
        plot_pts_cnt : int
            The number of points to be evaluated along the plot_domain to make a plot of density estimate.
        
        figsize : typle
            The size of the plot of density estimate.
            
        den_color : str or tuple
            The color for plotting the density estimate; see details at
            https://matplotlib.org/3.1.0/tutorials/colors/colors.html.
        
        hist_color : str or tuple
            The color for plotting the histogram; see details at
            https://matplotlib.org/3.1.0/tutorials/colors/colors.html.
        
        bins : int or sequence or str
            The bins used for plotting the histogram; see details at
            https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.hist.html.
        
        hist_alpha : float
            Set the alpha value used for blending in plotting the histogram.
        
    Returns
    -------
    dict
        A dictionary of x_vals, the values of the horizontal axis for plotting, and
        den_vals, the values of the vertial axis for plotting.
    
    """

    if len(data.shape) != 1:
        data = data.reshape(-1, 1)

    if data.shape[1] != 1:
        raise ValueError("The data should be of 1 column.")

    if len(coef) != data.shape[0] + 1:
        raise ValueError("There should be {np1} coefficients, but only {c} coefficients were got.".format(
            np1=len(data) + 1, c=len(coef)))
    
    if len(plot_kwargs['x_limit']) != 2:
        raise ValueError("The length of x_limit in plot_kwargs must be 2.")

    if len(plot_kwargs['y_limit']) != 2:
        raise ValueError("The length of y_limit in plot_kwargs must be 2.")

    if np.inf in plot_kwargs['x_limit'] or -np.inf in plot_kwargs['x_limit']:
        raise ValueError("The 'x_limit' in plot_kwargs contains non-finite values.")

    if np.inf in plot_kwargs['y_limit'] or -np.inf in plot_kwargs['y_limit']:
        raise ValueError("The 'y_limit' in plot_kwargs contains non-finite values.")

    coef = coef.reshape(-1, 1)
    
    plot_domain = [[plot_kwargs['x_limit'][0], plot_kwargs['x_limit'][1]]]
    plot_pts_cnt = plot_kwargs['plot_pts_cnt']

    unnorm = UnnormalizedDensity(
        data=data,
        kernel_function=kernel_function,
        base_density=base_density,
        coef=coef)
    
    # score matching basis
    natparam = unnorm.natural_param_eval_smbasis
    # unnorm_fun = unnorm.density_eval_smbasis
    # unnorm_fun_int = unnorm.density_eval_smbasis_1d
    
    # compute density values at x0_cand
    x0_cand = np.linspace(plot_domain[0][0], plot_domain[0][1], num=plot_pts_cnt).reshape(-1, 1)
    fx_val = natparam(x0_cand)
    baseden_val = base_density.baseden_eval(x0_cand).flatten()
    den_val = baseden_val * np.exp(fx_val)
    
    # score matching density for 1D
    def natparam_eval_sm_1d(x):
    
        n_obs = data.shape[0]
        n_basis = n_obs + 1
    
        # linear combination of first derivatives
        fx1 = np.sum([coef[l1] * kernel_function.kernel_x_1d_deriv1(data[l1, ])(x)
                      for l1 in range(n_basis - 1)])
        # xi part
        xi1 = np.sum([base_density.logbaseden_deriv1(data[l1, ], j=0) *
                      kernel_function.kernel_x_1d_deriv1(data[l1, ])(x)
                      for l1 in range(n_obs)])
        xi2 = np.sum([kernel_function.kernel_x_1d_deriv2(data[l1, ])(x)
                      for l1 in range(n_obs)])
        xi = -(xi1 + xi2) / n_obs
    
        output1 = xi * coef[-1] + fx1
    
        # simply return f(x)
        return output1
    
    if normalizing:
        integrand = lambda x: base_density.baseden_eval_1d(x) * np.exp(natparam_eval_sm_1d(x))
        
        norm_const, _ = integrate.nquad(integrand,
                                        base_density.domain,
                                        opts={'limit': 100})
        print(norm_const)
        den_val /= norm_const
        norm_flag = True if np.isinf(norm_const) else False
        den_flag = True if np.any(np.isinf(den_val)) else False
        
        if norm_flag or den_flag:
            print('Density values or the normalizing constant detects infinity.')
            minus_c = np.max(fx_val) - np.log(sys.float_info.max) + 1. + np.log(np.max(data) - np.min(data))
            print("The number subtracted from the exponent is " + str(np.round(minus_c, 10)) + ".")
            # print(np.max(fx_val - minus_c))
            
            den_val = baseden_val * np.exp(fx_val - minus_c)
            
            # approximate the normalizing constant
            mc_samples = base_density.sample(batch_size).reshape(-1, 1)
            den_mc_samples = natparam(mc_samples)
            # print(den_mc_samples.shape)
            # print('f(x) of MC\n')
            # print(pd.Series(den_mc_samples).describe())
            # print('exp(f(x)) of MC\n')
            # print(pd.Series(np.exp(den_mc_samples - minus_c)).describe())
            
            norm_const = np.mean(np.exp(den_mc_samples - minus_c))
            print(norm_const)
            # print(norm_const)
            den_val /= norm_const
    
    fig = plt.figure(figsize=plot_kwargs['figsize'])
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])
    
    plt.plot(x0_cand, den_val)
    plt.hist(data.flatten(),
             color=plot_kwargs['hist_color'],
             bins=plot_kwargs['bins'],
             range=plot_kwargs['x_limit'],
             density=True,
             alpha=plot_kwargs['hist_alpha'])
    
    # plt.plot(data, [0.01] * len(data), '|', color = 'k')
    
    ax.set_title('Density Plot (' + method + ')')
    ax.set_xlabel(x_label)
    ax.set_ylabel('density')
    ax.set_xlim(plot_kwargs['x_limit'])
    ax.set_ylim(plot_kwargs['y_limit'])
    if save_plot:
        plt.savefig(save_dir + save_filename + '.pdf')
    plt.show()
    
    return {"x_vals": x0_cand, "den_vals": den_val}
