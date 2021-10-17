import matplotlib.pyplot as plt
from dekef.unnormalized_density import *
from dekef.check import *


def plot_logdensity_1d_params(x_limit, y_limit, plot_pts_cnt=2000, figsize=(10, 10),
                              logden_color='tab:blue', fontsize=20):
    """
    Specifies and returns the plotting parameters used in the function plot_logdensity_1d.

    Parameters
    ----------
    x_limit : tuple
        The tuple to specify the plotting domain of the density estimate.
        Must be of length 2. Both components must be finite numbers.

    y_limit : tuple
        The tuple to specify the domain of the plot of density estimate in the vertical axis.
        Must be of length 2. Both components must be finite numbers.

    plot_pts_cnt : int, optional
        The number of points to be evaluated along the plot_domain to make a plot of density estimate;
        default is 2000.

    figsize : typle, optional
        The size of the plot of density estimate; default is (10, 10).

    logden_color : str or tuple, optional
        The color for plotting the log-density estimate; default is 'tab:blue'.
        See details at https://matplotlib.org/3.1.0/tutorials/colors/colors.html.

    fontsize : int, optional
        The font size in the plot; default is 20.

    Returns
    -------
    dict
        A dict containing all the plotting parameter inputs.

    """
    
    output = {'x_limit': x_limit,
              'y_limit': y_limit,
              'plot_pts_cnt': plot_pts_cnt,
              'figsize': figsize,
              'logden_color': logden_color,
              'fontsize': fontsize}
    
    return output


def plot_logdensity_1d(data, kernel_function, base_density, coef, method, x_label,
                       plot_kwargs, save_plot=False, save_dir=None, save_filename=None):

    """
    Plots the un-normalized log-density function over a bounded interval.
    
    Parameters
    ----------
    data : numpy.ndarray
        The array of observations with which the density function is estimated.

    kernel_function : kernel_function object
        The kernel function used to estimate the probability density function.
        __type__ must be 'kernel_function'.
        
    base_density : base_density object
        The base density function used to estimate the probability density function.
        __type__ must be 'base_density'.
    
    coef : numpy.ndarray
        The array of coefficients of basis functions in the estimated density function.

    method : str
        The density estimation method.
    
    x_label : str
        The label of the horizontal axis.
    
    plot_kwargs : dict
        The dict containing plotting parameters returned from the function plot_logdensity_1d_params.
    
    save_plot : bool, optional
        Whether to save the plot of the density estimate as a local file; default is False.
    
    save_dir : str, optional
        The directory path to which the plot of the density estimate is saved;
        only works when save_plot is set to be True. Default is None.
    
    save_filename : str, optional
        The file name for the plot of the density estimate saved as a local file;
        only works when save_plot is set to be True. Default is None.
        
    Returns
    -------
    dict
        A dictionary of x_vals, the values of the horizontal axis for plotting, and
        logden_vals, the values of the vertical axis for plotting.
    
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

    # prepare the newx for plotting
    x0_cand = np.linspace(plot_domain[0][0], plot_domain[0][1], num=plot_pts_cnt).reshape(-1, 1)

    unnorm = UnnormalizedDensity(
        data=data,
        kernel_function=kernel_function,
        base_density=base_density,
        coef=coef)
    
    if coef.shape[0] == n_obs:
        
        # Gu's basis
        unnorm_fun = unnorm.density_eval_gubasis
        
    elif coef.shape[0] == n_obs + 1:
        
        # score matching basis
        unnorm_fun = unnorm.density_eval_smbasis

    else:
    
        raise ValueError(("The length of coef is not correct and matches neither Gu's basis functions "
                          "nor score matching basis functions."))
    
    plot_val = np.log(unnorm_fun(x0_cand))
        
    fig = plt.figure(figsize=plot_kwargs['figsize'])
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])

    plt.plot(x0_cand, plot_val, plot_kwargs['logden_color'])
    
    ax.set_title('Unnormalized Log-density Plot (' + method + ')', fontsize=plot_kwargs['fontsize'])
    ax.set_xlabel(x_label, fontsize=plot_kwargs['fontsize'])
    ax.set_ylabel('unnormalized log-density', fontsize=plot_kwargs['fontsize'])
    ax.set_xlim(plot_kwargs['x_limit'])
    ax.set_ylim(plot_kwargs['y_limit'])
    ax.tick_params(axis='both', labelsize=plot_kwargs['fontsize'])
    if save_plot: 
        plt.savefig(save_dir + save_filename + '.pdf')
    plt.show()
    
    return {"x_vals": x0_cand, "logden_vals": plot_val}
