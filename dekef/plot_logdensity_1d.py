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
        The tuple to specify the plotting domain of the log-density estimate.
        Must be of length 2. Both components must be finite numbers.

    y_limit : tuple
        The tuple to specify the domain of the plot of the log-density estimate in the vertical axis.
        Must be of length 2. Both components must be finite numbers.

    plot_pts_cnt : int, optional
        The number of points to be evaluated along the plot_domain to make a plot of the log-density estimate;
        default is 2000.

    figsize : typle, optional
        The size of the plot of the log-density estimate; default is (10, 10).

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

    if len(x_limit) != 2:
        raise ValueError("The length of x_limit must be 2.")

    if len('y_limit') != 2:
        raise ValueError("The length of y_limit must be 2.")

    if np.inf in x_limit or -np.inf in x_limit:
        raise ValueError("x_limit contains non-finite values.")

    if np.inf in y_limit or -np.inf in y_limit:
        raise ValueError("y_limit contains non-finite values.")
    
    output = {'x_limit': x_limit,
              'y_limit': y_limit,
              'plot_pts_cnt': plot_pts_cnt,
              'figsize': figsize,
              'logden_color': logden_color,
              'fontsize': fontsize}
    
    return output


def plot_logdensity_1d(data, kernel_function, base_density, basis_type, coef, method, x_label,
                       plot_kwargs, grid_points=None, save_plot=False, save_dir=None, save_filename=None):

    """
    Plots the log-density function (up to an additive constant) over a bounded interval.
    
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
    
    basis_type : str
        The type of the basis functions in the natural parameter. Must be one of
            - 'gubasis', the basis functions being the kernel functions centered at data,
            - 'smbasis', the basis functions being the same as those in the score matching density estimator, i.e.,
                         a linear combination of the first two partial derivatives of the kernel functions centered
                         at data,
            - 'grid_points', the basis functions being the kernel functions centered at a set of
                             pre-specified grid points.
    
    coef : numpy.ndarray
        The array of coefficients of basis functions in the estimated log-density function.

    method : str
        The density estimation method.
    
    x_label : str
        The label of the horizontal axis.
    
    plot_kwargs : dict
        The dict containing plotting parameters returned from the function plot_logdensity_1d_params.
    
    grid_points : numpy.ndarray, optional
        The set of grid points at which the kernel functions are centered.
        Only need to supple when basis_type is 'grid_points'. Default is None.
        
    save_plot : bool, optional
        Whether to save the plot of the log-density estimate as a local file; default is False.
    
    save_dir : str, optional
        The directory path to which the plot of the log-density estimate is saved;
        only works when save_plot is set to be True. Default is None.
    
    save_filename : str, optional
        The file name for the plot of the log-density estimate saved as a local file;
        only works when save_plot is set to be True. Default is None.
        
    Returns
    -------
    dict
        A dictionary of 'x_vals', the values of the horizontal axis for plotting, and
        'logden_vals', the values of the vertical axis for plotting.
    
    """

    check_kernelfunction(kernel_function)
    check_basedensity(base_density)
    
    if len(data.shape) != 1:
        data = data.reshape(-1, 1)
    
    if data.shape[1] != 1:
        raise ValueError("The data should be of 1 column.")
    
    if len(grid_points.shape) == 1:
        grid_points = grid_points.reshape(-1, 1)
        
    coef = coef.reshape(-1, 1)
    
    plot_domain = [[plot_kwargs['x_limit'][0], plot_kwargs['x_limit'][1]]]
    plot_pts_cnt = plot_kwargs['plot_pts_cnt']

    # prepare the newx for plotting
    x0_cand = np.linspace(plot_domain[0][0], plot_domain[0][1], num=plot_pts_cnt).reshape(-1, 1)

    unnorm = UnnormalizedDensity(
        data=data,
        kernel_function=kernel_function,
        base_density=base_density,
        coef=coef,
        basis_type=basis_type,
        grid_points=grid_points
    )
    
    if basis_type == 'gubasis':
        
        # Gu's basis
        unnorm_fun = unnorm.density_eval_gubasis
        
    elif basis_type == 'smbasis':
        
        # score matching basis
        unnorm_fun = unnorm.density_eval_smbasis

    elif basis_type == 'grid_points':
    
        unnorm_fun = unnorm.density_eval_grid_points

    else:
    
        raise ValueError(f"basis_type must be one of 'gubasis', 'smbasis', and 'grid_points', but got {basis_type}.")
    
    plot_val = np.log(unnorm_fun(x0_cand))
        
    fig = plt.figure(figsize=plot_kwargs['figsize'])
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])

    plt.plot(x0_cand, plot_val, plot_kwargs['logden_color'])
    
    ax.set_title('Log-density Plot (' + method + ')', fontsize=plot_kwargs['fontsize'])
    ax.set_xlabel(x_label, fontsize=plot_kwargs['fontsize'])
    ax.set_ylabel('log-density', fontsize=plot_kwargs['fontsize'])
    ax.set_xlim(plot_kwargs['x_limit'])
    ax.set_ylim(plot_kwargs['y_limit'])
    ax.tick_params(axis='both', labelsize=plot_kwargs['fontsize'])
    if save_plot: 
        plt.savefig(save_dir + save_filename + '.pdf')
    plt.show()
    
    return {"x_vals": x0_cand, "logden_vals": plot_val}
