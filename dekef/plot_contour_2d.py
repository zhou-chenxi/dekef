import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from dekef.unnormalized_density import *


def plot_contour_2d(data, kernel_function, base_density, coef, normalizing, plot_domain, method,
					x_label, y_label, save_plot, save_dir, filled_contour=False, plot_pts_cnt=500, figsize=(10, 10)):
	
	"""
	Makes the contour plot of the density estimate over a bounded two-dimensional region.
	
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
        Whether to plot the contours of the normalized density estimate.
    
    plot_domain : list or numpy.ndarray
        The list or numpy.ndarray to specify the domain of the contour plot of the density estimate.
        Must be a list of length 2 with each component of length 2, or a numpy.ndarray of shape (2, 2).
        All components must be finite numbers.
    
    method : str
        The density estimation method.
    
    x_label : str
        The label of the horizontal axis.
    
    y_label : str
        The label of the vertical axis.
    
    save_plot : bool
        Whether to save the contour plot of the density estimate as a local file.
    
    save_dir : str
        The directory path to which the contour plot of the density estimate is saved;
        only works when save_plot is set to be True.
    
    filled_contour : bool, optional
        Whether to fill the contour plot of the density estimate; default is False.
    
    plot_pts_cnt : int, optional
        The number of points to be evaluated along each coordinate of plot_domain to make a contour plot
        of the density estimate; default is 500.
        
    figsize : typle, optional
        The size of the contour plot of the density estimate; default is (10, 10).
        
    """
	
	# check the data is of 2 columns
	if len(data.shape) != 2:
		raise ValueError("The dimensionality of data should be 2.")
	
	if data.shape[1] != 2:
		raise ValueError("The number of columns of data should be 2, but got {ll}.".format(ll=data.shape[1]))
	
	n_obs = data.shape[0]
	
	# check the plot_domain
	if isinstance(plot_domain, list):
		plot_domain1 = np.array(plot_domain)
	elif isinstance(plot_domain, np.ndarray):
		plot_domain1 = plot_domain
	else:
		raise TypeError("The plot_domain should be either a list or a numpy.ndarray, but got {type_pd}.".format(
			type_pd=type(plot_domain)))
	
	if plot_domain1.shape != (2, 2):
		raise ValueError("The plot_domain should be of length 2 and each component must be of length 2 as well!")
	
	if True in np.isinf(np.array(plot_domain1)).flatten():
		raise ValueError("The plot_domain contains np.inf values.")
	
	unnorm = UnnormalizedDensity(
		data=data,
		kernel_function=kernel_function,
		base_density=base_density,
		coef=coef)
	
	if coef.shape[0] == n_obs:
		
		# Gu's basis
		unnorm_fun = unnorm.density_eval_gubasis_2d
	
	elif coef.shape[0] == 2 * n_obs + 1:
		
		# score matching basis
		unnorm_fun = unnorm.density_eval_smbasis_2d
	
	else:
		
		raise ValueError(("The length of coef is not correct and matches neither Gu's basis functions "
						  "nor score matching basis functions."))
	
	x0_cand = np.linspace(plot_domain[0][0], plot_domain[0][1], num=plot_pts_cnt)
	x1_cand = np.linspace(plot_domain[1][0], plot_domain[1][1], num=plot_pts_cnt)
	
	x0_mesh, x1_mesh = np.meshgrid(x0_cand, x1_cand)
	plot_val = np.zeros((plot_pts_cnt, plot_pts_cnt))
	
	for i in range(plot_pts_cnt):
		for j in range(plot_pts_cnt):
			plot_val[i][j] = unnorm_fun(x0_mesh[i][j], x1_mesh[i][j])
	
	if normalizing:
		norm_const, _ = integrate.nquad(unnorm_fun, base_density.domain)
		print(norm_const)
		plot_val /= norm_const
	
	fig = plt.figure(figsize=figsize)
	left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
	ax = fig.add_axes([left, bottom, width, height])
	
	if filled_contour:
		cp = plt.contourf(x0_mesh, x1_mesh, plot_val)
		plt.colorbar(cp)
	else:
		cp = ax.contour(x0_mesh, x1_mesh, plot_val)
		ax.clabel(cp, inline=True, fontsize=10)
	
	ax.set_title('Contour Plot (' + method + ')', fontsize=plot_kwargs['fontsize'])
	ax.set_xlabel(x_label)
	ax.set_ylabel(y_label)
	if save_plot:
		plt.savefig(save_dir + '/contour_plot_' + method + '.pdf')
	plt.show()
