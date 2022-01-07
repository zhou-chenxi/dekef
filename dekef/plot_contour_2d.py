import matplotlib.pyplot as plt
from scipy import integrate
from dekef.unnormalized_density import *
from dekef.check import *


def plot_contour_2d_params(plot_domain, plot_pts_cnt=500, filled_contour=False, figsize=(10, 10), font_size=20):
	
	"""
	Specifies and returns the plotting parameters used in the function plot_contour_2d.
	
	Parameters
	----------
	plot_domain : list or numpy.ndarray
		The list or numpy.ndarray to specify the domain of the contour plot of the density estimate.
		Must be a list of length 2 with each component of length 2, or a numpy.ndarray of shape (2, 2).
		All components must be finite numbers.
	
	plot_pts_cnt : int, optional
		The number of points to be evaluated along each coordinate of plot_domain to make a contour plot
		of the density estimate; default is 500.
	
	filled_contour : bool, optional
		Whether to fill the contour plot of the density estimate; default is False.
		
	figsize : typle, optional
		The size of the contour plot of the density estimate; default is (10, 10).
	
	font_size : int, optional
		The font size in the plot; default is 20.
	
	"""
	
	output = {'plot_domain': plot_domain,
              'plot_pts_cnt': plot_pts_cnt,
			  'filled_contour': filled_contour,
              'figsize': figsize,
              'font_size': font_size}
	
	return output


def plot_contour_2d(data, kernel_function, base_density, basis_type, coef, normalizing, method,
					x_label, y_label, plot_kwargs, grid_points, save_plot=False, save_dir=None):
	"""
	Makes the contour plot of the density estimate over a bounded two-dimensional region.
	
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
		The array of coefficients for basis functions in the natural parameter in the estimated density function.
	
	normalizing : bool
		Whether to plot the contours of the normalized density estimate.
	
	method : str
		The density estimation method.
	
	x_label : str
		The label of the horizontal axis.
	
	y_label : str
		The label of the vertical axis.
	
	grid_points : numpy.ndarray, optional
		The set of grid points at which the kernel functions are centered.
		Only need to supple when basis_type is 'grid_points'. Default is None.
		
	save_plot : bool, optional
		Whether to save the contour plot of the density estimate to a local file; default is False.
	
	save_dir : str, optional
		The directory path to which the contour plot of the density estimate is saved;
		only works when save_plot is set to be True; default is None.
	
	plot_kwargs : dict
		The dict containing plotting parameters returned from the function plot_contour_2d_params.
	
	"""
	
	# check the data is of 2 columns
	if len(data.shape) != 2:
		raise ValueError("The dimensionality of data should be 2.")
	
	if data.shape[1] != 2:
		raise ValueError("The number of columns of data should be 2, but got {ll}.".format(ll=data.shape[1]))
	
	n_obs = data.shape[0]
	
	# check the plot_domain
	if isinstance(plot_kwargs['plot_domain'], list):
		plot_domain1 = np.array(plot_kwargs['plot_domain'])
	elif isinstance(plot_kwargs['plot_domain'], np.ndarray):
		plot_domain1 = plot_kwargs['plot_domain']
	else:
		raise TypeError("The plot_domain should be either a list or a numpy.ndarray, but got {type_pd}.".format(
			type_pd=type(plot_kwargs['plot_domain'])))
	
	if plot_domain1.shape != (2, 2):
		raise ValueError("The plot_domain should be of length 2 and each component must be of length 2 as well!")
	
	if True in np.isinf(np.array(plot_domain1)).flatten():
		raise ValueError("The plot_domain contains np.inf values.")
	
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
		unnorm_fun = unnorm.density_eval_gubasis_2d
	
	elif basis_type == 'smbasis':
		
		# score matching basis
		unnorm_fun = unnorm.density_eval_smbasis_2d
	
	elif basis_type == 'grid_points':
		
		# score matching basis
		unnorm_fun = unnorm.density_eval_grid_points_2d
	
	else:
		
		raise ValueError(f"basis_type must be one of 'gubasis', 'smbasis', and 'grid_points', but got {basis_type}.")
	
	x0_cand = np.linspace(plot_domain1[0][0], plot_domain1[0][1], num=plot_kwargs['plot_pts_cnt'])
	x1_cand = np.linspace(plot_domain1[1][0], plot_domain1[1][1], num=plot_kwargs['plot_pts_cnt'])
	
	x0_mesh, x1_mesh = np.meshgrid(x0_cand, x1_cand)
	plot_val = np.zeros((plot_kwargs['plot_pts_cnt'], plot_kwargs['plot_pts_cnt']))
	
	for i in range(plot_kwargs['plot_pts_cnt']):
		for j in range(plot_kwargs['plot_pts_cnt']):
			plot_val[i][j] = unnorm_fun(x0_mesh[i][j], x1_mesh[i][j])
	
	if normalizing:
		norm_const, _ = integrate.nquad(unnorm_fun, base_density.domain)
		print(norm_const)
		plot_val /= norm_const
	
	fig = plt.figure(figsize=plot_kwargs['figsize'])
	left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
	ax = fig.add_axes([left, bottom, width, height])
	
	if plot_kwargs['filled_contour']:
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
