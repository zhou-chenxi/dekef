from dekef.base_density import *
from dekef.kernel_function import *


def check_basedensity(base_density):
	
	"""
	Check the type of base_density.
	
	base_density : base_density object
		The base density function used to estimate the probability density function.
		__type__ must be 'base_density'; otherwise, TypeError is raised.
		
	"""

	input_type = base_density.__type__()
	
	if input_type != 'base_density':
		
		raise TypeError('The __type__ of base_density should be base_density, but got {}'.format(input_type))
	
	else:
		
		pass


def check_kernelfunction(kernel_function):

	"""
	Check the type of kernel_function.

	kernel_function : kernel_function object
		The kernel function used to estimate the probability density function.
		__type__ must be 'kernel_function'; otherwise, TypeError is raised.
		
	"""
	
	input_type = kernel_function.__type__()
	
	if input_type != 'kernel_function':
		
		raise TypeError('The __type__ of kernel_function should be kernel_function, but got {}'.format(input_type))
	
	else:
		
		pass
	

# def check_samedata(kernel_function, base_density):
#
# 	"""
# 	Check whether the data in kernel_function and the data in base_density are identical.
#
# 	kernel_function : kernel_function object
# 		The kernel function used to estimate the probability density function.
# 		__type__ must be 'kernel_function'.
#
# 	base_density : base_density object
# 		The base density function used to estimate the probability density function.
# 		__type__ must be 'base_density'.
#
# 	"""
#
# 	kf_data = kernel_function.data
# 	bd_data = base_density.data
#
# 	if not np.allclose(kf_data, bd_data):
#
# 		raise ValueError(('The data from kernel_function and base_density are different. '
# 						  'Please double check the data input.'))
#
# 	else:
#
# 		pass