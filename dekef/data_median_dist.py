import numpy as np
from scipy.spatial.distance import pdist


def data_median_dist(data):

    """
    Compute the median of the pairwise distances of data. 
    
    Parameters
    ----------
    data: numpy.ndarray
        The array of observations whose density function is to be estimated.
    
    Returns
    -------
    float
        The median of the pairwise distances of data.
        
    """
    
    dist_vec = pdist(data)
    output = np.median(dist_vec)

    return output
