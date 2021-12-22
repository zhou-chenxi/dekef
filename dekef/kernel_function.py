import numpy as np


class KernelFunction:
    
    """
    A parent class for the kernel function.

    __type__
        Set __type__ to be 'kernel_function'.

    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def __type__():
        
        return 'kernel_function'
        

class GaussianPoly2(KernelFunction):

    """
    A class to compute the Gaussian kernel function plus a polynomial kernel function of degree 2,
    k (x, y) = r1 * exp (- || x - y || ^ 2 / (2 * bw ^ 2)) + r2 * (x^\top y + c) ^ 2,
    and its derivatives.

    ...

    Attributes
    ----------
    data : numpy.ndarray
        The array of observations whose density function is to be estimated.
        
    N : int
        The number of rows of data.
        
    d : int
        The number of columns in of data.
        
    r1 : float
        The multiplicative coefficient associated with the Gaussian kernel function.
        
    r2 : float
        The multiplicative coefficient associated with the polynomial kernel function of degree 2.
        
    c : float
        The non-homogenous additive constant in the polynomial kernel function of degree 2.
        
    bw : float
        The bandwidth parameter in the Gaussian kernel function.
        
    kernel_type : str
        The type of the kernel function, 'gaussian_poly2'.

    Methods
    -------
    gaussian_kernel_gram_matrix(new_data)
        Computes the Gram matrix of shape (data.shape[0], new_data.shape[0]) using the Gaussian kernel function,
        with the (i, j)-th entry being exp(- ||X_i - Y_j||^2 / (2 * bw ^ 2)), where X_i is the i-th row in data,
        and Y_j is the j-th row in new_data.

    poly_kernel_gram_matrix(new_data)
        Computes the Gram matrix of shape (data.shape[0], new_data.shape[0]) using the polynomial kernel function
        of degree 2, with the (i, j)-th entry being (X_i^\top Y_j + c) ^ 2, where X_i is the i-th row in data,
        and Y_j is the j-th row in new_data.

    kernel_gram_matrix(new_data)
        Computes the Gram matrix of shape (data.shape[0], new_data.shape[0]) using the Gaussian kernel function plus
        the polynomial kernel function of degree 2, with the (i, j)-th entry being
        k (X_i, Y_j) = r1 * exp(- ||X_i - Y_j||^2 / (2 * bw ^ 2)) + r2 * (X_i^\top Y_j + c) ^ 2,
        where X_i is the i-th row in data, and Y_j is the j-th row in new_data.

    partial_kernel_matrix_10(new_data)
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0]) using
        the Gaussian kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, j)-th entry being partial_u k (X_i, Y_j),
        where partial_u k denotes the first partial derivative of k with respect to the u-th coordinate
        of its first argument, X_i is the i-th row in data, and Y_j is the j-th row in new_data.
        
    partial_kernel_matrix_01(new_data)
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0]) using
        the Gaussian kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, j)-th entry being partial_{d+u} k (X_i, Y_j),
        where partial_{d+u} k denotes the first partial derivative of k with respect to the u-th coordinate
        of its second argument, X_i is the i-th row in data, and Y_j is the j-th row in new_data.

    partial_kernel_matrix_20(new_data)
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0]) using
        the Gaussian kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, j)-th entry being partial_u^2 k (X_i, Y_j),
        where partial_u^2 k denotes the second partial derivative of k with respect to the u-th coordinate
        of its first argument, X_i is the i-th row in data, and Y_j is the j-th row in new_data.
    
    partial_kernel_matrix_02(new_data)
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0]) using
        the Gaussian kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, j)-th entry being partial_{d+u}^2 k (X_i, Y_j),
        where partial_{d+u}^2 k denotes the second partial derivative of k with respect to the u-th coordinate
        of its second argument, X_i is the i-th row in data, and Y_j is the j-th row in new_data.

    partial_kernel_matrix_11(new_data)
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) using
        the Gaussian kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u partial_{v+d} k (X_i, Y_j),
        where partial_u partial_{v+d} k denotes the second mixed partial derivative of k,
        one partial derivative is taken with respect to the u-th coordinate of the first argument,
        the other is taken with respect to the v-th coordinate of the second argument,
        X_i is the i-th row in data, and Y_j is the j-th row in new_data.

    partial_kernel_matrix_21(new_data)
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) using
        the Gaussian kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u^2 partial_{v+d} k (X_i, Y_j),
        where partial_u^2 partial_{v+d} k denotes the third mixed partial derivative of k,
        two partial derivatives are taken with respect to the u-th coordinate of the first argument,
        one partial derivative is taken with respect to the v-th coordinate of the second argument,
        X_i is the i-th row in data, and Y_j is the j-th row in new_data.

    partial_kernel_matrix_12(new_data)
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) using
        the Gaussian kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u partial_{v+d}^2 k (X_i, Y_j),
        where partial_u partial_{v+d}^2 k denotes the third mixed partial derivative of k,
        one partial derivative is taken with respect to the u-th coordinate of the first argument,
        two partial derivatives are taken with respect to the v-th coordinate of the second argument,
        X_i is the i-th row in data, and Y_j is the j-th row in new_data.

    partial_kernel_matrix_22(new_data)
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) using
        the Gaussian kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u^2 partial_{v+d}^2 k (X_i, Y_j),
        where partial_u^2 partial_{v+d}^2 k denotes the fourth mixed partial derivative of k,
        two partial derivatives are taken with respect to the u-th coordinate of the first argument,
        two partial derivatives are taken with respect to the v-th coordinate of the second argument,
        X_i is the i-th row in data, and Y_j is the j-th row in new_data.

    kernel_x_1d(landmark)
        Returns a function that computes k (landmark, y) at y, where k is the Gaussian kernel function
        plus the polynomial kernel function of degree 2, both landmark and y are 1-dimensional data points.

    kernel_x_1d_deriv1(landmark)
        Returns a function that computes partial_1 k (landmark, y) at y, where k is the Gaussian kernel function
        plus the polynomial kernel function of degree 2, both landmark and y are 1-dimensional data points.

    kernel_x_1d_deriv2(landmark)
        Returns a function that computes partial_1^2 k (x, landmark) at y, where k is the Gaussian kernel function
        plus the polynomial kernel function of degree 2, both landmark and y are 1-dimensional data points.

    kernel_x_2d(landmark)
        Returns a function that computes k (landmark, y) at y, where k is the Gaussian kernel function
        plus the polynomial kernel function of degree 2, both landmark and y are 2-dimensional data points.

    kernel_x_2d_deriv1_0(landmark)
        Returns a function that computes partial_1 k (landmark, y) at y, where k is the Gaussian kernel function
        plus the polynomial kernel function of degree 2, both landmark and y are 2-dimensional data points.

    kernel_x_2d_deriv1_1(landmark)
        Returns a function that computes partial_2 k (landmark, y) at y, where k is the Gaussian kernel function
        plus the polynomial kernel function of degree 2, both landmark and y are 2-dimensional data points.

    kernel_x_2d_deriv2_0(landmark)
        Returns a function that computes partial_1^2 k (landmark, y) at y, where k is the Gaussian kernel function
        plus the polynomial kernel function of degree 2, both landmark and y are 2-dimensional data points.

    kernel_x_2d_deriv2_1(landmark)
        Returns a function that computes partial_2^2 k (landmark, y) at y, where k is the Gaussian kernel function
        plus the polynomial kernel function of degree 2, both landmark and y are 2-dimensional data points.

    kernel_x_3d(landmark)
        Returns a function that computes k (landmark, y) at y, where k is the Gaussian kernel function
        plus the polynomial kernel function of degree 2, both landmark and y are 3-dimensional data points.

    """

    def __init__(self, data, r1, r2, c, bw):

        """
        Parameters
        ----------
        data : numpy.ndarray
            The array of observations whose density function is to be estimated.

        r1 : float
            The multiplicative coefficient associated with the Gaussian kernel function.
            Must be non-negative.

        r2 : float
            The multiplicative coefficient associated with the polynomial kernel function of degree 2.
            Must be non-negative.

        c : float
            The non-homogenous additive constant in the polynomial kernel function of degree 2.
            Must be non-negative.

        bw : float
            The bandwidth parameter in the Gaussian kernel function.
            Must be strictly positive.
            
        """
        
        super().__init__()

        assert r1 >= 0., "The parameter r1 must be non-negative."
        assert r2 >= 0., "The parameter r2 must be non-negative."
        assert c >= 0., "The parameter c must be non-negative."
        assert bw > 0., "The parameter bw must be strictly positive."
        
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        self.data = data
        self.N, self.d = self.data.shape
        
        self.r1 = r1
        self.r2 = r2
        self.c = c
        self.bw = bw
        self.kernel_type = 'gaussian_poly2'
        
    def gaussian_kernel_gram_matrix(self, new_data):
        
        """
        Computes the Gram matrix of shape (data.shape[0], new_data.shape[0]) using the Gaussian kernel function,
        with the (i, j)-th entry being exp(- ||X_i - Y_j||^2 / (2 * bw ^ 2)), where X_i is the i-th row in data,
        and Y_j is the j-th row in new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the Gaussian kernel function is to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (data.shape[0], new_data.shape[0]) with the (i, j)-th entry being
            exp(- ||X_i - Y_j||^2 / (2 * bw ^ 2)), where X_i is the i-th row in data,
            and Y_j is the j-th row in new_data.
            
        """
        
        n, d1 = new_data.shape

        assert self.d == d1, "The dimensionality of new_data does not match that of data. " 
        
        bw = self.bw

        tiled_data = np.tile(new_data, self.N).reshape(1, -1)
        tiled_land = np.tile(self.data.reshape(1, -1), n)
        
        diff = - (tiled_data - tiled_land) ** 2 / (2 * bw ** 2)
        power = np.sum(np.vstack(np.split(diff, self.N * n, axis=1)), axis=1)
        power = power.reshape(n, self.N)

        gauss_part = np.exp(power)

        return gauss_part.T
    
    def poly_kernel_gram_matrix(self, new_data): 
        
        """
        Computes the Gram matrix of shape (data.shape[0], new_data.shape[0]) using the polynomial kernel function
        of degree 2, with the (i, j)-th entry being (X_i^\top Y_j + c) ^ 2, where X_i is the i-th row in data,
        and Y_j is the j-th row in new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the polynomial kernel function of degree 2 is to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (data.shape[0], new_data.shape[0]) with the (i, j)-th entry being
            (X_i^\top Y_j + c) ^ 2, where X_i is the i-th row in data,
            and Y_j is the j-th row in new_data.

        """

        n, d1 = new_data.shape

        assert self.d == d1, "The dimensionality of new_data does not match that of data. " 

        tiled_data = np.tile(new_data, self.N).reshape(1, -1)
        tiled_land = np.tile(self.data.reshape(1, -1), n)
        
        prod_entry = tiled_data * tiled_land 
        split_prod = np.sum(np.vstack(np.split(prod_entry, self.N * n, axis=1)), axis=1)
        split_prod = split_prod.reshape(n, self.N)
        poly_part = split_prod

        return poly_part.T
    
    def kernel_gram_matrix(self, new_data): 
        
        """
        Computes the Gram matrix of shape (data.shape[0], new_data.shape[0]) using the Gaussian kernel function plus
        the polynomial kernel function of degree 2, with the (i, j)-th entry being
        k (X_i, Y_j) = r1 * exp(- ||X_i - Y_j||^2 / (2 * bw ^ 2)) + r2 * (X_i^\top Y_j + c) ^ 2,
        where X_i is the i-th row in data, and Y_j is the j-th row in new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the Gaussian kernel function plus the polynomial kernel function
            of degree 2 is to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (data.shape[0], new_data.shape[0]) with the (i, j)-th entry being
            k (X_i, Y_j) = r1 * exp(- ||X_i - Y_j||^2 / (2 * bw ^ 2)) + r2 * (X_i^\top Y_j + c) ^ 2,
            where X_i is the i-th row in data, and Y_j is the j-th row in new_data.
            
        """
        
        gauss_part = self.r1 * self.gaussian_kernel_gram_matrix(new_data=new_data)
        poly_part = self.r2 * (self.poly_kernel_gram_matrix(new_data=new_data) + self.c) ** 2

        output = gauss_part + poly_part 

        return output 
    
    def partial_kernel_matrix_10(self, new_data): 
        
        """
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0]) using
        the Gaussian kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, j)-th entry being partial_u k (X_i, Y_j),
        where partial_u k denotes the first partial derivative of k with respect to the u-th coordinate
        of its first argument, X_i is the i-th row in data, and Y_j is the j-th row in new_data.
        
        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which first partial of the Gaussian kernel function plus
            the polynomial kernel function of degree 2 is to be evaluated.
        
        Returns
        -------
        numpy.ndarray
            An array of shape (data.shape[0] * data.shape[1], new_data.shape[0])
            with the ((i-1)*d+u, j)-th entry being partial_u k (X_i, Y_j).
            
        """
        
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. " 
        
        # ----------------------------------------------------------------------
        # Gaussian kernel part 
        gauss_kernel = np.repeat(self.gaussian_kernel_gram_matrix(new_data=new_data),
                                 repeats=self.d, axis=0)
        multi_gauss1 = np.repeat(self.data.flatten(), n).reshape(self.N * self.d, n)
        multi_gauss2 = np.tile(new_data.T, (self.N, 1))
        gauss_partial = (-self.r1 * gauss_kernel * (multi_gauss1 - multi_gauss2) / self.bw ** 2)
        
        # ----------------------------------------------------------------------
        # Poly2 kernel part 
        poly_part = np.repeat((self.poly_kernel_gram_matrix(new_data=new_data) + self.c), repeats=self.d, axis=0)
        multi_poly = np.tile(new_data.T, (self.N, 1))
        poly_partial = 2 * self.r2 * poly_part * multi_poly
        
        output = gauss_partial + poly_partial
        
        return output

    def partial_kernel_matrix_01(self, new_data):
    
        """
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0]) using
        the Gaussian kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, j)-th entry being partial_{d+u} k (X_i, Y_j),
        where partial_{d+u} k denotes the first partial derivative of k with respect to the u-th coordinate
        of its second argument, X_i is the i-th row in data, and Y_j is the j-th row in new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which first partial of the Gaussian kernel function plus
            the polynomial kernel function of degree 2 is to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (data.shape[0] * data.shape[1], new_data.shape[0])
            with the ((i-1)*d+u, j)-th entry being partial_{d+u} k (X_i, Y_j).

        """
    
        n, d1 = new_data.shape
    
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
    
        # ----------------------------------------------------------------------
        # Gaussian kernel part
        gauss_kernel = np.repeat(self.gaussian_kernel_gram_matrix(new_data=new_data),
                                 repeats=self.d, axis=0)
        multi_gauss1 = np.repeat(self.data.flatten(), n).reshape(self.N * self.d, n)
        multi_gauss2 = np.tile(new_data.T, (self.N, 1))
        gauss_partial = (-self.r1 * gauss_kernel * (multi_gauss1 - multi_gauss2) / self.bw ** 2) * (-1.)

        # ----------------------------------------------------------------------
        # Poly2 kernel part
        poly_part = np.repeat((self.poly_kernel_gram_matrix(new_data=new_data) + self.c), repeats=self.d, axis=0)
        multi_poly = np.tile(self.data.flatten().reshape(-1, 1), (1, n))
        poly_partial = 2. * self.r2 * poly_part * multi_poly

        output = gauss_partial + poly_partial

        return output
    
    def partial_kernel_matrix_20(self, new_data): 
        
        """
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0]) using
        the Gaussian kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, j)-th entry being partial_u^2 k (X_i, Y_j),
        where partial_u^2 k denotes the second partial derivative of k with respect to the u-th coordinate
        of its first argument, X_i is the i-th row in data, and Y_j is the j-th row in new_data.
        
        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which second partial of the Gaussian kernel function plus
            the polynomial kernel function of degree 2 is to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (data.shape[0] * data.shape[1], new_data.shape[0])
            with the ((i-1)*d+u, j)-th entry being partial_u^2 k (X_i, Y_j).
            
        """
        
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. " 
        
        # ----------------------------------------------------------------------
        # Gaussian kernel part 
        gauss_kernel = np.repeat(self.gaussian_kernel_gram_matrix(new_data=new_data), repeats=self.d, axis=0)
        multi_gauss_part1 = np.repeat(self.data.flatten(), n).reshape(self.N * self.d, n)
        multi_gauss_part2 = np.tile(new_data.T, (self.N, 1))
        multi_gauss_part = ((multi_gauss_part1 - multi_gauss_part2) ** 2 / self.bw ** 4 - 
                            1 / self.bw ** 2)
        gauss_partial = self.r1 * multi_gauss_part * gauss_kernel
        
        # ----------------------------------------------------------------------
        # Poly2 kernel part 
        multi_poly = np.tile(new_data.T, (self.N, 1)) ** 2
        poly_partial = 2 * self.r2 * multi_poly
        
        output = gauss_partial + poly_partial
        
        return output

    def partial_kernel_matrix_02(self, new_data):
    
        """
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0]) using
        the Gaussian kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, j)-th entry being partial_{d+u}^2 k (X_i, Y_j),
        where partial_{d+u}^2 k denotes the second partial derivative of k with respect to the u-th coordinate
        of its second argument, X_i is the i-th row in data, and Y_j is the j-th row in new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which second partial of the Gaussian kernel function plus
            the polynomial kernel function of degree 2 is to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (data.shape[0] * data.shape[1], new_data.shape[0])
            with the ((i-1)*d+u, j)-th entry being partial_{d+u}^2 k (X_i, Y_j).

        """
    
        n, d1 = new_data.shape
    
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
    
        # ----------------------------------------------------------------------
        # Gaussian kernel part
        gauss_kernel = np.repeat(self.gaussian_kernel_gram_matrix(new_data=new_data), repeats=self.d, axis=0)
        multi_gauss_part1 = np.repeat(self.data.flatten(), n).reshape(self.N * self.d, n)
        multi_gauss_part2 = np.tile(new_data.T, (self.N, 1))
        multi_gauss_part = ((multi_gauss_part1 - multi_gauss_part2) ** 2 / self.bw ** 4 -
                            1 / self.bw ** 2)
        gauss_partial = self.r1 * multi_gauss_part * gauss_kernel
    
        # ----------------------------------------------------------------------
        # Poly2 kernel part
        multi_poly = np.tile(self.data.flatten().reshape(-1, 1), (1, n)) ** 2
        poly_partial = 2. * self.r2 * multi_poly
    
        output = gauss_partial + poly_partial
    
        return output
        
    def partial_kernel_matrix_11(self, new_data): 
        
        """
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) using
        the Gaussian kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u partial_{v+d} k (X_i, Y_j),
        where partial_u partial_{v+d} k denotes the second mixed partial derivative of k,
        one partial derivative is taken with respect to the u-th coordinate of the first argument,
        the other is taken with respect to the v-th coordinate of the second argument,
        X_i is the i-th row in data, and Y_j is the j-th row in new_data.
        
        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which second mixed partial of the Gaussian kernel function plus
            the polynomial kernel function of degree 2 is to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1])
            with the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u partial_{v+d} k (X_i, Y_j).
        
        """
        
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. " 
        
        # ----------------------------------------------------------------------
        # Gaussian kernel part 
        gauss_kernel = np.kron(self.gaussian_kernel_gram_matrix(new_data=new_data),
                               np.ones((self.d, self.d), dtype=np.float64))
        
        multi_gauss11 = np.tile(self.data.flatten().reshape(-1, 1), (1, n * self.d))
        multi_gauss12 = np.tile(np.repeat(new_data.T, repeats=self.d, axis=1), (self.N, 1))
        multi_gauss1  = multi_gauss11 - multi_gauss12
        
        multi_gauss21 = np.tile(np.repeat(self.data, repeats=self.d, axis=0), (1, n))
        multi_gauss22 = np.repeat(new_data.flatten(), self.N * self.d).reshape(n * self.d, self.N * self.d).T
        multi_gauss2  = multi_gauss21 - multi_gauss22
        
        multi_gauss   = multi_gauss1 * multi_gauss2 / self.bw ** 4
        
        # the matrix added when the coordinates wrt which the derivatives are taken are the same 
        add_mat = self.bw ** (-2) * np.tile(np.eye(self.d, dtype=np.float64), (self.N, n))
        
        gauss_partial = - self.r1 * gauss_kernel * (multi_gauss - add_mat)
        
        # ----------------------------------------------------------------------
        # Poly2 kernel part 
        poly_part11 = np.tile(np.repeat(self.data, repeats=self.d, axis=0), (1, n))
        poly_part12 = np.tile(np.repeat(new_data.T, repeats=self.d, axis=1), (self.N, 1))
        poly_part1  = poly_part11 * poly_part12
        
        poly_part2 = np.kron(self.poly_kernel_gram_matrix(new_data=new_data) + self.c,
                             np.eye(self.d, dtype=np.float64))
        
        poly_partial = 2. * self.r2 * (poly_part1 + poly_part2)
        
        output = gauss_partial + poly_partial
        
        return output
    
    def partial_kernel_matrix_21(self, new_data): 
        
        """
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) using
        the Gaussian kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u^2 partial_{v+d} k (X_i, Y_j),
        where partial_u^2 partial_{v+d} k denotes the third mixed partial derivative of k,
        two partial derivatives are taken with respect to the u-th coordinate of the first argument,
        one partial derivative is taken with respect to the v-th coordinate of the second argument,
        X_i is the i-th row in data, and Y_j is the j-th row in new_data.
        
        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which third mixed partial of the Gaussian kernel function plus
            the polynomial kernel function of degree 2 is to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) with
            the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u^2 partial_{v+d} k (X_i, Y_j).
            
        """
        
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. " 
        
        # ----------------------------------------------------------------------
        # Gaussian kernel part 
        gauss_kernel = np.kron(self.gaussian_kernel_gram_matrix(new_data=new_data),
                               np.ones((self.d, self.d), dtype=np.float64))
        
        multi_gauss11 = np.tile(self.data.flatten().reshape(-1, 1), (1, n * self.d))
        multi_gauss12 = np.tile(np.repeat(new_data.T, repeats=self.d, axis=1), (self.N, 1))
        multi_gauss1  = (multi_gauss11 - multi_gauss12) ** 2
        
        multi_gauss21 = np.tile(np.repeat(self.data, repeats=self.d, axis=0), (1, n))
        multi_gauss22 = np.repeat(new_data.flatten(), self.N * self.d).reshape(n * self.d, self.N * self.d).T
        multi_gauss2  = multi_gauss21 - multi_gauss22
        
        multi_gauss   = multi_gauss1 * multi_gauss2 / self.bw ** 6
        
        # the matrix added when the coordinates wrt which the derivatives are taken are the same 
        add_mat = ((multi_gauss2 + 2. * multi_gauss2 * np.tile(np.eye(self.d, dtype=np.float64), (self.N, n))) /
                   self.bw ** 4)
        
        gauss_partial = self.r1 * gauss_kernel * (multi_gauss - add_mat)
        
        # ----------------------------------------------------------------------
        # Poly2 kernel part 
        b = np.zeros((new_data.shape[0], new_data.shape[1], new_data.shape[1]))
        diag = np.arange(new_data.shape[1])
        b[:, diag, diag] = new_data
        b2 = np.hstack(b)
        poly_partial = 4 * self.r2 * np.tile(b2, (self.N, 1))
        
        output = gauss_partial + poly_partial
        
        return output
    
    def partial_kernel_matrix_12(self, new_data): 
        
        """
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) using
        the Gaussian kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u partial_{v+d}^2 k (X_i, Y_j),
        where partial_u partial_{v+d}^2 k denotes the third mixed partial derivative of k,
        one partial derivative is taken with respect to the u-th coordinate of the first argument,
        two partial derivatives are taken with respect to the v-th coordinate of the second argument,
        X_i is the i-th row in data, and Y_j is the j-th row in new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which third mixed partial of the Gaussian kernel function plus
            the polynomial kernel function of degree 2 is to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) with
            the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u partial_{v+d}^2 k (X_i, Y_j).
            
        """
        
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. " 
        
        # ----------------------------------------------------------------------
        # Gaussian kernel part 
        gauss_kernel = np.kron(self.gaussian_kernel_gram_matrix(new_data=new_data),
                               np.ones((self.d, self.d), dtype=np.float64))
        
        multi_gauss11 = np.tile(self.data.flatten().reshape(-1, 1), (1, n * self.d))
        multi_gauss12 = np.tile(np.repeat(new_data.T, repeats=self.d, axis=1), (self.N, 1))
        multi_gauss1  = multi_gauss11 - multi_gauss12
        
        multi_gauss21 = np.tile(np.repeat(self.data, repeats=self.d, axis=0), (1, n))
        multi_gauss22 = np.repeat(new_data.flatten(), self.N * self.d).reshape(n * self.d, self.N * self.d).T
        multi_gauss2  = (multi_gauss21 - multi_gauss22) ** 2
        
        multi_gauss   = multi_gauss1 * multi_gauss2 / self.bw ** 6
        
        # the matrix added when the coordinates wrt which the derivatives are taken are the same 
        add_mat = ((multi_gauss1 + 2. * multi_gauss1 * np.tile(np.eye(self.d, dtype=np.float64), (self.N, n))) /
                   self.bw ** 4)
        
        gauss_partial = self.r1 * gauss_kernel * (add_mat - multi_gauss)
        
        # ----------------------------------------------------------------------
        # Poly2 kernel part 
        b = np.zeros((self.data.shape[0], self.data.shape[1], self.data.shape[1]))
        diag = np.arange(self.data.shape[1])
        b[:, diag, diag] = self.data
        b2 = np.vstack(b)
        poly_partial = 4 * self.r2 * np.tile(b2, (1, n))
        
        output = gauss_partial + poly_partial
        
        return output
        
    def partial_kernel_matrix_22(self, new_data): 
        
        """
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) using
        the Gaussian kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u^2 partial_{v+d}^2 k (X_i, Y_j),
        where partial_u^2 partial_{v+d}^2 k denotes the fourth mixed partial derivative of k,
        two partial derivatives are taken with respect to the u-th coordinate of the first argument,
        two partial derivatives are taken with respect to the v-th coordinate of the second argument,
        X_i is the i-th row in data, and Y_j is the j-th row in new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which fourth mixed partial of the Gaussian kernel function plus
            the polynomial kernel function of degree 2 is to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) with
            the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u^2 partial_{v+d}^2 k (X_i, Y_j).
            
        """
        
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. " 
        
        # ----------------------------------------------------------------------
        # Gaussian kernel part 
        gauss_kernel = np.kron(self.gaussian_kernel_gram_matrix(new_data=new_data),
                               np.ones((self.d, self.d), dtype=np.float64))
        
        multi_gauss11 = np.tile(self.data.flatten().reshape(-1, 1), (1, n * self.d))
        multi_gauss12 = np.tile(np.repeat(new_data.T, repeats=self.d, axis=1), (self.N, 1))
        multi_gauss1  = (multi_gauss11 - multi_gauss12) ** 2
        
        multi_gauss21 = np.tile(np.repeat(self.data, repeats=self.d, axis=0), (1, n))
        multi_gauss22 = np.repeat(new_data.flatten(), self.N * self.d).reshape(n * self.d, self.N * self.d).T
        multi_gauss2  = (multi_gauss21 - multi_gauss22) ** 2
        
        multi_gauss   = (multi_gauss1 * multi_gauss2 / self.bw ** 8 - multi_gauss1 / self.bw ** 6 - 
                         multi_gauss2 / self.bw ** 6 + 1. / self.bw ** 4)
        
        # the matrix added when the coordinates wrt which the derivatives are taken are the same 
        add_mat = (- 4. * multi_gauss1 * np.tile(np.eye(self.d, dtype=np.float64), (self.N, n)) / self.bw ** 6
                   + 2. * np.tile(np.eye(self.d, dtype=np.float64), (self.N, n)) / self.bw ** 4)
        
        gauss_partial = self.r1 * gauss_kernel * (add_mat + multi_gauss)
        
        # ----------------------------------------------------------------------
        # Poly2 kernel part 
        poly_partial = 4 * self.r2 * np.tile(np.eye(self.d, dtype=np.float64), (self.N, n))
        
        output = gauss_partial + poly_partial
        
        return output
    
    def kernel_x_1d(self, landmark):
        
        """
        Returns a function that computes k (landmark, y) at y, where k is the Gaussian kernel function
        plus the polynomial kernel function of degree 2, both landmark and y are 1-dimensional data points.
        
        Parameters
        ----------
        landmark : float or np.ndarray
            A floating point number or a data array of shape (1,).
        
        Returns
        -------
        function
            A function that computes k (landmark, y) at y.
        
        """

        landmark = np.array(landmark).flatten()
        if len(landmark) != 1:
            raise ValueError("The length of landmark should be 1.")

        def output(x):
            
            y = (self.r1 * np.exp(- (x - landmark) ** 2 / (2 * self.bw ** 2)) +
                 self.r2 * (x * landmark + self.c) ** 2)
            return y

        return output
    
    def kernel_x_1d_deriv1(self, landmark):
        
        """
        Returns a function that computes partial_1 k (landmark, y) at y, where k is the Gaussian kernel function
        plus the polynomial kernel function of degree 2, both landmark and y are 1-dimensional data points.

        Parameters
        ----------
        landmark : float or np.ndarray
            A floating point number or a data array of shape (1,).

        Returns
        -------
        function
            A function that computes partial_1 k (landmark, y) at y.
            
        """

        landmark = np.array(landmark).flatten()
        if len(landmark) != 1:
            raise ValueError("The length of landmark should be 1.")
        
        def output(y): 
            
            normsq = (y - landmark) ** 2
            e_term = np.exp(- normsq / (2 * self.bw ** 2))
            ker = self.r1 * e_term * (- (landmark - y) / self.bw ** 2) + 2 * self.r2 * (y * landmark + self.c) * y
            
            return ker
        
        return output
    
    def kernel_x_1d_deriv2(self, landmark):
        
        """
        Returns a function that computes partial_1^2 k (landmark, y) at y, where k is the Gaussian kernel function
        plus the polynomial kernel function of degree 2, both landmark and y are 1-dimensional data points.

        Parameters
        ----------
        landmark : float or np.ndarray
            A floating point number or a data array of shape (1,).

        Returns
        -------
        function
            A function that computes partial_1^2 k (landmark, y) at y.
        
        """
        
        landmark = np.array(landmark).flatten()
        if len(landmark) != 1:
            raise ValueError("The length of landmark should be 1.")
        
        def output(y): 
            
            normsq = (y - landmark) ** 2
            e_term = np.exp(- normsq / (2 * self.bw ** 2))
            ker = (self.r1 * e_term * ((y - landmark) ** 2 / self.bw ** 4) - 
                   self.r1 * e_term / (self.bw ** 2) + 
                   2 * self.r2 * y ** 2)
            
            return ker
        
        return output

    def kernel_x_2d(self, landmark):
        
        """
        Returns a function that computes k (landmark, y) at y, where k is the Gaussian kernel function
        plus the polynomial kernel function of degree 2, both landmark and y are 2-dimensional data points.

        Parameters
        ----------
        landmark : np.ndarray
            A data array of shape (2,).

        Returns
        -------
        function
            A function that computes k (landmark, y) at y.
            
        """

        landmark = np.array(landmark).flatten()
        if len(landmark) != 2:
            raise ValueError("The length of landmark should be 2.")
        
        def output(x0, x1):
            y = (self.r1 * np.exp(- ((x0 - landmark[0]) ** 2 + (x1 - landmark[1]) ** 2) / (2 * self.bw ** 2)) +
                 self.r2 * (x0 * landmark[0] + x1 * landmark[1] + self.c) ** 2)
            return y

        return output

    def kernel_x_2d_deriv1_0(self, landmark):
        
        """
        Returns a function that computes partial_1 k (landmark, y) at y, where k is the Gaussian kernel function
        plus the polynomial kernel function of degree 2, both landmark and y are 2-dimensional data points.
    
        Parameters
        ----------
        landmark : np.ndarray
            A data array of shape (2,).

        Returns
        -------
        function
            A function that computes partial_1 k (landmark, y) at y.
            
        """
        
        landmark = np.array(landmark).flatten()
        if len(landmark) != 2:
            raise ValueError("The length of landmark should be 2.")
    
        def output(y0, y1): 

            normsq = (y0 - landmark[0]) ** 2 + (y1 - landmark[1]) ** 2
            e_term = np.exp(- normsq / (2 * self.bw ** 2))
            ker_val = (self.r1 * e_term * (- (landmark[0] - y0) / self.bw ** 2) + 
                       2 * self.r2 * (y0 * landmark[0] + y1 * landmark[1] + self.c) * y0)
            return ker_val
        
        return output
    
    def kernel_x_2d_deriv1_1(self, landmark):
    
        """
        Returns a function that computes partial_2 k (landmark, y) at y, where k is the Gaussian kernel function
        plus the polynomial kernel function of degree 2, both landmark and y are 2-dimensional data points.

        Parameters
        ----------
        landmark : np.ndarray
            A data array of shape (2,).

        Returns
        -------
        function
            A function that computes partial_2 k (landmark, y) at y.
        
        """
        
        landmark = np.array(landmark).flatten()
        if len(landmark) != 2:
            raise ValueError("The length of landmark should be 2.")
        
        def output(y0, y1): 

            normsq = (y0 - landmark[0]) ** 2 + (y1 - landmark[1]) ** 2
            e_term = np.exp(- normsq / (2 * self.bw ** 2))
            ker_val = (self.r1 * e_term * (- (landmark[1] - y1) / self.bw ** 2) + 
                       2 * self.r2 * (y0 * landmark[0] + y1 * landmark[1] + self.c) * y1)
            return ker_val
        
        return output
    
    def kernel_x_2d_deriv2_0(self, landmark):
        
        """
        Returns a function that computes partial_1^2 k (landmark, y) at y, where k is the Gaussian kernel function
        plus the polynomial kernel function of degree 2, both landmark and y are 2-dimensional data points.

        Parameters
        ----------
        landmark : np.ndarray
            A data array of shape (2,).
            
        Returns
        -------
        function
            A function that computes partial_1^2 k (landmark, y) at y.
            
        """
    
        landmark = np.array(landmark).flatten()
        if len(landmark) != 2:
            raise ValueError("The length of landmark should be 2.")
        
        def output(y0, y1): 

            normsq = (y0 - landmark[0]) ** 2 + (y1 - landmark[1]) ** 2
            e_term = np.exp(- normsq / (2 * self.bw ** 2))

            ker_val = ((self.r1 * e_term * ((landmark[0] - y0) ** 2 / self.bw ** 4) -
                        self.r1 * e_term / (self.bw ** 2) + 2 * self.r2 * y0 ** 2))
            return ker_val

        return output

    def kernel_x_2d_deriv2_1(self, landmark):
        
        """
        Returns a function that computes partial_2^2 k (landmark, y) at y, where k is the Gaussian kernel function
        plus the polynomial kernel function of degree 2, both landmark and y are 2-dimensional data points.

        Parameters
        ----------
        landmark : np.ndarray
            A data array of shape (2,).

        Returns
        -------
        function
            A function that computes partial_2^2 k (landmark, y) at y.
        
        """
        
        landmark = np.array(landmark).flatten()
        if len(landmark) != 2:
            raise ValueError("The length of landmark should be 2.")
        
        def output(y0, y1): 

            normsq = (y0 - landmark[0]) ** 2 + (y1 - landmark[1]) ** 2
            e_term = np.exp(- normsq / (2 * self.bw ** 2))
            
            ker_val = ((self.r1 * e_term * ((landmark[1] - y1) ** 2 / self.bw ** 4) -
                        self.r1 * e_term / (self.bw ** 2) + 2 * self.r2 * y1 ** 2))
            return ker_val

        return output

    def kernel_x_3d(self, landmark):
        
        """
        Returns a function that computes k (landmark, y) at y, where k is the Gaussian kernel function
        plus the polynomial kernel function of degree 2, both landmark and y are 3-dimensional data points.

        Parameters
        ----------
        landmark : np.ndarray
            A data array of shape (3,).

        Returns
        -------
        function
            A function that computes k (landmark, y) at y.
            
        """
    
        landmark = np.array(landmark).flatten()
        if len(landmark) != 3:
            raise ValueError("The length of landmark should be 3.")
        
        def output(x0, x1, x2):
            y = (self.r1 * np.exp(- ((x0 - landmark[0]) ** 2 + (x1 - landmark[1]) ** 2 +
                                     (x2 - landmark[2]) ** 2) / (2 * self.bw ** 2)) +
                 self.r2 * (x0 * landmark[0] + x1 * landmark[1] + x2 * landmark[2] + self.c) ** 2)
            return y
        
        return output


class RationalQuadPoly2(KernelFunction):
    
    """
    A class to compute the rational quadratic kernel function plus a polynomial kernel function of degree 2,
    k (x, y) = r1 * (1 + (||x - y|| ^ 2 / bw ^ 2)) ^ (-1) + r2 * (x^\top y + c) ^ 2,
    and its derivatives.

    ...

    Attributes
    ----------
    data : numpy.ndarray
        The array of observations whose density function is to be estimated.
        
    N : int
        The number of rows of data.
        
    d : int
        The number of columns of data.
        
    r1 : float
        The multiplicative coefficient associated with the rational quadratic kernel function.
        
    r2 : float
        The multiplicative coefficient associated with the polynomial kernel function of degree 2.
        
    c : float
        The non-homogenous additive constant in the polynomial kernel function of degree 2.
        
    bw : float
        The bandwidth parameter in the rational quadratic kernel function.
        
    kernel_type : str
        The type of the kernel function, 'rationalquad_poly2'.
        
    Methods
    -------
    rationalquad_kernel_gram_matrix(new_data)
        Computes the Gram matrix of shape (data.shape[0], new_data.shape[0]) using the rational quadratic kernel
        function, with the (i, j)-th entry being (1 + (||X_i - Y_j|| ^ 2 / (2 * bw ^ 2))) ^ (-1),
        where X_i is the i-th row in data, and Y_j is the j-th row in new_data.

    poly_kernel_gram_matrix(new_data)
        Computes the Gram matrix of shape (data.shape[0], new_data.shape[0]) using the polynomial kernel function of
        degree 2, with the (i, j)-th entry being (X_i^\top Y_j + c) ^ 2, where X_i is the i-th row in data,
        and Y_j is the j-th row in new_data.

    kernel_gram_matrix(new_data)
        Computes the Gram matrix of shape (data.shape[0], new_data.shape[0]) using the rational quadratic kernel
        function plus the polynomial kernel function of degree 2, with the (i, j)-th entry being
        k (X_i, Y_j) = r1 * (1 + (||X_i - Y_j|| ^ 2 / (2 * bw ^ 2))) ^ (-1) + r2 * (X_i^\top Y_j + c) ^ 2,
        where X_i is the i-th row in data, and Y_j is the j-th row in new_data.

    partial_kernel_matrix_10(new_data)
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0]) using
        the rational quadratic kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, j)-th entry being partial_u k (X_i, Y_j),
        where partial_u k denotes the first partial derivative of k with respect to the u-th coordinate
        of the first argument, X_i is the i-th row in data, and Y_j is the j-th row in new_data.

    partial_kernel_matrix_20(new_data)
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0]) using
        the rational quadratic kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, j)-th entry being partial_u^2 k (X_i, Y_j),
        where partial_u^2 k denotes the second partial derivative of k with respect to the u-th coordinate
        of the first argument, X_i is the i-th row in data, and Y_j is the j-th row in new_data.

    partial_kernel_matrix_11(new_data)
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) using
        the rational quadratic kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u partial_{v+d} k (X_i, Y_j),
        where partial_u partial_{v+d} k denotes the second mixed partial derivative of k,
        one partial derivative is taken with respect to the u-th coordinate of the first argument,
        the other is taken with respect to the v-th coordinate of the second argument,
        X_i is the i-th row in data, and Y_j is the j-th row in new_data.
    
    partial_kernel_matrix_21(new_data)
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) using
        the rational quadratic kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u^2 partial_{v+d} k (X_i, Y_j),
        where partial_u^2 partial_{v+d} k denotes the third mixed partial derivative of k,
        two partial derivatives are taken with respect to the u-th coordinate of the first argument,
        one partial derivative is taken with respect to the v-th coordinate of the second argument,
        X_i is the i-th row in data, and Y_j is the j-th row in new_data.
    
    partial_kernel_matrix_12(new_data)
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) using
        the rational quadratic kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u partial_{v+d}^2 k (X_i, Y_j),
        where partial_u partial_{v+d}^2 k denotes the third mixed partial derivative of k,
        one partial derivative is taken with respect to the u-th coordinate of the first argument,
        two partial derivatives are taken with respect to the v-th coordinate of the second argument,
        X_i is the i-th row in data, and Y_j is the j-th row in new_data.
    
    partial_kernel_matrix_22(new_data)
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) using
        the rational quadratic kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u^2 partial_{v+d}^2 k (X_i, Y_j),
        where partial_u^2 partial_{v+d}^2 k denotes the fourth mixed partial derivative of k,
        two partial derivatives are taken with respect to the u-th coordinate of the first argument,
        two partial derivatives are taken with respect to the v-th coordinate of the second argument,
        X_i is the i-th row in data, and Y_j is the j-th row in new_data.

    kernel_x_1d(landmark)
        Returns a function that computes k (landmark, y) at y, where k is the rational quadratic kernel function plus
        the polynomial kernel function of degree 2, both landmark and y are 1-dimensional data points.

    kernel_x_1d_deriv1(landmark)
        Returns a function that computes partial_1 k (landmark, y) at y, where k is the rational quadratic kernel
        function plus the polynomial kernel function of degree 2, both landmark and y are 1-dimensional data points.
        
    kernel_x_1d_deriv2(landmark)
        Returns a function that computes partial_1^2 k (x, landmark) at y, where k is the rational quadratic kernel
        function plus the polynomial kernel function of degree 2, both landmark and y are 1-dimensional data points.
        
    kernel_x_2d(landmark)
        Returns a function that computes k (landmark, y) at y, where k is the rational quadratic kernel function plus
        the polynomial kernel function of degree 2, both landmark and y are 2-dimensional data points.

    kernel_x_2d_deriv1_0(landmark)
        Returns a function that computes partial_1 k (landmark, y) at y, where k is the rational quadratic kernel
        function plus the polynomial kernel function of degree 2, both landmark and y are 2-dimensional data points.
        
    kernel_x_2d_deriv1_1(landmark)
        Returns a function that computes partial_2 k (landmark, y) at y, where k is the rational quadratic kernel
        function plus the polynomial kernel function of degree 2, both landmark and y are 2-dimensional data points.
        
    kernel_x_2d_deriv2_0(landmark)
        Returns a function that computes partial_1^2 k (landmark, y) at y, where k is the rational quadratic kernel
        function plus the polynomial kernel function of degree 2, both landmark and y are 2-dimensional data points.

    kernel_x_2d_deriv2_1(landmark)
        Returns a function that computes partial_2^2 k (landmark, y) at y, where k is the rational quadratic kernel
        function plus the polynomial kernel function of degree 2, both landmark and y are 2-dimensional data points.
        
    kernel_x_3d(landmark)
        Returns a function that computes k (landmark, y) at y, where k is the rational quadratic kernel function plus
        the polynomial kernel function of degree 2, both landmark and y are 3-dimensional data points.

    """
    
    def __init__(self, data, r1, r2, c, bw):
        
        """
        Parameters
        ----------
        data : numpy.ndarray
            The array of observations whose density function is to be estimated.

        r1 : float
            The multiplicative coefficient associated with the rational quadratic kernel function.
            Must be non-neagtive.

        r2 : float
            The multiplicative coefficient associated with the polynomial kernel function of degree 2.
            Must be non-neagtive.

        c : float
            The non-homogenous additive constant in the polynomial kernel function of degree 2.
            Must be non-neagtive.

        bw : float
            The bandwidth parameter in the rational quadratic kernel function.
            Must be strictly positive.

        """

        super().__init__()
        
        assert r1 >= 0., "The parameter r1 must be non-negative."
        assert r2 >= 0., "The parameter r2 must be non-negative."
        assert c >= 0., "The parameter c must be non-negative."
        assert bw > 0., "The parameter bw must be strictly positive."

        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        self.data = data
        self.N, self.d = self.data.shape
        
        self.r1 = r1
        self.r2 = r2
        self.c = c
        self.bw = bw
        self.kernel_type = 'rationalquad_poly2'
    
    def rationalquad_kernel_gram_matrix(self, new_data):
        
        """
        Computes the Gram matrix of shape (data.shape[0], new_data.shape[0]) using the rational quadratic kernel
        function, with the (i, j)-th entry being (1 + (||X_i - Y_j|| ^ 2 / (2 * bw ^ 2))) ^ (-1),
        where X_i is the i-th row in data, and Y_j is the j-th row in new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the rational quadratic kernel function is to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (data.shape[0], new_data.shape[0]) with the (i, j)-th entry being
            (1 + (||X_i - Y_j|| ^ 2 / (2 * bw ^ 2))) ^ (-1),
            where X_i is the i-th row in data, and Y_j is the j-th row in new_data.

        """
        
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        bw = self.bw
        
        tiled_data = np.tile(new_data, self.N).reshape(1, -1)
        tiled_land = np.tile(self.data.reshape(1, -1), n)
        
        ###############################
        # rational quadratic kernel part
        diff = (tiled_data - tiled_land) ** 2 / (bw ** 2)
        power = np.sum(np.vstack(np.split(diff, self.N * n, axis=1)), axis=1)
        power = power.reshape(n, self.N)
        
        rationalquad_part = (1. + power) ** (-1)
        
        return rationalquad_part.T
    
    def poly2_kernel_gram_matrix(self, new_data):
        
        """
        Computes the Gram matrix of shape (data.shape[0], new_data.shape[0]) using the polynomial kernel function
        of degree 2, with the (i, j)-th entry being (X_i^\top Y_j + c) ^ 2, where X_i is the i-th row in data,
        and Y_j is the j-th row in new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the polynomial kernel function of degree 2 is to be evaluated.
            
        Returns
        -------
        numpy.ndarray
            An array of shape (data.shape[0], new_data.shape[0]) with the (i, j)-th entry being
            (X_i^\top Y_j + c) ^ 2, where X_i is the i-th row in data,
            and Y_j is the j-th row in new_data.

        """
        
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        tiled_data = np.tile(new_data, self.N).reshape(1, -1)
        tiled_land = np.tile(self.data.reshape(1, -1), n)
        
        prod_entry = tiled_data * tiled_land
        split_prod = np.sum(np.vstack(np.split(prod_entry, self.N * n, axis=1)), axis=1)
        split_prod = split_prod.reshape(n, self.N)
        poly_part = split_prod
        
        return poly_part.T
    
    def kernel_gram_matrix(self, new_data):
        
        """
        Computes the Gram matrix of shape (data.shape[0], new_data.shape[0]) using the rational quadratic kernel
        function plus the polynomial kernel function of degree 2, with the (i, j)-th entry being
        k (X_i, Y_j) = r1 * (1 + (||X_i - Y_j|| ^ 2 / (2 * bw ^ 2))) ^ (-1) + r2 * (X_i^\top Y_j + c) ^ 2,
        where X_i is the i-th row in data, and Y_j is the j-th row in new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which the rational quadratic kernel function plus
            the polynomial kernel function of degree 2 is to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (data.shape[0], new_data.shape[0]) with the (i, j)-th entry being
            k (X_i, Y_j) = r1 * (1 + (||X_i - Y_j|| ^ 2 / (2 * bw ^ 2))) ^ (-1) + r2 * (X_i^\top Y_j + c) ^ 2,
            where X_i is the i-th row in data, and Y_j is the j-th row in new_data.

        """
        
        rationalquad_part = self.r1 * self.rationalquad_kernel_gram_matrix(new_data=new_data)
        poly_part = self.r2 * (self.poly2_kernel_gram_matrix(new_data=new_data) + self.c) ** 2
        
        output = rationalquad_part + poly_part
        
        return output
    
    def partial_kernel_matrix_10(self, new_data):
        
        """
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0]) using
        the rational quadratic kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, j)-th entry being partial_u k (X_i, Y_j),
        where partial_u k denotes the first partial derivative of k with respect to the u-th coordinate
        of the first argument, X_i is the i-th row in data, and Y_j is the j-th row in new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which first partial of the rational quadratic kernel function plus
            the polynomial kernel function of degree 2 is to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (data.shape[0] * data.shape[1], new_data.shape[0]) with the ((i-1)*d+u, j)-th entry being
            partial_u k (X_i, Y_j).

        """
        
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        # ----------------------------------------------------------------------
        # Rational quadratic kernel part
        rq_kernel = np.repeat(self.rationalquad_kernel_gram_matrix(new_data=new_data), repeats=self.d, axis=0)
        multi_rq1 = np.repeat(self.data.flatten(), n).reshape(self.N * self.d, n)
        multi_rq2 = np.tile(new_data.T, (self.N, 1))
        rq_partial = (-2 * self.r1 * rq_kernel ** 2 * (multi_rq1 - multi_rq2) / self.bw ** 2)
        
        # ----------------------------------------------------------------------
        # Poly2 kernel part
        poly_part = np.repeat((self.poly2_kernel_gram_matrix(new_data=new_data) + self.c), repeats=self.d, axis=0)
        multi_poly = np.tile(new_data.T, (self.N, 1))
        poly_partial = 2 * self.r2 * poly_part * multi_poly
        
        output = rq_partial + poly_partial
        
        return output
    
    def partial_kernel_matrix_20(self, new_data):
        
        """
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0]) using
        the rational quadratic kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, j)-th entry being partial_u^2 k (X_i, Y_j),
        where partial_u^2 k denotes the second partial derivative of k with respect to the u-th coordinate
        of the first argument, X_i is the i-th row in data, and Y_j is the j-th row in new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which second partial of the rational quadratic kernel function plus
            the polynomial kernel function of degree 2 is to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (data.shape[0] * data.shape[1], new_data.shape[0]) with the ((i-1)*d+u, j)-th entry being
            partial_u^2 k (X_i, Y_j).

        """
        
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        # ----------------------------------------------------------------------
        # Rational quadratic kernel part
        rq_kernel = np.repeat(self.rationalquad_kernel_gram_matrix(new_data=new_data), repeats=self.d, axis=0)
        multi_rq_part1 = np.repeat(self.data.flatten(), n).reshape(self.N * self.d, n)
        multi_rq_part2 = np.tile(new_data.T, (self.N, 1))
        multi_rq_part = (multi_rq_part1 - multi_rq_part2) ** 2
        rq_partial = self.r1 * (-2 * rq_kernel ** 2 / self.bw ** 2 +
                                8 * multi_rq_part * rq_kernel ** 3 / self.bw ** 4)
        
        # ----------------------------------------------------------------------
        # Poly2 kernel part
        multi_poly = np.tile(new_data.T, (self.N, 1)) ** 2
        poly_partial = 2 * self.r2 * multi_poly
        
        output = rq_partial + poly_partial
        
        return output
    
    def partial_kernel_matrix_11(self, new_data):
        
        """
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) using
        the rational quadratic kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u partial_{v+d} k (X_i, Y_j),
        where partial_u partial_{v+d} k denotes the second mixed partial derivative of k,
        one partial derivative is taken with respect to the u-th coordinate of the first argument,
        the other is taken with respect to the v-th coordinate of the second argument,
        X_i is the i-th row in data, and Y_j is the j-th row in new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which second mixed partial of the rational quadratic kernel function plus
            the polynomial kernel function of degree 2 is to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1])
            with the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u partial_{v+d} k (X_i, Y_j).

        """
        
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        # ----------------------------------------------------------------------
        # Rational quadratic kernel part
        rq_kernel = np.kron(self.rationalquad_kernel_gram_matrix(new_data=new_data),
                            np.ones((self.d, self.d), dtype=np.float32))
        
        multi_rq11 = np.tile(self.data.flatten().reshape(-1, 1), (1, n * self.d))
        multi_rq12 = np.tile(np.repeat(new_data.T, repeats=self.d, axis=1), (self.N, 1))
        multi_rq1 = multi_rq11 - multi_rq12
        
        multi_rq21 = np.tile(np.repeat(self.data, repeats=self.d, axis=0), (1, n))
        multi_rq22 = np.repeat(new_data.flatten(), self.N * self.d).reshape(n * self.d, self.N * self.d).T
        multi_rq2 = multi_rq21 - multi_rq22
        
        multi_rq = multi_rq1 * multi_rq2 / self.bw ** 4
        
        # the matrix added when the coordinates wrt which the derivatives are taken are the same
        add_mat = np.kron(self.rationalquad_kernel_gram_matrix(new_data=new_data), np.eye(self.d, dtype=np.float32))
        
        rq_partial = - 8 * self.r1 * multi_rq * rq_kernel ** 3 + 2 * self.r1 * add_mat ** 2 / self.bw ** 2
        
        # ----------------------------------------------------------------------
        # Poly2 kernel part
        poly_part11 = np.tile(np.repeat(self.data, repeats=self.d, axis=0), (1, n))
        poly_part12 = np.tile(np.repeat(new_data.T, repeats=self.d, axis=1), (self.N, 1))
        poly_part1 = poly_part11 * poly_part12
        
        poly_part2 = np.kron(self.poly2_kernel_gram_matrix(new_data=new_data) + self.c,
                             np.eye(self.d, dtype=np.float64))
        
        poly_partial = 2. * self.r2 * (poly_part1 + poly_part2)
        
        output = rq_partial + poly_partial
        
        return output
    
    def partial_kernel_matrix_21(self, new_data):
        
        """
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) using
        the rational quadratic kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u^2 partial_{v+d} k (X_i, Y_j),
        where partial_u^2 partial_{v+d} k denotes the third mixed partial derivative of k,
        two partial derivatives are taken with respect to the u-th coordinate of the first argument,
        one partial derivative is taken with respect to the v-th coordinate of the second argument,
        X_i is the i-th row in data, and Y_j is the j-th row in new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which third mixed partial of the rational quadratic kernel function plus
            the polynomial kernel function of degree 2 is to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) with
            the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u^2 partial_{v+d} k (X_i, Y_j).

        """
        
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        # ----------------------------------------------------------------------
        # Rational quadratic kernel part
        rq_kernel = np.kron(self.rationalquad_kernel_gram_matrix(new_data=new_data),
                            np.ones((self.d, self.d), dtype=np.float64))
        
        multi_rq11 = np.tile(self.data.flatten().reshape(-1, 1), (1, n * self.d))
        multi_rq12 = np.tile(np.repeat(new_data.T, repeats=self.d, axis=1), (self.N, 1))
        multi_rq1 = (multi_rq11 - multi_rq12) ** 2
        
        multi_rq21 = np.tile(np.repeat(self.data, repeats=self.d, axis=0), (1, n))
        multi_rq22 = np.repeat(new_data.flatten(), self.N * self.d).reshape(n * self.d, self.N * self.d).T
        multi_rq2 = multi_rq21 - multi_rq22
        
        multi_rq = multi_rq1 * multi_rq2 / self.bw ** 6
        
        # the matrix added when the coordinates wrt which the derivatives are taken are the same
        add_mat = multi_rq2 * np.tile(np.eye(self.d, dtype=np.float32), (self.N, n)) / self.bw ** 4
        
        rq_partial = (48 * self.r1 * multi_rq * rq_kernel ** 4 -
                      8 * self.r1 * multi_rq2 / self.bw ** 4 * rq_kernel ** 3 -
                      16 * self.r1 * add_mat * rq_kernel ** 3)
        
        # ----------------------------------------------------------------------
        # Poly2 kernel part
        b = np.zeros((new_data.shape[0], new_data.shape[1], new_data.shape[1]))
        diag = np.arange(new_data.shape[1])
        b[:, diag, diag] = new_data
        b2 = np.hstack(b)
        poly_partial = 4 * self.r2 * np.tile(b2, (self.N, 1))
        
        output = rq_partial + poly_partial
        
        return output
    
    def partial_kernel_matrix_12(self, new_data):
        
        """
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) using
        the rational quadratic kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u partial_{v+d}^2 k (X_i, Y_j),
        where partial_u partial_{v+d}^2 k denotes the third mixed partial derivative of k,
        one partial derivative is taken with respect to the u-th coordinate of the first argument,
        two partial derivatives are taken with respect to the v-th coordinate of the second argument,
        X_i is the i-th row in data, and Y_j is the j-th row in new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which third mixed partial of the rational quadratic kernel function plus
            the polynomial kernel function of degree 2 is to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) with
            the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u partial_{v+d}^2 k (X_i, Y_j).

        """
        
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        # ----------------------------------------------------------------------
        # Rational quadratic kernel part
        rq_kernel = np.kron(self.rationalquad_kernel_gram_matrix(new_data=new_data),
                            np.ones((self.d, self.d), dtype=np.float32))
        
        multi_rq11 = np.tile(self.data.flatten().reshape(-1, 1), (1, n * self.d))
        multi_rq12 = np.tile(np.repeat(new_data.T, repeats=self.d, axis=1), (self.N, 1))
        multi_rq1 = multi_rq11 - multi_rq12
        
        multi_rq21 = np.tile(np.repeat(self.data, repeats=self.d, axis=0), (1, n))
        multi_rq22 = np.repeat(new_data.flatten(), self.N * self.d).reshape(n * self.d, self.N * self.d).T
        multi_rq2 = (multi_rq21 - multi_rq22) ** 2
        
        multi_rq = multi_rq1 * multi_rq2 / self.bw ** 6
        
        # the matrix added when the coordinates wrt which the derivatives are taken are the same
        add_mat = multi_rq1 * np.tile(np.eye(self.d, dtype=np.float64), (self.N, n)) / self.bw ** 4
        
        rq_partial = (-48 * self.r1 * multi_rq * rq_kernel ** 4 +
                      8 * self.r1 * multi_rq1 / self.bw ** 4 * rq_kernel ** 3 +
                      16 * self.r1 * add_mat * rq_kernel ** 3)
        
        # ----------------------------------------------------------------------
        # Poly2 kernel part
        b = np.zeros((self.data.shape[0], self.data.shape[1], self.data.shape[1]))
        diag = np.arange(self.data.shape[1])
        b[:, diag, diag] = self.data
        b2 = np.vstack(b)
        poly_partial = 4 * self.r2 * np.tile(b2, (1, n))
        
        output = rq_partial + poly_partial
        
        return output
    
    def partial_kernel_matrix_22(self, new_data):
        
        """
        Computes the matrix of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) using
        the rational quadratic kernel function plus the polynomial kernel function of degree 2,
        with the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u^2 partial_{v+d}^2 k (X_i, Y_j),
        where partial_u^2 partial_{v+d}^2 k denotes the fourth mixed partial derivative of k,
        two partial derivatives are taken with respect to the u-th coordinate of the first argument,
        two partial derivatives are taken with respect to the v-th coordinate of the second argument,
        X_i is the i-th row in data, and Y_j is the j-th row in new_data.

        Parameters
        ----------
        new_data : numpy.ndarray
            A new data array at which fourth mixed partial of the rational quadratic kernel function plus
            the polynomial kernel function of degree 2 is to be evaluated.

        Returns
        -------
        numpy.ndarray
            An array of shape (data.shape[0] * data.shape[1], new_data.shape[0] * new_data.shape[1]) with
            the ((i-1)*d+u, (j-1)*d+v)-th entry being partial_u^2 partial_{v+d}^2 k (X_i, Y_j).

        """
        
        n, d1 = new_data.shape
        
        assert self.d == d1, "The dimensionality of new_data does not match that of data. "
        
        # ----------------------------------------------------------------------
        # Rational quadratic kernel part
        rq_kernel = np.kron(self.rationalquad_kernel_gram_matrix(new_data=new_data),
                            np.ones((self.d, self.d), dtype=np.float64))
        
        multi_rq11 = np.tile(self.data.flatten().reshape(-1, 1), (1, n * self.d))
        multi_rq12 = np.tile(np.repeat(new_data.T, repeats=self.d, axis=1), (self.N, 1))
        multi_rq1 = (multi_rq11 - multi_rq12) ** 2
        
        multi_rq21 = np.tile(np.repeat(self.data, repeats=self.d, axis=0), (1, n))
        multi_rq22 = np.repeat(new_data.flatten(), self.N * self.d).reshape(n * self.d, self.N * self.d).T
        multi_rq2 = (multi_rq21 - multi_rq22) ** 2
        
        rq_partial = (self.r1 * (384 * multi_rq1 * multi_rq2 * rq_kernel ** 5 / self.bw ** 8 -
                                 48 * multi_rq1 * rq_kernel ** 4 / self.bw ** 6 -
                                 48 * multi_rq2 * rq_kernel ** 4 / self.bw ** 6 +
                                 8 * rq_kernel ** 3 / self.bw ** 4) +
                      self.r1 * (-192 * multi_rq1 * rq_kernel ** 4 / self.bw ** 6 +
                                 16 * rq_kernel ** 3 / self.bw ** 4) *
                      np.tile(np.eye(self.d, dtype=np.float64), (self.N, n)))
        
        # ----------------------------------------------------------------------
        # Poly2 kernel part
        poly_partial = 4 * self.r2 * np.tile(np.eye(self.d, dtype=np.float64), (self.N, n))
        
        output = rq_partial + poly_partial
        
        return output
    
    def kernel_x_1d(self, landmark):
        
        """
        Returns a function that computes k (landmark, y) at y, where k is the rational quadratic kernel function plus
        the polynomial kernel function of degree 2, both landmark and y are 1-dimensional data points.

        Parameters
        ----------
        landmark : float or np.ndarray
            A floating point number or a data array of shape (1,).

        Returns
        -------
        function
            A function that computes k (landmark, y) at y.
        """
        
        landmark = np.array(landmark).flatten()
        if len(landmark) != 1:
            raise ValueError("The length of landmark should be 1.")
        
        def output(x):
            y = (self.r1 * (1 + (x - landmark) ** 2 / self.bw ** 2) ** (-1) +
                 self.r2 * (x * landmark + self.c) ** 2)
            return y
        
        return output
    
    def kernel_x_1d_deriv1(self, landmark):
        
        """
        Returns a function that computes partial_1 k (landmark, y) at y, where k is the rational quadratic kernel
        function plus the polynomial kernel function of degree 2, both landmark and y are 1-dimensional data points.
        
        Parameters
        ----------
        landmark : float or np.ndarray
            A floating point number or a data array of shape (1,).

        Returns
        -------
        function
            A function that computes partial_1 k (landmark, y) at y.

        """
        
        landmark = np.array(landmark).flatten()
        if len(landmark) != 1:
            raise ValueError("The length of landmark should be 1.")
        
        def output(y):
            normsq = (y - landmark) ** 2
            ker = (-2. * self.r1 * (landmark - y) / (self.bw ** 2 * (1 + normsq / self.bw ** 2) ** 2) +
                   2 * self.r2 * (y * landmark + self.c) * y)
            
            return ker
        
        return output
    
    def kernel_x_1d_deriv2(self, landmark):
        
        """
        Returns a function that computes partial_1^2 k (x, landmark) at y, where k is the rational quadratic kernel
        function plus the polynomial kernel function of degree 2, both landmark and y are 1-dimensional data points.
        
        Parameters
        ----------
        landmark : float or np.ndarray
            A floating point number or a data array of shape (1,).

        Returns
        -------
        function
            A function that computes partial_1^2 k (landmark, y) at y.

        """
        
        landmark = np.array(landmark).flatten()
        if len(landmark) != 1:
            raise ValueError("The length of landmark should be 1.")
        
        def output(y):
            normsq = (y - landmark) ** 2
            e_term = 1 + normsq / self.bw ** 2
            ker = (- self.r1 * 2 / (e_term ** 2 * self.bw ** 2) +
                   self.r1 * 8 * normsq / (e_term ** 3 * self.bw ** 4) +
                   2 * self.r2 * y ** 2)
            
            return ker
        
        return output
    
    def kernel_x_2d(self, landmark):
        
        """
        Returns a function that computes k (landmark, y) at y, where k is the rational quadratic kernel function plus
        the polynomial kernel function of degree 2, both landmark and y are 2-dimensional data points.

        Parameters
        ----------
        landmark : np.ndarray
            A data array of shape (2,).

        Returns
        -------
        function
            A function that computes k (landmark, y) at y.

        """
        
        landmark = np.array(landmark).flatten()
        if len(landmark) != 2:
            raise ValueError("The length of landmark should be 2.")
        
        def output(x0, x1):
            y = (self.r1 * (1 + ((x0 - landmark[0]) ** 2 + (x1 - landmark[1]) ** 2) / self.bw ** 2) ** (-1) +
                 self.r2 * (x0 * landmark[0] + x1 * landmark[1] + self.c) ** 2)
            return y
        
        return output
    
    def kernel_x_2d_deriv1_0(self, landmark):
        
        """
        Returns a function that computes partial_1 k (landmark, y) at y, where k is the rational quadratic kernel
        function plus the polynomial kernel function of degree 2, both landmark and y are 2-dimensional data points.
        
        Parameters
        ----------
        landmark : np.ndarray
            A data array of shape (2,).

        Returns
        -------
        function
            A function that computes partial_1 k (landmark, y) at y.

        """
        
        landmark = np.array(landmark).flatten()
        if len(landmark) != 2:
            raise ValueError("The length of landmark should be 2.")
        
        def output(y0, y1):
            e_term = 1 + ((y0 - landmark[0]) ** 2 + (y1 - landmark[1]) ** 2) / self.bw ** 2
            ker_val = (-2 * self.r1 * (landmark[0] - y0) / (self.bw ** 2 * e_term ** 2) +
                       2 * self.r2 * (y0 * landmark[0] + y1 * landmark[1] + self.c) * y0)
            return ker_val
        
        return output
    
    def kernel_x_2d_deriv1_1(self, landmark):
        
        """
        Returns a function that computes partial_2 k (landmark, y) at y, where k is the rational quadratic kernel
        function plus the polynomial kernel function of degree 2, both landmark and y are 2-dimensional data points.
        
        Parameters
        ----------
        landmark : np.ndarray
            A data array of shape (2,).

        Returns
        -------
        function
            A function that computes partial_2 k (landmark, y) at y.

        """
        
        landmark = np.array(landmark).flatten()
        if len(landmark) != 2:
            raise ValueError("The length of landmark should be 2.")
        
        def output(y0, y1):
            e_term = 1 + ((y0 - landmark[0]) ** 2 + (y1 - landmark[1]) ** 2) / self.bw ** 2
            ker_val = (-2 * self.r1 * (landmark[1] - y1) / (self.bw ** 2 * e_term ** 2) +
                       2 * self.r2 * (y0 * landmark[0] + y1 * landmark[1] + self.c) * y1)
            
            return ker_val
        
        return output
    
    def kernel_x_2d_deriv2_0(self, landmark):
        
        """
        Returns a function that computes partial_1^2 k (landmark, y) at y, where k is the rational quadratic kernel
        function plus the polynomial kernel function of degree 2, both landmark and y are 2-dimensional data points.
        
        Parameters
        ----------
        landmark : np.ndarray
            A data array of shape (2,).

        Returns
        -------
        function
            A function that computes partial_1^2 k (landmark, y) at y.

        """
        
        landmark = np.array(landmark).flatten()
        if len(landmark) != 2:
            raise ValueError("The length of landmark should be 2.")
        
        def output(y0, y1):
            e_term = 1 + ((y0 - landmark[0]) ** 2 + (y1 - landmark[1]) ** 2) / self.bw ** 2
            
            ker_val = (self.r1 * 8 * (landmark[0] - y0) ** 2 / (self.bw ** 4 * e_term ** 3) -
                       self.r1 * 2 / (self.bw ** 2 * e_term ** 2) +
                       2 * self.r2 * y0 ** 2)
            return ker_val
        
        return output
    
    def kernel_x_2d_deriv2_1(self, landmark):
        
        """
        Returns a function that computes partial_2^2 k (landmark, y) at y, where k is the rational quadratic kernel
        function plus the polynomial kernel function of degree 2, both landmark and y are 2-dimensional data points.

        Parameters
        ----------
        landmark : np.ndarray
            A data array of shape (2,).

        Returns
        -------
        function
            A function that computes partial_2^2 k (landmark, y) at y.

        """
        
        landmark = np.array(landmark).flatten()
        if len(landmark) != 2:
            raise ValueError("The length of landmark should be 2.")
        
        def output(y0, y1):
            e_term = 1 + ((y0 - landmark[0]) ** 2 + (y1 - landmark[1]) ** 2) / self.bw ** 2
            
            ker_val = (self.r1 * 8 * (landmark[1] - y1) ** 2 / (self.bw ** 4 * e_term ** 3) -
                       self.r1 * 2 / (self.bw ** 2 * e_term ** 2) +
                       2 * self.r2 * y1 ** 2)
            return ker_val
        
        return output
    
    def kernel_x_3d(self, landmark):
        
        """
        Returns a function that computes k (landmark, y) at y, where k is the rational quadratic kernel function plus
        the polynomial kernel function of degree 2, both landmark and y are 3-dimensional data points.

        Parameters
        ----------
        landmark : np.ndarray
            A data array of shape (3,).

        Returns
        -------
        function
            A function that computes k (landmark, y) at y.

        """
        
        landmark = np.array(landmark).flatten()
        if len(landmark) != 3:
            raise ValueError("The length of landmark should be 3.")
        
        def output(x0, x1, x2):
            y = (self.r1 * (1 + ((x0 - landmark[0]) ** 2 + (x1 - landmark[1]) ** 2 + (x2 - landmark[2]) ** 2) /
                            self.bw ** 2) ** (-1) +
                 self.r2 * (x0 * landmark[0] + x1 * landmark[1] + x2 * landmark[2] + self.c) ** 2)
            return y
        
        return output
