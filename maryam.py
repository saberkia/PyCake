import numpy as np

""" On behalf of Maryam Mirzakhani """


def gauss_kernel(x, mu, sigma):
    """Generating Gaussian Kernel with input values.

    Parameters
    ----------
    x : int or float
        input value x
    mu : ndarray
        mean of data (Gaussian Kernel)
    sigma : ndarray
        standard deviation (std) of data (Gaussian Kernel)

    Returns
    -------
    float
        the probability of x calculated by the Gaussian Kernel.
    """
    return np.exp(-(x-mu)**2/(2*sigma**2))
