""" Functions for helping to calculate PDF quantities. """
import numba
import numpy as np
import numba.extending
import ctypes

# make a numba-ready version of erfinv()
# This follows https://github.com/numba/numba/issues/3086#issuecomment-403469308
addr = numba.extending.get_cython_function_address("scipy.special.cython_special", "ndtri")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
ndtri_fn = functype(addr)
@numba.vectorize([numba.float64(numba.float64)])
def erfinv(y):
    """numba ufunc implementation of erfinv"""
    return ndtri_fn((y+1)/2.0)/np.sqrt(2)

@numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64)])
def log_normal_fast(x, mu, var):
    """  Returns the log of a normal distribution using numpy ufuncs
    
        input:
        ------
            
            x    : the input value to the log of the normal distribution
            
            mu   : the mean of the normal distribution
            
            var  : the variance of the normal distribution
            
        output:
        -------
            
            log_normal : the log of the normal distribution
    
    """
    
    return -0.5*np.log(var*2*np.pi) - (x - mu)**2 / (2*var)
    
@numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64)])
def normal_ppf_fast(F, mu, var):
    """  Returns the the quantile of percentile value F for a normal distribution
    
        input:
        ------
            
            F    : the percentile value [0-100]
            
            mu   : the mean of the normal distribution
            
            var  : the variance of the normal distribution
            
        output:
        -------
            
            quantile : the value corresponding to percentile F in the normal distribution
    
    """
    
    quantile = mu + np.sqrt(2*var)*erfinv(float(2*F/100 - 1))
    
    return quantile
    
@numba.vectorize([numba.float64(numba.float64, numba.float64)])
def log_poisson_pdf_fast(N,mu):
    """ Define a fast version of the log poisson. """
    return -mu + N*np.log(mu) -np.log(scipy.special.factorial(N, exact=False))


# A helper function for quickly calculating the PPF of a poisson distribution
addr = numba.extending.get_cython_function_address("scipy.special.cython_special", "pdtrik")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
pdtrik_fn = functype(addr)
addr = numba.extending.get_cython_function_address("scipy.special.cython_special", "pdtr")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
pdtr_fn = functype(addr)
@numba.vectorize([numba.float64(numba.float64, numba.float64)])
def poisson_ppf_fast(F, mu):
    """ A fast calculation of the quantile function of a poisson distribution. """
    q = F/100
    vals = np.ceil(pdtrik_fn(q, mu))
    vals1 = np.maximum(vals - 1, 0)
    temp = pdtr_fn(vals1, mu)
    return np.where(temp >= q, vals1, vals)





