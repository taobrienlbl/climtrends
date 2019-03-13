""" Functions for helping to calculate PDF quantities. """
import numba
import numpy as np
import numba.extending
import scipy.special
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
    
@numba.vectorize([numba.int64(numba.int64)],)
def fast_factorial(n):
    if n == 0:
        return 0
    
    retval = 1
    for n in range(2, n+1):
        retval *= n
        
    return retval
@numba.vectorize([numba.float64(numba.int64, numba.float64)])
def log_poisson_pdf_fast(N,mu):
    """ Define a fast version of the log poisson. """
    return -mu + N*np.log(mu) -np.log(fast_factorial(N))


# A helper function for quickly calculating the PPF of a poisson distribution
addr = numba.extending.get_cython_function_address("scipy.special.cython_special", "pdtrik")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
pdtrik_fn = functype(addr)
addr = numba.extending.get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_0pdtr")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
pdtr_fn = functype(addr)
@numba.vectorize([numba.float64(numba.float64, numba.float64)])
def poisson_ppf_fast(F, mu):
    """ A fast calculation of the quantile function of a poisson distribution. """
    q = F/100
    vals = np.ceil(pdtrik_fn(q, mu))
    vals1 = np.maximum(vals - 1, 0)
    temp = pdtr_fn(vals1, mu)
    
    retval = vals
    if temp >= q:
        retval = vals1
    return retval

@numba.vectorize([numba.float64(numba.float64, numba.float64)])
def log_exponential_pdf_fast(x,mu):
    """ Define a fast version of the log exponential. """
    return np.log(mu) - mu*x

@numba.vectorize([numba.float64(numba.float64, numba.float64)])
def exponential_ppf_fast(F,mu):
    """ Define a fast version of the exponential percentile function. """
    return -np.log(1 - F/100)/mu

addr = numba.extending.get_cython_function_address("scipy.special.cython_special", "gammaln")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
gammaln_fn = functype(addr)
addr = numba.extending.get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1xlogy")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
xlogy_fn = functype(addr)
@numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64)])
def log_gamma_pdf_fast(x, a, b):
    return xlogy_fn(a-1.0, x*b) - x*b - gammaln_fn(a) + np.log(b)

addr = numba.extending.get_cython_function_address("scipy.special.cython_special", "gammaincinv")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
gammaincinv_fn = functype(addr)
@numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64)])
def gamma_ppf_fast(F, alpha, beta):
    """ A fast version of the gamma percentile function"""
    return gammaincinv_fn(alpha, F/100)/beta


@numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64, numba.float64)])
def log_gev_pdf_fast(x, mu, sigma, xi):
    """ A fast version of the log of the GEV distribution. """
    if xi == 0:
        t = np.exp(-(x-mu)/sigma)
    else:
        t = (1 + xi*(x-mu)/sigma)**(-1/xi)
        
    return  xlogy_fn(xi + 1, t) - np.log(sigma) - t

@numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64, numba.float64)])
def gev_ppf_fast(F, mu, sigma, xi):
    """ A fast version of the GEV percentile function. """
    
    if xi == 0:
        coef = -np.log(-np.log(F/100))
    else:
        coef = ((-np.log(F/100))**(-xi) - 1)/xi
        
    return mu + sigma*coef