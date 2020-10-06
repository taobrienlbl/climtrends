""" A module for Bayesian parameter estimation of a normal distribution with trend in the mean. """

import numpy as np
import numba
import scipy.stats
from .ClimTrendModel import ClimTrendModel
from .utility_functions import log_normal_fast, normal_ppf_fast, normal_cdf_fast
       
class TrendNormalModel(ClimTrendModel):
    """ A simple normal distribution with trend.
    
    
        Parameters:
        
            theta = ( mean_slope, mean_intercept, variance )
    
    """
    
    def __init__(self, *args, **kwargs):
        
        # run the base class initialization
        super().__init__(*args, **kwargs)
        
        self.num_parameters = 3
        self.parameter_is_slope = [True, False, False]
        self.parameter_is_intercept = [False, True, False]
        self.parameter_is_positive_definite = [ False, False, True ]
        self.parameter_labels = ("$c_{\mu}$", "$\mu_0$", "$\sigma^2$")
        self.sanity_check()
        
    
    def log_likelihood(self, theta):
        
        cmu, mu0, var = theta
        
        mu = cmu*self.x + mu0
        
        return np.sum(log_normal_fast(self.y, mu, var))
    
    def calculate_mean_values(self, dates):
        
        # get the MCMC samples
        parameter_samples = self.get_mcmc_samples()
        
        # convert the input dates to times
        times = self.dates_to_xvalues(dates)
        
        # check that the MCMC sampler has been run
        if self.sampler is not None:
            
            # get the mean value
            mean_values = parameter_samples[0,:][:,np.newaxis]*times[np.newaxis,:] + parameter_samples[1][:,np.newaxis]
            
            # exponentiate the value if we are using an exponential trend model
            if self.use_exponential_model:
                mean_values = np.exp(mean_values)
        else:
            raise RuntimeError("the `run_mcmc_sampler()' method must be called prior to calling calculate_mean_values()'")
            
        return mean_values
    
    def get_percentile_of_percentile_at_time(self,
                                             dates,
                                             percentile,
                                             model_percentile,
                                             nskip = 100):
        
        # get the MCMC samples
        parameter_samples = self.get_mcmc_samples()
        
        # subset the MCMC samples to save time
        skip_slice = slice(None,None,nskip)
        parameter_samples = parameter_samples[:,skip_slice]
        
        # convert the input dates to times
        times = self.dates_to_xvalues(dates)
        
        # check that the MCMC sampler has been run
        if self.sampler is not None:
            
            # get the mean value
            
            mu  = parameter_samples[0][:,np.newaxis]*times[np.newaxis,:] + parameter_samples[1][:,np.newaxis]
            mu  = mu*np.ones([len(parameter_samples[0,:]),len(dates)])
            var = parameter_samples[2,:][:,np.newaxis]*np.ones([len(parameter_samples[0,:]),len(dates)])
            
            model_values = normal_ppf_fast(model_percentile, mu, var)
            
            # get the percentile of that value
            values = np.percentile(model_values, percentile, axis = 0)
            
            # exponentiate the value if we are using an exponential trend model
            if self.use_exponential_model:
                values = np.exp(values)
        else:
            raise RuntimeError("the `run_mcmc_sampler()' method must be called prior to calling get_percentile_of_mean_at_time()'")
            
        return values
 
    def generate_samples(self,
                         theta,
                         dates):
        # get the parameters
        cmu, mu0, var = theta
        
        # convert the dates to the internal years-since format
        times = self.dates_to_xvalues(dates)
        
        # vectorize the mean and variance
        mu  = cmu*times + mu0
        var = var*np.ones(np.shape(dates))
        
        # generate the samples
        samples = np.reshape(np.array( [ scipy.stats.norm.rvs(loc = m, scale = np.sqrt(v)) for m, v in zip(mu, var)] ), 
                             np.shape(dates))
        
        return samples
    
    def calculate_cdf_values(self,
                             value,
                             date):
        """ Calculates the posterior distribution of the probability of values greater than 'value' for all parameter samples at the given dates.
        
            input:
            ------
            
                value      : the value at which to evaluate the cumulative distribution function
            
                date       : the input abscissa (time) value.  This should be a datetime-like objects. 
                
            output:
            -------
            
                cdf_values : a numpy array of shape [num_samples] containing the CDF at `value`
                              for each MCMC sample at the given date
        
        
        """
        
        # exponentiate the value if we are using an exponential trend model
        if self.use_exponential_model:
            value = np.exp(value)
            
        # get the MCMC samples
        parameter_samples = self.get_mcmc_samples()
        
        # convert the input date to time
        time = self.dates_to_xvalues(date)
        
        # check that the MCMC sampler has been run
        if self.sampler is not None:
            
            # get the mean value
            
            mu  = parameter_samples[0,:]*time + parameter_samples[1,:]
            var = parameter_samples[2,:]
            
            cdf_values = normal_cdf_fast(value, mu, var)
        else:
            raise RuntimeError("the `run_mcmc_sampler()' method must be called prior to calling calculate_cdf_values()'")
            
        return cdf_values
        