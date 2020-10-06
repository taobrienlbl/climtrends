""" A module for Bayesian parameter estimation of a normal distribution. """
import numpy as np
import scipy.stats
import numba
from .ClimTrendModel import ClimTrendModel
from .utility_functions import log_normal_fast, normal_ppf_fast, normal_cdf_fast

class NoTrendNormalModel(ClimTrendModel):
    """ A simple normal distribution with no trend.
    
    
        Parameters:
        
            theta = ( mean, variance )
    
    """
    
    def __init__(self, *args, **kwargs):
        
        # run the base class initialization
        super().__init__(*args, **kwargs)
        
        self.num_parameters = 2
        self.parameter_is_slope = self.num_parameters*[False]
        self.parameter_is_intercept = [True, False]
        self.parameter_is_positive_definite = [ False, True ]
        self.parameter_labels = ("$\mu$", "$\sigma^2$")
        self.sanity_check()
        
    
    def log_likelihood(self, theta):
        
        mu, var = theta
        
        return np.sum(log_normal_fast(self.y, mu, var))
    
    def calculate_mean_values(self, dates):
        
        # get the MCMC samples
        parameter_samples = self.get_mcmc_samples()
        
        # check that the MCMC sampler has been run
        if self.sampler is not None:
            
            # get the mean value
            mu = parameter_samples[0,:]
            mean_values = mu[:, np.newaxis]*np.ones(len(dates))[np.newaxis, :]
            
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
        
        
        # check that the MCMC sampler has been run
        if self.sampler is not None:
            
            # get the mean value
            mu  = parameter_samples[0,:][:,np.newaxis]*np.ones([len(parameter_samples[0,:]),len(dates)])
            var = parameter_samples[1,:][:,np.newaxis]*np.ones([len(parameter_samples[0,:]),len(dates)])
            
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
        mu, var = theta
        
        # convert the dates to the internal years-since format
        times = self.dates_to_xvalues(dates)
        
        # vectorize the mean and variance
        mu  = mu*np.ones(np.shape(dates))
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
        
        # check that the MCMC sampler has been run
        if self.sampler is not None:
            
            # get the mean value
            mu  = parameter_samples[0,:]
            var = parameter_samples[1,:]
            
            cdf_values = normal_cdf_fast(value, mu, var)
        else:
            raise RuntimeError("the `run_mcmc_sampler()' method must be called prior to calling calculate_cdf_values()'")
            
        return cdf_values