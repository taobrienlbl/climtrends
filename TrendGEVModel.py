""" A module for Bayesian parameter estimation of a GEV distribution with trend in the location parameter. """

import numpy as np
import numba
import scipy.stats
from .ClimTrendModel import ClimTrendModel
from .utility_functions import log_gev_pdf_fast, gev_ppf_fast
       
class TrendGEVModel(ClimTrendModel):
    """ A GEV distribution with trend in the location parameter
    
        Parameters:
        
            theta = ( mu_slope, mu_intercept, sigma, xi )
    
    """
    
    def __init__(self, *args, **kwargs):
        
        # run the base class initialization
        super().__init__(*args, **kwargs)
        
        self.num_parameters = 4
        self.parameter_is_slope = [True, False, False, False]
        self.parameter_is_intercept = [False, True, False, False ]
        self.parameter_is_positive_definite = [ False, False, True, False ]
        self.parameter_labels = (r"$c_{\mu}$", r"$\mu_0$", r"$\sigma$", r"$\xi$")
        self.sanity_check()
        
    
    def log_likelihood(self, theta):
        
        cmu, mu0, sigma, xi = theta
        
        mu = cmu*self.x + mu0
        
        return np.sum(log_gev_pdf_fast(self.y, mu, sigma, xi))
    
    def calculate_mean_values(self, dates):
        
        # get the MCMC samples
        parameter_samples = self.get_mcmc_samples()
        
        # convert the input dates to times
        times = self.dates_to_xvalues(dates)
        
        # check that the MCMC sampler has been run
        if self.sampler is not None:
            
            # calculate the location, rate,  and shape parameters
            mu = parameter_samples[0,:][:,np.newaxis]*times[np.newaxis,:] + parameter_samples[1,:][:,np.newaxis]
            sigma = parameter_samples[2,:][:,np.newaxis]*np.ones([len(parameter_samples[0,:]),len(dates)])
            xi = parameter_samples[3,:][:,np.newaxis]*np.ones([len(parameter_samples[0,:]),len(dates)])
             
            # calculate the median instead of the mean, since the mean may be undefined
            coef = np.where(xi != 0, (np.log(2)**(-xi) - 1)/xi, -np.log(np.log(2)))
            
            median_values = mu + sigma*coef
            
            # exponentiate the value if we are using an exponential trend model
            if self.use_exponential_model:
                median_values = np.exp(median_values)
        else:
            raise RuntimeError("the `run_mcmc_sampler()' method must be called prior to calling calculate_mean_values()'")
            
        return median_values
    
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
            
            # calculate the location, rate,  and shape parameters
            mu = parameter_samples[0][:,np.newaxis]*times[np.newaxis,:] + parameter_samples[1][:,np.newaxis]
            sigma = parameter_samples[2,:][:,np.newaxis]*np.ones([len(parameter_samples[0,:]),len(dates)])
            xi = parameter_samples[3,:][:,np.newaxis]*np.ones([len(parameter_samples[0,:]),len(dates)])
            
            model_values = gev_ppf_fast(model_percentile, mu, sigma, xi)
            
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
        cmu, mu0, sigma, xi = theta
        
        # convert the dates to the internal years-since format
        times = self.dates_to_xvalues(dates)
        
        # vectorize the shape and rate parameters
        mu  = cmu*times + mu0
        
        # generate the samples
        samples = np.reshape(np.array( [ scipy.stats.genextreme.rvs(c = -xi, loc = m, scale = sigma) for m in mu ] ), 
                             np.shape(dates))
        
        return samples
    
    def get_starting_parameters(self):
        
        # do a first pass at the starting parameters
        starting_parameters = super().get_starting_parameters()
        
        # initialize xi specially
        starting_parameters[:,-1] = np.random.uniform(low=0, high = 1, size = self.num_walkers)
        
        return starting_parameters
        
        
 
