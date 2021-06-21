try:
    import sys
    sys.path.insert(0,"../")
    import climtrends.utility_functions as utility_functions
except:
    # if above import fails, assume we are testing an install
    import climtrends.utility_functions as utility_functions
import scipy.stats
import numpy as np


# generate a range of GEV parameter values
xi_vals = np.linspace(-0.5, 0.5, 10)
mu_vals = np.linspace(0.1, 10, 10)
sigma_vals = np.linspace(0.1, 5, 10)


# loop over parameter values
for xi in xi_vals:
    for mu in mu_vals:
        for sigma in sigma_vals:
            # **************************
            #  gev_cdf_fast()
            # **************************
            # generate a random value
            x = np.random.uniform(low = -100, high = 100)
            
            #calculate the CDF using scipy
            scipy_cdf = scipy.stats.genextreme.cdf(x, loc = mu, scale = sigma, c = xi)
            
            # calculate the CDF using our routines
            test_cdf = utility_functions.gev_cdf_fast(x, mu, sigma, xi)
            
            # test if they are close
            if not np.isclose(test_cdf - scipy_cdf, 0.0):
                print("gev_cdf_fast: ", test_cdf, scipy_cdf, x, mu, sigma, xi)
                quit()

            
            # **************************
            #  log_gev_pdf_fast()
            # **************************
            # generate a GEV sample with these parameters
            x = scipy.stats.genextreme.rvs(loc = mu, scale = sigma, c = xi)

            # calculate the log of the pdf using scipy
            scipy_pdf = scipy.stats.genextreme.logpdf(x, loc = mu, scale = sigma, c = xi)

            # calculate the log of the pdf using our routines
            test_pdf = utility_functions.log_gev_pdf_fast(x, mu, sigma, xi)
            
            # test if they are close
            if not np.isclose(test_pdf - scipy_pdf, 0.0):
                print("log_gev_pdf_fast: ", test_pdf, scipy_pdf, x, mu, sigma, xi)

            # **************************
            #  gev_ppf_fast()
            # **************************
            # generate a random value between 0 and 1
            q = np.random.uniform()
            # calculate the quantile using scipy
            scipy_ppf = scipy.stats.genextreme.ppf(q, loc = mu, scale = sigma, c = xi)

            # calculate the quantile using our routines
            test_ppf = utility_functions.gev_ppf_fast(q*100, mu, sigma, xi)

            # test that they are close
            if not np.isclose(test_ppf - scipy_ppf, 0.0):
                print("gev_ppf_fast: ", test_ppf, scipy_ppf, x, mu, sigma, xi)
                
