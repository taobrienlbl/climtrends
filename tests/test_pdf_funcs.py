try:
    import climtrends.utility_functions as utility_functions
except:
    # if import fails, assume we are running from the test directory
    import sys
    sys.path.insert("../")
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
            # generate a GEV sample with these parameters
            x = scipy.stats.genextreme.rvs(loc = mu, scale = sigma, c = xi)

            # calculate the log of the pdf using scipy
            scipy_pdf = scipy.stats.genextreme.logpdf(x, loc = mu, scale = sigma, c = xi)

            # calculate the log of the pdf using our routines
            test_pdf = utility_functions.log_gev_pdf_fast(x, mu, sigma, xi)
            
            # test if they are close
            if not np.isclose(test_pdf - scipy_pdf, 0.0):
                print(test_pdf, scipy_pdf, x, mu, sigma, xi)

            # generate a random value between 0 and 1
            q = np.random.uniform()
            # calculate the quantile using scipy
            scipy_ppf = scipy.stats.genextreme.ppf(q, loc = mu, scale = sigma, c = xi)

            # calculate the quantile using our routines
            test_ppf = utility_functions.gev_ppf_fast(q*100, mu, sigma, xi)

            # test that they are close
            if not np.isclose(test_ppf - scipy_ppf, 0.0):
                print(test_ppf, scipy_ppf, x, mu, sigma, xi)
