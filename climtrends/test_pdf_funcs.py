import utility_functions
import scipy.stats
import numpy as np


xi_vals = np.linspace(-0.5, 0.5, 10)
mu_vals = np.linspace(0.1, 10, 10)
sigma_vals = np.linspace(0.1, 5, 10)


for xi in xi_vals:
    for mu in mu_vals:
        for sigma in sigma_vals:
            x = scipy.stats.genextreme.rvs(loc = mu, scale = sigma, c = xi)
            
            scipy_pdf = scipy.stats.genextreme.logpdf(x, loc = mu, scale = sigma, c = xi)

            test_pdf = utility_functions.log_gev_pdf_fast(x, mu, sigma, xi)
            
            if not np.isclose(test_pdf - scipy_pdf, 0.0):
                print(test_pdf, scipy_pdf, x, mu, sigma, xi)

            q = np.random.uniform()
            scipy_ppf = scipy.stats.genextreme.ppf(q, loc = mu, scale = sigma, c = xi)

            test_ppf = utility_functions.gev_ppf_fast(q*100, mu, sigma, xi)

            if not np.isclose(test_ppf - scipy_ppf, 0.0):
                print(test_ppf, scipy_ppf, x, mu, sigma, xi)