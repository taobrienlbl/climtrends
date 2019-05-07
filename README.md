# ClimTrends

`ClimTrends` is a python package aimed at making it easy to calculate linear
trends in a variety of statistical modesl.  The current implementation
includes:

  * Normal Distribution - trend in mean
  * Poisson Distribution - trend in mean
  * Exponential Distribution - trend in mean
  * Gamma Distribution - trend in mean and standard deviation
  * GEV Distribution - trend in mean

The module utilizes `datetime`-like objects as the input time value, which
makes it easy to interoperate with data from
[netCDF4](http://unidata.github.io/netcdf4-python/) and
[xarray](http://xarray.pydata.org/en/stable/). All models use a Bayesian
framework (using the [emcee](http://dfm.io/emcee/current/) package) and assume
a uniform prior on all model parameters. This module is object-oriented and
designed to be easily extendable for regressions of other distributions.  To do
so, one needs to sub-class the `ClimTrendModel` class and implement a few
required routines; see `TrendNormalModel.py` or other `Trend*Model.py` files
for examples.

# Getting started

```python
# dates - a set of input dates
# data - corresponding data from those dates

import ClimTrends
import numpy as np

# initialize the MCMC model
linear_model = ClimTrends.TrendNormalModel(dates, data)

# run the sampler
linear_model.run_mcmc_sampler(num_samples = 1000)

# get samples of the slopes
slopes = linear_model.get_mean_trend_samples()

# get the 5th and 95th percentil slopes
slopes_5 = np.percentile(slopes, 5)
slopes_95 = np.percentile(slopes, 95)
```

# Known Issues

* The install process is not finalized yet; a setup.py needs to be written.
* The model assumes that input data are vectors; things will likely break if not.
* Probably other issues exist.  This code is tested and is verified to work in some base cases,
  but it is still in alpha stage and has not been tested across a range of settings.
