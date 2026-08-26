# A Black Carbon Power-law Degradation Model (BC-PDM)

This repository provides the Python code used to develop and constrain a quantitative degradation model for black carbon (BC) preserved in marine sediments over geological timescales.

The model integrates multi-site sedimentary BC records with a common regional input function and a power-law degradation function. The framework is designed to separate temporal variations in BC input from post-depositional degradation and preservation.

## Model framework

The observed BC concentration at site *i* and time *t* is represented as:

S_i(t) = a_i · f(t) · P(t)

where:

- `S_i(t)` is the observed BC signal at site *i* and time *t*;
- `a_i` is a site-specific scaling factor;
- `f(t)` represents the common regional BC input history;
- `P(t)` represents the post-depositional BC preservation function.

The degradation rate follows a power-law formulation:

k(t) = k₀ t^(-β)

which gives the cumulative preservation function:

P(t) = exp[-k₀/(1-β) · t^(1-β)]

for β ≠ 1.

The model therefore allows the observed sedimentary BC records to be represented as the product of a common input history, site-specific scaling, and an age-dependent preservation function.

## Code

Two main scripts are provided:

### 1. `MCMC_using map_f_fixed.py`

This script performs Bayesian parameter estimation using Markov chain Monte Carlo (MCMC).

The regional input function `f(t)` obtained from the MAP fitting procedure is treated as fixed, while the MCMC samples the remaining model parameters:

- `a1`, `a2`, `a3` — site-specific scaling factors;
- `k0` — initial degradation-rate parameter;
- `β` — power-law exponent.

The MCMC implementation uses the [`emcee`](https://emcee.readthedocs.io/) ensemble sampler.

The parameters are sampled in transformed space to ensure physically meaningful values:

- `a_i = exp(log a_i)`
- `k₀ = exp(log k₀)`
- `β = 1 / [1 + exp(-β_raw)]`

Laboratory-derived information on `k₀` and `β` is incorporated as prior information.

The script reads:

- the smoothed BC time series from `data.xlsx`;
- the MAP-derived regional input function `f(t)` from `data_with_model_smoothdata.csv`.

It then calculates the posterior distributions of the model parameters and generates posterior estimates and uncertainty envelopes for `P(t)`.

Main outputs include:

- posterior summaries of `a1`, `a2`, `a3`, `k0`, and `β`;
- posterior median and 16th–84th percentile estimates of `P(t)`;
- reconstructed model BC time series for the three sedimentary sites;
- diagnostic plots of `P(t)`, `f(t)`, parameter distributions, and the `k₀–β` relationship;
- CSV and text files containing posterior results.

The MCMC script therefore provides the posterior uncertainty of the degradation parameters conditional on the MAP-derived regional input history.

---

### 2. `bootstrap_full_free.py`

This script performs a full bootstrap analysis in which the regional input function `f(t)`, degradation parameters, and site-specific scaling factors are jointly re-fitted for each bootstrap realization.

Unlike the fixed-`f(t)` MCMC analysis, this procedure allows uncertainty in the regional input history to propagate into the inferred degradation function.

The model uses a cubic B-spline representation of `f(t)`. Smoothness and scaling regularization are applied to stabilize the reconstruction of the input history.

For each bootstrap realization:

1. The observed BC records are perturbed according to the specified observational uncertainty.
2. The common input function `f(t)` is represented using B-spline coefficients.
3. The site-specific scaling factors (`a1`, `a2`, `a3`), `k₀`, `β`, and spline coefficients are jointly optimized.
4. The degradation function `P(t)` is calculated from the fitted `k₀` and `β`.
5. The resulting `f(t)` and `P(t)` are stored for subsequent uncertainty analysis.

The default configuration uses 1200 bootstrap realizations and parallel processing.

This analysis is intended to quantify the uncertainty associated with the joint reconstruction of:

- regional BC input history `f(t)`;
- BC degradation/preservation function `P(t)`;
- site-specific scaling factors.

