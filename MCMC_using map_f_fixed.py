#!/usr/bin/env python3
"""
mcmc_using_map_f_fixed.py

Reads:
 - "先插值再平滑计算.xlsx" sheet "smooth data" (columns: t (Ma), BC-1501, BC-1148, BC-1146)
 - "data_with_model_smoothdata.csv" (MAP outputs) must contain columns t_yr (years), f(t) (or f_map)

Performs MCMC (emcee) to estimate parameters:
  log_a1, log_a2, log_a3, log_k0, beta_raw
where:
  a_i = exp(log_ai)
  k0 = exp(log_k0)
  beta = 1/(1+exp(-beta_raw))  (logistic transform -> in (0,1))

Model (fixed f_map):
  S_i(t) = a_i * f_map(t) * P(t)
  P(t) = exp( - k0/(1-beta) * t^(1-beta) )

Important numerical protections are included to avoid NaN.
"""
import os, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee

warnings.filterwarnings("ignore")

# --------- User settings ----------
EXCEL = "data.xlsx"
MAP_CSV = "data_with_model_smoothdata.csv"   # produced by your MAP run
sheet_name = "smooth data"
# MCMC settings (tune for production)
nwalkers = 48
nsteps = 2000   # increase for final runs
burn = int(0.3 * nsteps)
# observational noise model (relative)
obs_rel_sigma = 0.10   # observational sigma = 10% of site mean by default
# prior hyperparams (lab priors)
k0_lab_mean = 0.01148
k0_lab_sd = 0.0008
beta_lab_mean = 0.71
beta_lab_sd = 0.04
# ----------------------------------

# --- 1. load data ---
if not os.path.exists(EXCEL):
    raise FileNotFoundError(f"Excel file not found: {EXCEL}")
xls = pd.ExcelFile(EXCEL)
# find sheet
if sheet_name not in xls.sheet_names:
    # try lower-case match
    sheet_name = next((s for s in xls.sheet_names if "smooth" in s.lower()), xls.sheet_names[0])
df_smooth = pd.read_excel(EXCEL, sheet_name=sheet_name)
df_smooth = df_smooth.iloc[:, :4].copy()
df_smooth.columns = ["t_Ma", "BC_1501", "BC_1148", "BC_1146"]
# drop rows without t
df_smooth = df_smooth[~df_smooth["t_Ma"].isna()].reset_index(drop=True)

# load MAP f(t)
if not os.path.exists(MAP_CSV):
    raise FileNotFoundError(f"MAP CSV not found: {MAP_CSV}")
df_map = pd.read_csv(MAP_CSV)
# try to find t_yr or t_Ma and f column
if "t_yr" in df_map.columns:
    t_map = df_map["t_yr"].values
elif "t_Ma" in df_map.columns:
    t_map = df_map["t_Ma"].values * 1e6
else:
    raise ValueError("MAP CSV must contain 't_yr' or 't_Ma' column.")
# f column detection (common names)
f_col = None
for cand in ["f(t)", "f_map", "f_map", "f(t)_map", "f_map(t)", "f"]:
    if cand in df_map.columns:
        f_col = cand; break
# fallback: try any column with 'f' in name
if f_col is None:
    for c in df_map.columns:
        if 'f' in c.lower():
            f_col = c; break
if f_col is None:
    raise ValueError("Could not find an f(t) column in MAP CSV.")
f_map = df_map[f_col].values

# --- 2. align / filter: remove t==0 rows and ensure consistency ---
# we will use all data (no filtering)
# prepare smooth data times in years
t_smooth_yr = df_smooth["t_Ma"].values * 1e6
# match each smooth time to nearest MAP t
def match_indices(t_source, t_target):
    # returns indices into target for each source element
    idx = np.searchsorted(t_target, t_source)
    # refine by checking left/right neighbors
    out = []
    for i,s in enumerate(t_source):
        j = idx[i]
        candidates = []
        if j < len(t_target): candidates.append(j)
        if j-1 >= 0: candidates.append(j-1)
        # pick nearest
        dists = [abs(t_target[c]-s) for c in candidates]
        best = candidates[np.argmin(dists)]
        out.append(best)
    return np.array(out, dtype=int)

map_idx = match_indices(t_smooth_yr, t_map)
t_used = t_map[map_idx]   # in years
f_used = f_map[map_idx]

# observations S_i at those times (some values may be NaN)
S1 = df_smooth["BC_1501"].values
S2 = df_smooth["BC_1148"].values
S3 = df_smooth["BC_1146"].values

# final safe-times: ensure positive
t_used = np.maximum(t_used, 1.0)  # set minimum 1 year to avoid tiny powers

ntime = len(t_used)
print(f"Using {ntime} time points for MCMC (t>0).")

# observational sigma per site (use 10% of mean of non-NaN observations)
def site_sigma(S):
    m = np.nanmean(S)
    return max(1e-12, obs_rel_sigma * abs(m))
sigma1 = site_sigma(S1); sigma2 = site_sigma(S2); sigma3 = site_sigma(S3)
print("sigma1,sigma2,sigma3 =", sigma1, sigma2, sigma3)

# --- 3. define transforms and priors ---
# We'll sample x = [loga1, loga2, loga3, logk0, beta_raw]
# transforms:
def unpack_x(x):
    loga1, loga2, loga3, logk0, beta_raw = x
    a1 = np.exp(loga1); a2 = np.exp(loga2); a3 = np.exp(loga3)
    k0 = np.exp(logk0)
    # map beta_raw -> (0,1)
    beta = 1.0/(1.0 + np.exp(-beta_raw))
    return a1,a2,a3,k0,beta

# priors on transformed parameters:
# - logk0 ~ Normal(log(k0_lab_mean), k0_lab_sd/k0_lab_mean) approx on log-scale
logk0_prior_mean = math.log(k0_lab_mean)
logk0_prior_sd = k0_lab_sd / k0_lab_mean
# - beta_raw : map beta ~ N(beta_mean, beta_sd) => approximate on logit-scale
def beta_to_raw(mu_beta):
    return math.log(mu_beta/(1-mu_beta))
beta_raw_prior_mean = beta_to_raw(beta_lab_mean)
beta_raw_prior_sd = beta_lab_sd / (beta_lab_mean*(1-beta_lab_mean))  # approx delta-method

# a_i priors: use first non-NaN observations as center: put prior on log(a)
def first_non_nan(arr):
    for v in arr:
        if not np.isnan(v):
            return v
    return 1.0
loga1_center = math.log(max(1e-12, first_non_nan(df_smooth["BC_1501"].values)))
loga2_center = math.log(max(1e-12, first_non_nan(df_smooth["BC_1148"].values)))
loga3_center = math.log(max(1e-12, first_non_nan(df_smooth["BC_1146"].values)))
loga_prior_sd = 0.5  # fairly broad

# ---- log-prior (on x) ----
def log_prior_x(x):
    loga1, loga2, loga3, logk0, beta_raw = x
    # priors: Gaussians on transformed params
    lp = 0.0
    # loga
    lp += -0.5 * ((loga1 - loga1_center)/loga_prior_sd)**2
    lp += -0.5 * ((loga2 - loga2_center)/loga_prior_sd)**2
    lp += -0.5 * ((loga3 - loga3_center)/loga_prior_sd)**2
    # logk0
    lp += -0.5 * ((logk0 - logk0_prior_mean)/logk0_prior_sd)**2
    # beta_raw
    lp += -0.5 * ((beta_raw - beta_raw_prior_mean)/beta_raw_prior_sd)**2
    return lp

# --- 4. model and log-likelihood with strong numerical guards ---
def compute_P_t(k0, beta, tvec):
    # guard beta away from 1
    denom = 1.0 - beta
    if abs(denom) < 1e-6:
        return None   # indicate invalid
    # exponent = k0/denom * t^(1-beta)
    # compute exponent safely (clip t to reasonable range)
    # ensure tvec positive
    tvec_safe = np.maximum(tvec, 1.0)
    try:
        power = np.power(tvec_safe, 1.0 - beta)
        expo = (k0/denom) * power
        P = np.exp(-expo)
        # if any non-finite, return None
        if not np.all(np.isfinite(P)):
            return None
        return P
    except FloatingPointError:
        return None

def log_likelihood_x(x):
    # unpack
    a1,a2,a3,k0,beta = unpack_x(x)
    # compute P
    P = compute_P_t(k0, beta, t_used)
    if P is None:
        return -np.inf
    # model predictions at each time (may have NaNs if f_used has NaN)
    if np.any(~np.isfinite(f_used)):
        return -np.inf
    m1 = a1 * f_used * P
    m2 = a2 * f_used * P
    m3 = a3 * f_used * P
    # compute residuals only where obs present
    ll = 0.0
    # site1
    mask1 = ~np.isnan(S1)
    if np.any(mask1):
        r1 = (S1[mask1] - m1[mask1]) / sigma1
        ll += -0.5 * np.sum(r1**2)
    # site2
    mask2 = ~np.isnan(S2)
    if np.any(mask2):
        r2 = (S2[mask2] - m2[mask2]) / sigma2
        ll += -0.5 * np.sum(r2**2)
    # site3
    mask3 = ~np.isnan(S3)
    if np.any(mask3):
        r3 = (S3[mask3] - m3[mask3]) / sigma3
        ll += -0.5 * np.sum(r3**2)
    # return log-likelihood
    return ll

def log_prob_x(x):
    # quick finite check
    if not np.all(np.isfinite(x)):
        return -np.inf
    lp = log_prior_x(x)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_x(x)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

# --- 5. initialize walkers around MAP-informed values ---
# For loga, use log of first obs
x0 = np.array([loga1_center, loga2_center, loga3_center, logk0_prior_mean, beta_raw_prior_mean])
# small jitter
p0 = [x0 + 1e-3 * np.random.randn(x0.size) for _ in range(nwalkers)]

# Validate initial positions (if some invalid, re-draw)
for i in range(len(p0)):
    if log_prob_x(p0[i]) == -np.inf:
        p0[i] = x0 + 1e-4 * np.random.randn(x0.size)

# --- 6. run emcee ---
ndim = x0.size
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_x)

print("Starting MCMC: nwalkers", nwalkers, "nsteps", nsteps)
state = sampler.run_mcmc(p0, 200, progress=True)  # short burn-in to check
# check if any chain had all -inf (stall)
lnprobs = sampler.get_log_prob()
if np.all(np.isneginf(lnprobs)):
    raise RuntimeError("All initial positions invalid (log_prob -inf). Check priors and data.")
# continue
sampler.reset()
sampler.run_mcmc(state, nsteps, progress=True)

# --- 7. extract samples and summarize ---
samples = sampler.get_chain(flat=True)
logp = sampler.get_log_prob(flat=True)
print("Samples shape:", samples.shape)

# discard burn and compute posteriors
samples_post = samples[burn: , :]
# transform back
def transform_samples(samples_array):
    a1 = np.exp(samples_array[:,0])
    a2 = np.exp(samples_array[:,1])
    a3 = np.exp(samples_array[:,2])
    k0 = np.exp(samples_array[:,3])
    beta = 1.0/(1.0 + np.exp(-samples_array[:,4]))
    return a1,a2,a3,k0,beta

a1_s,a2_s,a3_s,k0_s,beta_s = transform_samples(samples_post)

def summarize(arr):
    return np.median(arr), np.percentile(arr,16), np.percentile(arr,84)

summary = {
    "a1": summarize(a1_s),
    "a2": summarize(a2_s),
    "a3": summarize(a3_s),
    "k0": summarize(k0_s),
    "beta": summarize(beta_s)
}
print("Posterior summary (median,16,84):")
for k,v in summary.items():
    print(f" {k}: {v}")

# --- 8. reconstruct posterior f(t) and P(t) credible bands ---
# here f_used is fixed (f_used). posterior uncertainty only from k0,beta
nsamps = samples_post.shape[0]
# sample a subset to save time
sel = np.random.choice(nsamps, size=min(2000, nsamps), replace=False)
P_samps = np.zeros((sel.size, t_used.size))
for i,ii in enumerate(sel):
    kk = k0_s[ii]; bb = beta_s[ii]
    Ptmp = compute_P_t(kk, bb, t_used)
    if Ptmp is None:
        Ptmp = np.full_like(t_used, np.nan)
    P_samps[i,:] = Ptmp
# median + 16/84
P_med = np.nanmedian(P_samps, axis=0)
P_lo = np.nanpercentile(P_samps, 16, axis=0)
P_hi = np.nanpercentile(P_samps, 84, axis=0)

# f(t) is fixed (f_used). we can show f_used and scale uncertainty via posterior a_i distributions:
# compute model median series for each site
a1_med = np.median(a1_s); a2_med = np.median(a2_s); a3_med = np.median(a3_s)
S1_model_med = a1_med * f_used * P_med
S2_model_med = a2_med * f_used * P_med
S3_model_med = a3_med * f_used * P_med

# --- 9. plots (PNG) ---
os.makedirs("mcmc_outputs", exist_ok=True)
import matplotlib
matplotlib.rcParams.update({'font.size':12})

# P(t) linear & log-log
plt.figure(figsize=(10,4), dpi=200)
plt.plot(t_used/1e3, P_med, label='P median')
plt.fill_between(t_used/1e3, P_lo, P_hi, alpha=0.25)
plt.xlabel("Time (kyr)")
plt.ylabel("P(t)")
plt.title("P(t) posterior (median + 16-84%)")
plt.grid(alpha=0.2)
plt.savefig("mcmc_outputs/P_t_linear.png", bbox_inches='tight'); plt.close()

plt.figure(figsize=(6,4), dpi=200)
plt.loglog(t_used/1e3, np.maximum(P_med,1e-20), label='P median')
plt.fill_between(t_used/1e3, np.maximum(P_lo,1e-20), np.maximum(P_hi,1e-20), alpha=0.25)
plt.xlabel("Time (kyr)")
plt.ylabel("P(t)")
plt.title("P(t) posterior (log-log)")
plt.grid(which='both', alpha=0.2)
plt.savefig("mcmc_outputs/P_t_loglog.png", bbox_inches='tight'); plt.close()

# f(t) (from MAP) scaled example
plt.figure(figsize=(10,4), dpi=200)
plt.plot(t_used/1e3, f_used, label='f(t) (MAP, fixed)')
plt.xlabel("Time (kyr)")
plt.ylabel("f(t) (relative)")
plt.title("f(t) used (from MAP)")
plt.grid(alpha=0.2)
plt.savefig("mcmc_outputs/f_map_used.png", bbox_inches='tight'); plt.close()

# parameter histograms
plt.figure(figsize=(10,6), dpi=200)
plt.subplot(2,2,1); plt.hist(a1_s, bins=40); plt.title("a1 posterior")
plt.subplot(2,2,2); plt.hist(a2_s, bins=40); plt.title("a2 posterior")
plt.subplot(2,2,3); plt.hist(a3_s, bins=40); plt.title("a3 posterior")
plt.subplot(2,2,4); plt.hist(k0_s, bins=40); plt.title("k0 posterior")
plt.savefig("mcmc_outputs/param_hists.png", bbox_inches='tight'); plt.close()

plt.figure(figsize=(6,6), dpi=200)
plt.plot(k0_s, beta_s, '.', ms=1, alpha=0.5)
plt.xlabel("k0"); plt.ylabel("beta"); plt.title("k0 vs beta samples")
plt.grid(alpha=0.2)
plt.savefig("mcmc_outputs/k0_beta_scatter.png", bbox_inches='tight'); plt.close()

# save summary CSV and sample subset
pd.DataFrame({
    "param":["a1","a2","a3","k0","beta"],
    "median":[summary["a1"][0], summary["a2"][0], summary["a3"][0], summary["k0"][0], summary["beta"][0]],
    "p16":[summary["a1"][1], summary["a2"][1], summary["a3"][1], summary["k0"][1], summary["beta"][1]],
    "p84":[summary["a1"][2], summary["a2"][2], summary["a3"][2], summary["k0"][2], summary["beta"][2]]
}).to_csv("mcmc_outputs/posterior_summary.csv", index=False)

# save a few outputs
np.savetxt("mcmc_outputs/P_med.txt", P_med)
np.savetxt("mcmc_outputs/t_used_years.txt", t_used)
pd.DataFrame({
    "t_yr": t_used,
    "f_map": f_used,
    "P_med": P_med, "P_lo": P_lo, "P_hi": P_hi,
    "S1_obs": S1, "S1_mod": S1_model_med,
    "S2_obs": S2, "S2_mod": S2_model_med,
    "S3_obs": S3, "S3_mod": S3_model_med
}).to_csv("mcmc_outputs/fP_and_models.csv", index=False)

print("MCMC finished. Outputs in ./mcmc_outputs/")
