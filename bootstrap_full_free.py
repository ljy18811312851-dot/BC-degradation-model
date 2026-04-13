#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bootstrap_full_free.py (Windows-safe)
Full-free bootstrap for joint f(t) (spline) + P(t) (k0,beta) + a_i fitting.

Key change: do NOT import matplotlib at module top-level to avoid MemoryError
on Windows when using multiprocessing (spawn). All plotting/imports happen
inside __main__ (main process only).
"""

import os
import time
import math
import sys
import traceback
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.interpolate import BSpline
import multiprocessing as mp
import zipfile
from collections import defaultdict

# ---------------------------
# User settings (tune these)
# ---------------------------
EXCEL = "data.xlsx"
SHEET_KEYWORD = "smooth"
MAP_OUT_DIR = "fit_results_smoothdata"    # directory where MAP outputs saved
OUTDIR = "fit_results_smoothdata_bootstrap_free"
# Bootstrap / parallel settings
nboot = 1200            # default; change to 1000+ for production
nproc = 3
random_seed = 123
max_nfev = 6000        # least_squares evaluations per bootstrap
# Regularization/prior hyperparams (should match MAP run)
lambda_scale = 1e-2
lambda_smooth = 1e-2
prior_sigma_rel = 0.5
# numeric guards
MIN_T = 1e-12
EXP_CLIP = 700.0
# ---------------------------

np.random.seed(random_seed)
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------
# Helper functions (no matplotlib here)
# ---------------------------
def load_map_outputs(map_dir):
    """Load MAP outputs: data_with_model_smoothdata.csv and param summary."""
    map_csv = os.path.join(map_dir, "data_with_model_smoothdata.csv")
    param_csv = os.path.join(map_dir, "param_summary_smoothdata.csv")
    if not os.path.exists(map_csv):
        raise FileNotFoundError(f"MAP CSV not found: {map_csv}")
    if not os.path.exists(param_csv):
        raise FileNotFoundError(f"MAP param summary not found: {param_csv}")
    df_map = pd.read_csv(map_csv)
    df_param = pd.read_csv(param_csv)
    # find t column (t_yr or t_Ma)
    tcol = None
    if "t_yr" in df_map.columns:
        tcol = "t_yr"
    elif "t_Ma" in df_map.columns:
        tcol = "t_Ma"
        df_map["t_yr"] = df_map["t_Ma"] * 1e6
        tcol = "t_yr"
    else:
        for c in df_map.columns:
            if c.lower().startswith("t"):
                tcol = c; break
    if tcol is None:
        raise RuntimeError("Could not find time column in MAP CSV")
    t = df_map[tcol].values.copy()
    # find f column
    f_col = None
    for cand in ["f(t)", "f_map", "f", "f_map(t)"]:
        if cand in df_map.columns:
            f_col = cand; break
    if f_col is None:
        for c in df_map.columns:
            if c.lower().startswith("f"):
                f_col = c; break
    if f_col is None:
        raise RuntimeError("Could not find f(t) column in MAP CSV")
    f_map = df_map[f_col].values.copy()
    # optional P
    P_col = None
    for cand in ["P(t)", "P", "p(t)"]:
        if cand in df_map.columns:
            P_col = cand; break
    P_map = df_map[P_col].values.copy() if P_col is not None else None
    return t, f_map, P_map, df_map, df_param

def find_smooth_sheet(excel_path):
    xls = pd.ExcelFile(excel_path)
    for nm in xls.sheet_names:
        if SHEET_KEYWORD in nm.lower():
            return nm
    return xls.sheet_names[0]

def build_S_obs_and_sigma(excel_path, map_t):
    """Read smooth-data sheet and align observations to map_t by nearest neighbor mapping."""
    sheet_name = find_smooth_sheet(excel_path)
    df_smooth = pd.read_excel(excel_path, sheet_name=sheet_name)
    df4 = df_smooth.iloc[:, :4].copy()
    df4.columns = ["t_Ma", "S1", "S2", "S3"]
    df4 = df4.dropna(subset=["t_Ma"]).sort_values("t_Ma").reset_index(drop=True)
    t_smooth_yr = df4["t_Ma"].values * 1e6

    def match_indices(t_source, t_target):
        idx = np.searchsorted(t_target, t_source)
        out = []
        for i,s in enumerate(t_source):
            j = idx[i]
            candidates = []
            if j < len(t_target): candidates.append(j)
            if j-1 >= 0: candidates.append(j-1)
            if not candidates:
                out.append(0)
            else:
                dists = [abs(t_target[c]-s) for c in candidates]
                best = candidates[int(np.argmin(dists))]
                out.append(best)
        return np.array(out, dtype=int)

    map_idx = match_indices(t_smooth_yr, map_t)
    S_obs = np.full((3, len(map_t)), np.nan)
    cols = [df4.iloc[:,1].values, df4.iloc[:,2].values, df4.iloc[:,3].values]
    agg = [defaultdict(list) for _ in range(3)]
    for i,mi in enumerate(map_idx):
        for s in range(3):
            val = cols[s][i]
            if not (val is np.nan):
                try:
                    if not np.isnan(val):
                        agg[s][mi].append(val)
                except Exception:
                    pass
    for s in range(3):
        for idx,vlist in agg[s].items():
            if vlist:
                S_obs[s, idx] = np.nanmean(vlist)
    mask_obs = ~np.isnan(S_obs)
    sigma = np.zeros_like(S_obs)
    for i in range(3):
        valid = mask_obs[i,:]
        if valid.any():
            sigma[i, valid] = 0.1 * np.nanmean(S_obs[i, valid])
        else:
            sigma[i, :] = 1.0
    sigma = np.where(sigma <= 0, 1.0, sigma)
    return S_obs, mask_obs, sigma, df4, sheet_name

def construct_knots(t, k=3, M_internal=None):
    t = np.array(t)
    t_min, t_max = t.min(), t.max()
    if M_internal is None:
        M_internal = min(8, max(1, len(t)//10))
    if M_internal > 0:
        t_knots_internal = np.quantile(t, np.linspace(0,1,M_internal+2)[1:-1])
    else:
        t_knots_internal = np.array([])
    t_knots = np.concatenate(([t_min]*(k+1), t_knots_internal, [t_max]*(k+1)))
    n_coeffs = len(t_knots) - (k+1)
    return t_knots, n_coeffs, k

def f_from_coeffs(c, tt, t_knots, k):
    spline = BSpline(t_knots, c, k, extrapolate=False)
    vals = spline(tt)
    t_min = t_knots[k]
    t_max = t_knots[-k-1]
    vals = np.where(np.isnan(vals) & (tt < t_min), c[0], vals)
    vals = np.where(np.isnan(vals) & (tt > t_max), c[-1], vals)
    return np.maximum(vals, 1e-12)

def retention_P(tt, k0, beta):
    tt = np.maximum(tt, MIN_T)
    denom = 1.0 - beta
    if abs(denom) > 1e-10:
        power = np.power(tt, 1.0 - beta)
        expo = (k0/denom) * power
        expo = np.minimum(expo, EXP_CLIP)
        return np.exp(-expo)
    else:
        return np.power(tt, -k0)

def build_D2(n_coeffs):
    if n_coeffs < 3:
        return np.zeros((0, n_coeffs))
    D2 = np.zeros((n_coeffs-2, n_coeffs))
    for i in range(n_coeffs-2):
        D2[i,i] = 1
        D2[i,i+1] = -2
        D2[i,i+2] = 1
    return D2

def residuals_full_factory(t, S_obs, mask_obs, sigma, t_knots, k, D2, a_prior):
    n_coeffs = D2.shape[1] if D2.size>0 else len(t_knots)-(k+1)
    def residuals_full(p, S_obs_local):
        loga = p[0:3]
        logk0 = p[3]
        beta_raw = p[4]
        c = p[5:5+n_coeffs]
        a = np.exp(loga)
        k0 = np.exp(logk0)
        beta = 1.0/(1.0 + np.exp(-beta_raw))
        fvals = f_from_coeffs(c, t, t_knots, k)
        f_norm = fvals / (np.nanmean(fvals) + 1e-16)
        P = retention_P(t, k0, beta)
        S_model = (a[:,None]) * f_norm[None,:] * P[None,:]
        res_list = []
        for i in range(3):
            mask = mask_obs[i,:]
            if np.any(mask):
                res_list.append((S_obs_local[i,mask] - S_model[i,mask]) / sigma[i,mask])
        res = np.concatenate(res_list) if len(res_list) > 0 else np.array([])
        reg_scale = np.sqrt(lambda_scale) * (c - 1.0)
        reg_smooth = np.sqrt(lambda_smooth) * (D2.dot(c)) if D2.size>0 else np.array([])
        prior_res = (np.exp(loga) - a_prior) / (prior_sigma_rel * np.maximum(a_prior, 1e-12))
        return np.concatenate([res, reg_scale, reg_smooth, prior_res])
    return residuals_full

# worker must be top-level (no matplotlib usage here)
def bootstrap_worker(task_tuple):
    try:
        (bidx, seed, p_opt, t, S_obs, mask_obs, sigma, t_knots, k, D2, a_prior) = task_tuple
        rng = np.random.RandomState(int(seed))
        S_pert = S_obs.copy()
        for i in range(3):
            mask = mask_obs[i,:]
            if np.any(mask):
                S_pert[i,mask] = S_obs[i,mask] + rng.normal(0.0, sigma[i,mask])
        p0 = p_opt + 1e-6 * rng.randn(len(p_opt))
        residuals_full = residuals_full_factory(t, S_obs, mask_obs, sigma, t_knots, k, D2, a_prior)
        nparam = len(p_opt)
        lower = np.full(nparam, -np.inf)
        upper = np.full(nparam, np.inf)
        lower[0:3] = -50; upper[0:3] = 50
        lower[3] = -30; upper[3] = 5
        lower[4] = -20; upper[4] = 20
        lower[5:] = 1e-8; upper[5:] = 1e8
        res = least_squares(lambda p: residuals_full(p, S_pert),
                            p0, bounds=(lower, upper),
                            max_nfev=max_nfev, xtol=1e-8, ftol=1e-8)
        psol = res.x
        loga = psol[0:3]; logk0 = psol[3]; beta_raw = psol[4]; c = psol[5:5+len(psol)-5]
        a = np.exp(loga); k0 = np.exp(logk0); beta = 1.0/(1.0 + np.exp(-beta_raw))
        fvals = f_from_coeffs(c, t, t_knots, k)
        f_norm = fvals / (np.nanmean(fvals) + 1e-16)
        Pvals = retention_P(t, k0, beta)
        return (bidx, True, psol, f_norm, Pvals, None)
    except Exception as e:
        tb = traceback.format_exc()
        return (task_tuple[0], False, None, None, None, tb)

# ---------------------------
# Main (plots imported and executed here)
# ---------------------------
if __name__ == "__main__":
    # Ensure Windows compatibility
    mp.freeze_support()

    # load MAP outputs
    try:
        t, f_map, P_map, df_map, df_param = load_map_outputs(MAP_OUT_DIR)
    except Exception as e:
        print("Error loading MAP outputs:", e)
        raise

    # build observations aligned to map times
    S_obs, mask_obs, sigma, df_smooth_partial, sheet_name = build_S_obs_and_sigma(EXCEL, t)

    # build spline knots and D2
    t_knots, n_coeffs, k = construct_knots(t, k=3, M_internal=None)
    D2 = build_D2(n_coeffs)

    # a_prior from param summary or fallback
    if {"a1","a2","a3"}.issubset(set(df_param.columns)):
        a_prior = np.array([df_param.loc[0,"a1"], df_param.loc[0,"a2"], df_param.loc[0,"a3"]])
    else:
        ap = []
        for i in range(3):
            idxs = np.where(mask_obs[i,:])[0]
            if idxs.size>0:
                ap.append(S_obs[i, idxs[0]])
            else:
                ap.append(np.nanmean(S_obs[i, mask_obs[i,:]]) if np.any(mask_obs[i,:]) else 1.0)
        a_prior = np.array(ap)

    # try to load p_opt from MAP, else construct approx
    p_opt_path = os.path.join(MAP_OUT_DIR, "p_opt.npy")
    if os.path.exists(p_opt_path):
        p_opt = np.load(p_opt_path)
        print("Loaded p_opt from", p_opt_path)
    else:
        if "a1" in df_param.columns:
            a0 = np.array([df_param.loc[0,"a1"], df_param.loc[0,"a2"], df_param.loc[0,"a3"]])
        else:
            a0 = np.array([np.nanmean(S_obs[i, mask_obs[i,:]]) if np.any(mask_obs[i,:]) else 1.0 for i in range(3)])
        k0_0 = df_param.loc[0, "k0 (yr^-1)"] if "k0 (yr^-1)" in df_param.columns else 1e-6
        beta_0 = df_param.loc[0, "beta"] if "beta" in df_param.columns else 0.7
        c0 = np.ones(n_coeffs)
        p_opt = np.concatenate([np.log(a0), [np.log(k0_0), np.log(beta_0/(1.0-beta_0))], c0])
        print("Constructed approximate p_opt from param_summary or defaults.")

    nparam = 5 + n_coeffs
    if len(p_opt) != nparam:
        if len(p_opt) > nparam:
            p_opt = p_opt[:nparam]
        else:
            pad = np.ones(nparam - len(p_opt))
            p_opt = np.concatenate([p_opt, pad])
        print("Adjusted p_opt length to", len(p_opt))

    # prepare tasks
    seeds = [random_seed + 7 + i for i in range(nboot)]
    tasks = []
    for i, sd in enumerate(seeds):
        tasks.append((i, sd, p_opt, t, S_obs, mask_obs, sigma, t_knots, k, D2, a_prior))

    # run bootstrap
    print(f"Starting full-free bootstrap: nboot={nboot}, nproc={nproc}, nparam={nparam}, ncoeffs={n_coeffs}")
    tstart = time.time()
    results = None
    if nproc <= 1:
        results = [bootstrap_worker(task) for task in tasks]
    else:
        pool = mp.Pool(processes=min(nproc, nboot))
        try:
            results = pool.map(bootstrap_worker, tasks)
        finally:
            pool.close()
            pool.join()
    tend = time.time()
    print(f"Bootstrap finished in {tend - tstart:.1f}s")

    # gather results
    samples = np.full((nboot, nparam), np.nan)
    f_list = []
    P_list = []
    fail_logs = {}
    for res in results:
        bidx, ok, psol, fvals, Pvals, tb = res
        if ok and psol is not None:
            samples[bidx,:] = psol
            f_list.append(fvals)
            P_list.append(Pvals)
        else:
            fail_logs[bidx] = tb

    n_valid = len(f_list)
    print(f"Successful fits: {n_valid}/{nboot}")

    if n_valid == 0:
        raise RuntimeError("No successful bootstrap fits. Consider lowering nboot, increasing max_nfev, or simplifying model.")

    f_ens = np.vstack(f_list)
    P_ens = np.vstack(P_list)
    np.save(os.path.join(OUTDIR, "bootstrap_samples_full.npy"), samples)
    np.save(os.path.join(OUTDIR, "f_ensemble_full.npy"), f_ens)
    np.save(os.path.join(OUTDIR, "P_ensemble_full.npy"), P_ens)

    # percentiles
    P_med = np.nanmedian(P_ens, axis=0)
    P_lo = np.nanpercentile(P_ens, 16, axis=0)
    P_hi = np.nanpercentile(P_ens, 84, axis=0)
    f_med = np.nanmedian(f_ens, axis=0)
    f_lo = np.nanpercentile(f_ens, 16, axis=0)
    f_hi = np.nanpercentile(f_ens, 84, axis=0)

    pd.DataFrame({
        "t_yr": t,
        "f_med": f_med, "f_lo": f_lo, "f_hi": f_hi,
        "P_med": P_med, "P_lo": P_lo, "P_hi": P_hi
    }).to_csv(os.path.join(OUTDIR, "fP_uncertainty_full.csv"), index=False)

    # parameter summaries
    valid_rows = np.isfinite(samples).all(axis=1)
    valid_samples = samples[valid_rows,:]
    a_samples = np.exp(valid_samples[:,0:3])
    k0_samples = np.exp(valid_samples[:,3])
    beta_samples = 1.0/(1.0 + np.exp(-valid_samples[:,4]))

    def ssum(x):
        return np.median(x), np.percentile(x,16), np.percentile(x,84)

    param_summary = {
        "a1_med": ssum(a_samples[:,0])[0], "a1_p16": ssum(a_samples[:,0])[1], "a1_p84": ssum(a_samples[:,0])[2],
        "a2_med": ssum(a_samples[:,1])[0], "a2_p16": ssum(a_samples[:,1])[1], "a2_p84": ssum(a_samples[:,1])[2],
        "a3_med": ssum(a_samples[:,2])[0], "a3_p16": ssum(a_samples[:,2])[1], "a3_p84": ssum(a_samples[:,2])[2],
        "k0_med": ssum(k0_samples)[0], "k0_p16": ssum(k0_samples)[1], "k0_p84": ssum(k0_samples)[2],
        "beta_med": ssum(beta_samples)[0], "beta_p16": ssum(beta_samples)[1], "beta_p84": ssum(beta_samples)[2],
        "nboot_success": valid_samples.shape[0], "nboot_requested": nboot
    }
    pd.DataFrame([param_summary]).to_csv(os.path.join(OUTDIR, "bootstrap_param_summary_full.csv"), index=False)

    # ---------------------------
    # Plotting (import matplotlib only here)
    # ---------------------------
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.size': 12})

    # P(t)
    plt.figure(figsize=(9,4))
    plt.fill_between(t/1e3, P_lo, P_hi, color='lightblue', alpha=0.5, label='68% CI')
    plt.plot(t/1e3, P_med, color='blue', lw=1.5, label='median')
    plt.xlabel('t (kyr)'); plt.ylabel('P(t)'); plt.title('P(t) bootstrap (full-free)')
    plt.legend(); plt.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "P_t_uncertainty_full.png"), dpi=200); plt.close()

    # f(t)
    plt.figure(figsize=(9,4))
    plt.fill_between(t/1e3, f_lo, f_hi, color='navajowhite', alpha=0.6, label='68% CI')
    plt.plot(t/1e3, f_med, color='orangered', lw=1.5, label='median')
    plt.xlabel('t (kyr)'); plt.ylabel('f(t)'); plt.title('f(t) bootstrap (full-free)')
    plt.legend(); plt.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "f_t_uncertainty_full.png"), dpi=200); plt.close()

    # parameter histograms
    plt.figure(figsize=(10,6))
    plt.subplot(2,2,1); plt.hist(a_samples[:,0], bins=40); plt.title('a1')
    plt.subplot(2,2,2); plt.hist(a_samples[:,1], bins=40); plt.title('a2')
    plt.subplot(2,2,3); plt.hist(a_samples[:,2], bins=40); plt.title('a3')
    plt.subplot(2,2,4); plt.hist(k0_samples, bins=40); plt.title('k0 (yr^-1)')
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "param_hists_full.png"), dpi=200); plt.close()

    plt.figure(figsize=(6,5))
    plt.hist(beta_samples, bins=40); plt.title('beta'); plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "beta_hist_full.png"), dpi=200); plt.close()

    plt.figure(figsize=(6,6))
    plt.plot(k0_samples, beta_samples, '.', ms=1, alpha=0.4)
    plt.xlabel('k0'); plt.ylabel('beta'); plt.title('k0 vs beta (bootstrap samples)')
    plt.grid(alpha=0.2); plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "k0_beta_scatter_full.png"), dpi=200); plt.close()

    # save medians and time arrays
    np.savetxt(os.path.join(OUTDIR, "P_med_full.txt"), P_med)
    np.savetxt(os.path.join(OUTDIR, "t_used_years_full.txt"), t)
    pd.DataFrame({"t_yr": t, "f_med": f_med, "f_lo": f_lo, "f_hi": f_hi,
                  "P_med": P_med, "P_lo": P_lo, "P_hi": P_hi}).to_csv(os.path.join(OUTDIR, "fP_uncertainty_full.csv"), index=False)

    # save zip
    zipf = "fit_results_smoothdata_full_bootstrap_free.zip"
    with zipfile.ZipFile(zipf, 'w') as z:
        for root, dirs, files in os.walk(OUTDIR):
            for file in files:
                z.write(os.path.join(root, file), arcname=file)
    print("Saved outputs to", OUTDIR, "and zipped to", zipf)

    # If there were failures, save some diagnostics
    if len(fail_logs) > 0:
        with open(os.path.join(OUTDIR, "bootstrap_failures.txt"), "w", encoding="utf-8") as fw:
            for idx, tb in fail_logs.items():
                fw.write(f"--- FAILURE {idx} ---\n")
                fw.write(str(tb) + "\n\n")
    print("Done.")
