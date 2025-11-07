

# Stochastic Volatility utilities (particle filter, fitting across multiple assets,
# and forecasting) — updated to be consistent with GARCH utilities scaling/annualization.

# Assumptions:
# - Input returns used for fitting are log returns (daily).
# - We compute scale_used = std(log_returns) per asset and store it in model CSVs.
# - h_particles returned by the particle filter are log-variances (shape (T, N)).
# - Reported vol series are annualized: daily_std * scale_used * sqrt(252).


import numpy as np
import pandas as pd
import os
from scipy.stats import t as t_dist
from scipy.optimize import minimize

# ---------------------------
# Particle filter utilities
# ---------------------------
def resample_particles(h_t, weights, N):
    cumulative = np.cumsum(weights)
    indices = np.searchsorted(cumulative, np.random.rand(N))
    return h_t[indices]

def sv_particle_filter(eps, mu, phi, sigma, N=500, dist='normal', nu_fixed=8):

   # Particle filter for the log-variance SV model:
   #   h_t = mu + phi*(h_{t-1} - mu) + sigma * eta_t,  eta ~ N(0,1)
   # Observation: eps_t ~ N(0, exp(h_t))  or t with df=nu_fixed

    # Returns:
    #   loglik : float (log-likelihood estimate)
    #   h_particles : np.array shape (T, N) containing log-variance samples per time

    T = len(eps)
    # Preallocate (T, N)
    h_particles = np.zeros((T, N))
    loglik = 0.0

    # Random shocks
    eta = np.random.randn(T, N)

    # Initialize stationary distribution for h (log-variance)
    h_particles[0] = mu + sigma * eta[0] / np.sqrt(max(1 - phi**2, 1e-12))

    # weights (will be overwritten each step)
    weights = np.ones(N) / N

    for t in range(T):
        if t > 0:
            h_particles[t] = mu + phi * (h_particles[t-1] - mu) + sigma * eta[t]

        # convert to variance (ensure numerically safe)
        h_exp = np.exp(h_particles[t])
        h_exp = np.maximum(h_exp, 1e-12)

        # compute log weights vectorized across particles
        if dist == 'normal':
            # observation density: eps[t] ~ N(0, h_exp)
            # log density for each particle: -0.5*(log(2pi) + log(h) + eps^2 / h)
            logw = -0.5 * (np.log(2.0 * np.pi) + np.log(h_exp) + (eps[t]**2) / h_exp)
        elif dist == 't':
            # standardized residual under each particle
            std_eps = eps[t] / np.sqrt(h_exp)
            logw = t_dist.logpdf(std_eps, df=nu_fixed) - 0.5 * np.log(h_exp)
        else:
            raise ValueError("dist must be 'normal' or 't'")

        # stabilize weights
        max_logw = np.max(logw)
        w = np.exp(logw - max_logw)
        w_sum = np.sum(w)
        if w_sum <= 0 or not np.isfinite(w_sum):
            w = np.ones_like(w) / len(w)
        else:
            w = w / w_sum

        # estimate loglik increment robustly (log-sum-exp)
        loglik += max_logw + np.log(np.mean(np.exp(logw - max_logw)) + 1e-300)

        # effective sample size
        ess = 1.0 / np.sum(w**2)
        if ess < N / 2.0:
            # multinomial resample
            h_particles[t] = resample_particles(h_particles[t], w, N)
            w = np.ones(N) / N

        weights = w  # store for potential diagnostics (not used later)

    return float(loglik), h_particles

# ---------------------------
# Negative log-likelihood wrappers
# ---------------------------
def sv_neg_loglik(params, eps, dist='normal', nu_fixed=8, N=500):
    mu, phi, sigma = params
    if sigma <= 0 or not (-1.0 < phi < 1.0):
        return 1e12
    loglik, _ = sv_particle_filter(eps, mu, phi, sigma, N=N, dist=dist, nu_fixed=nu_fixed)
    return -float(loglik)

def total_neg_loglik_vectorized(params, returns_dict, dist="normal", nu_fixed=None, N=500):
    mu, phi, sigma = params
    if sigma <= 0 or not (-1.0 < phi < 1.0):
        return 1e12

    total_nll = 0.0
    for eps in returns_dict.values():
        # per-asset returns in returns_dict are expected to be scaled (divided by scale)
        loglik, _ = sv_particle_filter(eps, mu, phi, sigma, N=N, dist=dist, nu_fixed=nu_fixed)
        total_nll += -loglik
    return total_nll

# ---------------------------
# Fit SV across assets (single param set for all assets)
# ---------------------------
def fit_sv_all_assets(
tickers,
dist="normal",
split_dir="data/splits",
nu_fixed=None,
N=500,
train_start=None,
train_end=None,
calibration_label="_train"
):
    
    #Jointly fit stochastic volatility (SV) models across multiple assets in-memory.
    #Uses only training data (train or traincovid) depending on calibration_label.
 

    
    returns_dict = {}
    scales = {}
    full_dfs = {}
    sv_results_dict = {}

    # Load and prepare returns for each asset
    for name in tickers.keys():
        if calibration_label in ["train", ""]:
            path = os.path.join(split_dir, f"{name}_train.csv")
        elif calibration_label == "traincovid":
            path = os.path.join(split_dir, f"{name}_traincovid.csv")
        else:
            raise ValueError(f"Unknown calibration label: {calibration_label}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] Missing file: {path}")

        df_full = pd.read_csv(path)
        df_full.dropna(subset=["date"], inplace=True)
        df_full["date"] = pd.to_datetime(df_full["date"], errors="coerce")

        # Compute log returns if not present
        if "return" in df_full.columns:
            rets_raw = df_full["return"].copy()
        else:
            if "price" not in df_full.columns:
                raise ValueError(f"File {path} must contain 'price' or 'return' column.")
            df_full["log_return"] = np.log(df_full["price"] / df_full["price"].shift(1))
            rets_raw = df_full["log_return"].copy()

        # Apply training window
        mask = np.ones(len(df_full), dtype=bool)
        if train_start is not None:
            mask &= df_full["date"] >= pd.Timestamp(train_start)
        if train_end is not None:
            mask &= df_full["date"] <= pd.Timestamp(train_end)

        df_train = df_full.loc[mask].copy()
        rets = df_train["return"].values if "return" in df_train.columns else df_train["log_return"].values
        rets = rets[~np.isnan(rets)]
        if len(rets) == 0:
            raise ValueError(f"No valid returns for {name} in training window.")

        # Standardize returns
        scale = np.std(rets) if np.std(rets) > 0 else 1.0
        returns_dict[name] = rets / scale
        scales[name] = scale

        # Prepare for output
        df_full["sv_vol"] = np.nan
        df_full["scale_used"] = np.nan
        full_dfs[name] = df_full

    # Fit SV parameters jointly
    initial_guess = [0.0, 0.95, 0.2]  # mu, phi, sigma
    bounds = [(-5, 5), (-0.999, 0.999), (1e-6, 5.0)]

    result = minimize(
        total_neg_loglik_vectorized,
        initial_guess,
        args=(returns_dict, dist, nu_fixed, N),
        method="L-BFGS-B",
        bounds=bounds
    )

    mu, phi, sigma = result.x
    total_nll = result.fun
    print(f"[OK] Joint SV fit ({dist}): mu={mu:.6f}, phi={phi:.6f}, sigma={sigma:.6f}, NLL={total_nll:.4f}")
    sv_params = (float(mu), float(phi), float(sigma))

    # Compute filtered volatilities for each asset
    for name, rets_std in returns_dict.items():
        _, h_particles = sv_particle_filter(rets_std, mu, phi, sigma, N=N, dist=dist, nu_fixed=nu_fixed)
        h_filtered = np.mean(h_particles, axis=1) if h_particles.ndim == 2 else h_particles

        df_full = full_dfs[name]

        # Assign to training indices
        mask = np.ones(len(df_full), dtype=bool)
        if train_start is not None:
            mask &= df_full["date"] >= pd.Timestamp(train_start)
        if train_end is not None:
            mask &= df_full["date"] <= pd.Timestamp(train_end)

        df_train_idx = np.where(mask)[0]
        scale_used = float(scales.get(name, 1.0))

        sv_vol_annual = np.exp(0.5 * h_filtered) * scale_used * np.sqrt(252)

        if len(sv_vol_annual) == len(df_train_idx):
            assign_idx = df_train_idx
        elif len(sv_vol_annual) == max(0, len(df_train_idx) - 1):
            assign_idx = df_train_idx[1:]
        else:
            raise ValueError(f"Length mismatch for {name}: sv_vol_annual={len(sv_vol_annual)}, mask={len(df_train_idx)}")

        df_full.loc[df_full.index[assign_idx], "sv_vol"] = sv_vol_annual
        df_full.loc[df_full.index[assign_idx], "scale_used"] = scale_used

        sv_results_dict[name] = df_full

    return sv_params, float(total_nll), sv_results_dict






# ---------------------------
# Forecast SV for out-of-sample test
# 
# ---------------------------

def forecast_sv_all_assets(name, mu_first, phi_first, sigma_first,
                           mu_second, phi_second, sigma_second,
                           split_dir="data/splits", train_dir="models/sv",
                           save_dir="results/sv", dist="normal", nu=5,
                           sv_window=20):
   
   #  Forecast stochastic volatility (SV) for all assets with one-step-ahead recursion.
    # Supports normal or student-t distributed returns and uses the end of the second training window for recalibration.

    os.makedirs(save_dir, exist_ok=True)

    # Load training data to get end of second training window
    df_train_second = pd.read_csv(os.path.join(train_dir, f"{name}_sv_traincovid.csv"))
    train_end_second = pd.to_datetime(df_train_second["date"].iloc[-1])

    # Load test data
    df_test = pd.read_csv(os.path.join(split_dir, f"{name}_test.csv"))
    df_test["date"] = pd.to_datetime(df_test["date"])
    df_test["log_return"] = np.log(df_test["price"] / df_test["price"].shift(1)).fillna(0)

    # Initialize SV latent states
    N = 1000  # number of particles
    h_particles = np.full(N, mu_first)
    sv_list = []

    for t in range(len(df_test)):
        r_prev = df_test["log_return"].iloc[t-1] if t > 0 else 0.0

        # Use recalibrated parameters based on train_end_second
        if df_test["date"].iloc[t] <= train_end_second:
            mu, phi, sigma = mu_first, phi_first, sigma_first
        else:
            mu, phi, sigma = mu_second, phi_second, sigma_second

        # Particle filter step
        if dist == "normal":
            weights = (1 / np.sqrt(2 * np.pi * np.exp(h_particles))) * \
                      np.exp(-0.5 * r_prev**2 / np.exp(h_particles))
        elif dist == "student_t":
            weights = t.pdf(r_prev / np.sqrt(np.exp(h_particles)), df=nu) / np.sqrt(np.exp(h_particles))
        else:
            raise ValueError("dist must be 'normal' or 'student_t'")

        # Normalize weights, fallback to uniform if sum=0
        weights_sum = np.sum(weights)
        if weights_sum == 0:
            weights = np.ones(N) / N
        else:
            weights /= weights_sum

        # Resample particles
        idx = np.random.choice(np.arange(N), size=N, p=weights)
        h_particles = h_particles[idx]

        # Propagate particles
        h_particles = mu + phi * (h_particles - mu) + sigma * np.random.randn(N)

        # Forecast SV: mean of exp(h) as variance, annualized
        sv_list.append(np.sqrt(np.mean(np.exp(h_particles))) * np.sqrt(252))

    df_test["sv_vol"] = sv_list
    df_test.to_csv(os.path.join(save_dir, f"{name}_sv_test.csv"), index=False)
    print(f"[OK] SV forecast saved for {name}")

    return df_test

