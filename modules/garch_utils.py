
# GARCH(1,1) 
# Supports: Normal / Student-t, Multi-asset, Shock reactivity
# =====================================================
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t
import os
#=====================================================
#1. Core GARCH(1,1) recursion
#=====================================================
def garch(returns, omega, alpha, beta, init_var=None, eps_floor=1e-16):
    T = len(returns)
    var_series = np.empty(T, dtype=float)
    if init_var is None:  
        if 0 < (alpha + beta) < 1:  
            init_var = omega / (1.0 - alpha - beta)  
        else:  
            init_var = np.var(returns) if T > 1 else (omega + 1e-8)  
    var_series[0] = max(init_var, eps_floor)  

    for t in range(1, T):  
        var_series[t] = omega + alpha * (returns[t - 1] ** 2) + beta * var_series[t - 1]  
    return np.maximum(var_series, eps_floor)  
#=====================================================
#2. Negative Log-Likelihood (Normal / Student-t)
# ====================================================
def garch_neg_loglik(params, eps, dist='normal', nu_fixed=8):
    omega, alpha, beta = params
    if not (omega > 0 and alpha >= 0 and beta >= 0):
        return 1e12
    if alpha + beta >= 1:
        return 1e12
    var = garch(eps, omega, alpha, beta)  
    var = np.maximum(var, 1e-16)  
    logvar = np.log(var)  

    if dist == 'normal':  
        ll = 0.5 * np.sum(np.log(2 * np.pi) + logvar + (eps ** 2) / var)  
    elif dist == 't':  
        standardized = eps / np.sqrt(var)  
        ll = -np.sum(t.logpdf(standardized, df=nu_fixed)) + 0.5 * np.sum(logvar)  
    else:  
        raise ValueError("dist must be 'normal' or 't'")  

    return float(ll)  

#=====================================================
#3. Multi-asset Negative Log-Likelihood
#=====================================================
def total_neg_loglik(params, returns_dict, dist='normal', nu_fixed=8):
    total = 0.0
    for _, eps in returns_dict.items():
        total += garch_neg_loglik(params, eps, dist=dist, nu_fixed=nu_fixed)
    return total
#=====================================================
#4. Fit GARCH across all assets (standard + global refit)
#=====================================================
def fit_garch_all_assets(
    tickers,
    dist="t",
    split_dir="data/splits",
    save_dir="models/garch",
    nu_fixed=8,
    train_start=None,
    train_end=None,
    calibration_label="_train"   # new argument
):
   

   #  Jointly fit GARCH(1,1) across multiple assets.
   #  Optimized to avoid redundant DataFrame reads and loops.

    os.makedirs(save_dir, exist_ok=True)

    returns_dict = {}
    scales = {}
    full_dfs = {}
    results_dict = {}

    # 1. Prepare returns and scaling
    for name in tickers.keys():
        calibration_label = calibration_label.lower()
        if calibration_label in ["train", ""]:
            path = os.path.join(split_dir, f"{name}_train.csv")
        elif calibration_label == "traincovid":
            path = os.path.join(split_dir, f"{name}_traincovid.csv")
        else:
            raise ValueError(f"Unknown calibration label: {calibration_label}")

        df_full = pd.read_csv(path)
        df_full["date"] = pd.to_datetime(df_full["date"], errors="coerce")

        df_full["log_return"] = np.log(df_full["price"] / df_full["price"].shift(1))
        df_full.dropna(subset=["log_return"], inplace=True)

        # Filter for training window
        mask = np.ones(len(df_full), dtype=bool)
        if train_start is not None:
            mask &= df_full["date"] >= train_start
        if train_end is not None:
            mask &= df_full["date"] <= train_end

        df_train = df_full.loc[mask]
        rets = df_train["log_return"].values
        scale = np.std(rets) if np.std(rets) > 0 else 1.0
        returns_dict[name] = rets / scale
        scales[name] = scale

        # Store full df for later assignment
        df_full["garch_vol"] = np.nan
        df_full["scale_used"] = np.nan
        full_dfs[name] = df_full

    # 2. Initial guess and bounds
    initial_guess = [1e-6, 0.05, 0.9]
    bounds = [(1e-12, None), (1e-12, 1.0 - 1e-6), (1e-12, 1.0 - 1e-6)]

    # 3. Fit GARCH parameters jointly
    result = minimize(
        total_neg_loglik,
        initial_guess,
        args=(returns_dict, dist, nu_fixed),
        method="L-BFGS-B",
        bounds=bounds
    )
    omega, alpha, beta = result.x
    total_nll = result.fun
    print(f"[OK] Joint GARCH fit ({dist}): omega={omega:.3e}, alpha={alpha:.3f}, beta={beta:.3f}, NLL={total_nll:.2f}")

    # 4. Compute and assign volatilities
    for name, rets_std in returns_dict.items():
        var_std = garch(rets_std, omega, alpha, beta)  # make sure garch() is vectorized
        scale = scales[name]
        var_orig = var_std * (scale ** 2)
        vol_annual = np.sqrt(var_orig) * np.sqrt(252)

        df_full = full_dfs[name]

        # Assign to training indices
        mask = np.ones(len(df_full), dtype=bool)
        if train_start is not None:
            mask &= df_full["date"] >= train_start
        if train_end is not None:
            mask &= df_full["date"] <= train_end

        df_train_idx = np.where(mask)[0]
        # Align lengths exactly
        if len(df_train_idx) == len(vol_annual):
            df_full.iloc[df_train_idx, df_full.columns.get_loc("garch_vol")] = vol_annual
            df_full.iloc[df_train_idx, df_full.columns.get_loc("scale_used")] = scale
        else:
            # If off by one due to returns shift
            df_full.iloc[df_train_idx[1:], df_full.columns.get_loc("garch_vol")] = vol_annual
            df_full.iloc[df_train_idx[1:], df_full.columns.get_loc("scale_used")] = scale

     
    
        results_dict[name] = df_full


    return (float(omega), float(alpha), float(beta)), float(total_nll), results_dict



#=====================================================
#5. Forecast with local scaling (causal)

#=====================================================


def forecast_garch_all_assets(name, omega_first, alpha_first, beta_first,
                              omega_second, alpha_second, beta_second,
                              split_dir="data/splits", train_dir="models/garch",
                              save_dir="results/garch", dist="student_t", nu=8, rv_window=20):
 
   # Forecast GARCH(1,1) volatility for all assets with one-step-ahead recursion.
  #  Supports normal or student-t distributed returns and uses the end of the second training window for recalibration.

    os.makedirs(save_dir, exist_ok=True)

    # Load training data to get end of second training window
    df_train_second = pd.read_csv(os.path.join(train_dir, f"{name}_garch_traincovid.csv"))
    train_end_second = pd.to_datetime(df_train_second["date"].iloc[-1])

    # Load test data
    df_test = pd.read_csv(os.path.join(split_dir, f"{name}_test.csv"))
    df_test["date"] = pd.to_datetime(df_test["date"])
    df_test["log_return"] = np.log(df_test["price"] / df_test["price"].shift(1)).fillna(0)

    # Initialize GARCH variance
    var_forecast = np.zeros(len(df_test))
    var_forecast[0] = omega_first / (1 - alpha_first - beta_first)
    garch_vol_list = []

    for t in range(len(df_test)):
        r_prev = df_test["log_return"].iloc[t-1] if t > 0 else 0.0

        # Choose parameters based on recalibration
        if df_test["date"].iloc[t] <= train_end_second:
            omega, alpha, beta = omega_first, alpha_first, beta_first
        else:
            omega, alpha, beta = omega_second, alpha_second, beta_second

        # Standard GARCH recursion
        if t > 0:
            var_prev = var_forecast[t-1]

            if dist == "normal":
                var_new = omega + alpha * r_prev**2 + beta * var_prev
            elif dist == "student_t":
                # Adjust squared return by factor (nu-2)/nu to account for Student-t variance
                # This keeps unconditional variance consistent
                scale_factor = (nu - 2) / nu
                var_new = omega + alpha * r_prev**2 * scale_factor + beta * var_prev
            else:
                raise ValueError("dist must be 'normal' or 'student_t'")
        else:
            var_new = var_forecast[0]

        var_forecast[t] = var_new
        garch_vol_list.append(np.sqrt(var_new) * np.sqrt(252))  # annualized

    df_test["garch_vol"] = garch_vol_list
    df_test.to_csv(os.path.join(save_dir, f"{name}_garch_test.csv"), index=False)
    print(f"[OK] GARCH forecast saved for {name}")

    return df_test
