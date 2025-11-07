import os
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime


# -------------------------------------------
# Global shock event windows
# -------------------------------------------
SHOCK_WINDOWS = {
            "covid_shock": ("2020-02-15", "2020-06-30"),
            "inflation_energy_shock": ("2022-02-01", "2022-12-31"),
            "oil_spike_shock": ("2024-07-01", "2025-03-31"),
        }


# Ticker dictionary
tickers = {
    "FX_USDINR": "USDINR=X",
    # USD to Indian Rupee exchange rate (Forex)
    "BOND_BNDX": "BNDX",
    # International bond ETF (tracks investment-grade bonds outside the US)
    "COM_CRUDE": "BZ=F"
    # Brent Crude Oil futures (global crude oil benchmark)
}


def fetch_data(ticker, start="2013-06-05", end="2025-10-30"):
    # returns: pandas DataFrame with columns for price, return, log_return, cumulative

    # Download daily price data from Yahoo Finance and compute returns.
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    # Ensure data is available
    if df.empty:
        print(f"{ticker}: no data returned")
        return df

    # Reset index so we have a 'date' column
    df = df.reset_index()
    df.rename(columns={'Date': 'date'}, inplace=True)

    # Flatten MultiIndex if it exists
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = df.columns.str.strip()  # Remove extra spaces

    # Choose the adjusted close price if it exists, else normal Close
    if "Adj Close" in df.columns:
        df.rename(columns={"Adj Close": "price"}, inplace=True)
    elif "Close" in df.columns:
        df.rename(columns={"Close": "price"}, inplace=True)
    else:
        raise ValueError(f"No valid price column found for ticker {ticker}")

    # Convert price to numeric and drop NaNs
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df.dropna(subset=["price"], inplace=True)

    # Compute daily returns: \( r_t = \frac{P_t}{P_{t-1}} - 1 \)
    df["return"] = df["price"].pct_change()

    # Compute log returns : \( \ln\left(\frac{P_t}{P_{t-1}}\right) \)
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    df.dropna(subset=["return", "log_return"], inplace=True)

    # Compute cumulative returns: \( R_t = \frac{P_t}{P_0} - 1 \)
    if not df.empty:
        df["cumulative"] = df["price"] / df["price"].iloc[0] - 1
    else:
        df["cumulative"] = np.nan
    df.dropna(subset=["cumulative"], inplace=True)

    return df


def save_data(start="2013-06-05", end="2025-10-30"):
    # Fetch and save raw data for all tickers in the global dictionary.
    os.makedirs("data", exist_ok=True)

    for name, ticker in tickers.items():
        df = fetch_data(ticker, start, end)
        df = df.copy()
        if df.empty:
            continue
        df["asset"] = name  # Add asset label column
        df.to_csv(f"data/{name}.csv", index=False)
        print(f"Saved {name}.csv")


def load_data(name):
    # returns: pandas DataFrame with datetime index and cleaned columns
    df = pd.read_csv(f"data/{name}.csv")

    # Make all column names lowercase for consistency
    df.columns = df.columns.str.lower()

    # Parse date column if present, otherwise infer from first column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        first_col = df.columns[0]
        try:
            parsed_dates = pd.to_datetime(df[first_col], errors="coerce")
            if parsed_dates.notna().any():
                df["date"] = parsed_dates
            else:
                raise KeyError
        except Exception:
            raise KeyError(
                f"No valid date column found in {name}.csv. "
                f"Columns: {list(df.columns)}"
            )

    # Clean, set index, and sort
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df


def compute_volatility(df, window=20):
    # returns: DataFrame with added 'volatility' column
    # Formula: \( \sigma_t = \text{std}(r_{t - w : t}) \times \sqrt{252} \)
    # (252 ~ number of trading days per year)
    df["volatility"] = df["return"].rolling(window).std() * np.sqrt(252)
    return df

def tag_shock_periods(df, shock_windows=SHOCK_WINDOWS):
    # Adds Boolean flags for pre/during/post for each shock
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["shock_period"] = "normal"

    for name, (start, end) in shock_windows.items():
        start, end = pd.to_datetime(start), pd.to_datetime(end)
        df[f"is_pre_{name}"] = df["date"] < start
        df[f"is_during_{name}"] = (df["date"] >= start) & (df["date"] <= end)
        df[f"is_post_{name}"] = df["date"] > end

        df.loc[df[f"is_during_{name}"], "shock_period"] = name

    return df


def split_data_global(
    data_dir="data",
    train_end="2019-06-05",
    extra_train_windows=None,
    save_dir="data/splits",
    min_train_length=252,
    verbose=True
):
 
   #  Perform a global, time-based train/test split for all assets,
   #  and optionally create extra training windows (e.g., COVID).
   
    os.makedirs(save_dir, exist_ok=True)
    train_end = pd.to_datetime(train_end)
    manifest_records = []

    for file in os.listdir(data_dir):
        if not file.endswith(".csv"):
            continue

        name = file.replace(".csv", "")
        df = load_data(name)  

        # Standard pre-COVID split
        train_df = df.loc[df.index <= train_end].copy()
        test_df = df.loc[df.index > train_end].copy()

        test_df = test_df.reset_index()
        test_df = tag_shock_periods(test_df) 
        test_df = test_df.set_index("date")

        if len(train_df) < min_train_length:
            if verbose:
                print(f"Skipping {name}: insufficient training data ({len(train_df)} rows).")
            continue

        train_path = os.path.join(save_dir, f"{name}_train.csv")
        test_path = os.path.join(save_dir, f"{name}_test.csv")
        train_df.to_csv(train_path)
        test_df.to_csv(test_path)

        manifest_records.append({
            "asset": name,
            "train_start": train_df.index.min().strftime("%Y-%m-%d"),
            "train_end": train_df.index.max().strftime("%Y-%m-%d"),
            "test_start": test_df.index.min().strftime("%Y-%m-%d"),
            "test_end": test_df.index.max().strftime("%Y-%m-%d"),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "train_path": train_path,
            "test_path": test_path,
        })

        # Extra windows (e.g., COVID / post-COVID)
        if extra_train_windows:
            for suffix, start, end in extra_train_windows:
                start = pd.to_datetime(start)
                end = pd.to_datetime(end)
                window_df = df[(df.index >= start) & (df.index <= end)].copy()

                if len(window_df) < min_train_length:
                    if verbose:
                        print(f"Skipping {name} ({suffix}): insufficient window data.")
                    continue

                window_path = os.path.join(save_dir, f"{name}_train{suffix}.csv")
                testcovid_df = df[df.index > end].copy()
                testcovid_path = os.path.join(save_dir, f"{name}_test{suffix}.csv")

                window_df.to_csv(window_path)
                testcovid_df.to_csv(testcovid_path)

                manifest_records.append({
                    "asset": name,
                    "train_start": window_df.index.min().strftime("%Y-%m-%d"),
                    "train_end": window_df.index.max().strftime("%Y-%m-%d"),
                    "test_start": testcovid_df.index.min().strftime("%Y-%m-%d") if not testcovid_df.empty else None,
                    "test_end": testcovid_df.index.max().strftime("%Y-%m-%d") if not testcovid_df.empty else None,
                    "n_train": len(window_df),
                    "n_test": len(testcovid_df),
                    "train_path": window_path,
                    "test_path": testcovid_path if not testcovid_df.empty else None,
                })

                if verbose:
                    print(f"{name} ({suffix}): train={len(window_df)} rows")

        if verbose:
            print(f"{name}: train={len(train_df)}, test={len(test_df)}")

    # Final manifest
    manifest = pd.DataFrame(manifest_records)
    manifest_path = os.path.join(save_dir, "manifest.csv")
    save_split_manifest(manifest, manifest_path)

    if verbose:
        print(f"\nSaved manifest to {manifest_path}")
    return manifest

# ===============================================================
# Save and Load Manifest Utilities (Auto-selects training periods)
# ===============================================================

def save_split_manifest(manifest, path):
    
   #  Save the manifest DataFrame to CSV and confirm on success.
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    manifest.to_csv(path, index=False)
    if os.path.exists(path):
        print(f"[OK] Manifest saved: {path}")


def load_split_manifest(period="pre_covid", path="data/splits/manifest.csv"):

    #Load the manifest CSV and automatically filter paths by training period.
    
    # period options:
        #- "pre_covid"  -> uses *_train.csv
        #- "covid"      -> uses *_traincovid.csv
        #- "post_covid" -> uses *_postshock.csv
  
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found at {path}")

    manifest = pd.read_csv(
        path,
        parse_dates=["train_start", "train_end", "test_start", "test_end"]
    )

    # Map period name to suffix
    period_suffix = {
        "pre_covid": "_train.csv",
        "covid": "_traincovid.csv",
        "post_covid": "_postshock.csv",
    }.get(period, "_train.csv")

    # Filter manifest rows that match this period
    filtered = manifest[manifest["train_path"].str.endswith(period_suffix)]
    if filtered.empty:
        print(f"[WARN] No entries found for period '{period}' in manifest.")
    else:
        print(f"[OK] Loaded manifest for {period} ({len(filtered)} assets).")

    return filtered


def slice_shock_window(df, shock_name="covid_shock", recalibration_frac=None, recalibration_period=None):

# Split data around a defined shock window into pre-shock, shock pre-recalibration, and shock post-recalibration.


# Returns:
    # Tuple[DataFrame, DataFrame, DataFrame]: (pre_shock, shock_pre_cal, shock_post_cal)

    if shock_name not in SHOCK_WINDOWS:
        raise ValueError(f"{shock_name} not found in SHOCK_WINDOWS")

    start, end = pd.to_datetime(SHOCK_WINDOWS[shock_name][0]), pd.to_datetime(SHOCK_WINDOWS[shock_name][1])
    shock_df = df[(df['date'] >= start) & (df['date'] <= end)].copy()
    if shock_df.empty:
        raise ValueError(f"No data found in shock window {shock_name}")

    if recalibration_period is not None:
        cal_end = shock_df['date'].min() + recalibration_period
        shock_pre_cal = shock_df[shock_df['date'] <= cal_end].copy()
        shock_post_cal = shock_df[shock_df['date'] > cal_end].copy()
    elif recalibration_frac is not None:
        cal_idx = int(len(shock_df) * recalibration_frac)
        shock_pre_cal = shock_df.iloc[:cal_idx].copy()
        shock_post_cal = shock_df.iloc[cal_idx:].copy()
    else:
        raise ValueError("Either recalibration_frac or recalibration_period must be provided.")

    pre_shock = df[df['date'] < start].copy()
    return pre_shock, shock_pre_cal, shock_post_cal
    