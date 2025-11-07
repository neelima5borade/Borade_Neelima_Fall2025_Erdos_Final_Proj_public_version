import os
import sys

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from modules.data_utils import tickers, load_data, fetch_data, compute_volatility


# ======================================================
#  SHOCK PERIODS CONFIG
# ======================================================
SHOCK_PERIODS =  {
            "covid_shock": ("2020-02-15", "2020-06-30"),
            "inflation_energy_shock": ("2022-02-01", "2022-12-31"),
            "oil_spike_shock": ("2024-07-01", "2025-03-31"),
        }


# ======================================================
#  Helper: Shade shocks on an axis
# ======================================================
def _shade_shocks(ax, df=None, add_labels=True):
 
    # Shade known or data-embedded shock regions on a plot.
    # Case 1: Dynamic shocks from dataframe (preferred)
    if df is not None and "shock_period" in df.columns:
        df = df.dropna(subset=["shock_period"])
        if df.empty:
            return

        for label, color in [
            ("during_shock", "red"),
            ("post_shock", "orange"),
        ]:
            mask = df["shock_period"] == label
            if not mask.any():
                continue
            start = df.loc[mask, "index"].min()
            end = df.loc[mask, "index"].max()
            ax.axvspan(
                start,
                end,
                color=color,
                alpha=0.15,
                label=label.replace("_", " ").title() if add_labels else None,
            )

    # Case 2: Static global SHOCK_PERIODS fallback
    else:
        for name, (start, end) in SHOCK_PERIODS.items():
            label = f"{name.replace('_', ' ').title()}" if add_labels else None
            ax.axvspan(
                pd.to_datetime(start),
                pd.to_datetime(end),
                color="red",
                alpha=0.12,
                label=label,
            )



# ======================================================
#  Helper: Save + Show Plot
# ======================================================
def _save_and_show_plot(fig, asset_name, plot_name):
    os.makedirs(f"plots/{asset_name}", exist_ok=True)
    save_path = f"plots/{asset_name}/{plot_name}.png"
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show(block=True)  # <- force figure to display and block



# ======================================================
# 1. Price, Returns, and Cumulative Returns
# ======================================================
def plot_data(train_end="2019-06-05", rolling_window=20):
    # Plot price, returns, and cumulative returns for all assets with shaded shocks.
    global_train_start = "2013-06-05"
    train_end = pd.to_datetime(train_end)

    for name in tickers.keys():
        df = load_data(name)
        df = df.loc[df.index >= global_train_start].copy()

        df.columns = df.columns.str.strip().str.lower()
        if "date" in df.columns:
            df.rename(columns={"date": "Date"}, inplace=True)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df.set_index("Date", inplace=True)

        for col in ["price", "return", "cumulative"]:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in {name} CSV")
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["price", "return", "cumulative"], inplace=True)

        df["price_MA"] = df["price"].rolling(window=rolling_window).mean()
        test_end = df.index.max()

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

        # (1) Price
        axes[0].plot(df.index, df["price"], label="Price", color="blue")
        axes[0].plot(
            df.index,
            df["price_MA"],
            label=f"{rolling_window}-Day MA",
            color="red",
            linestyle="--",
        )
        if test_end > train_end:
            axes[0].axvspan(train_end, test_end, color="gray", alpha=0.15, label="Test Period")
        _shade_shocks(axes[0])
        axes[0].set_title(f"{name} Price")
        axes[0].set_ylabel("Price")
        axes[0].legend()
        axes[0].grid(True)

        # (2) Daily Returns
        axes[1].plot(df.index, df["return"], label="Daily Return", color="orange")
        if test_end > train_end:
            axes[1].axvspan(train_end, test_end, color="gray", alpha=0.15)
        _shade_shocks(axes[1])
        axes[1].set_title(f"{name} Daily Returns")
        axes[1].set_ylabel("Return")
        axes[1].legend()
        axes[1].grid(True)

        # (3) Cumulative Returns
        axes[2].plot(df.index, df["cumulative"] * 100, label="Cumulative Return (%)", color="green")
        if test_end > train_end:
            axes[2].axvspan(train_end, test_end, color="gray", alpha=0.15)
        _shade_shocks(axes[2])
        axes[2].set_title(f"{name} Cumulative Returns (%)")
        axes[2].set_ylabel("Cumulative (%)")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        _save_and_show_plot(fig, name, "price_returns")

    return


# ======================================================
# 2. Rolling Volatility Plot
# ======================================================
def plot_print_volatility(train_end="2019-06-05", window=20):
    # Compute and plot rolling volatility with shock shading.
    global_train_start = "2013-06-05"
    train_end = pd.to_datetime(train_end)

    for ticker in tickers.keys():
        df = load_data(ticker)
        df = df.loc[df.index >= global_train_start]
        df = compute_volatility(df, window=window)

        test_end = df.index.max()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df["volatility"], label=f"{window}D Rolling Volatility", color="purple")
        if test_end > train_end:
            ax.axvspan(train_end, test_end, color="gray", alpha=0.15, label="Test Period")
        _shade_shocks(ax)

        ax.set_title(f"{ticker}: Rolling {window}-Day Volatility (Annualized)")
        ax.set_ylabel("Volatility ($σ$)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        _save_and_show_plot(fig, ticker, f"rolling_volatility_{window}d")

    return


# ======================================================
# 3. Rolling vs GARCH vs SV Comparison
# ======================================================

def plot_train_vols(name, base_dir="models", split_dir="data/splits", label=None, suffix=""):


    # Merge otrain, GARCH train, and SV train CSVs for one asset
    base_path = os.path.join(split_dir, f"{name}_train{suffix}.csv")
    garch_path = os.path.join(base_dir, "garch", f"{name}_garch_train{suffix}.csv")
    sv_path = os.path.join(base_dir, "sv", f"{name}_sv_train{suffix}.csv")

    # Check files exist
    for p in [base_path, garch_path, sv_path]:
        if not os.path.exists(p):
            print(f"[WARN] Missing file: {p}")
            return

    # --- Load and parse dates
    df_base = pd.read_csv(base_path, parse_dates=["date"]).sort_values("date")
    df_garch = pd.read_csv(garch_path, parse_dates=["date"]).sort_values("date")
    df_sv = pd.read_csv(sv_path, parse_dates=["date"]).sort_values("date")

    # --- Align all by intersection of available dates
    df_base.set_index("date", inplace=True)
    df_garch.set_index("date", inplace=True)
    df_sv.set_index("date", inplace=True)

    # Keep only overlapping periods across all three
    df = df_base.join(df_garch[["garch_vol"]], how="inner")
    df = df.join(df_sv[["sv_vol"]], how="inner")

    # Restore a clean date column
    df.reset_index(inplace=True)

    if df.empty:
        print(f"[WARN] No overlapping dates for {name} — skipping plot.")
        return

    # Ensure numeric types and drop NaNs
    df[["garch_vol", "sv_vol"]] = df[["garch_vol", "sv_vol"]].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=["garch_vol", "sv_vol"], inplace=True)

    # --- Compute log returns and rolling vol
    if "price" in df.columns:
        df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    df["rolling_vol"] = df["log_return"].rolling(10).std() * np.sqrt(252)
    df.dropna(subset=["rolling_vol", "garch_vol", "sv_vol"], inplace=True)

    # --- Plot comparison
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["date"], df["rolling_vol"], label="Rolling 10D Vol", color="black", lw=1)
    ax.plot(df["date"], df["garch_vol"], label="GARCH Vol", linestyle="--", color="tab:blue", lw=1.2)
    ax.plot(df["date"], df["sv_vol"], label="SV Vol", linestyle="--", color="tab:orange", lw=1.2)

    # Format x-axis as years
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    title_label = f" ({label})" if label is not None else ""
    ax.set_title(f"{name}: Training Volatility Comparison{title_label}", fontsize=13)
    ax.set_ylabel("Annualized Volatility)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_name = name + (f"_{label}" if label else "")
    _save_and_show_plot(fig, save_name, "train_vol_comparison")


# ----------------------------------------
# 1. Utility: Plot Volatility Differences, Ratios, and Overlay
# ----------------------------------------

def plot_rolling_garch_vs_sv(
    name,
    df,
    train_end="2019-06-05",
    sv_smooth_window=5,
    spike_threshold=1.5,
    calib_date="2020-09-30",
    ax_dict=None
):

    # Plots volatility diagnostics with difference and ratio.
    # Highlights volatility spikes automatically.
    

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    train_end = pd.to_datetime(train_end)
    df = df[df['date'] >= train_end]
    if df.empty:
        print(f"No data to plot for {name}")
        return

    # Smooth SV
    df['sv_vol_smooth'] = df['sv_vol'].rolling(window=sv_smooth_window, min_periods=1).mean()
    # Rolling realized volatility (annualized)
    df['rolling_vol'] = df['log_return'].rolling(window=30, min_periods=1).std() * np.sqrt(252)

    # Differences and ratios
    df['garch_diff'] = df['garch_vol'] - df['rolling_vol']
    df['sv_diff'] = df['sv_vol_smooth'] - df['rolling_vol']
    df['garch_ratio'] = df['garch_vol'] / df['rolling_vol']
    df['sv_ratio'] = df['sv_vol_smooth'] / df['rolling_vol']

    # Volatility spikes
    rolling_mean = df['rolling_vol'].rolling(window=30, min_periods=1).mean()
    spikes = df['rolling_vol'] > spike_threshold * rolling_mean

    # Axes
    ax_diff = ax_dict.get("ax_diff")
    ax_ratio = ax_dict.get("ax_ratio")

    # Difference plot
    ax_diff.plot(df['date'], df['garch_diff'], label='GARCH - Rolling', color='red', lw=1.5)
    ax_diff.plot(df['date'], df['sv_diff'], label='SV - Rolling', color='green', lw=1.5)
    ax_diff.axhline(0, color='gray', linestyle='--', alpha=0.7)
    for date in df['date'][spikes]:
        ax_diff.axvspan(date, date, color='orange', alpha=0.3)
    ax_diff.axvline(pd.to_datetime(calib_date), color='black', linestyle='--')
    ax_diff.set_title(f"{name} Vol Differences (Shocks Highlighted)")
    ax_diff.set_ylabel("Forecast - Rolling Vol")
    ax_diff.grid(True)
    ax_diff.legend()

    # Ratio plot
    ax_ratio.plot(df['date'], df['garch_ratio'], label='GARCH / Rolling', color='red', lw=1.5)
    ax_ratio.plot(df['date'], df['sv_ratio'], label='SV / Rolling', color='green', lw=1.5)
    ax_ratio.axhline(1, color='gray', linestyle='--', alpha=0.7)
    ax_ratio.set_ylabel("Forecast / Rolling")
    ax_ratio.set_xlabel("Date")
    ax_ratio.set_title(f"{name} Vol Ratios")
    ax_ratio.grid(True)
    ax_ratio.legend()
