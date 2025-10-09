#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from graphics_style import apply_graphics_style
from config_utils import PLOT_DIR, DATA_DIR

warnings.simplefilter(action='ignore', category=FutureWarning)
REPORT_FILE = os.path.join(PLOT_DIR, "stats_report.txt")


def descriptive_stats(df):
    stats = df.describe().T
    stats["IQR"] = stats["75%"] - stats["25%"]  # Interquartile Range
    stats["CV (%)"] = (stats["std"] / stats["mean"]) * 100  # Coefficient of Variation

    print("\n=== Descriptive Statistics ===\n", stats)
    return stats


def plot_remediation_cdf(df, time_col="age", event_col="fixed"):
    from matplotlib.ticker import PercentFormatter

    apply_graphics_style()

    # Filter to closed vulnerabilities only
    closed = df[df[event_col] == 1][time_col].dropna()

    # Sort and compute empirical CDF
    sorted_days = np.sort(closed.values)
    y = np.arange(1, len(sorted_days) + 1) / len(sorted_days)
    y_pct = y * 100

    # Convert to months
    x = sorted_days / 30.44

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y_pct, color='#005566', linewidth=2)
    ax.fill_between(x, y_pct, 100, color='lightblue', alpha=0.3)  # Fill down from 100%

    # Y-axis: 100% at top, 0% at bottom
    ax.set_ylim(100, 0)
    ax.yaxis.set_major_formatter(PercentFormatter())

    # Milestones (in days)
    milestones = {
        "1 month": 30,
        "3 months": 91,
        "6 months": 182,
        "1 year": 365
    }

    for label, days in milestones.items():
        pct = (closed <= days).mean() * 100
        x_val = days / 30.44
        y_val = pct

        # Slightly lift dot for milestones except 1 month
        y_dot = y_val - 0.8 if label != "1 month" else y_val
        x_dot = x_val - 0.2 if label != "1 month" else x_val
        # Marker dot
        ax.plot(x_dot, y_dot, 'o', color='darkred', markersize=6)

        if label == "1 month":
            ax.annotate(
                f"{pct:.1f}% remediated\nin the first {label}",
                (x_val, y_val),
                xytext=(15, 0),  # shift right
                textcoords='offset points',
                ha='left',
                va='center',
                fontsize=8,
                fontfamily='DejaVu Sans',
                color='#4D4D4D'
            )
        else:
            ax.annotate(
                f"{pct:.1f}% remediated\nin the first {label}",
                (x_val, y_val),
                xytext=(0, 30),  # above
                textcoords='offset points',
                ha='center',
                va='top',
                fontsize=8,
                fontfamily='DejaVu Sans',
                color='#4D4D4D'
            )
    # Axis formatting
    ax.set_xlim(0, 12)
    ax.set_xlabel("Time (months)", fontname="DejaVu Sans", fontsize=9)
    ax.set_ylabel("Percentage of vulnerabilities remediated", fontname="DejaVu Sans", fontsize=9)
    # ax.set_title("Cumulative Remediation Over Time (Closed Vulnerabilities Only)")
    for label in ax.get_xticklabels():
        label.set_fontname("DejaVu Sans")

    for label in ax.get_yticklabels():
        label.set_fontname("DejaVu Sans")
    ax.tick_params(labelsize=8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Output
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "cdf.png"), dpi=300)
    plt.close()


def main():
    # Ensure the outputs directory exists
    os.makedirs(PLOT_DIR, exist_ok=True)

    data_path = os.path.join(DATA_DIR, "clean_vm_data_pandas.csv")
    df = pd.read_csv(data_path)
    datetime_cols = ["first_found_datetime", "last_found_datetime", "published_datetime"]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)

    print("\ndf.head()")
    print(df.head(2))

    descriptive_stats(df)
    plot_remediation_cdf(df, time_col="age", event_col="fixed")


if __name__ == "__main__":
    main()
