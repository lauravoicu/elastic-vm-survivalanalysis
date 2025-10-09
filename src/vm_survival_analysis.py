#!/usr/bin/env python3
import os
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from graphics_style import apply_graphics_style
from config_utils import PLOT_DIR, DATA_DIR, log_message

import warnings
warnings.filterwarnings("ignore", message="Approximating using `survival_function_`")

REPORT_FILE = os.path.join(PLOT_DIR, "survival_analysis_report.txt")


def explain_survival_metrics():
    msg = """
=== Survival Analysis Metrics Explained ===
- Median Survival Time: 50% of vulnerabilities remain open longer than this time.
- Survival Probability Beyond X Days: Percentage of vulns still open after X days.
- 25th Percentile Time: 75% remain unfixed.
- 75th Percentile Time: 25% remain unfixed.
"""
    log_message(msg, report_file=REPORT_FILE, to_report=True)


def interpolate_percentile(kmf, p):
    """Interpolates time t where survival function S(t) = 1 - p."""
    sf = kmf.survival_function_[kmf._label]
    times = sf.index.values
    surv_probs = sf.values.flatten()
    target = 1 - p

    if surv_probs[-1] > target:
        return np.nan  # Never gets low enough

    return float(np.interp(target, surv_probs[::-1], times[::-1]))


def fit_km(df, time_col="age", event_col="fixed", outname=None, time_unit="months"):
    if outname is None:
        outname = f"km_survival_{time_unit}.png"

    UNIT_DIVISOR = {"days": 1, "weeks": 7, "months": 30}
    UNIT_LABEL = {"days": "Time (Days)", "weeks": "Time (Weeks)", "months": "Time (Months)"}

    if time_unit not in UNIT_DIVISOR:
        raise ValueError(f"Unsupported time_unit: {time_unit}")

    scale = UNIT_DIVISOR[time_unit]
    xlabel = UNIT_LABEL[time_unit]
    TICK_SPACING = {
        "days": 100,
        "weeks": 10,
        "months": 2
    }
    kmf = KaplanMeierFitter(label="KM_estimate")

    durations = df[time_col]              # ← Always use original durations (in days)
    event_observed = df[event_col]

    max_time = df[time_col].max()
    if not np.isfinite(max_time):
        log_message(f"[WARNING] Max {time_col} is not finite: {max_time}. Using default max of 365 days.",
                    report_file=REPORT_FILE, to_report=True)
        max_time = 365
    timeline = np.arange(0, max_time + 1, 1)  # Daily increments
    # Fit without manual timeline to preserve step shape
    kmf.fit(durations, event_observed, timeline)
    print(kmf.survival_function_.loc[30:350])
    survival_df = kmf.survival_function_

    final_val = kmf.survival_function_.iloc[-1, 0]
    print("KM fit final survival value:", final_val)

    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    plt.step(
        survival_df.index / scale,
        survival_df[kmf._label],
        where="post",
        color="#005566"
    )

    upper_x = np.ceil(durations.max() / scale * 1.1)
    ax.set_xlim(0, upper_x)
    tick_spacing = TICK_SPACING[time_unit]
    ax.set_xticks(np.arange(0, upper_x + tick_spacing, tick_spacing))
    ax.set_ylim(0, 1)

    ax.set_xlabel(xlabel, fontname="DejaVu Sans", fontsize=10)
    ax.set_ylabel("Probability vulnerability is still open", fontname="DejaVu Sans", fontsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("DejaVu Sans")

    legend = ax.get_legend()
    if legend:
        legend.remove()
    plt.title("")

    # Annotations
    closed = df[df[event_col] == 1]

    stats = {
        "Median (closed only)": {
            "time": closed[time_col].median(),
            "color": "#1f77b4",
            "x_shift": 50.0,
            "y_offset": -0.06
        },
        "Mean (closed only)": {
            "time": closed[time_col].mean(),
            "color": "#aec7e8",
            "x_shift": 150.0,
            "y_offset": -0.25
        },
        "Median (observed so far)": {
            "time": durations.median(),
            "color": "#ff7f0e",
            "x_shift": 50.0,
            "y_offset": 0.08
        },
        "Mean (observed so far)": {
            "time": durations.mean(),
            "color": "#ffbb78",
            "x_shift": 150.0,
            "y_offset": 0.14
        }
    }

    for label, item in stats.items():
        t = item["time"]
        color = item["color"]
        x_shift = item["x_shift"]
        y_offset = item["y_offset"]

        x = t / scale
        y = kmf.predict(t)

        plt.axvline(x, color=color, linestyle="--", alpha=0.6, linewidth=0.75)
        plt.plot(x, y, 'o', color=color)

        # Annotate
        plt.annotate(
            f"{label}\n{int(t)} days\n{y * 100:.1f}% open",
            xy=(x, y),
            xytext=(x + x_shift / scale, y + y_offset),
            arrowprops=dict(arrowstyle="->", lw=1.5, color=color),
            fontsize=9,
            color=color,
            ha="left"
        )

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, outname), dpi=300)
    plt.close()

    # Logging
    log_message(f"Closed-only mean: {closed[time_col].mean():.3f}  "
                f"median: {closed[time_col].median():.3f}", report_file=REPORT_FILE, to_report=True)
    log_message(f"Observed-so-far mean: {durations.mean():.3f}  "
                f"median: {durations.median():.3f}", report_file=REPORT_FILE, to_report=True)
    log_message(f"\nKM Median Survival Time: {kmf.median_survival_time_:.1f} days", report_file=REPORT_FILE, to_report=True)
    log_message(f"Confidence Interval:\n{median_survival_times(kmf.confidence_interval_)}", report_file=REPORT_FILE, to_report=True)

    for p in [0.75, 0.25, 0.20, 0.15, 0.10, 0.05, 0.01]:
        try:
            val = interpolate_percentile(kmf, 1-p)
            log_message(f"{int((1 - p) * 100)}th percentile: {val:.1f} days", report_file=REPORT_FILE, to_report=True)
        except Exception:
            continue
    s_closed = df.loc[df['fixed'] == 1, 'age'].astype(float)
    s_all = df['age'].astype(float)

    def summarize(s, name):
        qs = s.quantile([.1, .25, .5, .75, .9, .95, .99])
        print(f"\n{name}  n={len(s)}  mean={s.mean():.3f}  median={s.median():.3f}  "
              f"std={s.std():.2f}  skew={s.skew():.3f}")
        print(qs)

    summarize(s_closed, "Closed-only")
    summarize(s_all, "Observed-so-far")

    # Tail mass checks
    for k in [30, 60, 90, 180]:
        print(f"Closed-only  >{k}d: {(s_closed > k).mean() * 100:.2f}%")
        print(f"Observed-so-far >{k}d: {(s_all > k).mean() * 100:.2f}%")

    # Top tail inspection
    print(s_closed.max(), s_all.max())
    print(s_closed.nlargest(10))
    print(s_all.nlargest(10))

    # Quartile skewness: Bowley
    def bowley_skew(s):
        q1, q2, q3 = s.quantile([.25, .5, .75])
        return (q3 + q1 - 2 * q2) / (q3 - q1)

    print("Bowley (closed):", bowley_skew(s_closed))
    print("Bowley (all):   ", bowley_skew(s_all))

    # Trimmed/Winsorized means to show tail impact
    from scipy import stats
    print("Trimmed mean 99.5% (closed):", stats.trim_mean(s_closed, 0.005))
    print("Trimmed mean 99.5% (all):   ", stats.trim_mean(s_all, 0.005))
    return kmf


# Main
def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write("=== Survival Analysis Report ===\n\n")
    apply_graphics_style()

    data_path = os.path.join(DATA_DIR, "clean_vm_data_pandas.csv")
    df = pd.read_csv(data_path)
    datetime_cols = ["first_found_datetime", "last_found_datetime", "published_datetime"]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)
    explain_survival_metrics()

    # Standardize durations
    df["first_found_datetime"] = pd.to_datetime(df["first_found_datetime"], errors="coerce")
    df["last_found_datetime"] = pd.to_datetime(df["last_found_datetime"], errors="coerce")

    # Deduplication
    df["first_found_datetime"] = pd.to_datetime(df["first_found_datetime"], errors="coerce")
    df["last_found_datetime"] = pd.to_datetime(df["last_found_datetime"], errors="coerce")
    df = df.sort_values(by=["unique_vuln_id", "last_found_datetime"])
    df = df.drop_duplicates(subset="unique_vuln_id", keep="last")

    # fit_km(df, time_col="age", event_col="fixed", time_unit="days")
    # fit_km(df, time_col="age", event_col="fixed", time_unit="weeks")
    fit_km(df, time_col="age", event_col="fixed", time_unit="months")


if __name__ == "__main__":
    main()