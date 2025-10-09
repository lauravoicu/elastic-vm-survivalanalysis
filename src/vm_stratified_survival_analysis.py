#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI backend for Matplotlib.
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from graphics_style import apply_graphics_style
from config_utils import PLOT_DIR, DATA_DIR, log_message
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Suppress ApproximationWarning specifically
warnings.filterwarnings("ignore", message="Approximating using `survival_function_`")
# Categories to stratify survival analysis
STRATIFY_COLS = ["severity"]
REPORT_FILE = os.path.join(PLOT_DIR, "stratified_survival_analysis_report.txt")


def explain_stratified_survival_metrics():
    """Standard explanations of stratified survival analysis metrics."""
    explanations = """
    === Stratified Survival Analysis Explained ===
    - **Kaplan-Meier Survival Curves**: Compare how long vulnerabilities remain open across different categories.
    - **Cox Proportional Hazards Model**: Identifies risk factors influencing time-to-fix.
    - **Log-Rank Test**: Determines if survival curves between different groups are statistically different.
    """
    log_message(explanations, report_file=REPORT_FILE, to_report=True)


def stratified_kaplan_meier(df, stratify_col):

    unique_groups = df[stratify_col].dropna().unique()
    if len(unique_groups) < 2:
        log_message(f"[INFO] Not enough unique values in '{stratify_col}' to stratify. Skipping KM analysis.",
                    report_file=REPORT_FILE, to_report=True)
        return

    fig = plt.figure(figsize=(12, 6), facecolor='white')
    ax = fig.add_subplot(111)

    log_message("[DEBUG] Running updated stratified_kaplan_meier function", report_file=REPORT_FILE, to_report=False)
    severity_order = ["critical", "high", "medium", "low"]
    label_map = {s: s.upper() for s in severity_order}
    for group in severity_order:
        if group not in df[stratify_col].dropna().unique():
            continue
        df_subset_raw = df[df[stratify_col] == group].copy()
        log_message(f"[DEBUG] {stratify_col}='{group}': Raw subset size = {len(df_subset_raw)}",
                    report_file=REPORT_FILE, to_report=False)

        df_subset = df_subset_raw.dropna(subset=["age", "fixed"])
        df_subset["fixed"] = df_subset["fixed"].astype(int)

        if len(df_subset) < 5:
            log_message(f"[INFO] Skipping '{group}' in {stratify_col} due to low sample size ({len(df_subset)}).",
                        report_file=REPORT_FILE, to_report=True)
            continue

        kmf = KaplanMeierFitter()
        fit_success = False
        try:
            max_time = df['age'].max()
            if not np.isfinite(max_time):
                log_message(f"[WARNING] Max {'time_to_fix_days'} is not finite: {max_time}. Using default max of 365 days.",
                            report_file=REPORT_FILE, to_report=True)
                max_time = 365
            timeline = np.arange(0, max_time + 1, 1)  # Daily increments
            kmf.fit(durations=df_subset["age"], event_observed=df_subset["fixed"], timeline=timeline, label=label_map[group])
            fit_success = True
        except Exception as e:
            log_message(
                f"[WARNING] Kaplan-Meier fit failed for {group} in {stratify_col} "
                f"at fit step: {type(e).__name__}: {str(e)}",
                report_file=REPORT_FILE, to_report=True)
            continue

        if fit_success:
            try:
                # kmf.plot_survival_function(ax=ax, ci_show=False, lw=2.5)
                print(kmf.survival_function_.loc[30:350])
                survival_df = kmf.survival_function_

                final_val = kmf.survival_function_.iloc[-1, 0]
                print("KM fit final survival value:", final_val)

                UNIT_DIVISOR = {"days": 1, "weeks": 7, "months": 30}
                scale = UNIT_DIVISOR["months"]

                ax.step(
                    survival_df.index / scale,
                    survival_df[kmf._label],
                    where="post",
                    lw=2.5,
                    label=label_map[group]
                )
            except Exception as e:
                log_message(f"[WARNING] Plotting failed for {group} in {stratify_col}: {type(e).__name__}: {str(e)}",
                            report_file=REPORT_FILE, to_report=True)

            try:
                median_ = kmf.median_survival_time_
                log_message(f"[INFO] {stratify_col}='{group}' Median Survival Time: {median_} days",
                            report_file=REPORT_FILE, to_report=True)
                # 25th and 75th percentiles
                time_25th = kmf.percentile(0.75)  # 75% still unfixed (25th percentile of survival)
                time_75th = kmf.percentile(0.25)  # 25% still unfixed (75th percentile of survival)
                log_message(f"25th percentile survival time (75% unfixed): {time_25th:.1f} days",
                            report_file=REPORT_FILE, to_report=True)
                log_message(f"75th percentile survival time (25% unfixed): {time_75th:.1f} days",
                            report_file=REPORT_FILE, to_report=True)

                if stratify_col == "owner.team" and median_ == float('inf'):
                    log_message(f"[DEBUG] Survival function for {group}:",
                                report_file=REPORT_FILE, to_report=True)
                    survival_times = kmf.survival_function_.index
                    survival_probs = kmf.survival_function_[group]
                    for t, s in zip(survival_times[:5], survival_probs[:5]):
                        log_message(f"[DEBUG] t={t:.1f}, S(t)={s:.3f}",
                                    report_file=REPORT_FILE, to_report=True)
                    log_message(f"[DEBUG] Final S(t) at t={survival_times[-1]:.1f}: {survival_probs.iloc[-1]:.3f}",
                                report_file=REPORT_FILE, to_report=True)


            except Exception as e:
                log_message(
                    f"[WARNING] Median or survival function access failed for {group} "
                    f"in {stratify_col}: {type(e).__name__}: {str(e)}",
                    report_file=REPORT_FILE, to_report=True)
                continue

    TICK_SPACING = {"days": 100, "weeks": 10, "months": 2}
    UNIT_LABEL = {"days": "Time (Days)", "weeks": "Time (Weeks)", "months": "Time (Months)"}
    tick_spacing = TICK_SPACING["months"]
    upper_x_months = df["age"].max() / 30
    ax.set_xlim(0, upper_x_months)
    ax.set_xticks(np.arange(0, upper_x_months + tick_spacing, tick_spacing))

    ax.set_xlabel(UNIT_LABEL["months"], fontname="DejaVu Sans", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability vulnerability is still open", fontname="DejaVu Sans", fontsize=10)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("DejaVu Sans")

    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.title("")
    handles, labels = ax.get_legend_handles_labels()
    sorted_labels = [label_map[s] for s in severity_order if s in df[stratify_col].unique()]
    sorted_handles = [handles[labels.index(label_map[s])] for s in severity_order if s in df[stratify_col].unique()]
    ax.legend(sorted_handles, sorted_labels, title="Severity", fontsize="small", title_fontsize="medium")
    plt.tight_layout()
    try:
        output_path = os.path.join(PLOT_DIR, f"km_stratified_survival_{stratify_col}.png")
        plt.savefig(output_path, dpi=300, facecolor='white')
        log_message(f"[INFO] Kaplan-Meier plot saved for {stratify_col}: {output_path}",
                    report_file=REPORT_FILE, to_report=False)
    except Exception as e:
        log_message(f"[WARNING] Plot save failed for {stratify_col}: {type(e).__name__}: {str(e)}",
                    report_file=REPORT_FILE, to_report=True)
    plt.close()


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Clear previous report
    with open(REPORT_FILE, "w") as f:
        f.write("=== Stratified Survival Analysis Report ===\n\n")
    apply_graphics_style()

    data_path = os.path.join(DATA_DIR, "clean_vm_data_pandas.csv")
    df = pd.read_csv(data_path)
    datetime_cols = ["first_found_datetime", "last_found_datetime", "published_datetime"]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)
    explain_stratified_survival_metrics()

    for stratify_col in STRATIFY_COLS:
        if stratify_col in df.columns:
            # Pass export config matching the stratify_col
            stratified_kaplan_meier(df, stratify_col)
        else:
            log_message(f"[WARNING] Column '{stratify_col}' missing in dataset. Skipping.",
                        report_file=REPORT_FILE, to_report=True)


if __name__ == "__main__":
    main()
