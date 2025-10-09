import os

# Configuration constants
PLOT_DIR = "../outputs"
DATA_DIR = "../data"
REPORT_FILE = os.path.join(PLOT_DIR, "report.txt")  # Default report file, can be overridden


def log_message(message, report_file=REPORT_FILE, to_report=True):
    print(message)  # Always print to console
    if to_report:   # Write to report only if flagged
        os.makedirs(os.path.dirname(report_file), exist_ok=True)  # Ensure directory exists
        with open(report_file, "a") as f:
            f.write(message + "\n")