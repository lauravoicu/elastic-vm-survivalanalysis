# Vulnerability Management Survival Analysis

This repository provides a toolkit for analyzing vulnerability remediation performance using survival analysis techniques.
For detailed methodology, see the accompanying [blog post](https://www.elastic.co/security-labs/time-to-patch-metrics). The code: 

1. Extracts vulnerability data from Elasticsearch
2. Applies survival analysis to understand time-to-patch behavior
3. Generates survival curves, stratified analysis, visualizations and statistical reports


## Features

1. **Data Extraction**

- Connects to Elasticsearch with Qualys VMDR data
- Exports vulnerability data to CSV/Parquet formats
- Located in [`src/es_connect.py`](./src/es_connect.py)


2. **Survival Analysis**

- Kaplan-Meier estimation for vulnerability remediation
- Handles censored data (open vulnerabilities)
- Located in [`src/vm_survival_analysis.py`](./src/vm_survival_analysis.py)


3. **Stratified Analysis**

- Compare remediation performance across groups (severity, teams, environments, etc.)
- Generate separate survival curves for different categories
- Located in [`src/vm_stratified_survival_analysis.py`](./src/vm_stratified_survival_analysis.py)


4. **Statistical Analysis**

- Descriptive statistics and cumulative distribution functions
- Remediation velocity metrics for closed vulnerabilities
- Located in [`src/vm_statistical_analysis.py`](./src/vm_statistical_analysis.py)



## Repository Structure
```bash
elastic-vm-survival-analysis/
├── src/
│   ├── es_connect.py                      # Elasticsearch data extraction
│   ├── vm_survival_analysis.py            # Core survival analysis
│   ├── vm_stratified_survival_analysis.py # Stratified analysis by groups
│   ├── vm_statistical_analysis.py         # Descriptive stats and CDF plots
│   ├── graphics_style.py                  # Plot styling configuration
│   └── config_utils.py                    # Configuration and logging utilities
├── data/                                  # Data storage directory
├── outputs/                               # Generated plots and reports
├── .env                                   # Environment variables
├── requirements.txt
└── README.md
```

## Installation & Setup

1. **Clone the Repository**

```bash
git clone <repository-url>
cd vulnerability-survival-analysis
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```
Or using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Environment Variables**

Create a .env file with your Elasticsearch configuration:

```bash
ES_HOST=your-elasticsearch-host.com
ES_PORT=9200
ES_API_KEY=your-api-key
ES_INDEX=your-qualys-log-index
```

## Usage

1. **Extract Data from Elasticsearch**

```bash
python src/es_connect.py
```
Connects to Elasticsearch and saves vulnerability data to `data/clean_vm_data_pandas.csv`

Optional step: perform additional data cleaning per your specific needs.

2. **Run Core Survival Analysis**

```bash
python src/vm_survival_analysis.py
```
Generates Kaplan-Meier survival curves and saves plots to outputs/

3. **Generate Stratified Analysis**

```bash
python src/vm_stratified_survival_analysis.py
```
Creates survival curves grouped by severity levels (configurable)

4. **Statistical Analysis**

```bash
python src/vm_statistical_analysis.py
```
Produces descriptive statistics and CDF plots for remediation velocity

## Outputs

- Survival curves: `km_survival_months.png` - Shows probability vulnerabilities remain open over time (time horizon can be days/months/years)
- Stratified curves: `km_stratified_survival_severity.png` - Compares performance across severity levels
- CDF plots: `cdf.png` - Remediation velocity for successfully closed vulnerabilities
- Reports: Text files with statistical summaries and metrics

## Data Requirements
The Elasticsearch index should contain fields:

- `qualys_vmdr.asset_host_detection.vulnerability.unique_vuln_id`
- `qualys_vmdr.asset_host_detection.vulnerability.first_found_datetime`
- `qualys_vmdr.asset_host_detection.vulnerability.last_found_datetime`
- `qualys_vmdr.asset_host_detection.vulnerability.status`
- `vulnerability.severity`
- `elastic.owner.team` and `elastic.environment` (for filtering)

## Requirements

- Python 3.8+
- Elasticsearch cluster with vulnerability data
- See requirements.txt for Python dependencies
