# 🛡️ Behavioural Fraud Detection for Mobile Money

An end-to-end, production-grade fraud detection system engineered for high-volume mobile money networks. The model uses LightGBM decision trees over five behavioral features to detect **Account Takeovers (ATO)**, **dormancy awakening**, and **money-muling funnels** with strict zero-leakage account-split validation.

Includes a feature extraction pipeline, rigorous reproducibility audits, a curated test suite, and an interactive **Streamlit web application** for real-time transaction scoring and explainability.

---

## 📑 Table of Contents
- [Executive Overview](#-executive-overview)
- [Why Traditional Fraud Systems Fail](#-why-traditional-fraud-systems-fail)
- [Model Architecture & Methodology](#-model-architecture--methodology)
  - [The 5 Behavioral Features](#the-5-behavioral-features)
  - [Strict Account-Split Validation (Zero Leakage)](#strict-account-split-validation-zero-leakage)
  - [Operational Thresholds & Policy Tradeoffs](#operational-thresholds--policy-tradeoffs)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Environment Setup](#-installation--environment-setup)
- [How to Run](#-how-to-run)
  - [1. Launch the Interactive Web Dashboard](#1-launch-the-interactive-web-dashboard)
  - [2. Run the Fast Parity Check](#2-run-the-fast-parity-check)
  - [3. Run the Full 2.5M Transaction Verification](#3-run-the-full-25m-transaction-verification)
- [Streamlit Web Interface Walkthrough](#-streamlit-web-interface-walkthrough)
- [Curated Test Cases (`streamlittest.csv`)](#-curated-test-cases-streamlittestcsv)
- [Benchmark Results](#-benchmark-results)

---

## 🚀 Executive Overview

Mobile money ecosystems handle millions of peer-to-peer transfers, bill payments, and cash-outs daily. Fraudulent activity in these systems is rare (~0.1% – 1% prevalence) and dynamic. This project provides:

1. **LightGBM Gradient Boosted Decision Tree**: A compact, portable text model (`model_strict_split_acct.txt`) optimized for millisecond-latency inference.
2. **Behavioral Feature Engineering (`features.py`)**: Computes running historical user baselines with zero forward-looking data leakage.
3. **Streamlit Intelligence App (`app.py`)**: An analyst-facing front end featuring manual parameter input, live fraud flagging, automated investigation notes, and batch CSV processing.
4. **Reproducibility Suite (`local_test.py` & `verify_features.py`)**: Validates that all scores and features match original benchmarks across 2,510,889 transfers down to floating-point precision ($< 10^{-9}$).

---

## 🔍 Why Traditional Fraud Systems Fail

| Static Rule Approach | Behavioral ML Approach (This Project) |
| :--- | :--- |
| **Fixed Thresholds** (e.g. Flag if amount > $5,000) causes high false alarms for legitimate high-net-worth users or merchants. | **Account-Specific Z-Scores**: Evaluates how unusual the amount is relative to that individual account's personal history. |
| **Blind to Micro-Draining**: Fraudsters draining accounts in multiple small amounts ($50–$100) fly under static limits. | **Cadence & Dormancy Detection**: Flags rapid-fire bursts or transfers that abruptly break long dormancy periods. |
| **Static Blacklists**: Stolen or purchased mule accounts have clean histories prior to the attack. | **Network Inflow Velocity**: Detects recipient wallets suddenly aggregating transfers from dozens of unique senders. |

---

## 🧠 Model Architecture & Methodology

### The 5 Behavioral Features

Rather than raw account IDs or dollar values, the model relies on five behavioral signals:

| Feature | Type | Formula / Logic | Fraud Detection Intuition |
| :--- | :--- | :--- | :--- |
| **`amt_z_own`** | Float | $$\frac{\ln(1 + \text{amt}) - \mu_{\text{own}}}{\sigma_{\text{own}} + 0.1}$$ | Deviation from initiator's own transfer history. A z-score of `+3.0` indicates an extreme transfer spike for that specific user. |
| **`hour`** | Integer | $\text{step} \pmod{24}$ | Time of day $(0 - 23)$. Fraud spikes disproportionately during early morning / overnight hours (1:00 AM – 4:00 AM). |
| **`steps_since_last`** | Float | $t_{\text{current}} - t_{\text{previous}}$ | Hours since this account's previous transfer. Detects "sleeper mules" or dormant compromised accounts suddenly waking up. |
| **`is_new_recipient`** | Binary | $1 \text{ if first transfer else } 0$ | Flag indicating the initiator has never sent money to this recipient before. High correlation with account takeover drainings. |
| **`recip_cnt_24`** | Float | Rolling 24h count | Number of transfers received by the destination account across the entire network in the past 24 hours (mule funnel detection). |

### Strict Account-Split Validation (Zero Leakage)
A common pitfall in tabular fraud modeling is random train/test splitting. If an account has 20 transactions and 15 are in training while 5 are in testing, the model merely memorizes the account identity rather than learning generalized fraud behavior.

- **Split Policy**: Evaluated on accounts **never seen during training**.
- **Time Scope**: Evaluated on subsequent time periods ($\text{step} \ge 24$) with a required minimum history ($\ge 5$ prior transfers) to ensure statistical validity.

### Operational Thresholds & Policy Tradeoffs
Fraud operations teams cannot manually review 10% of all transactions. Detection thresholds are calibrated on validation false-positive rates:

- **1% FPR Policy (Default: `0.8457`)**: Flags approximately **11.9 alerts per 1,000 transactions** at 1% fraud prevalence. Minimizes analyst alert fatigue while catching anomalous transfers.
- **5% FPR Policy (`0.8023`)**: Captures higher fraud recall when higher operational capacity is available.

---

## 📂 Project Directory Structure

```text
FraudDetection/
├── app.py                     # Streamlit interactive web dashboard
├── features.py                # Core feature engineering & scoring pipeline
├── local_test.py              # Fast parity verification script
├── verify_features.py         # Full 2.5M row end-to-end dataset audit
├── streamlittest.csv          # Sample batch test file (15 curated transactions)
├── dataaudit.py               # Exploratory data profiling script
├── README.md                  # Project documentation
├── .venv/                     # Python 3.11 virtual environment
└── Synthetic Mobile Money Transaction Dataset/
    ├── MoMTSim_20240722202413_1000_dataset.csv     # Raw dataset (2.5M rows)
    ├── synthetic_mobile_money_transaction_dataset.csv
    └── artifacts/
        ├── model_strict_split_acct.txt             # LightGBM booster model
        ├── scoring_bundle.json                     # Feature list, thresholds, reference quantiles
        ├── dashboard_alerts.csv                    # 11,812 pre-scored test alerts with SHAP values
        ├── example_alerts.csv                      # Sample alert narratives
        ├── model_data.parquet                      # Scored test dataset (155,176 test rows)
        └── final_results.csv                       # Benchmark evaluation metrics
```

---

## ⚙️ Installation & Environment Setup

### Prerequisites
- Python 3.11 (64-bit recommended)
- PowerShell (Windows) or Bash (macOS/Linux)

### Setup Virtual Environment
```powershell
# Navigate to the project directory
cd C:\Users\linzs\Projects\FraudDetection

# Create virtual environment with Python 3.11 (if not already present)
py -3.11 -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

### Install Required Dependencies
```powershell
pip install lightgbm pandas pyarrow scipy numpy streamlit
```

---

## 🏃 How to Run

### 1. Launch the Interactive Web Dashboard
Run the Streamlit application to test and visualize the model in your browser:
```powershell
streamlit run app.py
```
Your default browser will automatically open **`http://localhost:8501`**.

---

### 2. Run the Fast Parity Check
Verifies model predictions, evaluation thresholds, and test-set row counts:
```powershell
python local_test.py
```
**Expected Output:**
```text
PASS - feature columns present 
PASS - no missing or infinite feature values 
PASS - test row count (155176)
PASS - scores match saved dashboard scores (max difference 2.22e-16)
PASS - alert flags match saved flags 
PASS - every test alert is in the saved file (local 6812, saved 6812)
PASS - false-positive rate matches the table (0.0114)
PASS - recall matches the table (0.0618)
Alerts per 1,000 transactions at 1% prevalence: 11.9 (table: 11.9)
Single-row score: 0.7041768731878637

All checks passed
```

---

### 3. Run the Full 2.5M Transaction Verification
Processes the entire 366 MB raw CSV dataset (`MoMTSim`), re-derives all 5 behavioral features from scratch, and checks values against precomputed parquet tables:
```powershell
python verify_features.py
```
**Expected Output:**
```text
PASS - cleaned transfer count (2510889)
Status counts: {'insufficient_history': 1663179, 'scored': 788922, 'warm_up': 58788}
PASS - scored rows 
PASS - warm-up rows 
PASS - insufficient-history rows 
PASS - same rows in scope 
PASS - feature amt_z_own matches (max difference 2.34e-13)
PASS - feature hour matches
PASS - feature steps_since_last matches
PASS - feature is_new_recipient matches
PASS - feature recip_cnt_24 matches
PASS - all dashboard rows found (11812 of 11812)
PASS - scores match dashboard file
PASS - alert flags match dashboard file 
PASS - unscored rows have no score or alert 

Elapsed: ~45 s
All checks passed
```

---

## 🖥️ Streamlit Web Interface Walkthrough

The web dashboard (`app.py`) is structured into five functional modules:

### 1. 🎮 Manual Transaction Testing & Flagging
- **Manual Input**: Direct numeric/text input fields for all 5 features (`amt_z_own`, `hour`, `steps_since_last`, `is_new_recipient`, `recip_cnt_24`).
- **Real-time Scoring Verdict**:
  - 🚨 **CRITICAL FRAUD ALERT** (Red): Score > active threshold. Suggests immediate block or step-up authentication.
  - ⚠️ **ELEVATED SUSPICION** (Amber): Score elevated ($> 0.65$) but below cut-off. Suggests background monitoring.
  - ✅ **TRANSACTION CLEARED** (Green): Standard behavioral pattern.
- **Percentile Benchmarking**: Compares submitted inputs against legitimate transfer medians (e.g., *"Amount deviation is higher than 99.2% of legitimate transfers"*).
- **Automated Case Notes**: Synthesizes investigator-ready narrative findings.

### 2. 🧮 Raw-to-Feature Calculator
- Enter raw figures (e.g. *Current Transfer: \$2,500*, *Account Average: \$100*, *Dormancy: 18 hours*).
- Automatically calculates mathematical log-z-scores and tests against the model.

### 3. 🧠 How the Model Works
- Comprehensive interactive documentation on behavioral fraud modeling, zero-leakage validation, and operational economics.

### 4. 📊 Historical Alerts Explorer
- Filter and search through 11,812 pre-scored test alerts from `dashboard_alerts.csv`.
- Inspect model explanation text (`alert_text`) and individual feature values.

### 5. 📁 Batch CSV Scorer
- Drag-and-drop any CSV containing the 5 features.
- Evaluates transactions in bulk, adds `risk_score` and `alert` columns, and provides a one-click **"Download Scored CSV"** button.

---

## 🧪 Curated Test Cases (`streamlittest.csv`)

A test dataset with 15 representative transactions is provided in the root directory:

| Transaction ID | Profile Description | `amt_z_own` | `hour` | `steps_since_last` | `is_new_recipient` | `recip_cnt_24` | Expected Score | Alert Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `TXN_001` | Normal recurring bill payment | `+0.05` | 14 | 1.0 | 0 | 15.0 | `0.6366` | 🟢 Cleared |
| `TXN_002` | Standard peer transfer | `-0.41` | 12 | 0.0 | 0 | 12.0 | `0.2980` | 🟢 Cleared |
| `TXN_003` | Small grocery payment | `-0.33` | 16 | 0.0 | 1 | 12.0 | `0.4442` | 🟢 Cleared |
| `TXN_004` | Rent / salary to known contact | `+0.85` | 10 | 24.0 | 0 | 10.0 | `0.6987` | 🟢 Cleared |
| `TXN_005` | Elevated amount to new peer | `+0.98` | 13 | 0.0 | 1 | 8.0 | `0.8094` | 🟡 Monitor |
| `TXN_011` | **Dormant flare (8h inactivity)** | `+0.09` | 2 | 8.0 | 1 | 7.0 | `0.9559` | 🚨 **ALERT** |
| `TXN_012` | **Night transfer (10h gap)** | `-0.25` | 3 | 10.0 | 1 | 12.0 | `0.9264` | 🚨 **ALERT** |
| `TXN_013` | **Massive amount spike (z=+3.1)** | `+3.10` | 1 | 7.0 | 1 | 3.0 | `0.9494` | 🚨 **ALERT** |
| `TXN_014` | **Long dormant flare (18h gap)** | `+0.20` | 4 | 18.0 | 1 | 8.0 | `0.9605` | 🚨 **ALERT** |
| `TXN_015` | **11h inactivity to new recipient** | `-0.17` | 16 | 11.0 | 1 | 12.0 | `0.8949` | 🚨 **ALERT** |

---

## 📈 Benchmark Results

Evaluation on the holdout test set (unseen accounts at $\text{step} \ge 24$, 155,176 evaluated transactions):

- **False Positive Rate at 1% Target**: `0.0114` (~1.1%)
- **True Positive Recall at 1% Target**: `0.0618` (~6.2%)
- **Alert Rate**: **11.9 alerts per 1,000 transactions** (at 1% fraud prevalence)
- **Feature Computation Throughput**: ~55,000 transactions/second in pandas/numpy
- **Model Inference Speed**: $< 0.1$ ms per transaction

---

## 📄 License
This project is open-source under the MIT License.
