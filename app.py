"""Comprehensive Fraud Detection Web Application & Investigation Interface.

Features:
- Interactive Manual Feature Input (text fields & numeric inputs) with Live Fraud Flagging
- Real-time Explainability: Percentiles, Anomaly Scores & Analyst Investigation Summary
- Raw-to-Feature Calculator (converts raw currency amounts to account z-scores)
- Complete Model Architecture & Behavioral Methodology Documentation
- Historical Alerts Inspector with One-Click Load to Simulator
- Batch CSV Processor with Downloadable Scored Reports
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import streamlit as st

# -----------------------------------------------------------------------------
# App Layout & Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Mobile Money Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished dashboard look
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .card { background-color: #F9FAFB; padding: 1.25rem; border-radius: 0.5rem; border: 1px solid #E5E7EB; margin-bottom: 1rem; }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .alert-box { padding: 1rem; border-radius: 0.5rem; font-weight: 600; margin-bottom: 1rem; }
    .alert-high { background-color: #FEE2E2; color: #991B1B; border: 1px solid #F87171; }
    .alert-medium { background-color: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; }
    .alert-low { background-color: #D1FAE5; color: #065F46; border: 1px solid #34D399; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Paths and Resource Loading
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = (
    BASE_DIR / "Synthetic Mobile Money Transaction Dataset"
    if (BASE_DIR / "Synthetic Mobile Money Transaction Dataset").exists()
    else BASE_DIR
)
ARTIFACTS_DIR = DATASET_DIR / "artifacts"
MODEL_FILE = ARTIFACTS_DIR / "model_strict_split_acct.txt"
BUNDLE_FILE = ARTIFACTS_DIR / "scoring_bundle.json"
ALERTS_FILE = ARTIFACTS_DIR / "dashboard_alerts.csv"


@st.cache_resource
def load_booster():
    if not MODEL_FILE.exists():
        st.error(f"Model file not found at: {MODEL_FILE}")
        return None
    return lgb.Booster(model_file=str(MODEL_FILE))


@st.cache_data
def load_scoring_bundle():
    if not BUNDLE_FILE.exists():
        st.error(f"Scoring bundle not found at: {BUNDLE_FILE}")
        return None
    with open(BUNDLE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_alerts_data():
    if not ALERTS_FILE.exists():
        return None
    return pd.read_csv(ALERTS_FILE)


booster = load_booster()
bundle = load_scoring_bundle()

if not booster or not bundle:
    st.stop()

FEATS = bundle["features"]
THR_1PCT = bundle["thresholds"]["fpr_1pct"]
THR_5PCT = bundle["thresholds"]["fpr_5pct"]
QUANTILES = bundle.get("reference_quantiles", {})


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def compute_percentiles(input_dict):
    """Calculate where the inputs lie relative to reference legitimate transactions."""
    percentiles = {}
    for feat in ["amt_z_own", "steps_since_last", "recip_cnt_24"]:
        if feat in QUANTILES:
            val = input_dict[feat]
            q_arr = np.array(QUANTILES[feat])
            pct = float((q_arr < val).mean() * 100)
            percentiles[feat] = pct
        else:
            percentiles[feat] = 50.0
    return percentiles


def generate_analyst_report(input_dict, score, threshold):
    """Generate concise case notes for fraud operations."""
    reasons = []
    pcts = compute_percentiles(input_dict)

    if input_dict["amt_z_own"] > 1.5:
        reasons.append(
            f"**Amount Deviation (z = {input_dict['amt_z_own']:+.2f})**: Exceeds {pcts['amt_z_own']:.1f}% of legitimate transfers. The transaction size is exceptionally higher than the account's historical baseline."
        )
    elif input_dict["amt_z_own"] < -1.0:
        reasons.append(
            f"**Amount Deviation (z = {input_dict['amt_z_own']:+.2f})**: Lower than usual transfer size."
        )

    if input_dict["steps_since_last"] >= 6:
        reasons.append(
            f"**Dormancy Awakening ({input_dict['steps_since_last']:.0f} hours)**: Longer inactivity than {pcts['steps_since_last']:.1f}% of legitimate transfers. Sudden activity following extended dormancy is a key indicator of compromised accounts or sleeper mules."
        )

    if input_dict["is_new_recipient"] == 1:
        reasons.append(
            "**Unfamiliar Counterparty**: First recorded transaction between this initiating account and this recipient."
        )

    if input_dict["recip_cnt_24"] <= 2:
        reasons.append(
            f"**Low Recipient Activity ({input_dict['recip_cnt_24']:.0f} incoming txns/24h)**: Counterparty has sparse incoming velocity, common in freshly registered cash-out endpoints."
        )
    elif input_dict["recip_cnt_24"] >= 25:
        reasons.append(
            f"**High Recipient Velocity ({input_dict['recip_cnt_24']:.0f} incoming txns/24h)**: Recipient is rapidly aggregating transfers across multiple senders (mule aggregation pattern)."
        )

    if input_dict["hour"] in [1, 2, 3, 4, 5]:
        reasons.append(
            f"**Unusual Time Window ({int(input_dict['hour']):02d}:00 hours)**: Execution during off-peak overnight hours when account holder may not notice unauthorized activity."
        )

    return reasons


# -----------------------------------------------------------------------------
# Sidebar: Settings & Preloaded Presets
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 🛡️ Fraud Guard UI")
st.sidebar.caption("Behavioural Fraud Detection & Decision Engine")

threshold_mode = st.sidebar.selectbox(
    "Alert Sensitivity Threshold",
    [
        f"1% FPR Strict Policy ({THR_1PCT:.4f})",
        f"5% FPR Sensitive Policy ({THR_5PCT:.4f})",
        "Custom Manual Threshold",
    ],
)

if "Custom" in threshold_mode:
    ACTIVE_THR = st.sidebar.slider("Custom Threshold", 0.0, 1.0, float(THR_1PCT), 0.005)
elif "5% FPR" in threshold_mode:
    ACTIVE_THR = THR_5PCT
else:
    ACTIVE_THR = THR_1PCT

st.sidebar.info(
    f"**Active Alert Cut-off:** `{ACTIVE_THR:.4f}`\n\n"
    "Transactions with a predicted probability above this cut-off will be flagged as **FRAUD ALERTS**."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Preset Scenarios")
st.sidebar.caption("Click to load archetypal transaction profiles into the manual test fields:")

if "input_amt_z" not in st.session_state:
    st.session_state.input_amt_z = 0.0
    st.session_state.input_hour = 14
    st.session_state.input_steps = 1.0
    st.session_state.input_new_recip = 0
    st.session_state.input_recip_cnt = 12.0

col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    if st.button("🟢 Normal Peer"):
        st.session_state.input_amt_z = 0.05
        st.session_state.input_hour = 14
        st.session_state.input_steps = 1.0
        st.session_state.input_new_recip = 0
        st.session_state.input_recip_cnt = 14.0
        st.rerun()

    if st.button("🔴 Dormant Flare"):
        st.session_state.input_amt_z = 0.15
        st.session_state.input_hour = 2
        st.session_state.input_steps = 18.0
        st.session_state.input_new_recip = 1
        st.session_state.input_recip_cnt = 8.0
        st.rerun()

with col_p2:
    if st.button("🔴 Amount Spike"):
        st.session_state.input_amt_z = 3.20
        st.session_state.input_hour = 1
        st.session_state.input_steps = 7.0
        st.session_state.input_new_recip = 1
        st.session_state.input_recip_cnt = 3.0
        st.rerun()

    if st.button("🟡 High Velocity"):
        st.session_state.input_amt_z = 0.90
        st.session_state.input_hour = 16
        st.session_state.input_steps = 0.0
        st.session_state.input_new_recip = 1
        st.session_state.input_recip_cnt = 32.0
        st.rerun()


# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_test, tab_calc, tab_explain, tab_explore, tab_batch = st.tabs(
    [
        "🎮 Manual Transaction Testing & Flagging",
        "🧮 Raw Transaction to Feature Calculator",
        "🧠 How the Model Works",
        "📊 Historical Alerts Inspector",
        "📁 Batch CSV Scorer",
    ]
)


# =============================================================================
# TAB 1: Manual Feature Input & Real-time Fraud Flagging
# =============================================================================
with tab_test:
    st.markdown('<div class="main-header">Manual Transaction Scorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Enter feature values directly into the text/number fields to evaluate transactions in real time against the LightGBM behavioral model.</div>',
        unsafe_allow_html=True,
    )

    with st.form("manual_scoring_form"):
        st.markdown("### 1. Input Transaction Features")
        st.caption("You can type exact values into the input boxes below or adjust them using the step buttons.")

        c1, c2, c3 = st.columns(3)

        with c1:
            amt_z_own = st.number_input(
                "Amount Deviation Z-score (`amt_z_own`)",
                value=float(st.session_state.input_amt_z),
                min_value=-10.0,
                max_value=30.0,
                step=0.1,
                format="%.2f",
                help="Normalized log-amount deviation from the account's historical average: (ln(amt) - mean) / (std + 0.1). 0.0 = exact average; +3.0 = extreme spike.",
            )
            hour = st.number_input(
                "Hour of Day (`hour`)",
                value=int(st.session_state.input_hour),
                min_value=0,
                max_value=23,
                step=1,
                help="Hour of the transfer (0 to 23). Overnight transfers (1:00 - 4:00 AM) often correlate with fraud.",
            )

        with c2:
            steps_since_last = st.number_input(
                "Hours Since Last Transfer (`steps_since_last`)",
                value=float(st.session_state.input_steps),
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                format="%.1f",
                help="Hours of account inactivity before this transfer. Values >= 6 indicate awakening from dormancy.",
            )
            is_new_recipient = st.selectbox(
                "Recipient Relationship (`is_new_recipient`)",
                options=[0, 1],
                index=int(st.session_state.input_new_recip),
                format_func=lambda x: "1: New Recipient (First time sending to this account)" if x == 1 else "0: Repeat Recipient (Prior transfer history)",
                help="Binary flag: 1 if this account has never sent to this recipient before; 0 otherwise.",
            )

        with c3:
            recip_cnt_24 = st.number_input(
                "Recipient 24h Inflow Count (`recip_cnt_24`)",
                value=float(st.session_state.input_recip_cnt),
                min_value=0.0,
                max_value=150.0,
                step=1.0,
                format="%.1f",
                help="Number of incoming transfers the recipient received across all accounts in the preceding 24 hours.",
            )
            st.markdown("<br>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("🔍 Evaluate Transaction & Score", use_container_width=True)

    # Prediction computation
    current_input = {
        "amt_z_own": float(amt_z_own),
        "hour": int(hour),
        "steps_since_last": float(steps_since_last),
        "is_new_recipient": int(is_new_recipient),
        "recip_cnt_24": float(recip_cnt_24),
    }
    input_df = pd.DataFrame([current_input])[FEATS]
    score = float(booster.predict(input_df)[0])
    is_alert = score > ACTIVE_THR

    # Decision Banner
    st.markdown("---")
    st.markdown("### 2. Live Model Verdict")

    if is_alert:
        st.markdown(
            f"""
            <div class="alert-box alert-high">
                🚨 FRAUD ALERT FLAGGED — High Risk Anomaly Detected<br>
                <span style="font-size:0.95rem; font-weight:normal;">
                Risk Score (<b>{score:.4f}</b>) exceeds the detection threshold (<b>{ACTIVE_THR:.4f}</b>). 
                Recommended Action: <b>Block transaction or trigger step-up multi-factor authentication</b>.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif score > 0.65:
        st.markdown(
            f"""
            <div class="alert-box alert-medium">
                ⚠️ ELEVATED SUSPICION — Below Alert Cut-off<br>
                <span style="font-size:0.95rem; font-weight:normal;">
                Risk Score (<b>{score:.4f}</b>) is elevated but below the strict cut-off (<b>{ACTIVE_THR:.4f}</b>). 
                Recommended Action: <b>Permit transfer with background telemetry monitoring</b>.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="alert-box alert-low">
                ✅ TRANSACTION CLEARED — Normal Behavioral Profile<br>
                <span style="font-size:0.95rem; font-weight:normal;">
                Risk Score (<b>{score:.4f}</b>) is well within normal behavioral bounds (Cut-off: <b>{ACTIVE_THR:.4f}</b>). 
                Recommended Action: <b>Authorize immediately</b>.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Metric Cards
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Fraud Probability", f"{score:.4f}", f"{score * 100:.1f}%")
    with m_col2:
        st.metric("Active Cut-off", f"{ACTIVE_THR:.4f}")
    with m_col3:
        st.metric("Alert Status", "ALERT" if is_alert else "PASS")
    with m_col4:
        decision_label = "BLOCK / REVIEW" if is_alert else ("MONITOR" if score > 0.65 else "APPROVE")
        st.metric("Policy Action", decision_label)

    st.progress(min(max(score, 0.0), 1.0))

    # Feature Breakdown & Benchmarking Table
    st.markdown("### 3. Feature Breakdown & Reference Benchmarks")
    pcts = compute_percentiles(current_input)

    breakdown_data = [
        {
            "Feature Name": "amt_z_own (Amount Deviation)",
            "Submitted Value": f"{amt_z_own:+.2f}",
            "Legitimate Population Median": "-0.26",
            "Percentile Rank": f"{pcts['amt_z_own']:.1f}%",
            "Impact on Risk": "🔴 Pushes UP" if amt_z_own > 1.0 else ("🟢 Normal" if amt_z_own < 0.5 else "🟡 Slight UP"),
        },
        {
            "Feature Name": "hour (Execution Time)",
            "Submitted Value": f"{hour:02d}:00",
            "Legitimate Population Median": "13:00",
            "Percentile Rank": f"Hour {hour}",
            "Impact on Risk": "🔴 Pushes UP (Night)" if hour in [1, 2, 3, 4] else "🟢 Normal Daytime",
        },
        {
            "Feature Name": "steps_since_last (Hours Dormant)",
            "Submitted Value": f"{steps_since_last:.1f} hrs",
            "Legitimate Population Median": "0.0 hrs (active)",
            "Percentile Rank": f"{pcts['steps_since_last']:.1f}%",
            "Impact on Risk": "🔴 Pushes UP (Long gap)" if steps_since_last >= 6 else "🟢 Normal cadence",
        },
        {
            "Feature Name": "is_new_recipient (First-time Send)",
            "Submitted Value": "Yes (1)" if is_new_recipient == 1 else "No (0)",
            "Legitimate Population Median": "No (0)",
            "Percentile Rank": "N/A (Binary)",
            "Impact on Risk": "🔴 Pushes UP" if is_new_recipient == 1 else "🟢 Familiar recipient",
        },
        {
            "Feature Name": "recip_cnt_24 (Recipient Inflow Velocity)",
            "Submitted Value": f"{recip_cnt_24:.0f} txns",
            "Legitimate Population Median": "10.0 txns",
            "Percentile Rank": f"{pcts['recip_cnt_24']:.1f}%",
            "Impact on Risk": "🔴 Pushes UP" if recip_cnt_24 <= 2 or recip_cnt_24 >= 30 else "🟢 Standard inflow",
        },
    ]

    st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)

    # Analyst Narrative Notes
    st.markdown("### 4. Automated Fraud Analyst Notes")
    report_reasons = generate_analyst_report(current_input, score, ACTIVE_THR)
    if is_alert and report_reasons:
        for r in report_reasons:
            st.markdown(f"- {r}")
    elif not is_alert and score < 0.35:
        st.info("No anomalous risk factors identified. The transfer displays normal consumer transaction patterns.")
    elif report_reasons:
        for r in report_reasons:
            st.markdown(f"- {r}")


# =============================================================================
# TAB 2: Raw Transaction to Feature Calculator
# =============================================================================
with tab_calc:
    st.markdown('<div class="main-header">Raw Transaction to Feature Calculator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Convert real-world transaction figures (currency amounts, account history, hours since last transfer) into the model\'s 5 input features.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        The model does not consume raw dollar/shilling amounts directly because transfer sizes vary vastly across users ($10 for a student vs $10,000 for a business).
        Instead, it evaluates **how unusual the amount is relative to that specific account's historical baseline**.
        """
    )

    c_c1, c_c2 = st.columns(2)

    with c_c1:
        st.subheader("Account History & Current Transfer")
        raw_amt = st.number_input("Current Transfer Amount ($)", value=2500.0, min_value=1.0, step=50.0)
        typical_amt = st.number_input("Account's Average Historical Transfer ($)", value=100.0, min_value=1.0, step=10.0)
        account_spread = st.number_input("Account's Historical Variance/Spread (Std Dev)", value=0.5, min_value=0.01, step=0.1)

    with c_c2:
        st.subheader("Cadence & Destination")
        calc_hour = st.slider("Hour of Transaction (0-23)", 0, 23, 2)
        calc_hours_dormant = st.number_input("Hours Since Account's Previous Transfer", value=18.0, min_value=0.0, step=1.0)
        calc_is_new = st.checkbox("First time transferring to this recipient?", value=True)
        calc_inflow = st.number_input("Recipient's Inflow Transactions in Last 24 Hours", value=4.0, min_value=0.0, step=1.0)

    # Compute z-score
    log_amt = np.log1p(raw_amt)
    log_typical = np.log1p(typical_amt)
    calculated_z = (log_amt - log_typical) / (account_spread + 0.1)

    st.markdown("---")
    st.subheader("Calculated Feature Vector")

    cf_col1, cf_col2, cf_col3, cf_col4, cf_col5 = st.columns(5)
    cf_col1.metric("`amt_z_own`", f"{calculated_z:+.2f}")
    cf_col2.metric("`hour`", f"{calc_hour}")
    cf_col3.metric("`steps_since_last`", f"{calc_hours_dormant:.1f}")
    cf_col4.metric("`is_new_recipient`", f"{1 if calc_is_new else 0}")
    cf_col5.metric("`recip_cnt_24`", f"{calc_inflow:.0f}")

    # Immediate Test with Calculated Features
    calc_feat_dict = {
        "amt_z_own": calculated_z,
        "hour": calc_hour,
        "steps_since_last": calc_hours_dormant,
        "is_new_recipient": 1 if calc_is_new else 0,
        "recip_cnt_24": calc_inflow,
    }
    calc_score = float(booster.predict(pd.DataFrame([calc_feat_dict])[FEATS])[0])
    calc_alert = calc_score > ACTIVE_THR

    st.markdown("### Resulting Fraud Score")
    if calc_alert:
        st.error(f"🚨 **FRAUD ALERT TRIGGERED** — Risk Score: **{calc_score:.4f}** (Cut-off: {ACTIVE_THR:.4f})")
    else:
        st.success(f"✅ **CLEARED** — Risk Score: **{calc_score:.4f}** (Cut-off: {ACTIVE_THR:.4f})")


# =============================================================================
# TAB 3: Model Architecture & Behavioral Methodology Documentation
# =============================================================================
with tab_explain:
    st.markdown('<div class="main-header">How the Fraud Detection Model Works</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Full architectural breakdown: features, strict account-split validation, and alert thresholding.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### 1. Problem Formulation: Behavioral Fraud Detection
        Mobile money fraud commonly occurs via **Account Takeover (ATO)**, **credential theft**, and **money muling**. 
        Traditional rule engines (e.g., "flag all transactions over $5,000") suffer from two critical flaws:
        1. **High False Alarms**: Normal high-net-worth users or small businesses making legitimate large transfers get blocked.
        2. **Blind to Micro-Fraud**: Fraudsters who drain accounts in small batches bypass static thresholds.

        This model evaluates **deviation from the initiating account's own established behavioral baseline**.
        """
    )

    st.markdown(
        """
        ### 2. The 5 Predictive Behavioral Features
        | Feature Name | Definition | Fraud Intuition |
        | :--- | :--- | :--- |
        | `amt_z_own` | Normalized z-score of $\ln(1+\\text{amount})$ against the account's historical mean and variance. | Measures if the transfer is abnormally large **for this specific user**. |
        | `hour` | Hour of day $(0 - 23)$ calculated as $\\text{step} \\pmod{24}$. | Fraudsters frequently operate during late-night / early-morning windows. |
        | `steps_since_last` | Time elapsed (in hours) since this account's previous transfer. | Sleeper accounts or compromised dormant wallets suddenly becoming active. |
        | `is_new_recipient` | Binary flag $(0 \\text{ or } 1)$ indicating first transaction to this destination. | Draining funds to a previously unseen recipient wallet. |
        | `recip_cnt_24` | Rolling 24-hour count of all incoming transfers to this recipient platform-wide. | Captures mule accumulation funnels (recipients receiving funds from dozens of victims). |
        """
    )

    st.markdown(
        """
        ### 3. Strict Account-Split Methodology (Zero Leakage)
        In tabular fraud modeling, randomly splitting rows into train/test sets causes **account leakage**: the model memorizes account IDs rather than generalizable fraud behavior.
        - **Training Data**: Scored exclusively on a subset of accounts.
        - **Test Data**: Scored on completely unseen accounts evaluated over future time steps $(\\text{step} \\ge 24)$.
        - This guarantees that the model detects behavioral anomalies on **brand-new accounts** that were never seen during training.
        """
    )

    st.markdown(
        """
        ### 4. Operating Points & Threshold Selection
        Fraud detection is a cost-tradeoff problem. If you flag 20% of all transactions, the fraud operations team will be overwhelmed with false alarms.
        - **Strict Policy (`fpr_1pct = 0.8457`)**: Tuned so that only **1%** of legitimate transactions trigger an alert. At this threshold, the model delivers ~11.9 alerts per 1,000 transactions at 1% fraud prevalence.
        - **Sensitive Policy (`fpr_5pct = 0.8023`)**: Captures higher fraud recall while accepting a 5% false-positive rate.
        """
    )


# =============================================================================
# TAB 4: Historical Alerts Explorer
# =============================================================================
with tab_explore:
    st.markdown('<div class="main-header">Historical Alerts Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Inspect real pre-scored transactions from the test set, view fraud investigation notes, and load cases into the simulator.</div>',
        unsafe_allow_html=True,
    )

    df_alerts = load_alerts_data()

    if df_alerts is None:
        st.warning("`dashboard_alerts.csv` was not found in artifacts directory.")
    else:
        tot_alerts = (df_alerts["alert"] == 1).sum()
        c_e1, c_e2, c_e3 = st.columns(3)
        c_e1.metric("Sampled Validation Rows", f"{len(df_alerts):,}")
        c_e2.metric("Flagged Fraud Alerts", f"{tot_alerts:,}", f"{(tot_alerts/len(df_alerts))*100:.1f}%")
        c_e3.metric("Max Risk Score", f"{df_alerts['risk_score'].max():.4f}")

        filter_choice = st.radio(
            "Filter transactions by category:",
            ["Flagged Alerts Only (alert == 1)", "Confirmed Fraud Cases (eval_only_true_label == 1)", "All Rows"],
            horizontal=True,
        )

        if "Flagged Alerts" in filter_choice:
            view_df = df_alerts[df_alerts["alert"] == 1]
        elif "Confirmed Fraud" in filter_choice:
            view_df = df_alerts[df_alerts["eval_only_true_label"] == 1]
        else:
            view_df = df_alerts

        display_columns = [
            "row_id",
            "risk_score",
            "alert",
            "amt_z_own",
            "hour",
            "steps_since_last",
            "is_new_recipient",
            "recip_cnt_24",
            "alert_text",
        ]
        valid_cols = [c for c in display_columns if c in view_df.columns]

        st.dataframe(
            view_df[valid_cols].sort_values("risk_score", ascending=False).head(100),
            use_container_width=True,
            height=400,
        )


# =============================================================================
# TAB 5: Batch CSV Scorer
# =============================================================================
with tab_batch:
    st.markdown('<div class="main-header">Batch CSV Transaction Scorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Upload a batch CSV file (such as streamlittest.csv) to evaluate hundreds or thousands of transactions in one click.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(batch_df):,} rows.")

            has_required_feats = all(f in batch_df.columns for f in FEATS)

            if has_required_feats:
                # Compute scores
                scores = booster.predict(batch_df[FEATS])
                batch_df["risk_score"] = np.round(scores, 4)
                batch_df["alert"] = (scores > ACTIVE_THR).astype(int)
                batch_df["verdict"] = np.where(batch_df["alert"] == 1, "🚨 FRAUD ALERT", "✅ CLEARED")

                alert_cnt = int(batch_df["alert"].sum())
                b_c1, b_c2, b_c3 = st.columns(3)
                b_c1.metric("Rows Processed", f"{len(batch_df):,}")
                b_c2.metric("Alerts Triggered", f"{alert_cnt:,}", f"{(alert_cnt/len(batch_df))*100:.1f}%")
                b_c3.metric("Threshold Applied", f"{ACTIVE_THR:.4f}")

                st.markdown("### Preview of Scored Results")
                st.dataframe(batch_df.head(100), use_container_width=True)

                out_csv = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Complete Scored CSV",
                    data=out_csv,
                    file_name="scored_fraud_batch.csv",
                    mime="text/csv",
                )
            else:
                st.error(
                    f"Uploaded CSV must contain the model feature columns: `{FEATS}`.\n\n"
                    f"Columns found in uploaded file: `{list(batch_df.columns)}`"
                )
        except Exception as err:
            st.error(f"Error processing CSV: {err}")
