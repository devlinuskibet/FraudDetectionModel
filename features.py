"""Feature pipeline and scoring for the strict behavioural fraud model.

Input: raw MoMTSim-style transactions with the columns step, transactionType, amount,
initiator and recipient. The same rules as the training pipeline are applied, so the
features match the saved training table.
"""

# Imports used by the module
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

# Constants that must match the training pipeline
Z_FLOOR = 0.1          # floor added to the account's standard deviation in the z-score
WINDOW = 24            # recipient activity window, in steps
MIN_HISTORY = 5        # minimum earlier transfers by the initiator for a row to be scored
FEATS = ["amt_z_own", "hour", "steps_since_last", "is_new_recipient", "recip_cnt_24"]   # model inputs, in order

# Default paths to model artifacts and raw dataset
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Synthetic Mobile Money Transaction Dataset" if (BASE_DIR / "Synthetic Mobile Money Transaction Dataset").exists() else BASE_DIR
DEFAULT_DIR = DATASET_DIR / "artifacts"
DEFAULT_MODEL_FILE = DEFAULT_DIR / "model_strict_split_acct.txt"
DEFAULT_BUNDLE_FILE = DEFAULT_DIR / "scoring_bundle.json"
DEFAULT_RAW_FILE = DATASET_DIR / "MoMTSim_20240722202413_1000_dataset.csv"


def clean_transfers(raw):
    """Keep TRANSFER rows, drop exact duplicates and non-positive amounts, and keep a row_id for tracing."""
    # row_id is the position of the row in the raw file (the index of a freshly read CSV)
    df = raw if "row_id" in raw.columns else raw.rename_axis("row_id").reset_index()
    df = df[df["transactionType"] == "TRANSFER"]                  # only transfers are modelled
    dup = df.drop(columns="row_id").duplicated()                  # exact duplicates, ignoring row_id
    df = df[~dup & (df["amount"] > 0)]                            # drop duplicates and empty transfers
    return df.reset_index(drop=True)


def build_features(tr):
    """Compute the five model inputs and the history count from cleaned TRANSFER rows."""
    # Work on the needed columns only, with identifiers replaced by integer codes
    tr = tr[["row_id", "step", "amount", "initiator", "recipient"]].copy()
    tr["initiator"] = pd.factorize(tr["initiator"])[0]            # integer code per initiating account
    tr["recipient"] = pd.factorize(tr["recipient"])[0]            # integer code per recipient

    # Log-scaled amount, its square (for the running variance) and hour of day
    tr["amount_log"] = np.log1p(tr["amount"])
    tr["amount_log_sq"] = tr["amount_log"] ** 2
    tr["hour"] = tr["step"] % 24                                  # assumes one step is one hour

    # Sort by account then time, so every running statistic sees only earlier transfers
    tr = tr.sort_values(["initiator", "step", "row_id"], kind="stable").reset_index(drop=True)
    g = tr.groupby("initiator")

    # Account history: earlier transfer count, running mean and spread of the log amount
    tr["prior_count"] = g.cumcount()                              # earlier transfers by this account
    n = tr["prior_count"].where(tr["prior_count"] > 0)            # NaN when there is no history
    s1 = g["amount_log"].cumsum() - tr["amount_log"]              # sum of earlier log amounts
    s2 = g["amount_log_sq"].cumsum() - tr["amount_log_sq"]        # sum of earlier squared log amounts
    mean = s1 / n                                                 # account's typical amount so far
    var = (s2 / n) - mean ** 2                                    # running variance
    std = np.sqrt(var.clip(lower=0))                              # clip guards against tiny negative values

    # Deviation from the account's own norm, steps since its last transfer, first send to this recipient
    tr["amt_z_own"] = (tr["amount_log"] - mean) / (std + Z_FLOOR)
    tr["steps_since_last"] = g["step"].diff()
    tr["is_new_recipient"] = (~tr.duplicated(["initiator", "recipient"])).astype("int8")

    # Transfers received by the recipient in the previous WINDOW steps (current step excluded)
    rc = tr["recipient"].to_numpy().astype(np.int64)
    st = tr["step"].to_numpy().astype(np.int64)
    base = int(st.max()) + WINDOW + 2                             # multiplier larger than any step, so keys never collide
    keys = np.sort(rc * base + st)                                # one sorted key per transfer: recipient code and step
    hi = np.searchsorted(keys, rc * base + st, side="left")                           # received before this step
    lo = np.searchsorted(keys, rc * base + np.maximum(st - WINDOW, 0), side="left")   # received before the window start
    tr["recip_cnt_24"] = hi - lo

    return tr[["row_id", "step", "prior_count"] + FEATS]


def add_status(feat):
    """Label each row as scored, warm_up (incomplete recipient window) or insufficient_history."""
    start = feat["step"].min()                                    # first step in the supplied data
    feat["status"] = np.select(
        [feat["prior_count"] < MIN_HISTORY, feat["step"] - start < WINDOW],
        ["insufficient_history", "warm_up"], default="scored")
    return feat


def score(feat, model_file=None, bundle_file=None, threshold="fpr_1pct"):
    """Score the eligible rows and raise an alert when the score passes the saved threshold."""
    if model_file is None:
        model_file = DEFAULT_MODEL_FILE
    if bundle_file is None:
        bundle_file = DEFAULT_BUNDLE_FILE
    with open(bundle_file, encoding="utf-8") as f:
        bundle = json.load(f)                         # feature list and thresholds saved on Kaggle
    assert bundle["features"] == FEATS, "feature list differs from the module"
    booster = lgb.Booster(model_file=str(model_file))             # portable text model
    thr = bundle["thresholds"][threshold]                         # alert cut-off on the risk score
    feat = add_status(feat)
    ok = feat["status"] == "scored"                               # only these rows are scored
    feat["risk_score"] = np.nan                                   # other rows stay unscored
    if ok.any():
        feat.loc[ok, "risk_score"] = booster.predict(feat.loc[ok, FEATS])
    feat["alert"] = (feat["risk_score"] > thr).astype("int8")     # NaN compares as False, so unscored rows never alert
    return feat


def score_transactions(raw, model_file=None, bundle_file=None, threshold="fpr_1pct"):
    """Full path: clean raw transactions, build features, score and flag alerts."""
    return score(build_features(clean_transfers(raw)), model_file, bundle_file, threshold)


if __name__ == "__main__":
    import time
    print(f"Loading sample transactions from {DEFAULT_RAW_FILE.name}...")
    t0 = time.time()
    sample_raw = pd.read_csv(DEFAULT_RAW_FILE, nrows=20000)
    print(f"Scoring transactions...")
    scored_sample = score_transactions(sample_raw)
    alerts = int((scored_sample["alert"] == 1).sum())
    print(f"Done in {time.time() - t0:.2f}s: processed {len(scored_sample)} transactions, {alerts} alerts flagged.")