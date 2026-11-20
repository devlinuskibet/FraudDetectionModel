# Imports used by the local test
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

# Folder holding the files downloaded from Kaggle
DIR = Path(__file__).resolve().parent / "artifacts"
if not DIR.exists():
    DIR = Path(__file__).resolve().parent / "Synthetic Mobile Money Transaction Dataset" / "artifacts"

# Load the scoring bundle (feature list, thresholds) and the portable model
with open(DIR / "scoring_bundle.json", encoding="utf-8") as f:
    bundle = json.load(f)
FEATS = bundle["features"]                                   # feature order the model expects
THR = bundle["thresholds"]["fpr_1pct"]                       # alert threshold at 1% validation FPR
booster = lgb.Booster(model_file=str(DIR / "model_strict_split_acct.txt"))

# Load the prepared data and keep the account-split test rows in the final scope
data = pd.read_parquet(DIR / "model_data.parquet")
test = data[(data["step"] >= 24) & (data["split_acct"] == "test")].reset_index(drop=True)
y = test["isFraud"].to_numpy()                               # labels, used for evaluation only

results = []                                                 # (test name, passed) pairs
def check(name, ok, detail=""):
    # Record and print one PASS/FAIL line
    results.append(ok)
    print(("PASS" if ok else "FAIL"), "-", name, detail)

# Test 1: inputs are complete (right columns, no missing or infinite values)
check("feature columns present", all(f in test.columns for f in FEATS))
check("no missing or infinite feature values", bool(np.isfinite(test[FEATS].to_numpy()).all()))

# Test 2: the saved test-set size matches the final evaluation (155,176 rows)
check("test row count", len(test) == 155176, f"({len(test)})")

# Test 3: scores reproduce the risk scores saved on Kaggle
test["score"] = booster.predict(test[FEATS])                 # fraud probability for every test row
saved = pd.read_csv(DIR / "dashboard_alerts.csv")            # alerts and a random sample of unflagged rows
merged = saved.merge(test[["row_id", "score"]], on="row_id", how="left")
max_diff = float(np.abs(merged["risk_score"] - merged["score"]).max())
check("scores match saved dashboard scores", max_diff < 1e-6, f"(max difference {max_diff:.2e})")

# Test 4: the alert flag at the saved threshold matches the saved alert column
local_alert = (merged["score"] > THR).astype(int)
check("alert flags match saved flags", bool((local_alert == merged["alert"]).all()))
check("every test alert is in the saved file",
      int((test["score"] > THR).sum()) == int(saved["alert"].sum()),
      f"(local {int((test['score'] > THR).sum())}, saved {int(saved['alert'].sum())})")

# Test 5: the operating point reproduces the final table (strict model, account split, 1% target)
flag = test["score"].to_numpy() > THR
fpr = flag[y == 0].mean()                                    # table: 0.011
recall = flag[y == 1].mean()                                 # table: 0.062
check("false-positive rate matches the table", abs(fpr - 0.011) < 0.001, f"({fpr:.4f})")
check("recall matches the table", abs(recall - 0.062) < 0.001, f"({recall:.4f})")
print("Alerts per 1,000 transactions at 1% prevalence:",
      round(1000 * (0.01 * recall + 0.99 * fpr), 1), "(table: 11.9)")

# Test 6: smoke test on a single transaction, as a dashboard would score it
one = pd.DataFrame([test.loc[0, FEATS].to_dict()])           # one row built from a dictionary of feature values
print("Single-row score:", float(booster.predict(one[FEATS])[0]))

# Summary
print("\nAll checks passed" if all(results) else "\nSome checks FAILED")