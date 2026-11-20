# Imports used by the verification
import time
from pathlib import Path
import numpy as np
import pandas as pd
import features as F

# Paths to the raw CSV and the folder with the downloaded Kaggle files
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Synthetic Mobile Money Transaction Dataset" if (BASE_DIR / "Synthetic Mobile Money Transaction Dataset").exists() else BASE_DIR
RAW = DATASET_DIR / "MoMTSim_20240722202413_1000_dataset.csv"
DIR = DATASET_DIR / "artifacts"

results = []                                                  # pass/fail record
def check(name, ok, detail=""):
    # Record and print one PASS/FAIL line
    results.append(ok)
    print(("PASS" if ok else "FAIL"), "-", name, detail)

t0 = time.time()

# 1. Read the raw file and remove the label, so the pipeline is tested without it
raw = pd.read_csv(RAW).drop(columns=["isFraud"])
cleaned = F.clean_transfers(raw)
del raw                                                       # free memory
check("cleaned transfer count", len(cleaned) == 2510889, f"({len(cleaned)})")

# 2. Build features, label each row and compare the status counts with the training scope
feat = F.add_status(F.build_features(cleaned))
del cleaned
counts = feat["status"].value_counts().to_dict()
print("Status counts:", counts)
check("scored rows", counts.get("scored") == 788922)
check("warm-up rows", counts.get("warm_up") == 58788)
check("insufficient-history rows", counts.get("insufficient_history") == 1663179)

# 3. Compare every feature with the table saved on Kaggle, row by row (matched on row_id)
saved = pd.read_parquet(DIR / "model_data.parquet", columns=["row_id", "step"] + F.FEATS)
saved = saved[saved["step"] >= 24]                            # final scope
ours = feat[feat["status"] == "scored"]
check("same rows in scope", set(ours["row_id"]) == set(saved["row_id"]))
m = ours.merge(saved, on="row_id", suffixes=("_new", "_saved"))
for f in F.FEATS:
    d = float(np.abs(m[f + "_new"] - m[f + "_saved"]).max())  # largest difference for this feature
    check(f"feature {f} matches", d < 1e-9, f"(max difference {d:.2e})")

# 4. Score through the module and compare with the dashboard file saved on Kaggle
scored = F.score(feat, DIR / "model_strict_split_acct.txt", DIR / "scoring_bundle.json")
dash = pd.read_csv(DIR / "dashboard_alerts.csv", usecols=["row_id", "risk_score", "alert"])
mm = dash.merge(scored[["row_id", "risk_score", "alert"]], on="row_id", suffixes=("_saved", "_new"))
check("all dashboard rows found", len(mm) == len(dash), f"({len(mm)} of {len(dash)})")
d = float(np.abs(mm["risk_score_saved"] - mm["risk_score_new"]).max())
check("scores match dashboard file", d < 1e-6, f"(max difference {d:.2e})")
check("alert flags match dashboard file", bool((mm["alert_saved"] == mm["alert_new"]).all()))

# 5. Unscored rows must carry no score and no alert
unscored = scored[scored["status"] != "scored"]
check("unscored rows have no score or alert",
      bool(unscored["risk_score"].isna().all() and (unscored["alert"] == 0).all()))

print(f"\nElapsed: {time.time() - t0:.0f} s")
print("All checks passed" if all(results) else "Some checks FAILED")