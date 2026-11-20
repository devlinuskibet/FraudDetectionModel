import pandas as pd

files = ["file1.csv", "file2.csv"]   # your actual paths

for f in files:
    print("=" * 60, f)
    df = pd.read_csv(f)
    print("Rows:", len(df))
    print(df.dtypes)
    print(df.head())
    print("Nulls:\n", df.isna().sum())

    # Edit these names to match what the header shows
    fraud_col = "isFraud"
    type_col = "action"
    orig_col = "nameOrig"

    print("Fraud rate:", df[fraud_col].mean())
    print(df.groupby(type_col)[fraud_col].agg(["count", "sum", "mean"]))

    per_acct = df.groupby(orig_col).size()
    print("Transactions per initiator:")
    print(per_acct.describe())
    print("Accounts with >=10 txns:", (per_acct >= 10).mean())