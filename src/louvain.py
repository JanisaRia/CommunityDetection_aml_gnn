import pandas as pd
import numpy as np

# Load transaction data
df = pd.read_csv("../data/hiPatterns.csv")  # Ensure this contains 'amount' column

# Compute Z-score for transaction amounts
df["z_score"] = (df["amount"] - df["amount"].mean()) / df["amount"].std()

# Flag high-risk transactions (Z-score > 3 means anomaly)
anomalous_transactions = df[df["z_score"].abs() > 3]

# Print suspicious transactions
print(f"🚨 Detected {len(anomalous_transactions)} high-risk transactions based on amount deviation")
print(anomalous_transactions.head(10))
# Count transactions per sender
transaction_counts = df["sender_id"].value_counts()

# Compute Z-score for transaction frequency
df["txn_z_score"] = (transaction_counts - transaction_counts.mean()) / transaction_counts.std()

# Flag high-risk accounts (Z-score > 3 means anomaly)
suspicious_accounts = df[df["txn_z_score"].abs() > 3]["sender_id"].unique()

print(f"🚨 Detected {len(suspicious_accounts)} high-risk accounts based on transaction frequency")
print(suspicious_accounts[:10])
