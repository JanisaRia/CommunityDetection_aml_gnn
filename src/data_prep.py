import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
# print("Step 1: Merging datasets...")

original_data = pd.read_csv("hi_patterns.csv")
synthetic_data = pd.read_csv("Synthetic_Data.csv")

merged_data = pd.concat([original_data, synthetic_data], ignore_index=True)
merged_data.to_csv("hiPatterns.csv", index=False)

print(f"Merged dataset saved. Shape: {merged_data.shape}")

# print("Step 2: Cleaning data...")

df = pd.read_csv("hiPatterns.csv")

# Convert to proper types
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["converted_amount"] = pd.to_numeric(df["converted_amount"], errors="coerce")

# Handle missing values
df.fillna({
    "amount": 0, 
    "converted_amount": 0, 
    "currency": "Unknown", 
    "transaction_type": "Unknown"
}, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Create new feature
df["amount_diff"] = df["converted_amount"] - df["amount"]

print(f"Data cleaned. Shape: {df.shape}")

# print("Step 3: Feature Engineering...")

# One-Hot Encoding for categorical variables
df = pd.get_dummies(df, columns=["currency", "transaction_type"], drop_first=True)

# Log Transformations to normalize amount values
df['amount_log'] = np.log1p(df['amount'])
df['converted_amount_log'] = np.log1p(df['converted_amount'])
df['amount_diff_log'] = np.log1p(df['amount_diff'])

# Extract time-based features
df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# Drop original columns that are no longer needed
df.drop(columns=['transaction_id', 'receiver_id', 'amount', 'converted_amount', 'date'], inplace=True)

print(f"Feature Engineering completed. Shape: {df.shape}")
# print("Step 4: Scaling Features...")

num_features = ['time', 'amount_log', 'converted_amount_log', 'amount_diff_log']

scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])

print("Feature scaling applied.")

# print("Step 5: Analyzing Collinearity...")

# Compute Correlation Matrix
corr_matrix = df.corr()

# Plot heatmap for collinearity visualization
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()

print("Collinearity analysis completed.")

df.to_csv("hiPatterns.csv", index=False)
print(f"Final processed dataset saved as 'hiPatterns.csv'. Shape: {df.shape}")
