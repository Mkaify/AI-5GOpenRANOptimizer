import pandas as pd

# Load your downloaded dataset
df = pd.read_csv("data/5g_network_data.csv")

# 1. Standardize or remove non-numeric categorical text attributes for scaling safety
# (Or use pd.get_dummies(df) if you want to keep them as numeric features)
text_cols = ['Location', 'Network Type', 'Device Model', 'Carrier', 'Video Streaming Quality']
df.drop(columns=text_cols, errors='ignore', inplace=True)

# 2. Ensure an 'id' column exists to satisfy the baseline preprocessor expectations
df['id'] = range(len(df))

# 3. Rename 'Timestamp' to lowercase 'timestamp' to match code drop sequences
df.rename(columns={'Timestamp': 'timestamp'}, inplace=True)

# 4. Save back over your raw data folder
df.to_csv("data/5g_network_data.csv", index=False)
print("Dataset header formatting fully aligned with repository constraints!")