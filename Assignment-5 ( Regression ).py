# applying Data preprocessing and feature engineering on the following data..
# file = 'house-prices-advanced-regression-techniques-publicleaderboard-2025-07-07T16_40_20.csv'

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'house-prices-advanced-regression-techniques-publicleaderboard-2025-07-07T16_40_20.csv'
df = pd.read_csv(file_path)

# Step 1: Convert LastSubmissionDate to datetime
df['LastSubmissionDate'] = pd.to_datetime(df['LastSubmissionDate'])

# Step 2: Extract datetime features
df['SubmissionYear'] = df['LastSubmissionDate'].dt.year
df['SubmissionMonth'] = df['LastSubmissionDate'].dt.month
df['SubmissionDay'] = df['LastSubmissionDate'].dt.day
df['SubmissionHour'] = df['LastSubmissionDate'].dt.hour
df['SubmissionWeekday'] = df['LastSubmissionDate'].dt.weekday  # Monday=0, Sunday=6
df['IsWeekend'] = df['SubmissionWeekday'].isin([5, 6]).astype(int)

# Step 3: Feature engineering - Number of team members
df['TeamSize'] = df['TeamMemberUserNames'].apply(lambda x: len(str(x).split()))

# Step 4: Encode categorical features using Label Encoding
le_team_name = LabelEncoder()
le_team_members = LabelEncoder()

df['TeamNameEncoded'] = le_team_name.fit_transform(df['TeamName'])
df['TeamMembersEncoded'] = le_team_members.fit_transform(df['TeamMemberUserNames'])

# Optional: Drop original text columns if not needed
df_processed = df.drop(columns=['LastSubmissionDate', 'TeamName', 'TeamMemberUserNames'])

# Display the processed dataframe
print(df_processed.head())
