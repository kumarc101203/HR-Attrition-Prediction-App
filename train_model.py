# ============================================
# HR ATTRITION MODEL TRAINING SCRIPT (FIXED)
# ============================================

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"   # 🔥 FIX: prevents hanging

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

print("Step 1: Loading dataset...")

# ================================
# LOAD DATASET
# ================================
df = pd.read_csv("HR-Employee-Attrition.csv")

print("Step 2: Dropping useless columns...")

# ================================
# DROP USELESS COLUMNS
# ================================
df = df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'])

print("Step 3: Encoding data...")

# ================================
# CLEAN + ENCODE
# ================================
df['Attrition'] = df['Attrition'].str.strip().map({'Yes': 1, 'No': 0})
df['OverTime'] = df['OverTime'].str.strip().map({'Yes': 1, 'No': 0})

df = pd.get_dummies(df, drop_first=True)

print("Step 4: Splitting data...")

# ================================
# SPLIT FEATURES & TARGET
# ================================
X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Step 5: Scaling data...")

# ================================
# SCALING
# ================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Step 6: Applying SMOTE...")

# ================================
# HANDLE IMBALANCE (SMOTE)
# ================================
smote = SMOTE(random_state=42)   # 🔥 FIX: no parallel threads
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("Step 7: Training model...")

# ================================
# TRAIN MODEL
# ================================
model = LogisticRegression(max_iter=1000)  # 🔥 FIX
model.fit(X_train_sm, y_train_sm)

print("Model training completed!")

# ================================
# SAVE FEATURES
# ================================
feature_columns = X.columns

with open("features.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

print("Features saved successfully!")

# ================================
# SAVE MODEL + SCALER
# ================================
with open("attrition_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")

print("🎯 DONE — Script finished successfully!")