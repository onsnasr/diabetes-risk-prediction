import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

# Load the raw data
df = pd.read_csv('data/diabetes.csv')

# Step 1: Replace the fake zeros with actual NaN (mark them as truly missing)
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)

# Step 2: Separate features (X) from target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Step 3: Split into train/test BEFORE imputing (important - avoids data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Impute missing values using the median (learned from training data only)
imputer = SimpleImputer(strategy='median')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

# Step 5: Scale the features (normalize to similar ranges)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X.columns)

# Step 6: Save everything for the next stage (model building)
X_train_scaled.to_csv('data/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('data/X_test_scaled.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

joblib.dump(imputer, 'data/imputer.pkl')
joblib.dump(scaler, 'data/scaler.pkl')

print("Train shape:", X_train_scaled.shape)
print("Test shape:", X_test_scaled.shape)
print("Preprocessing complete. Files saved to data/")