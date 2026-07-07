import pandas as pd

df = pd.read_csv('data/diabetes.csv')

print("Shape:", df.shape)
print()
print(df.info())
print()
print(df.describe())
print()
print("Missing values (NaN):")
print(df.isnull().sum())
print()
print("Outcome distribution:")
print(df['Outcome'].value_counts())

print()
print("Checking for zeros in medical columns (likely disguised missing values):")
cols_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_check:
    n_zero = (df[col] == 0).sum()
    pct = n_zero / len(df) * 100
    print(f"{col}: {n_zero} zeros ({pct:.1f}%)")