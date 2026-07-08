import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load the final trained model and the training data (for column names)
model = joblib.load('data/final_model.pkl')
X_train = pd.read_csv('data/X_train_scaled.csv')

# Random Forest tracks how much each feature helped reduce prediction error
importances = pd.Series(model.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=False)

print("Feature Importance (most to least influential):")
print(importances)

# Create a horizontal bar chart
plt.figure(figsize=(8, 5))
importances.plot(kind='barh')
plt.gca().invert_yaxis()  # most important at the top
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('data/feature_importance.png', dpi=120)
plt.close()

print("\nChart saved to data/feature_importance.png")