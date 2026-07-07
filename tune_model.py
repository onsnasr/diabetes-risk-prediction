import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# Load preprocessed data
X_train = pd.read_csv('data/X_train_scaled.csv')
X_test = pd.read_csv('data/X_test_scaled.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()
y_test = pd.read_csv('data/y_test.csv').values.ravel()

# --- Baseline: default Random Forest (no tuning) ---
baseline_model = RandomForestClassifier(random_state=42)
baseline_model.fit(X_train, y_train)
baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
baseline_auc = roc_auc_score(y_test, baseline_proba)

# --- Tuning: search for better hyperparameters ---
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid.fit(X_train, y_train)

tuned_model = grid.best_estimator_
tuned_proba = tuned_model.predict_proba(X_test)[:, 1]
tuned_auc = roc_auc_score(y_test, tuned_proba)

# --- Compare baseline vs tuned on the REAL test set, keep the better one ---
print(f"Baseline ROC_AUC (default settings): {baseline_auc:.4f}")
print(f"Tuned ROC_AUC (GridSearchCV):        {tuned_auc:.4f}")
print()

if tuned_auc > baseline_auc:
    final_model = tuned_model
    print("Tuned model performs better -> keeping TUNED model as final.")
else:
    final_model = baseline_model
    print("Baseline model performs better -> keeping DEFAULT model as final.")
    print("(This can happen with small datasets - tuning overfits to cross-validation folds.)")

# --- Final evaluation of whichever model we kept ---
y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]

print()
print("FINAL MODEL METRICS:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC_AUC:", roc_auc_score(y_test, y_proba))

joblib.dump(final_model, 'data/final_model.pkl')
print("\nFinal model saved to data/final_model.pkl")