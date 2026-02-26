# train_model.py
# Train a Decision Tree classifier on collected stroke data.
# Run this after collect_data.py has gathered enough samples.
#
# Usage:
#   python3 train_model.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np

CSV_FILE  = "stroke_data.csv"
MODEL_OUT = "stroke_model.pkl"

FEATURES = ["peak_accel", "mean_gyro_y", "var_gyro_y", "mean_gyro_z", "var_gyro_z", "duration"]

df = pd.read_csv(CSV_FILE)
print(f"Loaded {len(df)} samples  ({df['label'].value_counts().to_dict()})")

X = df[FEATURES]
y = df["label"]

# Cross-validation gives a much more honest accuracy estimate than a single split
clf = DecisionTreeClassifier(max_depth=3, random_state=42)   # shallower = less overfit
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="f1")
print(f"\n5-fold cross-validation F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"  Per-fold: {[f'{s:.2f}' for s in cv_scores]}")

# Train final model on all data
clf.fit(X, y)

# Show leave-one-out sanity check on training data
y_train_pred = clf.predict(X)
print("\nTraining set (sanity check):")
print(classification_report(y, y_train_pred, target_names=["Bad(0)", "Good(1)"]))
print("Confusion matrix:")
print(confusion_matrix(y, y_train_pred))

joblib.dump(clf, MODEL_OUT)
print(f"\nModel saved to {MODEL_OUT}")

print("\nFeature importances:")
for name, imp in sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1]):
    bar = "#" * int(imp * 30)
    print(f"  {name:20s} {imp:.3f}  {bar}")
