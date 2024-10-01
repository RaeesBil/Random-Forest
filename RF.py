import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('rf.csv')

print("First few rows of the dataset:\n", df.head())
print("\nColumn names:\n", df.columns)

if df['class'].dtype == 'object':
    df['class'] = df['class'].apply(lambda x: 1 if x == 'desired_class' else 0)  # Replace 'desired_class' with the class you want to set as 1

df_numeric = df.select_dtypes(include=['number'])

if 'class' not in df_numeric.columns:
    raise KeyError("'class' column not found in numeric data. Please ensure 'class' is properly encoded and included.")

X = df_numeric.drop('class', axis=1)
y = df_numeric['class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

cv_scores = cross_val_score(clf, X, y, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

importances = clf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print("Feature Importances:\n", feature_importance_df.sort_values(by='Importance', ascending=False))