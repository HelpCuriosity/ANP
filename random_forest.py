import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


DATA = Path("Final_Dataset_without_duplicate.csv")
df = pd.read_csv(DATA)

# Label encoding
df['label'] = df['Class'].map({'Benign': 0, 'Malware': 1})
drop_cols = ['md5', 'sha1', 'Class']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

X = df.drop(columns=['label'])
y = df['label'].values


cat_cols = [c for c in X.columns if X[c].dtype == object]
num_cols = [c for c in X.columns if X[c].dtype != object]

if 'file_extension' in cat_cols:
    top_ext = X['file_extension'].value_counts().nlargest(30).index
    X['file_extension'] = X['file_extension'].where(X['file_extension'].isin(top_ext), other='__OTHER__')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ],
    remainder='drop'
)


clf = Pipeline([
    ('pre', preprocessor),
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)


clf.fit(X_train, y_train)

preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))


joblib.dump(clf, "C:/Vs_code/ANP/samples/model_pipeline.joblib")
print("\n✅ Model saved to samples/model_pipeline.joblib")


try:
    new_sample = pd.read_csv("new_sample.csv")
    preds_new = clf.predict(new_sample)
    print("\nPrediction for new sample(s):", ["Malware" if p == 1 else "Benign" for p in preds_new])
except FileNotFoundError:
    print("\n⚠️ No 'new_sample.csv' found — skipping test prediction.")
