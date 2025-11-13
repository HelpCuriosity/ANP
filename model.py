import pandas as pd
from pathlib import Path
import joblib

MODEL_PATH = Path("C:/Vs_code/ANP/samples/model_pipeline.joblib")
NEW_PATH = Path("new_sample.csv")
OUT_PATH = Path("new_sample_results.csv")

if not MODEL_PATH.exists():
    print(f"Model not found: {MODEL_PATH}")
    raise SystemExit(1)

if not NEW_PATH.exists():
    print(f"No new_sample.csv found at {NEW_PATH}")
    raise SystemExit(1)

clf = joblib.load(MODEL_PATH)

new_df = pd.read_csv(NEW_PATH)

# Try to get expected column lists from the fitted ColumnTransformer
pre = clf.named_steps.get("pre", None)
if pre is None:
    print("Loaded model pipeline does not contain 'pre' step. Exiting.")
    raise SystemExit(1)

# transformers_ is available after fitting
transformers = pre.transformers_
# Find numeric and categorical column lists (assumes same order as training)
num_cols = []
cat_cols = []
for name, transformer, cols in transformers:
    if transformer.__class__.__name__.lower().startswith("robust") or "scale" in transformer.__class__.__name__.lower():
        num_cols = list(cols)
    elif transformer.__class__.__name__.lower().startswith("onehotencoder") or "onehot" in transformer.__class__.__name__.lower():
        cat_cols = list(cols)
    else:
        # fallback: if cols is list of names
        # we try to infer by type
        try:
            if isinstance(cols, (list, tuple)):
                # naive assignment if still empty
                if not num_cols:
                    num_cols = list(cols)
                elif not cat_cols:
                    cat_cols = list(cols)
        except Exception:
            pass

expected_cols = list(num_cols) + list(cat_cols)

# Ensure columns exist in new_df. For missing numeric cols -> fill 0, missing categorical -> fill '__MISSING__'
for c in num_cols:
    if c not in new_df.columns:
        new_df[c] = 0.0
for c in cat_cols:
    if c not in new_df.columns:
        new_df[c] = "__MISSING__"

# Reorder to expected columns (ColumnTransformer expects these names)
X_new = new_df[expected_cols]

# Predict
try:
    preds = clf.predict(X_new)
except Exception as e:
    print("Prediction failed:", e)
    raise

probs = None
if hasattr(clf, "predict_proba"):
    try:
        proba_all = clf.predict_proba(X_new)
        # if binary, take column 1
        if proba_all.shape[1] >= 2:
            probs = proba_all[:, 1]
        else:
            probs = proba_all[:, 0]
    except Exception:
        probs = None

# Prepare results
results = X_new.copy()
results["prediction"] = ["Malware" if p == 1 else "Benign" for p in preds]
if probs is not None:
    results["score"] = probs
else:
    results["score"] = ""

results.to_csv(OUT_PATH, index=False)
print(f"Saved predictions to {OUT_PATH}")
print(results[["prediction", "score"]].head(20))
