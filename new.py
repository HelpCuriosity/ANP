import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
RESULTS_PATH = Path("new_sample_results.csv")

if not RESULTS_PATH.exists():
    print(f"Results file not found at {RESULTS_PATH}")
    raise SystemExit(1)

df = pd.read_csv(RESULTS_PATH)

# Display a quick preview
print(df.head())

# Plot malware probability scores
plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(df)), df['score'], color=['red' if p == 'Malware' else 'green' for p in df['prediction']])
plt.title("Model Prediction Confidence for Each Sample", fontsize=16)
plt.xlabel("Sample Index", fontsize=12)
plt.ylabel("Malware Probability (Score)", fontsize=12)
plt.ylim(0, 1)

# Add text labels for prediction
for i, (bar, label) in enumerate(zip(bars, df['prediction'])):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, label, 
             ha='center', fontsize=9, rotation=45)

plt.tight_layout()
plt.show()
