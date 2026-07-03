"""
generate_sample_dataset.py
──────────────────────────
Generates a realistic synthetic CSV dataset that mirrors
the structure of the Kaggle Garbage Classification dataset.

Use this to test the pipeline immediately without Kaggle credentials.
Run:  python generate_sample_dataset.py
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)
os.makedirs("data", exist_ok=True)

RECYCLABLE_MAP = {
    "cardboard": "Recyclable",
    "glass":     "Recyclable",
    "metal":     "Recyclable",
    "paper":     "Recyclable",
    "plastic":   "Recyclable",
    "trash":     "Non-Recyclable",
}

# Approximate sample counts from actual Kaggle dataset
CATEGORY_COUNTS = {
    "cardboard": 403,
    "glass":     501,
    "metal":     410,
    "paper":     594,
    "plastic":   482,
    "trash":     137,
}

# Feature profiles per category (mean, std)
FEATURE_PROFILES = {
    #                  hardness   transparency  smoothness  weight   organic  color_uni
    "cardboard": dict(h=(0.20,0.05), t=(0.05,0.03), s=(0.30,0.07), w=(0.20,0.05), o=(0.05,0.03), c=(0.60,0.08)),
    "glass":     dict(h=(0.90,0.04), t=(0.80,0.07), s=(0.92,0.04), w=(0.75,0.07), o=(0.01,0.01), c=(0.80,0.06)),
    "metal":     dict(h=(0.95,0.03), t=(0.02,0.02), s=(0.85,0.05), w=(0.85,0.07), o=(0.01,0.01), c=(0.85,0.05)),
    "paper":     dict(h=(0.15,0.05), t=(0.08,0.04), s=(0.55,0.08), w=(0.10,0.04), o=(0.08,0.04), c=(0.70,0.07)),
    "plastic":   dict(h=(0.55,0.08), t=(0.45,0.10), s=(0.80,0.06), w=(0.25,0.06), o=(0.02,0.02), c=(0.65,0.08)),
    "trash":     dict(h=(0.35,0.12), t=(0.10,0.06), s=(0.40,0.12), w=(0.50,0.12), o=(0.75,0.10), c=(0.25,0.10)),
}

rows = []
for cat, n in CATEGORY_COUNTS.items():
    p = FEATURE_PROFILES[cat]
    for i in range(n):
        def sample(key):
            mean, std = p[key]
            return float(np.clip(np.random.normal(mean, std), 0, 1))

        rows.append({
            "filename":            f"{cat}/{cat}_{i+1:04d}.jpg",
            "category":            cat,
            "label":               RECYCLABLE_MAP[cat],
            "material_hardness":   sample("h"),
            "transparency":        sample("t"),
            "surface_smoothness":  sample("s"),
            "estimated_weight":    sample("w"),
            "organic_content":     sample("o"),
            "color_uniformity":    sample("c"),
        })

df = pd.DataFrame(rows)
out_path = "data/garbage_dataset.csv"
df.to_csv(out_path, index=False)

print(f"✅ Sample dataset generated → '{out_path}'")
print(f"   Total samples : {len(df)}")
print(f"\n   Category distribution:")
print(df["category"].value_counts().to_string())
print(f"\n   Binary label distribution:")
print(df["label"].value_counts().to_string())
print(f"\n🚀 Run the classifier:")
print(f"   python naive_bayes_classifier.py")
