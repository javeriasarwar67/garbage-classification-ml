# 🗑️ Garbage Classification — Naive Bayes

Binary classification of waste items: **Recyclable** vs **Non-Recyclable**
using the Gaussian Naive Bayes algorithm.

---

## 📦 Dataset

| Field        | Detail |
|---|---|
| Source       | [Kaggle — mostafaabla/garbage-classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) |
| Classes      | cardboard, glass, metal, paper, plastic, trash |
| Binary Label | Recyclable (cardboard/glass/metal/paper/plastic) vs Non-Recyclable (trash) |
| Size         | ~2,500 images across 6 categories |

---

## 📂 Project Structure

```
garbage_classification/
├── naive_bayes_classifier.py   ← Main classifier (from scratch + sklearn)
├── download_dataset.py         ← Kaggle downloader
├── generate_sample_dataset.py  ← Synthetic data for quick testing
├── requirements.txt
├── notebooks/
│   └── garbage_classification.ipynb
├── data/
│   └── garbage_dataset.csv     ← Created after download/generate
├── models/                     ← Auto-created; saved .pkl models
└── results/                    ← Auto-created; plots & metrics
```

---

## 🚀 Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2A — Download real Kaggle dataset
```bash
# First place kaggle.json at ~/.kaggle/kaggle.json
# Get it from: kaggle.com → Account → Settings → API → Create New Token

python download_dataset.py
```

### Step 2B — OR use synthetic data (no Kaggle needed)
```bash
python generate_sample_dataset.py
```

### Step 3 — Run the classifier
```bash
python naive_bayes_classifier.py
```

### Step 4 — Optional: explore in Jupyter
```bash
jupyter notebook notebooks/garbage_classification.ipynb
```

---

## 🧠 How Naive Bayes Works Here

```
P(Recyclable | Features) ∝ P(Features | Recyclable) × P(Recyclable)
```

For each feature, we model its distribution per class using a **Gaussian** (normal) distribution:

```
P(x | class) = (1 / √(2πσ²)) × exp(-(x-μ)² / 2σ²)
```

We combine all features using the **"naive" independence assumption**:

```
P(all features | class) = P(f1|class) × P(f2|class) × ... × P(fn|class)
```

### Features Used

| Feature              | Description                        |
|---|---|
| `material_hardness`  | How rigid/hard the material is     |
| `transparency`       | How see-through the item is        |
| `surface_smoothness` | Surface texture                    |
| `estimated_weight`   | Relative weight                    |
| `organic_content`    | Likelihood of organic/food content |
| `color_uniformity`   | How uniform the color is           |

---

## 📊 Output Files

After running `naive_bayes_classifier.py`, check the `results/` folder:

| File                            | Description                       |
|---|---|
| `confusion_matrix_binary.png`   | TP/TN/FP/FN heatmap               |
| `roc_curve.png`                 | ROC curve with AUC score          |
| `feature_importance.png`        | Which features matter most        |
| `binary_nb_model.pkl`           | Saved binary classifier           |
| `multiclass_nb_model.pkl`       | Saved 6-class classifier          |
| `scaler.pkl`                    | Saved StandardScaler              |

---

## 🔄 Recyclability Rules

| Category   | Label          |
|---|---|
| Cardboard  | ✅ Recyclable  |
| Glass      | ✅ Recyclable  |
| Metal      | ✅ Recyclable  |
| Paper      | ✅ Recyclable  |
| Plastic    | ✅ Recyclable  |
| Trash      | ❌ Non-Recyclable |

---

## 📈 Expected Results

```
              precision    recall  f1-score

Non-Recyclable     0.85      0.82      0.83
    Recyclable     0.96      0.97      0.97

      accuracy                         0.95
```

*(Accuracy will vary with real Kaggle image features vs synthetic features)*
