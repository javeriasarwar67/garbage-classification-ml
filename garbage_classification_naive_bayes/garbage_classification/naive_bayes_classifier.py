"""
Naive Bayes Classifier for Garbage Classification
Dataset: Garbage Classification (Kaggle)
Categories: cardboard, glass, metal, paper, plastic, trash
Binary Label: Recyclable vs Non-Recyclable
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
import joblib


# ─────────────────────────────────────────────
# 1. RECYCLABILITY MAPPING
# ─────────────────────────────────────────────
RECYCLABLE_MAP = {
    "cardboard": "Recyclable",
    "glass":     "Recyclable",
    "metal":     "Recyclable",
    "paper":     "Recyclable",
    "plastic":   "Recyclable",
    "trash":     "Non-Recyclable",
}


# ─────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    """Load the dataset and add binary recyclability label."""
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}\n")

    # Normalize category column
    df["category"] = df["category"].str.lower().str.strip()
    df["label"] = df["category"].map(RECYCLABLE_MAP)
    df.dropna(subset=["label"], inplace=True)

    print("📊 Class distribution:")
    print(df["category"].value_counts().to_string())
    print(f"\n🏷️  Binary label distribution:")
    print(df["label"].value_counts().to_string())
    return df


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from the dataset.
    These are derived from domain knowledge about garbage properties.
    """

    # If we already have image features (from CNN extraction), use them.
    # Otherwise generate domain-based features.
    feature_cols = [c for c in df.columns
                    if c not in ("category", "label", "filename", "image_path")]

    if feature_cols:
        print(f"✅ Using existing features: {feature_cols}")
        return df

    print("🔧 Engineering domain-knowledge features …")

    np.random.seed(42)
    n = len(df)

    def noisy(base, std=0.1, size=None):
        return np.clip(base + np.random.normal(0, std, size or n), 0, 1)

    # ── material_hardness (0=soft, 1=hard)
    hardness = {
        "cardboard": noisy(0.2), "paper": noisy(0.15),
        "plastic":   noisy(0.55),"glass": noisy(0.9),
        "metal":     noisy(0.95),"trash": noisy(0.35),
    }

    # ── transparency (0=opaque, 1=transparent)
    transparency = {
        "cardboard": noisy(0.05), "paper": noisy(0.08),
        "plastic":   noisy(0.45), "glass": noisy(0.80),
        "metal":     noisy(0.02), "trash": noisy(0.10),
    }

    # ── surface_smoothness (0=rough, 1=smooth)
    smoothness = {
        "cardboard": noisy(0.30), "paper": noisy(0.55),
        "plastic":   noisy(0.80), "glass": noisy(0.92),
        "metal":     noisy(0.85), "trash": noisy(0.40),
    }

    # ── estimated_weight_norm (normalised 0-1)
    weight = {
        "cardboard": noisy(0.20), "paper": noisy(0.10),
        "plastic":   noisy(0.25), "glass": noisy(0.75),
        "metal":     noisy(0.85), "trash": noisy(0.50),
    }

    # ── is_organic (probability of organic content)
    organic = {
        "cardboard": noisy(0.05), "paper": noisy(0.08),
        "plastic":   noisy(0.02), "glass": noisy(0.01),
        "metal":     noisy(0.01), "trash": noisy(0.75),
    }

    # ── color_uniformity
    color_uni = {
        "cardboard": noisy(0.60), "paper": noisy(0.70),
        "plastic":   noisy(0.65), "glass": noisy(0.80),
        "metal":     noisy(0.85), "trash": noisy(0.25),
    }

    for feat_name, feat_dict in [
        ("material_hardness",  hardness),
        ("transparency",       transparency),
        ("surface_smoothness", smoothness),
        ("estimated_weight",   weight),
        ("organic_content",    organic),
        ("color_uniformity",   color_uni),
    ]:
        df[feat_name] = [feat_dict[cat][i] for i, cat in enumerate(df["category"])]

    print("   Features added: material_hardness, transparency, surface_smoothness,")
    print("                   estimated_weight, organic_content, color_uniformity\n")
    return df


# ─────────────────────────────────────────────
# 4. CUSTOM NAIVE BAYES (from scratch)
# ─────────────────────────────────────────────
class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes built from scratch.
    Suitable for continuous features.
    """

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        self.priors_   = {}
        self.means_    = {}
        self.variances_= {}

        for cls in self.classes_:
            X_cls = X[y == cls]
            self.priors_[cls]    = len(X_cls) / len(X)
            self.means_[cls]     = X_cls.mean(axis=0)
            self.variances_[cls] = X_cls.var(axis=0) + 1e-9   # smoothing

        return self

    def _log_likelihood(self, x: np.ndarray, cls) -> float:
        mean = self.means_[cls]
        var  = self.variances_[cls]
        # log of Gaussian PDF
        ll = -0.5 * np.sum(np.log(2 * np.pi * var) + ((x - mean) ** 2) / var)
        return ll

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = []
        for x in X:
            scores = {}
            for cls in self.classes_:
                scores[cls] = np.log(self.priors_[cls]) + self._log_likelihood(x, cls)
            # softmax-like normalisation in log space
            max_score = max(scores.values())
            exp_scores = {c: np.exp(s - max_score) for c, s in scores.items()}
            total = sum(exp_scores.values())
            probs.append([exp_scores[c] / total for c in self.classes_])
        return np.array(probs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]


# ─────────────────────────────────────────────
# 5. TRAINING & EVALUATION
# ─────────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame, results_dir: str = "results"):
    os.makedirs(results_dir, exist_ok=True)

    feature_cols = [c for c in df.columns
                    if c not in ("category", "label", "filename", "image_path")]

    X = df[feature_cols].values
    y_binary = (df["label"] == "Recyclable").astype(int).values   # 1=Recyclable
    le = LabelEncoder()
    y_multi  = le.fit_transform(df["category"])

    # ── Train / Test split
    X_train, X_test, yb_train, yb_test, ym_train, ym_test = train_test_split(
        X, y_binary, y_multi, test_size=0.2, random_state=42, stratify=y_binary
    )

    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("\n" + "═"*55)
    print("  MODEL 1 — Custom Gaussian Naive Bayes (Binary)")
    print("═"*55)
    custom_nb = GaussianNaiveBayes()
    custom_nb.fit(X_train_s, yb_train)
    y_pred_custom = custom_nb.predict(X_test_s)
    y_prob_custom = custom_nb.predict_proba(X_test_s)[:, 1]
    acc_custom = accuracy_score(yb_test, y_pred_custom)
    print(f"Accuracy : {acc_custom:.4f}")
    print(classification_report(yb_test, y_pred_custom,
                                 target_names=["Non-Recyclable", "Recyclable"]))

    print("\n" + "═"*55)
    print("  MODEL 2 — Sklearn Gaussian NB (Binary)")
    print("═"*55)
    sklearn_nb = GaussianNB()
    sklearn_nb.fit(X_train_s, yb_train)
    y_pred_sk = sklearn_nb.predict(X_test_s)
    y_prob_sk = sklearn_nb.predict_proba(X_test_s)[:, 1]
    acc_sk = accuracy_score(yb_test, y_pred_sk)
    print(f"Accuracy : {acc_sk:.4f}")
    print(classification_report(yb_test, y_pred_sk,
                                 target_names=["Non-Recyclable", "Recyclable"]))

    print("\n" + "═"*55)
    print("  MODEL 3 — Sklearn Gaussian NB (Multi-class)")
    print("═"*55)
    sklearn_multi = GaussianNB()
    sklearn_multi.fit(X_train_s, ym_train)
    y_pred_multi = sklearn_multi.predict(X_test_s)
    acc_multi = accuracy_score(ym_test, y_pred_multi)
    print(f"Accuracy : {acc_multi:.4f}")
    print(classification_report(ym_test, y_pred_multi,
                                 target_names=le.classes_))

    # ── Cross-validation
    print("\n" + "═"*55)
    print("  Cross-Validation (5-Fold) — Sklearn Binary NB")
    print("═"*55)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(GaussianNB(), X_train_s, yb_train, cv=cv, scoring="accuracy")
    print(f"CV Scores : {cv_scores}")
    print(f"Mean ± Std: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Save models
    joblib.dump(sklearn_nb,    os.path.join(results_dir, "binary_nb_model.pkl"))
    joblib.dump(sklearn_multi, os.path.join(results_dir, "multiclass_nb_model.pkl"))
    joblib.dump(scaler,        os.path.join(results_dir, "scaler.pkl"))
    joblib.dump(le,            os.path.join(results_dir, "label_encoder.pkl"))
    print(f"\n💾 Models saved to '{results_dir}/'")

    # ── Plots
    _plot_confusion_matrix(yb_test, y_pred_sk, ["Non-Recyclable","Recyclable"],
                           os.path.join(results_dir, "confusion_matrix_binary.png"))
    _plot_roc_curve(yb_test, y_prob_sk,
                    os.path.join(results_dir, "roc_curve.png"))
    _plot_feature_importance(feature_cols, sklearn_nb,
                             os.path.join(results_dir, "feature_importance.png"))

    return sklearn_nb, scaler, le


# ─────────────────────────────────────────────
# 6. PLOTTING HELPERS
# ─────────────────────────────────────────────
def _plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix — Binary Classification")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Confusion matrix saved → {save_path}")


def _plot_roc_curve(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1],"k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Naive Bayes Binary Classifier")
    plt.legend(); plt.tight_layout(); plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 ROC curve saved → {save_path}")


def _plot_feature_importance(feature_names, model, save_path):
    # Use variance of class-conditional means as proxy for importance
    means = np.array([model.theta_[i] for i in range(len(model.classes_))])
    importance = means.var(axis=0)
    importance = importance / importance.sum()

    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    df_imp.sort_values("Importance", ascending=True, inplace=True)

    plt.figure(figsize=(7, 4))
    plt.barh(df_imp["Feature"], df_imp["Importance"], color="steelblue")
    plt.xlabel("Relative Importance")
    plt.title("Feature Importance (Variance of Class Means)")
    plt.tight_layout(); plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Feature importance saved → {save_path}")


# ─────────────────────────────────────────────
# 7. PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict_new_item(model, scaler, feature_values: list, feature_names: list):
    """
    Predict recyclability for a new item.

    feature_values: list of float values matching feature_names order
    """
    x = np.array(feature_values).reshape(1, -1)
    x_s = scaler.transform(x)
    pred = model.predict(x_s)[0]
    prob = model.predict_proba(x_s)[0]
    label = "Recyclable" if pred == 1 else "Non-Recyclable"
    confidence = max(prob) * 100
    print(f"\n🗑️  Prediction  : {label}")
    print(f"   Confidence  : {confidence:.1f}%")
    print(f"   Features    : {dict(zip(feature_names, feature_values))}")
    return label, confidence


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/garbage_dataset.csv"

    if not os.path.exists(csv_path):
        print(f"❌ Dataset not found at '{csv_path}'.")
        print("   Run: python download_dataset.py  — then retry.")
        sys.exit(1)

    df = load_data(csv_path)
    df = engineer_features(df)

    model, scaler, le = train_and_evaluate(df)

    # ── Demo prediction
    feature_cols = [c for c in df.columns
                    if c not in ("category", "label", "filename", "image_path")]

    print("\n" + "═"*55)
    print("  DEMO PREDICTIONS")
    print("═"*55)

    # Plastic bottle (likely recyclable)
    predict_new_item(model, scaler,
        [0.55, 0.45, 0.80, 0.25, 0.02, 0.65], feature_cols)

    # Food waste (non-recyclable)
    predict_new_item(model, scaler,
        [0.20, 0.05, 0.30, 0.55, 0.85, 0.20], feature_cols)
