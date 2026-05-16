"""
churn_analysis.py
=================
Full pipeline for Customer Churn Prediction:
  1. Data Loading & Cleaning
  2. Exploratory Data Analysis (EDA)
  3. Data Visualization (6 charts)
  4. Machine Learning Models (Logistic Regression + Decision Tree)
  5. Model Evaluation (Accuracy, Confusion Matrix, Precision, Recall)
  6. Feature Importance (bonus)

Run:  python3 churn_analysis.py
Outputs: charts saved to ./charts/   model saved to churn_model.pkl
"""

import os
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, roc_curve,
)

warnings.filterwarnings("ignore")

# ── Aesthetic theme ───────────────────────────────────────────────────────────
PALETTE_CHURN = {"Yes": "#FF6B6B", "No": "#4ECDC4"}
DARK_BG       = "#1a1a2e"
ACCENT        = "#e94560"
ACCENT2       = "#0f3460"
FONT_COLOR    = "#eaeaea"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    "#16213e",
    "axes.edgecolor":    "#444",
    "axes.labelcolor":   FONT_COLOR,
    "xtick.color":       FONT_COLOR,
    "ytick.color":       FONT_COLOR,
    "text.color":        FONT_COLOR,
    "grid.color":        "#333",
    "grid.alpha":        0.4,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    14,
    "axes.labelsize":    11,
})

os.makedirs("charts", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & CLEANING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  CUSTOMER CHURN PREDICTION — Full Pipeline")
print("═"*60)

print("\n[1] Loading data …")
df = pd.read_csv("telco_churn.csv")
print(f"    Shape: {df.shape}")
print(f"    Columns: {list(df.columns)}")

# ── Missing values ────────────────────────────────────────
print(f"\n    Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# ── Remove irrelevant column ──────────────────────────────
df.drop(columns=["customerID"], inplace=True)

# ── Encode target ─────────────────────────────────────────
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print(f"\n    After cleaning shape: {df.shape}")
print(f"    Churn rate: {df['Churn'].mean():.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2] EDA Insights …")

total      = len(df)
churned    = df["Churn"].sum()
retained   = total - churned

print(f"    Total customers  : {total:,}")
print(f"    Churned          : {churned:,}  ({churned/total:.1%})")
print(f"    Retained         : {retained:,}  ({retained/total:.1%})")

# Churn by contract
print("\n    Churn rate by Contract:")
print(df.groupby("Contract")["Churn"].mean().mul(100).round(1).to_string())

# Churn by internet service
print("\n    Churn rate by InternetService:")
print(df.groupby("InternetService")["Churn"].mean().mul(100).round(1).to_string())

# Avg tenure churned vs retained
print("\n    Avg tenure (months):")
print(df.groupby("Churn")["tenure"].mean().round(1).to_string())

# Senior citizen churn
sc_churn = df.groupby("SeniorCitizen")["Churn"].mean().mul(100).round(1)
print(f"\n    Senior citizen churn rate : {sc_churn[1]}%")
print(f"    Non-senior churn rate    : {sc_churn[0]}%")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATA VISUALIZATION  (6 charts)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3] Generating visualizations …")

df_plot = df.copy()
df_plot["Churn_label"] = df_plot["Churn"].map({1: "Yes", 0: "No"})

# ── Chart 1: Overall Churn Distribution (pie / donut) ────
fig, ax = plt.subplots(figsize=(7, 7))
sizes  = [retained, churned]
labels = ["Retained", "Churned"]
colors = ["#4ECDC4", "#FF6B6B"]
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, colors=colors,
    autopct="%1.1f%%", startangle=140,
    wedgeprops=dict(width=0.55, edgecolor="#1a1a2e", linewidth=3),
    textprops={"color": FONT_COLOR, "fontsize": 13},
)
for at in autotexts:
    at.set_fontsize(15); at.set_fontweight("bold")
ax.set_title("Overall Churn Distribution", pad=20, fontsize=16, fontweight="bold")
fig.tight_layout()
fig.savefig("charts/01_churn_distribution.png", dpi=150, bbox_inches="tight",
            facecolor=DARK_BG)
plt.close(fig)
print("    ✓ charts/01_churn_distribution.png")

# ── Chart 2: Churn by Contract Type (bar chart) ──────────
fig, ax = plt.subplots(figsize=(8, 5))
ct = df_plot.groupby(["Contract", "Churn_label"]).size().unstack(fill_value=0)
ct.plot(kind="bar", ax=ax, color=["#4ECDC4", "#FF6B6B"],
        edgecolor="#1a1a2e", linewidth=1.5, width=0.6)
ax.set_title("Churn Count by Contract Type", fontsize=15, fontweight="bold")
ax.set_xlabel("Contract Type")
ax.set_ylabel("Number of Customers")
ax.legend(["Retained", "Churned"], labelcolor=FONT_COLOR, framealpha=0.3)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
ax.yaxis.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("charts/02_churn_by_contract.png", dpi=150, bbox_inches="tight",
            facecolor=DARK_BG)
plt.close(fig)
print("    ✓ charts/02_churn_by_contract.png")

# ── Chart 3: Churn by Internet Service (count plot) ──────
fig, ax = plt.subplots(figsize=(8, 5))
order = df_plot["InternetService"].value_counts().index
sns.countplot(data=df_plot, x="InternetService", hue="Churn_label",
              order=order, palette=PALETTE_CHURN, ax=ax,
              edgecolor="#1a1a2e", linewidth=1.2)
ax.set_title("Churn by Internet Service Type", fontsize=15, fontweight="bold")
ax.set_xlabel("Internet Service")
ax.set_ylabel("Customer Count")
ax.legend(title="Churn", labelcolor=FONT_COLOR, framealpha=0.3)
ax.yaxis.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("charts/03_churn_by_internet.png", dpi=150, bbox_inches="tight",
            facecolor=DARK_BG)
plt.close(fig)
print("    ✓ charts/03_churn_by_internet.png")

# ── Chart 4: Tenure Distribution (histogram) ─────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df_plot[df_plot["Churn"] == 0]["tenure"], bins=30, alpha=0.75,
        color="#4ECDC4", label="Retained", edgecolor="#1a1a2e")
ax.hist(df_plot[df_plot["Churn"] == 1]["tenure"], bins=30, alpha=0.75,
        color="#FF6B6B", label="Churned",  edgecolor="#1a1a2e")
ax.set_title("Tenure Distribution: Churned vs Retained", fontsize=15, fontweight="bold")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Count")
ax.legend(labelcolor=FONT_COLOR, framealpha=0.3)
ax.yaxis.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("charts/04_tenure_distribution.png", dpi=150, bbox_inches="tight",
            facecolor=DARK_BG)
plt.close(fig)
print("    ✓ charts/04_tenure_distribution.png")

# ── Chart 5: Monthly Charges boxplot ─────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=df_plot, x="Churn_label", y="MonthlyCharges",
            palette=PALETTE_CHURN, ax=ax,
            linewidth=1.5, flierprops=dict(marker="o", markersize=3, alpha=0.5))
ax.set_title("Monthly Charges: Churned vs Retained", fontsize=15, fontweight="bold")
ax.set_xlabel("Churn")
ax.set_ylabel("Monthly Charges ($)")
ax.yaxis.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("charts/05_monthly_charges_box.png", dpi=150, bbox_inches="tight",
            facecolor=DARK_BG)
plt.close(fig)
print("    ✓ charts/05_monthly_charges_box.png")

# ── Chart 6: Correlation Heatmap ─────────────────────────
# encode categoricals for correlation
df_enc = df_plot.drop(columns=["Churn_label"]).copy()
le = LabelEncoder()
for col in df_enc.select_dtypes("object").columns:
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))

corr = df_enc.corr()
fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5, linecolor="#1a1a2e",
            annot_kws={"size": 7},
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold", pad=20)
fig.tight_layout()
fig.savefig("charts/06_correlation_heatmap.png", dpi=150, bbox_inches="tight",
            facecolor=DARK_BG)
plt.close(fig)
print("    ✓ charts/06_correlation_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING & TRAIN/TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4] Feature engineering …")
df_ml = df_enc.copy()   # already label-encoded

X = df_ml.drop(columns=["Churn"])
y = df_ml["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"    Train size: {X_train_s.shape[0]:,}  |  Test size: {X_test_s.shape[0]:,}")
print(f"    Features  : {X.shape[1]}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MACHINE LEARNING MODELS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5] Training models …")

# ── Logistic Regression (mandatory) ──────────────────────
lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr.fit(X_train_s, y_train)
y_pred_lr  = lr.predict(X_test_s)
y_prob_lr  = lr.predict_proba(X_test_s)[:, 1]
print("    ✓ Logistic Regression trained")

# ── Decision Tree (bonus) ─────────────────────────────────
dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt  = dt.predict(X_test)
y_prob_dt  = dt.predict_proba(X_test)[:, 1]
print("    ✓ Decision Tree trained")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(name, y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)
    print(f"\n    ── {name} ──")
    print(f"       Accuracy  : {acc:.4f}")
    print(f"       Precision : {prec:.4f}")
    print(f"       Recall    : {rec:.4f}")
    print(f"       F1 Score  : {f1:.4f}")
    print(f"       ROC-AUC   : {auc:.4f}")
    print(f"       Confusion Matrix:\n{cm}")
    return acc, prec, rec, f1, auc, cm

print("\n[6] Model Evaluation …")
acc_lr, prec_lr, rec_lr, f1_lr, auc_lr, cm_lr = evaluate(
    "Logistic Regression", y_test, y_pred_lr, y_prob_lr
)
acc_dt, prec_dt, rec_dt, f1_dt, auc_dt, cm_dt = evaluate(
    "Decision Tree", y_test, y_pred_dt, y_prob_dt
)


# ── Confusion matrix charts ───────────────────────────────
def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="magma", ax=ax,
                linewidths=2, linecolor="#1a1a2e",
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"],
                annot_kws={"size": 16, "weight": "bold"})
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)

plot_cm(cm_lr, "Confusion Matrix — Logistic Regression",
        "charts/07_cm_logistic.png")
print("    ✓ charts/07_cm_logistic.png")

plot_cm(cm_dt, "Confusion Matrix — Decision Tree",
        "charts/08_cm_decision_tree.png")
print("    ✓ charts/08_cm_decision_tree.png")

# ── ROC Curve comparison ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for label, y_prob, color in [
    ("Logistic Regression", y_prob_lr, "#FF6B6B"),
    ("Decision Tree",       y_prob_dt, "#4ECDC4"),
]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{label}  (AUC={auc_val:.3f})")
ax.plot([0, 1], [0, 1], "w--", lw=1.5, alpha=0.6)
ax.set_title("ROC Curve Comparison", fontsize=15, fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(framealpha=0.3, labelcolor=FONT_COLOR)
ax.yaxis.grid(True, alpha=0.3)
ax.xaxis.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("charts/09_roc_curve.png", dpi=150, bbox_inches="tight",
            facecolor=DARK_BG)
plt.close(fig)
print("    ✓ charts/09_roc_curve.png")


# ── Feature Importance (Logistic Regression coefficients) ─
feat_importance = pd.Series(np.abs(lr.coef_[0]), index=X.columns)
feat_importance = feat_importance.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors_fi = [ACCENT if v > feat_importance.median() else ACCENT2
             for v in feat_importance.values]
ax.barh(feat_importance.index[::-1], feat_importance.values[::-1],
        color=colors_fi[::-1], edgecolor="#1a1a2e", linewidth=1)
ax.set_title("Feature Importance (Logistic Regression |Coefficients|)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("|Coefficient|")
ax.xaxis.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("charts/10_feature_importance.png", dpi=150, bbox_inches="tight",
            facecolor=DARK_BG)
plt.close(fig)
print("    ✓ charts/10_feature_importance.png")

# ── Model comparison bar chart ────────────────────────────
metrics_data = {
    "Metric":    ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
    "Logistic":  [acc_lr, prec_lr, rec_lr, f1_lr, auc_lr],
    "DecTree":   [acc_dt, prec_dt, rec_dt, f1_dt, auc_dt],
}
dfm = pd.DataFrame(metrics_data).set_index("Metric")

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(dfm))
w = 0.35
ax.bar(x - w/2, dfm["Logistic"], w, color="#FF6B6B", label="Logistic Regression",
       edgecolor="#1a1a2e")
ax.bar(x + w/2, dfm["DecTree"],  w, color="#4ECDC4", label="Decision Tree",
       edgecolor="#1a1a2e")
ax.set_xticks(x); ax.set_xticklabels(dfm.index)
ax.set_ylim(0, 1.1)
ax.set_title("Model Performance Comparison", fontsize=15, fontweight="bold")
ax.set_ylabel("Score")
ax.legend(framealpha=0.3, labelcolor=FONT_COLOR)
ax.yaxis.grid(True, alpha=0.3)
for rect in ax.patches:
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, h + 0.01,
            f"{h:.3f}", ha="center", va="bottom", fontsize=9, color=FONT_COLOR)
fig.tight_layout()
fig.savefig("charts/11_model_comparison.png", dpi=150, bbox_inches="tight",
            facecolor=DARK_BG)
plt.close(fig)
print("    ✓ charts/11_model_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SAVE MODEL & METADATA (used by Streamlit app)
# ═══════════════════════════════════════════════════════════════════════════════
model_bundle = {
    "model":          lr,
    "scaler":         scaler,
    "feature_names":  list(X.columns),
    "label_encoder_map": {                   # store original categories
        col: sorted(df_plot[col].unique().tolist())
        for col in df_plot.select_dtypes("object").columns
        if col != "Churn_label"
    },
}
with open("churn_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)
print("\n    ✓ churn_model.pkl saved")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  SUMMARY")
print("═"*60)
print(f"  Dataset           : 7,043 customers, 20 features")
print(f"  Churn rate        : {df['Churn'].mean():.1%}")
print(f"\n  Logistic Regression:")
print(f"    Accuracy        : {acc_lr:.4f}")
print(f"    Precision       : {prec_lr:.4f}")
print(f"    Recall          : {rec_lr:.4f}")
print(f"    F1 Score        : {f1_lr:.4f}")
print(f"    ROC-AUC         : {auc_lr:.4f}")
print(f"\n  Decision Tree:")
print(f"    Accuracy        : {acc_dt:.4f}")
print(f"    Precision       : {prec_dt:.4f}")
print(f"    Recall          : {rec_dt:.4f}")
print(f"    F1 Score        : {f1_dt:.4f}")
print(f"    ROC-AUC         : {auc_dt:.4f}")
print(f"\n  Charts saved → ./charts/  (11 files)")
print(f"  Model saved  → churn_model.pkl")
print("═"*60)
