"""
app.py  –  Customer Churn Prediction  |  Streamlit UI
Run:  streamlit run app.py
"""

import pickle, os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSight AI",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0f0f1a; }

/* metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg,#1a1a2e,#16213e);
    border: 1px solid #e9456033;
    border-radius: 12px;
    padding: 16px;
}

/* sidebar */
[data-testid="stSidebar"] { background: #0d0d1f; border-right: 1px solid #222; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: #0f0f1a; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    background: #1a1a2e;
    color: #aaa;
    border: 1px solid #333;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#e94560,#c0392b) !important;
    color: white !important;
    border: none !important;
}

/* buttons */
.stButton > button {
    background: linear-gradient(135deg,#e94560,#c0392b);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-weight: 600;
    font-size: 15px;
    transition: all .2s;
    width: 100%;
}
.stButton > button:hover { opacity: .85; transform: translateY(-1px); }

/* prediction result cards */
.result-churn {
    background: linear-gradient(135deg,#e9456022,#c0392b22);
    border: 2px solid #e94560;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}
.result-safe {
    background: linear-gradient(135deg,#4ecdc422,#1abc9c22);
    border: 2px solid #4ecdc4;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}
.badge {
    display:inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    margin: 4px 2px;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg,#e94560,#f8a5c2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-sub { color: #888; font-size: 1.1rem; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Dark plot theme ───────────────────────────────────────
DARK="#1a1a2e"; ACCENT="#e94560"; TEAL="#4ecdc4"
plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": "#16213e",
    "axes.edgecolor": "#444", "axes.labelcolor": "#eee",
    "xtick.color": "#eee", "ytick.color": "#eee",
    "text.color": "#eee", "grid.color": "#333", "grid.alpha": .4,
})

# ═══════════════════════════════════════════════════════════
# Data loading & preprocessing (cached)
# ═══════════════════════════════════════════════════════════
@st.cache_data
def load_and_prepare():
    df = pd.read_csv("telco_churn.csv")
    df.drop(columns=["customerID"], inplace=True, errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    return df

@st.cache_resource
def train_models(df):
    df2 = df.copy()
    df2["Churn"] = df2["Churn"].map({"Yes": 1, "No": 0})
    le = LabelEncoder()
    cat_cols = df2.select_dtypes("object").columns.tolist()
    for col in cat_cols:
        df2[col] = le.fit_transform(df2[col].astype(str))
    X = df2.drop(columns=["Churn"])
    y = df2["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced", solver="liblinear")
    lr.fit(Xs_train, y_train)
    dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                                random_state=42, class_weight="balanced")
    dt.fit(X_train, y_train)

    return lr, dt, scaler, X.columns.tolist(), X_test, Xs_test, y_test

df_raw  = load_and_prepare()
lr, dt, scaler, feature_cols, X_test, Xs_test, y_test = train_models(df_raw)

df_vis = df_raw.copy()   # keep original labels for charts

# ═══════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════
st.markdown('<p class="hero-title">📉 ChurnSight AI</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Customer Churn Prediction System — built with Machine Learning</p>',
            unsafe_allow_html=True)
st.markdown("---")

# ═══════════════════════════════════════════════════════════
# KPI row
# ═══════════════════════════════════════════════════════════
total    = len(df_raw)
churned  = (df_raw["Churn"] == "Yes").sum()
retained = total - churned
churn_rate = churned / total * 100

k1, k2, k3, k4 = st.columns(4)
k1.metric("👥 Total Customers", f"{total:,}")
k2.metric("🚪 Churned",         f"{churned:,}",   f"{churn_rate:.1f}% rate")
k3.metric("✅ Retained",        f"{retained:,}",  f"{100-churn_rate:.1f}%")
k4.metric("📊 Features Used",   len(feature_cols))

st.markdown("---")

# ═══════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA & Insights", "📈 Model Results", "🔮 Predict Churn", "📋 Dataset"
])

# ─── TAB 1 : EDA ─────────────────────────────────────────
with tab1:
    st.subheader("Exploratory Data Analysis")

    c1, c2 = st.columns(2)

    # Churn donut
    with c1:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie([retained, churned], labels=["Retained", "Churned"],
               colors=[TEAL, ACCENT], autopct="%1.1f%%", startangle=140,
               wedgeprops=dict(width=.55, edgecolor=DARK, linewidth=3),
               textprops={"color": "#eee", "fontsize": 11})
        ax.set_title("Overall Churn Distribution", fontsize=13, fontweight="bold")
        st.pyplot(fig); plt.close(fig)

    # Churn by contract
    with c2:
        ct = df_vis.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(5, 5))
        ct.plot(kind="bar", ax=ax, color=[TEAL, ACCENT],
                edgecolor=DARK, linewidth=1.2, width=.6)
        ax.set_title("Churn by Contract Type", fontsize=13, fontweight="bold")
        ax.set_xlabel("Contract"); ax.set_ylabel("Customers")
        ax.legend(["Retained", "Churned"], framealpha=.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
        ax.yaxis.grid(True, alpha=.3)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    c3, c4 = st.columns(2)

    # Tenure histogram
    with c3:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(df_vis[df_vis["Churn"]=="No"]["tenure"],  bins=25, alpha=.75, color=TEAL,   label="Retained", edgecolor=DARK)
        ax.hist(df_vis[df_vis["Churn"]=="Yes"]["tenure"], bins=25, alpha=.75, color=ACCENT, label="Churned",  edgecolor=DARK)
        ax.set_title("Tenure Distribution", fontsize=13, fontweight="bold")
        ax.set_xlabel("Tenure (months)"); ax.set_ylabel("Count")
        ax.legend(framealpha=.3); ax.yaxis.grid(True, alpha=.3)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # Monthly charges boxplot
    with c4:
        fig, ax = plt.subplots(figsize=(5, 4))
        data_no  = df_vis[df_vis["Churn"]=="No"]["MonthlyCharges"]
        data_yes = df_vis[df_vis["Churn"]=="Yes"]["MonthlyCharges"]
        bp = ax.boxplot([data_no, data_yes], labels=["Retained","Churned"],
                        patch_artist=True, medianprops={"color":"white","linewidth":2})
        bp["boxes"][0].set_facecolor(TEAL + "88")
        bp["boxes"][1].set_facecolor(ACCENT + "88")
        ax.set_title("Monthly Charges", fontsize=13, fontweight="bold")
        ax.set_ylabel("Monthly Charges ($)"); ax.yaxis.grid(True, alpha=.3)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    df_enc2 = df_vis.copy()
    le2 = LabelEncoder()
    for col in df_enc2.select_dtypes("object").columns:
        df_enc2[col] = le2.fit_transform(df_enc2[col].astype(str))
    fig, ax = plt.subplots(figsize=(13, 9))
    sns.heatmap(df_enc2.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=.4, linecolor=DARK,
                annot_kws={"size": 7},
                mask=np.triu(np.ones(df_enc2.shape[1], dtype=bool)))
    ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # Key insights
    st.subheader("💡 Key Insights")
    contract_churn = df_vis.groupby("Contract")["Churn"].apply(lambda x: (x=="Yes").mean()*100).round(1)
    inet_churn     = df_vis.groupby("InternetService")["Churn"].apply(lambda x: (x=="Yes").mean()*100).round(1)
    senior_yes = df_vis[df_vis["SeniorCitizen"]==1]["Churn"].eq("Yes").mean()*100
    senior_no  = df_vis[df_vis["SeniorCitizen"]==0]["Churn"].eq("Yes").mean()*100

    i1, i2, i3 = st.columns(3)
    i1.info(f"📑 Month-to-month customers churn at **{contract_churn.get('Month-to-month',0):.1f}%** vs only **{contract_churn.get('Two year',0):.1f}%** for 2-year contracts")
    i2.warning(f"📡 Fiber optic users churn at **{inet_churn.get('Fiber optic',0):.1f}%** — highest among all internet types")
    i3.error(f"👴 Senior citizens churn at **{senior_yes:.1f}%** vs **{senior_no:.1f}%** for non-seniors")

# ─── TAB 2 : MODEL RESULTS ───────────────────────────────
with tab2:
    st.subheader("Model Performance Evaluation")

    y_pred_lr = lr.predict(Xs_test)
    y_prob_lr = lr.predict_proba(Xs_test)[:,1]
    y_pred_dt = dt.predict(X_test)
    y_prob_dt = dt.predict_proba(X_test)[:,1]

    def metrics_dict(yt, yp, ypr):
        return {
            "Accuracy":  round(accuracy_score(yt,yp),4),
            "Precision": round(precision_score(yt,yp),4),
            "Recall":    round(recall_score(yt,yp),4),
            "F1 Score":  round(f1_score(yt,yp),4),
            "ROC-AUC":   round(roc_auc_score(yt,ypr),4),
        }

    m_lr = metrics_dict(y_test, y_pred_lr, y_prob_lr)
    m_dt = metrics_dict(y_test, y_pred_dt, y_prob_dt)

    st.markdown("#### 📊 Metrics Comparison")
    mdf = pd.DataFrame({"Logistic Regression": m_lr, "Decision Tree": m_dt})
    st.dataframe(mdf.style.highlight_max(axis=1, color="#e9456033")
                 .format("{:.4f}"), use_container_width=True)

    c1, c2 = st.columns(2)

    # Confusion matrices
    with c1:
        cm = confusion_matrix(y_test, y_pred_lr)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="magma", ax=ax,
                    xticklabels=["Retained","Churned"],
                    yticklabels=["Retained","Churned"],
                    annot_kws={"size":14,"weight":"bold"})
        ax.set_title("Confusion Matrix — Logistic Regression", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with c2:
        cm2 = confusion_matrix(y_test, y_pred_dt)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm2, annot=True, fmt="d", cmap="magma", ax=ax,
                    xticklabels=["Retained","Churned"],
                    yticklabels=["Retained","Churned"],
                    annot_kws={"size":14,"weight":"bold"})
        ax.set_title("Confusion Matrix — Decision Tree", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ROC curves
    st.markdown("#### 📈 ROC Curve")
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, yp, c in [("Logistic Regression", y_prob_lr, ACCENT),
                          ("Decision Tree", y_prob_dt, TEAL)]:
        fpr, tpr, _ = roc_curve(y_test, yp)
        auc = roc_auc_score(y_test, yp)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],"w--",lw=1.5,alpha=.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison", fontsize=13, fontweight="bold")
    ax.legend(framealpha=.3); ax.grid(True,alpha=.3)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # Feature importance
    st.markdown("#### 🔑 Feature Importance (Logistic Regression)")
    fi = pd.Series(np.abs(lr.coef_[0]), index=feature_cols).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [ACCENT if v > fi.median() else "#0f3460" for v in fi.values]
    ax.barh(fi.index, fi.values, color=colors, edgecolor=DARK, linewidth=.8)
    ax.set_xlabel("|Coefficient|"); ax.set_title("Feature Importance", fontsize=13, fontweight="bold")
    ax.xaxis.grid(True, alpha=.3)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ─── TAB 3 : PREDICT ─────────────────────────────────────
with tab3:
    st.subheader("🔮 Predict Churn for a New Customer")
    st.markdown("Fill in customer details below and click **Predict**.")

    # encode helper
    def encode_yes_no(val): return 1 if val == "Yes" else 0
    def encode_service(val):
        return {"Yes":2,"No":1,"No internet service":0,"No phone service":0}[val]

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Demographics**")
            gender         = st.selectbox("Gender",          ["Male","Female"])
            senior         = st.selectbox("Senior Citizen",  ["No","Yes"])
            partner        = st.selectbox("Partner",         ["Yes","No"])
            dependents     = st.selectbox("Dependents",      ["No","Yes"])

        with c2:
            st.markdown("**Account Info**")
            tenure         = st.slider("Tenure (months)",    0, 72, 12)
            contract       = st.selectbox("Contract",        ["Month-to-month","One year","Two year"])
            paperless      = st.selectbox("Paperless Billing",["Yes","No"])
            payment        = st.selectbox("Payment Method",  ["Electronic check","Mailed check",
                                                               "Bank transfer (automatic)","Credit card (automatic)"])
            monthly        = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
            total_c        = st.number_input("Total Charges ($)", min_value=0.0, value=monthly*tenure)

        with c3:
            st.markdown("**Services**")
            phone_svc      = st.selectbox("Phone Service",    ["Yes","No"])
            multi_lines    = st.selectbox("Multiple Lines",   ["No","Yes","No phone service"])
            internet_svc   = st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
            online_sec     = st.selectbox("Online Security",  ["No","Yes","No internet service"])
            online_bk      = st.selectbox("Online Backup",   ["No","Yes","No internet service"])
            dev_prot       = st.selectbox("Device Protection",["No","Yes","No internet service"])
            tech_sup       = st.selectbox("Tech Support",     ["No","Yes","No internet service"])
            stream_tv      = st.selectbox("Streaming TV",     ["No","Yes","No internet service"])
            stream_mv      = st.selectbox("Streaming Movies", ["No","Yes","No internet service"])

        submitted = st.form_submit_button("🔮 Predict Churn Risk")

    if submitted:
        # Build input row matching feature_cols order
        le_map = {
            "gender":          {"Male":0,"Female":1},
            "Partner":         {"Yes":1,"No":0},
            "Dependents":      {"Yes":1,"No":0},
            "PhoneService":    {"Yes":1,"No":0},
            "MultipleLines":   {"No":1,"No phone service":0,"Yes":2},
            "InternetService": {"DSL":0,"Fiber optic":1,"No":2},
            "OnlineSecurity":  {"No":1,"No internet service":0,"Yes":2},
            "OnlineBackup":    {"No":1,"No internet service":0,"Yes":2},
            "DeviceProtection":{"No":1,"No internet service":0,"Yes":2},
            "TechSupport":     {"No":1,"No internet service":0,"Yes":2},
            "StreamingTV":     {"No":1,"No internet service":0,"Yes":2},
            "StreamingMovies": {"No":1,"No internet service":0,"Yes":2},
            "Contract":        {"Month-to-month":0,"One year":1,"Two year":2},
            "PaperlessBilling":{"Yes":1,"No":0},
            "PaymentMethod":   {"Bank transfer (automatic)":0,"Credit card (automatic)":1,
                                "Electronic check":2,"Mailed check":3},
        }
        row = {
            "gender":          le_map["gender"][gender],
            "SeniorCitizen":   1 if senior=="Yes" else 0,
            "Partner":         le_map["Partner"][partner],
            "Dependents":      le_map["Dependents"][dependents],
            "tenure":          tenure,
            "PhoneService":    le_map["PhoneService"][phone_svc],
            "MultipleLines":   le_map["MultipleLines"][multi_lines],
            "InternetService": le_map["InternetService"][internet_svc],
            "OnlineSecurity":  le_map["OnlineSecurity"][online_sec],
            "OnlineBackup":    le_map["OnlineBackup"][online_bk],
            "DeviceProtection":le_map["DeviceProtection"][dev_prot],
            "TechSupport":     le_map["TechSupport"][tech_sup],
            "StreamingTV":     le_map["StreamingTV"][stream_tv],
            "StreamingMovies": le_map["StreamingMovies"][stream_mv],
            "Contract":        le_map["Contract"][contract],
            "PaperlessBilling":le_map["PaperlessBilling"][paperless],
            "PaymentMethod":   le_map["PaymentMethod"][payment],
            "MonthlyCharges":  monthly,
            "TotalCharges":    total_c,
        }
        X_input = pd.DataFrame([row])[feature_cols]
        X_scaled = scaler.transform(X_input)
        pred  = lr.predict(X_scaled)[0]
        prob  = lr.predict_proba(X_scaled)[0][1]

        st.markdown("---")
        if pred == 1:
            st.markdown(f"""
            <div class="result-churn">
              <h2 style="color:#e94560;margin:0">⚠️ HIGH CHURN RISK</h2>
              <p style="font-size:3rem;margin:12px 0;font-weight:700;color:#e94560">{prob*100:.1f}%</p>
              <p style="color:#ccc">This customer is likely to churn. Consider offering a retention incentive.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
              <h2 style="color:#4ecdc4;margin:0">✅ LOW CHURN RISK</h2>
              <p style="font-size:3rem;margin:12px 0;font-weight:700;color:#4ecdc4">{prob*100:.1f}%</p>
              <p style="color:#ccc">This customer is likely to stay. Keep up the good experience!</p>
            </div>""", unsafe_allow_html=True)

        # Risk factors
        st.markdown("#### 🔍 Key Risk Factors")
        risks = []
        if contract == "Month-to-month": risks.append("📑 Month-to-month contract (highest churn risk)")
        if internet_svc == "Fiber optic": risks.append("📡 Fiber optic internet (higher churn rate)")
        if senior == "Yes": risks.append("👴 Senior citizen (elevated churn rate)")
        if tenure < 12: risks.append("🕐 Short tenure (<12 months) — new customer")
        if monthly > 80: risks.append(f"💰 High monthly charges (${monthly:.0f})")

        if risks:
            for r in risks:
                st.warning(r)
        else:
            st.success("✅ No major risk factors detected for this customer.")

# ─── TAB 4 : DATASET ─────────────────────────────────────
with tab4:
    st.subheader("📋 Dataset Preview")
    st.caption(f"Shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

    col_filter = st.multiselect("Filter columns to display",
                                df_raw.columns.tolist(),
                                default=df_raw.columns.tolist()[:8])
    churn_filter = st.radio("Show customers:", ["All","Churned only","Retained only"], horizontal=True)
    df_show = df_raw.copy()
    if churn_filter == "Churned only":  df_show = df_show[df_show["Churn"]=="Yes"]
    if churn_filter == "Retained only": df_show = df_show[df_show["Churn"]=="No"]
    st.dataframe(df_show[col_filter].head(500), use_container_width=True)

    @st.cache_data
    def convert_csv(df): return df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Dataset (CSV)", convert_csv(df_raw),
                       "telco_churn.csv", "text/csv")

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#555;font-size:13px'>"
    "ChurnSight AI · Built for SQROCK Week 3 Task · "
    "Logistic Regression + Decision Tree · Scikit-learn · Streamlit"
    "</p>", unsafe_allow_html=True
)
