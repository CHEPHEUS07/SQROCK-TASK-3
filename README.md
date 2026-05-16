# Customer Churn Prediction System

## SQROCK Week 3 Task — Machine Learning Project

A complete **Customer Churn Prediction System** built with Python and Streamlit.

### 🚀 Live App
> Live at: [https://churnsight-ai.onrender.com/](https://churnsight-ai.onrender.com/)

### 📊 Features
- **EDA & Insights** — Churn distribution, contract analysis, tenure & charges
- **ML Models** — Logistic Regression + Decision Tree
- **Model Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- **Live Predictions** — Input customer data and get instant churn probability
- **Dataset Explorer** — Filter and download the dataset

### 🛠️ Tech Stack
- Python · Pandas · NumPy
- Matplotlib · Seaborn
- Scikit-learn
- Streamlit

### 📁 Files
| File | Description |
|---|---|
| `app.py` | Main Streamlit application |
| `generate_data.py` | Synthetic dataset generator |
| `churn_analysis.py` | Full EDA + ML pipeline script |
| `telco_churn.csv` | Dataset (7,043 customers, 21 features) |
| `requirements.txt` | Python dependencies |

### ▶️ Run Locally
```bash
pip install -r requirements.txt
python generate_data.py   # generates telco_churn.csv
streamlit run app.py
```
