"""
Streamlit Web Application: Bank Marketing Term Deposit Prediction Demo
23CSE301 Machine Learning Capstone — Review 2 (Bonus GUI & Public Deployment)
"""
import sys
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Ensure repository root is on sys.path so BankFeatureEngineering can be unpickled
app_dir = Path(__file__).resolve().parent
repo_root = app_dir.parent

if str(repo_root) in sys.path:
    sys.path.remove(str(repo_root))
sys.path.insert(0, str(repo_root))

if str(app_dir) not in sys.path:
    sys.path.append(str(app_dir))

# Import and alias custom transformer so joblib unpickling resolves under either namespace
try:
    import app.bank_features
    sys.modules['bank_features'] = app.bank_features
    from app.bank_features import BankFeatureEngineering
except ImportError:
    import bank_features
    sys.modules['app.bank_features'] = bank_features
    from bank_features import BankFeatureEngineering

# Streamlit Page Configuration
st.set_page_config(
    page_title="Bank Term Deposit Prediction — ML Capstone",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache model loading for instant page re-renders
@st.cache_resource
def load_model_pipeline():
    possible_paths = [
        repo_root / "models" / "bank_marketing_pipeline.joblib",
        app_dir / "models" / "bank_marketing_pipeline.joblib",
        Path("models/bank_marketing_pipeline.joblib")
    ]
    for path in possible_paths:
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError(f"Model artifact not found in any of: {[str(p) for p in possible_paths]}")

# Header & Introduction
st.title("🏦 Bank Term Deposit Subscription Predictor")
st.caption("23CSE301 Machine Learning Capstone Project — Review 2 Interactive GUI")

# Prominent Operational Notice on Duration Feature
st.warning(
    "⚠️ **Operational Caveat on Call Duration (`duration`)**:\n\n"
    "In telemarketing campaigns, call duration is known **strictly ex-post** (only after the conversation concludes). "
    "Prior to dialling ($t_0$), duration is fundamentally unknown. This application operates as a "
    "**Completed-Call Diagnostic System** to evaluate conversion likelihood given client profile and "
    "conversation engagement, rather than an ex-ante outbound prospecting selector."
)

# Sidebar with model information
with st.sidebar:
    st.header("📌 Model Information")
    try:
        loaded_pipe = load_model_pipeline()
        clf_step = loaded_pipe.named_steps.get('clf', None)
        clf_display_name = clf_step.__class__.__name__ if clf_step is not None else "Tuned Classifier"
    except Exception:
        clf_display_name = "Multi-Layer Perceptron (Tuned)"

    st.markdown(f"""
    - **Dataset**: Bank Marketing (45,211 instances)
    - **Champion Model**: `{clf_display_name}`
    - **Selection Metric**: 5-Fold Stratified CV Macro F1
    - **Preprocessing**: Leak-free ColumnTransformer with `BankFeatureEngineering`, `StandardScaler`, and `OneHotEncoder`.
    """)
    st.divider()
    st.markdown("**Team**: Machine Learning Capstone (Review 2)")
    st.markdown("**Course**: 23CSE301 | AY 2026–27")

# Main Interface Tabs
tab_predict, tab_about = st.tabs(["🎯 Real-Time Prediction", "📊 Model Benchmark & Methodology"])

with tab_predict:
    st.subheader("Client Demographic & Campaign Profile")
    st.write("Configure customer attributes below and click **Run Prediction** to estimate term deposit subscription propensity.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 👤 Demographics")
        age = st.slider("Age", min_value=18, max_value=95, value=38, step=1, help="Client age in years")
        job = st.selectbox(
            "Job Category",
            options=['management', 'technician', 'entrepreneur', 'blue-collar', 'retired', 'admin.', 
                     'services', 'self-employed', 'unemployed', 'housemaid', 'student', 'unknown'],
            index=0
        )
        marital = st.selectbox(
            "Marital Status",
            options=['married', 'single', 'divorced'],
            index=0
        )
        education = st.selectbox(
            "Education Level",
            options=['secondary', 'tertiary', 'primary', 'unknown'],
            index=1
        )

    with col2:
        st.markdown("#### 💰 Financial Profile")
        default = st.selectbox("Credit in Default?", options=['no', 'yes'], index=0)
        balance = st.number_input("Average Yearly Balance (€)", min_value=-8000, max_value=105000, value=1200, step=100)
        housing = st.selectbox("Has Housing Loan?", options=['yes', 'no'], index=0)
        loan = st.selectbox("Has Personal Loan?", options=['no', 'yes'], index=0)

    with col3:
        st.markdown("#### 📞 Campaign Outreach Context")
        contact = st.selectbox("Contact Communication Type", options=['cellular', 'telephone', 'unknown'], index=0)
        month = st.selectbox(
            "Last Contact Month",
            options=['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr', 'sep'],
            index=0
        )
        day = st.slider("Last Contact Day of Month", min_value=1, max_value=31, value=15, step=1)
        duration = st.slider("Call Duration (seconds)", min_value=0, max_value=3000, value=240, step=10, 
                             help="Recorded conversation length in seconds (Ex-Post metric)")

    st.markdown("#### 📜 Previous Marketing Campaigns")
    col_prev1, col_prev2, col_prev3 = st.columns(3)
    
    with col_prev1:
        campaign = st.number_input("Number of Contacts in Current Campaign", min_value=1, max_value=50, value=1, step=1)
    
    with col_prev2:
        has_prior_contact = st.checkbox("Client was contacted in previous campaigns?", value=False)
        if has_prior_contact:
            pdays = st.number_input("Days Since Prior Campaign Contact", min_value=1, max_value=871, value=90, step=1)
            previous = st.number_input("Number of Prior Campaign Contacts", min_value=1, max_value=50, value=1, step=1)
        else:
            pdays = -1
            previous = 0
            st.info("Pdays is fixed to `-1` and Previous to `0` (never contacted prior).")

    with col_prev3:
        poutcome = st.selectbox(
            "Outcome of Previous Campaign",
            options=['unknown', 'failure', 'other', 'success'],
            index=0 if not has_prior_contact else 1
        )

    st.divider()

    # Predict Button
    if st.button("🚀 Evaluate Subscription Propensity", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'balance': balance,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'day': day,
            'month': month,
            'duration': duration,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'poutcome': poutcome
        }])

        try:
            pipeline = load_model_pipeline()
            pred_class = pipeline.predict(input_data)[0]
            pred_probs = pipeline.predict_proba(input_data)[0]
            prob_yes = pred_probs[1]
            prob_no = pred_probs[0]

            st.subheader("Prediction Result")
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                if pred_class == 1:
                    st.success("### ✅ SUBSCRIBED\n**Likely to subscribe to term deposit**")
                else:
                    st.error("### ❌ NOT SUBSCRIBED\n**Unlikely to subscribe to term deposit**")

            with res_col2:
                st.markdown(f"**Subscription Probability: `{prob_yes * 100:.2f}%`**")
                st.progress(float(prob_yes))
                st.caption(f"Estimated probabilities: P(Yes) = {prob_yes:.4f} | P(No) = {prob_no:.4f}")

                if prob_yes >= 0.50:
                    st.info("High engagement likelihood: Recommend priority client outreach and personalized deposit terms.")
                elif prob_yes >= 0.25:
                    st.warning("Moderate engagement: Conversion possible with follow-up or customized incentive package.")
                else:
                    st.write("Low conversion probability: Standard digital marketing touchpoint recommended.")

            st.caption("ℹ️ *Note: Posterior probabilities are uncalibrated model decision estimates and reflect conversion propensity under the 11.7% historical conversion rate.*")

        except Exception as e:
            st.error(f"Error during model evaluation: {e}")

with tab_about:
    st.subheader("Review 2 Classification Methodology")
    st.markdown("""
    ### 10-Algorithm Comparative Benchmark
    In Review 2, we evaluated ten distinct classification algorithms on the identical 80:20 stratified split ($N_{test} = 9,043$):
    - **Part A Baselines**: Logistic Regression, K-Nearest Neighbors, Gaussian Naive Bayes, Decision Tree, Support Vector Classifier (RBF).
    - **Part B Ensembles & Neural Networks**: Random Forest, AdaBoost, Gradient Boosting, Bagging (Decision Tree base), Multi-Layer Perceptron (MLP).
    
    ### Preprocessing & Leakage Elimination
    - Custom deterministic feature engineering (`pdays_contacted` and `balance_negative`) implemented via reusable `BankFeatureEngineering` transformer.
    - Numerical features standardized via `StandardScaler()`.
    - Categorical features one-hot encoded via `OneHotEncoder(drop='first', handle_unknown='ignore')`.
    - 5-Fold Stratified Cross-Validation executed with end-to-end pipelines to prevent preprocessing leakage into validation folds.
    
    ### Model Selection Rationale
    - Because the dataset has an 11.7% conversion imbalance, candidate models were evaluated strictly using **5-Fold Stratified CV Macro F1** on the training partition.
    - **Multi-Layer Perceptron (MLP)** achieved top training CV Macro F1 (`0.7383`), yielding peak held-out test Macro F1 (`0.7573`), while **Gradient Boosting** achieved top test Weighted F1 (`0.8996`) and ROC-AUC (`0.9284`).
    """)
