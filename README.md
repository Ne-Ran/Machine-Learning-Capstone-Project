# Machine Learning Capstone Project (23CSE301)

Academic Year 2026–27 | B.Tech. Computer Science and Engineering  
**Review 1 Deliverable**: Full Regression Track & Classification Track (Part A)

---

## Project Overview
This repository contains the official Machine Learning Capstone Project submission for **23CSE301**. The capstone project requires building modular, mathematically sound, leak-free, reproducible machine learning pipelines across three primary paradigms:
1. **Regression Track**: Continuous demand forecasting on the Seoul Bike Sharing Dataset (`data/Regression/SeoulBikeData.csv`).
2. **Classification Track**: Binary prediction of term deposit subscriptions on the Bank Marketing Dataset (`data/Classification/bank.csv`).
3. **Clustering Track**: Unsupervised customer segmentation (scheduled for Review 2).

The project adheres strictly to the official course rubric, emphasizing data integrity, systematic exploratory analysis across all features, leak-free preprocessing, benchmark reproducibility, and transparent reporting.

---

## Review 1 Scope
Review 1 encompasses the complete implementation and evaluation of:
- **Full Regression Track**:
  - Systematic audit and complete EDA across every predictor feature.
  - Data cleaning, outlier investigation, and quantile-stratified train/test splitting.
  - Feature engineering (cyclical hour encoding, calendar features, rush-hour indicator).
  - Implementation and benchmarking of all **10 required regression algorithms**.
  - Hyperparameter tuning via `GridSearchCV` on selected models (Random Forest and SVR) with a dedicated tuning results table.
  - 5-fold cross-validation on the training set for the top 2 models.
  - Diagnostic visualizations (Predicted vs. Actual with $y=x$, Residual plot, and Tree feature importances).
- **Classification Track (Part A)**:
  - Systematic audit and complete EDA across all 16 predictor features.
  - Evaluation of class imbalance (88.3% non-subscribers vs. 11.7% subscribers) and operational limitations of the call `duration` feature.
  - Stratified 80:20 train/test splitting and dense one-hot encoding for algorithm compatibility.
  - Feature engineering (`pdays_contacted` and `balance_negative` indicators).
  - Implementation and benchmarking of all **5 required Part-A classifiers**.
  - Metric reporting: Accuracy, Positive-Class Precision, Positive-Class Recall, Weighted F1-score, ROC-AUC, and individual formatted confusion matrix heatmaps.
  - Consolidated comparison table sorted by **Weighted F1 descending**.

Review 2 components (Classification Part B ensemble algorithms and the full Clustering Track) are structurally accommodated but deferred to Review 2.

---

## Datasets

### 1. Seoul Bike Sharing Dataset (Regression Track)
- **Path**: `data/Regression/SeoulBikeData.csv`
- **Observations**: 8,760 hourly records spanning a complete annual cycle (Dec 1, 2017 – Nov 30, 2018).
- **Features**: 14 attributes including meteorological observations (Temperature, Humidity, Wind Speed, Solar Radiation, Rainfall, Snowfall, Dew Point Temperature, Visibility) and operational calendar indicators (Hour, Seasons, Holiday, Functioning Day).
- **Target**: `Rented Bike Count` (continuous hourly rental demand, range: 0 – 3,556).
- **Missing Values & Duplicates**: 0 missing entries, 0 duplicate records.

### 2. Bank Marketing Dataset (Classification Track Part A)
- **Path**: `data/Classification/bank.csv`
- **Observations**: 45,211 client records from direct telemarketing campaigns of a Portuguese banking institution.
- **Features**: 16 attributes spanning client demographics (age, job, marital, education, default, balance, housing, loan), current contact details (contact, day, month, duration), and previous campaign outcomes (campaign, pdays, previous, poutcome).
- **Target**: `y` (binary classification: `'yes'` = subscribed to term deposit [5,289 / 11.7%], `'no'` = did not subscribe [39,922 / 88.3%]).
- **Missing Values & Duplicates**: 0 missing entries, 0 duplicate records.

---

## Repository Structure
```text
ML Capstone/
│
├── .gitignore                                      # Excludes checkpoints, cache, and virtual environments
├── 23CSE301_ML_26_27_Capstone_Guidelines (1).pdf    # Official course guidelines & rubric
├── Master Prompt for Anti-gravity.md                # System prompt specification
├── README.md                                       # Master project documentation
├── requirements.txt                                # Pinned runtime dependencies
│
├── data/
│   ├── Classification/
│   │   └── bank.csv                                # Genuine Bank Marketing CSV dataset
│   └── Regression/
│       └── SeoulBikeData.csv                       # Seoul Bike Sharing CSV dataset
│
├── Notebooks/
│   ├── regression.ipynb                            # Fully executed Regression track notebook
│   └── classification.ipynb                        # Fully executed Classification Part-A notebook
│
└── Models/                                         # Directory allocated for serialized model artifacts
```

---

## Regression Track: Summary of Results

### Master Baseline Comparison Table (Ranked by Test $R^2$)
All ten baseline algorithms were trained on the identical training partition (7,008 samples) and evaluated on the identical held-out test partition (1,752 samples) using a quantile-stratified 80:20 split:

| Rank | Model | Test $R^2$ | Test RMSE | Test MAE |
| :---: | :--- | :---: | :---: | :---: |
| 1 | **Random Forest Regressor** | **0.9429** | **154.03** | **93.60** |
| 2 | **Gradient Boosting Regressor** | **0.8894** | **214.32** | **148.36** |
| 3 | **Decision Tree Regressor** | **0.8688** | **233.45** | **130.15** |
| 4 | **Polynomial Regression (deg 2)** | **0.8450** | **253.71** | **181.14** |
| 5 | **K-Nearest Neighbors Regressor** | **0.7798** | **302.40** | **180.77** |
| 6 | **Linear Regression** | **0.6191** | **397.72** | **300.34** |
| 7 | **Ridge Regression** | **0.6190** | **397.79** | **300.39** |
| 8 | **Lasso Regression** | **0.6172** | **398.70** | **301.17** |
| 9 | **ElasticNet Regression** | **0.5004** | **455.47** | **333.30** |
| 10 | **Support Vector Regressor (SVR)** | **0.2794** | **547.03** | **375.43** |

### Hyperparameter Tuning Results
Hyperparameter tuning via `GridSearchCV` was applied to two selected models:
- **Support Vector Regressor (SVR)**: Tuning $C=1000, \\epsilon=20.0$ lifted $R^2$ from **0.2794 to 0.9046** ($+0.6252$ improvement), resolving the severe scale penalty of default $C=1.0$.
- **Random Forest Regressor**: Tuning `n_estimators=200`, `max_depth=None`, `min_samples_split=2` yielded $R^2 = 0.9434$ ($\text{RMSE} = 153.25$, $\text{MAE} = 93.30$).

### 5-Fold Cross-Validation (Training Split)
Cross-validation on the training set confirmed strong generalization stability:
- **Tuned Random Forest**: Mean CV $R^2 = \mathbf{0.9318 \pm 0.0087}$ (Held-out Test $R^2 = 0.9434$).
- **Gradient Boosting Regressor**: Mean CV $R^2 = \mathbf{0.8861 \pm 0.0051}$ (Held-out Test $R^2 = 0.8894$).

---

## Classification Track (Part A): Summary of Results

### Master Preliminary Comparison Table (Ranked by Weighted F1 Descending)
All five Part-A classifiers were evaluated on the identical stratified test partition (9,043 samples; 7,985 non-subscribers and 1,058 subscribers). The positive class is explicitly defined as `1` (`'yes'`):

| Rank | Model | Accuracy | Precision (`yes`) | Recall (`yes`) | Weighted F1 | ROC-AUC | Training Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Decision Tree Classifier** | 0.9001 | 0.6106 | 0.4045 | **0.8911** | 0.8588 | 0.14s |
| 2 | **Logistic Regression** | 0.9012 | 0.6445 | 0.3478 | **0.8879** | **0.9055** | 0.12s |
| 3 | **Support Vector Classifier (SVC)** | **0.9024** | **0.6642** | 0.3346 | **0.8878** | 0.8912 | 25.87s |
| 4 | **K-Nearest Neighbors** | 0.8959 | 0.6039 | 0.3214 | **0.8816** | 0.8536 | 2.55s |
| 5 | **Gaussian Naive Bayes** | 0.8555 | 0.4036 | **0.4924** | **0.8616** | 0.8048 | 0.07s |

### Key Diagnostic Takeaways
- **Class Imbalance Dynamics**: While overall accuracy exceeds 90%, positive-class recall ranges between 32% and 49%, reflecting the asymmetric cost of false negatives in telemarketing.
- **Model Divergence**:
  - **Logistic Regression** and **SVC** provide high precision (~64–66%) and top-tier ranking discrimination ($\text{ROC-AUC} \ge 0.89$).
  - **Gaussian Naive Bayes** delivers the highest subscriber detection rate ($\text{Recall} = 49.24\%$), proving advantageous when missing a potential subscriber carries a high penalty.
- **Operational Limitation of `duration`**: Call duration is a powerful retrospective predictor ($r = +0.39$), but cannot be used for prospective lead filtering prior to placing calls.

---

## Environment Setup
The project requires Python 3.10+ (tested on Python 3.13.6). To install the exact dependencies, run:

```bash
pip install -r requirements.txt
```

### Dependencies
```text
numpy==2.2.6
pandas==2.3.3
matplotlib==3.10.7
seaborn==0.13.2
scikit-learn==1.8.0
jupyter
notebook
```

---

## How to Run

### Interactive Exploration
Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook
```
Open and execute:
- `Notebooks/regression.ipynb`
- `Notebooks/classification.ipynb`

### Non-Interactive Headless Execution
To re-run and re-render both notebooks top-to-bottom from the command line:
```bash
# Execute Regression track notebook in place
jupyter nbconvert --to notebook --execute --inplace Notebooks/regression.ipynb

# Execute Classification Part-A notebook in place
jupyter nbconvert --to notebook --execute --inplace Notebooks/classification.ipynb
```

---

## Reproducibility & Anti-Leakage Protocol
1. **Global Seed**: `RANDOM_STATE = 42` is enforced across all train/test splits, cross-validation folds, and stochastic estimators.
2. **Strict Featurization Ordering**: All scaling (`StandardScaler`) and encoding (`OneHotEncoder`) transformers are fitted **strictly on the training partition** (`X_train`) and applied downstream to the test partition (`X_test`). Full-dataset preprocessing fits are strictly prohibited.
3. **Target Isolation**: Target quantile bins used for regression stratified splitting are constructed temporarily and dropped immediately prior to feature transformation.

---

## AI Assistance Disclosure
In accordance with academic guidelines (Section 7.5 of course rubrics):
Generative AI assistance (Google Antigravity / Gemini) was utilized strictly for codebase scaffolding, markdown formatting, and pipeline architecture. All exploratory data analyses, statistical observations, machine learning implications, and metric interpretations are derived directly from actual model execution on the official datasets.