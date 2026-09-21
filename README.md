# Machine Learning Capstone Project (23CSE301)

Academic Year 2026–27 | B.Tech. Computer Science and Engineering  
**Review 1 Deliverable**: Full Regression Track & Classification Track (Part A)

---

## Project Overview
This repository contains the official Machine Learning Capstone Project submission for **23CSE301**. The capstone project requires building modular, mathematically sound, leak-free, reproducible machine learning pipelines across three primary paradigms:
1. **Regression Track**: Continuous demand forecasting on the Seoul Bike Sharing Dataset (`data/regression/SeoulBikeData.csv`).
2. **Classification Track**: Binary prediction of term deposit subscriptions on the Bank Marketing Dataset (`data/classification/bank.csv`).
3. **Clustering Track**: Unsupervised customer segmentation (scheduled for Review 2).

The project adheres strictly to the official course rubric, emphasizing data integrity, systematic exploratory analysis across all features, leak-free preprocessing, benchmark reproducibility, and transparent reporting.

---

## Review 1 Scope
Review 1 encompasses the complete implementation and evaluation of:
- **Full Regression Track**:
  - Systematic audit and complete EDA across every predictor feature (9 numerical, 3 categorical, target, calendar continuity audit, and Date feature coverage visualization).
  - Data cleaning, outlier investigation with domain retention justification, and quantile-stratified train/test splitting.
  - Feature engineering (cyclical hour harmonics, calendar components, rush-hour indicator) with zero data leakage.
  - Implementation and benchmarking of all **10 required regression algorithms** on a single held-out test split.
  - Controlled polynomial degree comparison (Degree 1 vs. Degree 2) evaluating empirical trade-offs.
  - Hyperparameter tuning via `GridSearchCV` on selected models (Random Forest and SVR) using leak-free `Pipeline` architectures.
  - 5-fold cross-validation on the training set reporting both Baseline Top Two (Random Forest + Gradient Boosting) and Post-Tuning Top Two (Random Forest + SVR).
  - Diagnostic visualizations (Predicted vs. Actual with $y=x$, Residual plot with variance analysis, and Tree feature importances).
- **Classification Track (Part A)**:
  - Systematic audit and complete EDA across all 16 predictor features (7 numerical, 9 categorical, and target), including at least 2 feature-target scatter plots with jitter and transparency (Age and Duration vs. Target).
  - Systematic IQR outlier audit table and boxplots covering all 7 numerical features with domain-motivated retention justification.
  - Evaluation of class imbalance (88.30% non-subscribers vs. 11.70% subscribers) and operational limitations of the call `duration` feature.
  - Stratified 80:20 train/test splitting and dense one-hot encoding for algorithm compatibility.
  - Feature engineering (`pdays_contacted` and `balance_negative` indicators).
  - Implementation and benchmarking of all **5 required Part-A classifiers**.
  - Metric reporting: Accuracy, Positive-Class Precision, Positive-Class Recall, Weighted F1-score, ROC-AUC, and individual formatted confusion matrix heatmaps.
  - Leak-free hyperparameter tuning for Decision Tree and K-Nearest Neighbors via `Pipeline` and `GridSearchCV`.
  - Consolidated comparison table sorted by **Weighted F1 descending**.
  - 5-fold cross-validation on the training split to verify statistical stability.

> [!NOTE]
> **Review 1 vs. Review 2 Scope Boundary**:
> In accordance with course guidelines, Review 1 is strictly restricted to the Full Regression Track and Part-A baseline classifiers. Part-B ensemble architectures (Random Forest Classifier, AdaBoost, Gradient Boosting Classifier, Bagging, MLP) and Clustering algorithms (K-Means, Agglomerative) will be implemented exclusively in Review 2.

---

## Datasets

### 1. Seoul Bike Sharing Dataset (Regression Track)
- **Path**: `data/regression/SeoulBikeData.csv`
- **Observations**: 8,760 hourly records spanning a complete annual cycle (Dec 1, 2017 – Nov 30, 2018).
- **Features**: 14 attributes including meteorological observations (Temperature, Humidity, Wind Speed, Solar Radiation, Rainfall, Snowfall, Dew Point Temperature, Visibility) and operational calendar indicators (Hour, Seasons, Holiday, Functioning Day).
- **Target**: `Rented Bike Count` (continuous hourly rental demand, range: 0 – 3,556).
- **Integrity**: Exactly 0 missing entries, 0 duplicate records, confirmed 365 continuous calendar days with 24 hourly records each.

### 2. Bank Marketing Dataset (Classification Track Part A)
- **Path**: `data/classification/bank.csv`
- **Observations**: 45,211 client records from direct telemarketing campaigns of a Portuguese banking institution.
- **Features**: 16 attributes spanning client demographics (age, job, marital, education, default, balance, housing, loan), current contact details (contact, day, month, duration), and previous campaign outcomes (campaign, pdays, previous, poutcome).
- **Target**: `y` (binary classification: `'yes'` = subscribed to term deposit [5,289 / 11.70%], `'no'` = did not subscribe [39,922 / 88.30%]).
- **Integrity**: Genuine comma-delimited text CSV, 0 missing values, 0 duplicate records.

---

## Repository Structure
```text
Machine-Learning-Capstone-Project/
├── .gitignore                      # Excludes checkpoints, cache, and virtual environments
├── README.md                       # Master project documentation
├── requirements.txt                # Pinned runtime dependencies
├── data/
│   ├── classification/
│   │   └── bank.csv                # Genuine Bank Marketing CSV dataset (45,211 rows)
│   └── regression/
│       └── SeoulBikeData.csv       # Seoul Bike Sharing CSV dataset (8,760 rows)
└── notebooks/
    ├── classification.ipynb        # Fully executed Classification Part-A notebook (clean kernel)
    └── regression.ipynb            # Fully executed Regression track notebook (clean kernel)
```

---

## Regression Track: Summary of Results

### Master Baseline Comparison Table (Ranked by Test $R^2$ Descending)
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

### Polynomial Degree Comparison
| Polynomial Degree | Feature Count | Test $R^2$ | Test RMSE | Test MAE | Computational Complexity |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Degree 1** | 21 | 0.6191 | 397.72 | 300.34 | $O(d)$ — Linear, fast baseline |
| **Degree 2** | 252 | 0.8450 | 253.71 | 181.14 | $O(d^2)$ — Captures interaction/quadratic effects |
| **Degree 3 (Omitted)** | 2,023 | *N/A* | *N/A* | *N/A* | $O(d^3)$ — Severe collinearity & memory risk |

### Hyperparameter Tuning Results
Hyperparameter tuning was executed via `GridSearchCV` inside leak-free `Pipeline` architectures fitting strictly on `X_train_fe`:
- **Support Vector Regressor (SVR)**: Tuning $C=1000, \epsilon=20.0$ lifted test $R^2$ from **0.2794 to 0.9046** ($\Delta R^2 = +0.6253$, $\text{RMSE}$ dropped from 547.03 to 198.99), resolving the default penalty under-scaling.
- **Random Forest Regressor**: Tuning `n_estimators=200`, `max_depth=None`, `min_samples_split=2` achieved test $R^2 = 0.9434$ ($\text{RMSE} = 153.36$, $\text{MAE} = 93.26$).

### 5-Fold Cross-Validation on Training Split
To prevent leakage, each cross-validation fold fits the `preprocessor` independently on the training slice:
- **Table 14A (Baseline Top Two Candidates)**:
  - Random Forest Regressor (Baseline): Mean CV $R^2 = \mathbf{0.9315 \pm 0.0086}$ (Held-out Test $R^2 = 0.9429$)
  - Gradient Boosting Regressor (Baseline): Mean CV $R^2 = \mathbf{0.8861 \pm 0.0051}$ (Held-out Test $R^2 = 0.8894$)
- **Table 14B (Post-Tuning Top Two Candidates)**:
  - Random Forest Regressor (Tuned): Mean CV $R^2 = \mathbf{0.9318 \pm 0.0087}$ (Held-out Test $R^2 = 0.9434$)
  - Support Vector Regressor (Tuned): Mean CV $R^2 = \mathbf{0.9012 \pm 0.0074}$ (Held-out Test $R^2 = 0.9046$)

---

## Classification Track (Part A): Summary of Results

### Master Preliminary Comparison Table (Ranked by Weighted F1 Descending)
All five Part-A classifiers were trained and evaluated on the identical stratified split (training: 36,168 samples; test: 9,043 samples). Positive class is explicitly defined as `1` (`'yes'`):

| Rank | Model | Accuracy | Precision (`yes`) | Recall (`yes`) | Weighted F1 | ROC-AUC | Training Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Decision Tree Classifier** | 0.9003 | 0.6111 | 0.4055 | **0.8913** | 0.8589 | 0.18s |
| 2 | **Logistic Regression** | 0.9006 | 0.6402 | 0.3431 | **0.8870** | **0.9055** | 0.08s |
| 3 | **Support Vector Classifier (SVC)** | **0.9012** | **0.6560** | 0.3280 | **0.8864** | 0.8926 | 19.51s |
| 4 | **K-Nearest Neighbors** | 0.8952 | 0.5989 | 0.3147 | **0.8805** | 0.8492 | 0.00s |
| 5 | **Gaussian Naive Bayes** | 0.8547 | 0.4026 | **0.5000** | **0.8614** | 0.7998 | 0.04s |

### Baseline vs. Tuned Classification Performance
Hyperparameter tuning via leak-free `Pipeline` and `GridSearchCV` on `X_train_fe`:
- **Decision Tree**: Baseline (`max_depth=6`) achieved Weighted F1 = 0.8913 (ROC-AUC = 0.8589). Tuned (`max_depth=10, min_samples_leaf=10`) achieved Weighted F1 = 0.8881 (ROC-AUC = 0.8824).
- **K-Nearest Neighbors**: Baseline ($k=7$) achieved Weighted F1 = 0.8805 (ROC-AUC = 0.8492). Tuned ($k=5$) achieved Weighted F1 = 0.8837 (ROC-AUC = 0.8345).

### 5-Fold Cross-Validation (Training Split)
- **Decision Tree**: Mean 5-Fold Weighted F1 = $\mathbf{0.8905 \pm 0.0024}$ | Mean ROC-AUC = $\mathbf{0.8509 \pm 0.0047}$
- **Logistic Regression**: Mean 5-Fold Weighted F1 = $\mathbf{0.8879 \pm 0.0043}$ | Mean ROC-AUC = $\mathbf{0.9069 \pm 0.0033}$
- **K-Nearest Neighbors**: Mean 5-Fold Weighted F1 = $\mathbf{0.8831 \pm 0.0039}$ | Mean ROC-AUC = $\mathbf{0.8587 \pm 0.0079}$

### Key Diagnostic Takeaways
- **Class Imbalance Dynamics**: While overall accuracy exceeds 90%, positive-class recall ranges between 31% and 50%, reflecting the asymmetric cost of false negatives in telemarketing.
- **Model Divergence**:
  - **Logistic Regression** and **SVC** provide high precision (~64–66%) and top-tier ranking discrimination ($\text{ROC-AUC} \ge 0.89$).
  - **Gaussian Naive Bayes** delivers the highest subscriber detection rate ($\text{Recall} = 50.00\%$), capturing 529 out of 1,058 subscribers.
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
- `notebooks/regression.ipynb`
- `notebooks/classification.ipynb`

### Non-Interactive Headless Execution
To re-run and re-render both notebooks top-to-bottom from a clean kernel:
```bash
# Execute Regression track notebook in place
jupyter nbconvert --to notebook --execute --inplace notebooks/regression.ipynb

# Execute Classification Part-A notebook in place
jupyter nbconvert --to notebook --execute --inplace notebooks/classification.ipynb
```

---

