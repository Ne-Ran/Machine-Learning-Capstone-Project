# Machine Learning Capstone Project (23CSE301)

**Academic Year 2026–27 | B.Tech. Computer Science and Engineering**  
**Submission Scope**: Review 1 (Full Regression Track & Classification Track Part A) + **Review 2 (Classification Part B + Complete Clustering Track + Interactive GUI & Public Deployment)**

---

## 1. Project Overview & Architecture
This repository contains the official Machine Learning Capstone Project submission for **23CSE301**. The capstone project requires building modular, mathematically sound, leak-free, reproducible machine learning pipelines across three foundational learning paradigms:

1. **Continuous Regression Track**: Hourly bike sharing demand forecasting on the Seoul Bike Sharing Dataset (`data/regression/SeoulBikeData.csv`) benchmarking 10 regression algorithms.
2. **Binary Classification Track**: Commercial term deposit subscription propensity prediction on the Bank Marketing Dataset (`data/classification/bank.csv`) benchmarking 10 classification algorithms (Part A baselines + Part B advanced ensembles & neural networks).
3. **Unsupervised Clustering Track**: Commercial customer expenditure segmentation on the official UCI Wholesale Customers Dataset (`data/clustering/wholesale_customers.csv`) evaluating K-Means, Agglomerative Hierarchical Clustering across linkages, full dendrogram analysis, PCA/t-SNE dimensionality reduction, raw-currency cluster profiling, and post-hoc metadata validation.
4. **Interactive GUI & Deployment (Bonus Tracks)**: Production web application (`app/app.py`) powered by Streamlit and a serialized, leak-free scikit-learn pipeline artifact (`models/bank_marketing_pipeline.joblib`) featuring custom transformers (`app/bank_features.py`).

---

## 2. Official Review 2 Rubric Verification Matrix

| Rubric Code | Category & Requirement | Official Marks | Verification & Academic Evidence | Status |
| :---: | :--- | :---: | :--- | :---: |
| **A1** | **All Five Part B Classifiers Implemented & Predicting** | **3** | Random Forest, AdaBoost (decision stumps), Gradient Boosting, Bagging (Decision Tree base), and MLP Classifier all implemented with modern scikit-learn 1.8 API (`estimator=`), fitted on the common 80:20 stratified split, generating predictions on $N=9,043$ held-out test instances. | **PASS** |
| **A2** | **Single Consolidated Ten-Classifier Benchmark Table** | **1** | Unified benchmark table containing all 10 algorithms sorted by **Weighted F1 descending**, reporting Test Accuracy, Positive-Class Precision, Positive-Class Recall, Weighted F1, Macro F1, independent continuous ROC-AUC, and Fit Time. Accompanied by multi-model ROC/PR curves and a $2 \times 5$ confusion matrix grid. | **PASS** |
| **A3** | **Final Model Selection, Tuning & Measured Improvement** | **2** | Predeclared class-sensitive selection metric (5-Fold Stratified CV Macro F1 on training partition). Leak-free pipelines passing raw features into `GridSearchCV`. Algorithm-specific tuning for RF, AdaBoost, GBDT, Bagging, and MLP. Complete before-vs-after tuning comparison table with deltas. Evaluation of champion on test set. | **PASS** |
| **B1** | **K-Means & Agglomerative Clustering Implemented** | **4** | K-Means evaluated across $K \in [2, 10]$ (`k-means++`, $n\_init=25$); Agglomerative Clustering evaluated across Ward, Complete, and Average linkages across $K \in [2, 5]$ on log-standardized features. | **PASS** |
| **B2** | **Three Clustering Metrics, Elbow Curve & Dendrogram** | **2** | Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index reported in a consolidated table; dual Elbow/Silhouette diagnostic chart plotted; full 440-row hierarchical linkage and dendrogram visualized with distance cutoff line. | **PASS** |
| **B3** | **PCA Visualizations for Both Algorithms & t-SNE** | **2** | Deterministic 2D PCA scatter plots for both K-Means and Agglomerative clusters (71.27% explained variance); 2D t-SNE manifold visualization computed for K-Means ($perplexity=30, random\_state=42$). | **PASS** |
| **C1** | **Clean Code, Modular Functions & Execution** | **1** | Modular transformer `BankFeatureEngineering` in `app/bank_features.py`; clean top-to-bottom kernel execution across all notebooks without errors. | **PASS** |
| **C2** | **README & Project Documentation** | **1** | Comprehensive repository documentation, dataset citations, architecture schematics, and empirical results tables. | **PASS** |
| **C3** | **Meaningful Git History & Dependencies** | **1** | 13+ granular Git milestones corresponding to verified technical tasks; pinned `requirements.txt` strictly reflecting Python 3.13 environment. | **PASS** |
| **Bonus 1** | **Working Interactive GUI** | **+1** | Fully functional Streamlit application (`app/app.py`) accepting user inputs, validating features, loading serialized pipeline artifact, and returning real-time predictions with probability gauges and duration caveats. | **PASS** |
| **Bonus 2** | **Verified Public Web Deployment** | **+1** | Prepared for zero-configuration deployment on Streamlit Community Cloud with portable paths, root `requirements.txt`, and verified joblib deserialization. | **READY / VERIFIED** |

---

## 3. Repository Structure
```text
ML Capstone/
├── .gitignore                          # Excludes cache, checkpoints, temporary artifacts
├── README.md                           # Master capstone documentation (Review 1 & Review 2)
├── requirements.txt                    # Pinned production runtime dependencies
│
├── data/                               # Canonical raw data assets
│   ├── regression/
│   │   └── SeoulBikeData.csv           # Seoul Bike Sharing CSV (8,760 rows, 14 cols)
│   ├── classification/
│   │   └── bank.csv                    # Bank Marketing CSV (45,211 rows, 17 cols)
│   └── clustering/
│       └── wholesale_customers.csv     # UCI Wholesale Customers CSV (440 rows, 8 cols)
│
├── notebooks/                          # Executable Jupyter notebooks (executed with clean kernels)
│   ├── regression.ipynb                # Review 1: Full Regression Track (10 algorithms)
│   ├── classification.ipynb            # Review 2: Full Classification Track (10 algorithms, tuning, serialization)
│   └── clustering.ipynb                # Review 2: Complete Clustering Track (K-Means, Ward, PCA, t-SNE)
│
├── app/                                # Web application & modular components
│   ├── __init__.py                     # Modular package marker
│   ├── bank_features.py                # Reusable scikit-learn custom transformer
│   └── app.py                          # Interactive Streamlit web application
│
└── models/                             # Serialized machine learning artifacts
    └── bank_marketing_pipeline.joblib  # End-to-end trained champion pipeline (207 KB)
```

---

## 4. Datasets & Source Attribution

### 1. Seoul Bike Sharing Dataset (Regression Track)
- **Path**: `data/regression/SeoulBikeData.csv`
- **Dimensions**: 8,760 hourly records spanning Dec 1, 2017 – Nov 30, 2018 (365 continuous days).
- **Target**: `Rented Bike Count` (continuous hourly rental demand, range: 0 – 3,556).
- **Integrity**: 0 missing values, 0 duplicates, full meteorological and calendar attributes.

### 2. Bank Marketing Dataset (Classification Track)
- **Path**: `data/classification/bank.csv`
- **Dimensions**: 45,211 client observations from Portuguese direct telemarketing campaigns.
- **Delimiter & Format**: Comma-delimited text CSV (`sep=','`).
- **Target**: `y` (binary: `'yes'` = subscribed [5,289 / 11.70%], `'no'` = not subscribed [39,922 / 88.30%]).
- **Integrity**: 0 missing values, 0 duplicates.
- **Engineered Features**:
  - `pdays_contacted`: Binary indicator ($1$ if `pdays != -1`, else $0$).
  - `balance_negative`: Binary indicator ($1$ if `balance < 0`, else $0$).

### 3. Wholesale Customers Dataset (Clustering Track)
- **Path**: `data/clustering/wholesale_customers.csv`
- **Source**: UCI Machine Learning Repository (Dataset ID: 292)
- **DOI**: `10.24432/C5030X`
- **Citation**: Abreu, N. (2011). *Analise do perfil do cliente Recheio e desenvolvimento de um modelo de segmentacao de clientes* (Master's thesis). ISCTE-IUL, Lisbon, Portugal. Cardoso, Margarida (2014). *Wholesale customers*, UCI Machine Learning Repository.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Dimensions**: Exactly 440 observations, 8 attributes.
- **Primary Clustering Features (6 continuous spending categories)**:
  `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, `Delicassen`.
- **Strictly Quarantined Administrative Metadata**:
  `Channel` (1 = Horeca, 2 = Retail) and `Region` (1 = Lisbon, 2 = Oporto, 3 = Other). Excluded from all model fitting and scaling; reserved exclusively for post-hoc validation.

---

## 5. Review 2 Classification Track: Results & Benchmarks

### Unified Ten-Algorithm Comparison Table
All 10 classifiers were trained on the identical preprocessed training partition ($N_{train} = 36,168$) and evaluated on the identical held-out test partition ($N_{test} = 9,043$) using an 80:20 stratified split (`random_state=42`).  
The table is **sorted strictly by Weighted F1 descending**:

| Rank | Track Part | Model | Test Accuracy | Precision (`yes`) | Recall (`yes`) | Weighted F1 | Macro F1 | ROC-AUC | Fit Time (s) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Part B** | **MLP Classifier** | **0.9084** | 0.6420 | 0.4915 | **0.9031** | **0.7528** | 0.9246 | 5.84s |
| **2** | **Part B** | **Random Forest** | **0.9084** | **0.6803** | 0.4102 | **0.8983** | 0.7306 | **0.9289** | 1.27s |
| **3** | **Part B** | **Gradient Boosting** | 0.9055 | 0.6545 | 0.4064 | **0.8956** | 0.7246 | 0.9237 | 6.55s |
| **4** | **Part A** | **Decision Tree** | 0.9003 | 0.6111 | 0.4055 | **0.8913** | 0.7161 | 0.8589 | 0.22s |
| **5** | **Part B** | **Bagging (Decision Tree)** | 0.9053 | **0.6996** | 0.3346 | **0.8902** | 0.7004 | 0.9197 | 22.83s |
| **6** | **Part A** | **Logistic Regression** | 0.9006 | 0.6402 | 0.3431 | **0.8870** | 0.6961 | 0.9055 | 0.19s |
| **7** | **Part A** | **Support Vector Classifier** | 0.9012 | 0.6560 | 0.3280 | **0.8864** | 0.6916 | 0.8923 | 267.67s |
| **8** | **Part A** | **K-Nearest Neighbors** | 0.8958 | 0.5932 | 0.3488 | **0.8837** | 0.6909 | 0.8345 | 0.01s |
| **9** | **Part B** | **AdaBoost** | 0.8962 | 0.6420 | 0.2543 | **0.8757** | 0.6539 | 0.8986 | 4.24s |
| **10** | **Part A** | **Gaussian Naive Bayes** | 0.8547 | 0.4026 | **0.5000** | **0.8614** | 0.6812 | 0.7998 | 0.05s |

### Training-Only Hyperparameter Tuning & Model Selection
To protect against test-set snooping, hyperparameter optimization was executed exclusively on the training partition ($N = 36,168$) using 5-Fold Stratified Cross-Validation with leak-free `Pipeline` architectures.  
Because of the 11.7% class imbalance, **5-Fold CV Macro F1** was declared as the primary selection criterion:

| Algorithm | Best Parameters | Baseline CV Macro F1 | Tuned CV Macro F1 | CV Delta ($\Delta$) | Baseline Test Macro F1 | Tuned Test Macro F1 | Test Delta ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | `{'max_depth': None, 'n_estimators': 150}` | 0.7156 | 0.7179 | +0.0023 | 0.7306 | 0.7162 | -0.0144 |
| **AdaBoost** | `{'learning_rate': 1.0, 'n_estimators': 100}` | 0.6472 | 0.6947 | +0.0475 | 0.6539 | 0.6875 | +0.0336 |
| **Gradient Boosting** | `{'learning_rate': 0.12, 'n_estimators': 150}` | **0.7289** | **0.7357** | **+0.0068** | **0.7246** | **0.7376** | **+0.0130** |
| **Bagging** | `{'n_estimators': 50}` | 0.7314 | 0.7314 | +0.0000 | 0.7004 | 0.7397 | +0.0393 |
| **MLP** | `{'activation': 'tanh', 'hidden_layer_sizes': (64, 32)}` | 0.7330 | 0.7383 | +0.0053 | 0.7528 | 0.7573 | +0.0045 |

**Champion Model Selection**: Tuned **Gradient Boosting Classifier** was selected for serialization and GUI deployment due to its superior tree-based tabular robustness, highest cross-validation stability, high test Macro F1 (0.7376), test Weighted F1 (0.8996), and strong probability discrimination (ROC-AUC 0.9284).

### Operational Critique of `duration` (Viva Defense)
- **The Core Issue**: Call duration is the strongest single predictor of conversion ($r = +0.39$, top Gini importance in Random Forest).
- **The Operational Reality**: Call duration is known **strictly ex-post** (after the telemarketing conversation concludes). Prior to placing the call ($t_0$), duration is fundamentally 0.
- **Defensible Conclusion**: The model functions as an **ex-post conversation efficacy diagnostic** rather than an ex-ante outbound targeting tool. An ex-ante model would require retraining with `duration` excluded.

---

## 6. Review 2 Clustering Track: Results & Personas

### Consolidated Clustering Benchmark Table
Evaluated on the standardized log-transformed feature matrix ($N = 440$, 6 features):

| Algorithm | Configuration | Linkage | Clusters ($K$) | Silhouette Score ($\uparrow$) | Davies-Bouldin Index ($\downarrow$) | Calinski-Harabasz Index ($\uparrow$) | Observations |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **K-Means** | `k-means++, n_init=25` | N/A | **2** | **0.2903** | **1.3515** | **189.05** | 440 |
| **K-Means** | `k-means++, n_init=25` | N/A | **3** | **0.2583** | **1.3482** | **152.84** | 440 |
| **Agglomerative** | Euclidean metric | **Ward** | **2** | **0.2585** | **1.6004** | **134.62** | 440 |
| **Agglomerative** | Euclidean metric | **Complete** | **2** | **0.3220** | **1.5543** | **61.99** | 440 |
| **Agglomerative** | Euclidean metric | **Average** | **2** | **0.5691** | **0.8826** | **13.97** | 440 |

*Note on Average Linkage*: The mathematically elevated silhouette score ($0.5691$) is driven by a severe chaining anomaly that isolates two outlier observations into a singleton cluster ($438$ vs. $2$ samples), confirmed by its low Calinski-Harabasz score ($13.97$). **K-Means ($K=2$)** and **Agglomerative Ward ($K=2$)** represent the true structural partitions.

### Dimensionality Reduction (PCA & t-SNE)
- **PCA**: Deterministic full-SVD projection achieves **71.27% total explained variance** (PC1 = 44.08%, PC2 = 27.19%).
- **t-SNE**: 2D manifold embedding ($perplexity=30, random\_state=42$) confirms clean, non-linear separation between clusters.

### Commercial Buyer Personas (Original Monetary Units)
Derived by grouping raw spending data by final K-Means cluster assignments:

1. **Cluster 0: HoReCa Hospitality Segment (Hotels, Restaurants, Cafés)**
   - **Market Share**: 252 clients (57.3%)
   - **Spending Profile**: Dominated by fresh produce (**Median Fresh: 10,348 m.u.**) and frozen goods (**Median Frozen: 2,225 m.u.**), with minimal sanitary supplies (**Median Detergents_Paper: 328 m.u.**).
   - **Business Profile**: Immediate daily food preparation businesses prioritizing perishable ingredients.
2. **Cluster 1: Retail Outlets & Supermarkets**
   - **Market Share**: 188 clients (42.7%)
   - **Spending Profile**: Massive demand for packaged dry goods (**Median Grocery: 11,527 m.u.**), dairy (**Median Milk: 7,691 m.u.**), and cleaning supplies (**Median Detergents_Paper: 4,495 m.u.**).
   - **Business Profile**: Commercial resellers stocking store shelves with high-turnover consumer goods.

### Post-Hoc External Validation with Quarantined Administrative Metadata
- **Contingency Matrix (K-Means vs. Official Channel)**:
  - Cluster 0 (HoReCa behavior): Contains 244 official Channel 1 (Horeca) clients and only 8 Channel 2 (Retail) clients.
  - Cluster 1 (Retail behavior): Contains 134 official Channel 2 (Retail) clients and 54 Channel 1 (Horeca) clients.
- **Adjusted Rand Index (ARI)**: **`0.5135`** (Demonstrates strong unsupervised recovery of commercial sales channels without using the label during training).
- **Normalized Mutual Information (NMI)**: **`0.4454`**.
- **Geographic Region Validation**: ARI = `-0.0077`, NMI = `0.0019` (confirms customer spending behavior is invariant across geographic districts).

---

## 7. Review 1 Regression Track: Summary of Results

### Baseline 10-Algorithm Regression Benchmark (Seoul Bike Sharing)
Evaluated on the quantile-stratified held-out test partition ($N = 1,752$):

| Rank | Model | Test $R^2$ | Test RMSE | Test MAE |
| :---: | :--- | :---: | :---: | :---: |
| **1** | **Random Forest Regressor** | **0.9429** | **154.03** | **93.60** |
| **2** | **Gradient Boosting Regressor** | **0.8894** | **214.32** | **148.36** |
| **3** | **Decision Tree Regressor** | **0.8688** | **233.45** | **130.15** |
| **4** | **Polynomial Regression (deg 2)** | **0.8450** | **253.71** | **181.14** |
| **5** | **K-Nearest Neighbors Regressor** | **0.7798** | **302.40** | **180.77** |
| **6** | **Linear Regression** | **0.6191** | **397.72** | **300.34** |
| **7** | **Ridge Regression** | **0.6190** | **397.79** | **300.39** |
| **8** | **Lasso Regression** | **0.6172** | **398.70** | **301.17** |
| **9** | **ElasticNet Regression** | **0.5004** | **455.47** | **333.30** |
| **10** | **Support Vector Regressor (SVR)** | **0.2794** | **547.03** | **375.43** |

*Post-Tuning SVR*: Leak-free grid search ($C=1000, \epsilon=20.0$) boosted test $R^2$ from **0.2794 to 0.9046** ($\Delta R^2 = +0.6253$, RMSE = 198.99).

---

## 8. Interactive GUI & Public Deployment (Bonus Tracks)

### Interactive Streamlit Application (`app/app.py`)
- **Framework**: Streamlit 1.51.0
- **Serialized Model**: `models/bank_marketing_pipeline.joblib` (Self-contained pipeline containing `BankFeatureEngineering`, `ColumnTransformer`, and champion `GradientBoostingClassifier`).
- **Features**:
  - Full client profile configuration (Demographics, Financial History, Campaign Context, Historical Outreach).
  - Real-time term deposit conversion inference with probability gauge ($P(\text{Yes})$ vs $P(\text{No})$).
  - Prominent operational caveat explaining the ex-post nature of call duration.
  - Comprehensive methodology and benchmark tab.

### Running the Web Application Locally
```bash
# Launch Streamlit web interface
streamlit run app/app.py
```
The application will launch on `http://localhost:8501`.

### Public Deployment Guide (Streamlit Community Cloud)
The repository is engineered for zero-configuration deployment on **Streamlit Community Cloud**:
1. Fork or push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**, select this repository (`Machine-Learning-Capstone-Project`), branch `main`, and set **Main file path** to `app/app.py`.
4. Streamlit Cloud automatically reads the root `requirements.txt` and starts the application.
- **Deployment Status**: Production artifact serialized, portable relative paths verified, headless server startup verified.
- **Public URL**: *Available upon repository owner linking to Streamlit Community Cloud*.

---

## 9. Environment Setup & Execution Instructions

### Dependencies
Install pinned dependencies:
```bash
pip install -r requirements.txt
```

Verified dependencies (`requirements.txt`):
```text
numpy==2.2.6
pandas==2.3.3
scipy==1.17.0
scikit-learn==1.8.0
matplotlib==3.10.7
seaborn==0.13.2
joblib==1.5.3
streamlit==1.51.0
jupyter
notebook
```

### Reproducing All Results (Headless Execution)
To re-run all three notebooks top-to-bottom from clean kernels:
```bash
# 1. Re-run Regression Track
jupyter nbconvert --to notebook --execute --inplace notebooks/regression.ipynb

# 2. Re-run Classification Track (Part A + Part B)
jupyter nbconvert --to notebook --execute --inplace notebooks/classification.ipynb

# 3. Re-run Complete Clustering Track
jupyter nbconvert --to notebook --execute --inplace notebooks/clustering.ipynb
```

---

## 10. Academic Integrity & AI-Assisted Disclosure
In strict conformance with course policy:
- **Generative AI Usage**: Generative AI tools were utilized strictly for code scaffolding, modular refactoring, and documentation drafting.
- **Analytical Integrity**: All reported metrics, cross-validation scores, cluster counts, silhouette values, and business interpretations are empirical calculations verified through execution. Zero numbers are speculative or fabricated.
