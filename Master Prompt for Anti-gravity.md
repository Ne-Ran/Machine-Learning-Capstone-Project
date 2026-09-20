 # MASTER PROMPT — ML CAPSTONE REVIEW 1



You are the primary ML engineer responsible for completing our \*\*23CSE301 Machine Learning Capstone Project\*\*.



Your job is to build a \*\*fully working, academically defensible, reproducible Review 1 submission\*\* that maximizes our score against the official capstone guidelines and rubric.



Do NOT treat this as a generic ML tutorial.



Treat the official project guideline PDF available in the project/context as the \*\*source of truth\*\* for requirements.

all guidelines are available in D:\\ML Capstone\\23CSE301\_ML\_26\_27\_Capstone\_Guidelines (1).pdf



We are currently preparing ONLY for:



\- \*\*Review 1\*\*

\- Full Regression Track

\- Classification Track — Part A



Do NOT implement Review 2 algorithms yet unless something must be structurally prepared for future integration.



\---



\# 1. CURRENT PROJECT STATE



The repository already exists.



Current structure:





ML Capstone/

│

├── Data/

│   ├── regression/

│   │   └── SeoulBikeData.csv

│   │

│   └── classification/

│       └── bank.csv

│

├── Models/

│

├── Notebooks/

│

├── ReadME.md

│

└── Requirements.txt



Important:



The datasets already exist.

Do NOT download replacement datasets.

Do NOT rename the datasets.

Do NOT move the datasets.

Do NOT create duplicate datasets.

Do NOT create unnecessary README files.

We want one main README at the repository root.

Do NOT create README files inside Data/, Models/, or Notebooks/.

Keep the repository structure clean and lecturer-friendly.



Use the exact existing folders unless there is a genuine technical reason to change something.



3\. IMPORTANT REQUIREMENT



Build the code and analysis framework.

However, the final written interpretations must be based on the actual outputs produced by the notebook.

Never invent observations.

Never fabricate metric values.

Never claim a model improved unless the measured result proves it.

Never hard-code results.

Never manually type fake results into tables.

All numerical results must be generated dynamically from the actual dataset and model outputs.

Markdown interpretations should be generated only after examining the actual results.



The analysis must be original and specific to our datasets.



Do NOT produce generic textbook observations that are unsupported by the plots or metrics.



4\. FILES TO CREATE / MODIFY



Create or modify ONLY the following main files:



Notebooks/

├── regression.ipynb

└── classification.ipynb



Update:



ReadME.md

Requirements.txt



Do NOT create additional README files.



Do NOT create unnecessary Python files unless absolutely necessary.



Do NOT create unnecessary notebooks.



Do NOT create model files unless saving a model is genuinely useful.



For Review 1, the primary deliverables are the notebooks.



5\. NOTEBOOK DESIGN PRINCIPLE



Both notebooks must be:



Clean

Professional

Easy for a lecturer to inspect

Easy to present

Easy for students to explain during viva

Fully executable from top to bottom

Logically ordered

Clearly separated into sections using Markdown

Free of unnecessary code

Free of giant unexplained code blocks



Do NOT write one giant cell containing the entire pipeline.



Use modular cells and reusable functions where appropriate.



Every major stage should have a Markdown heading and a short explanation.



6\. REPRODUCIBILITY



Use:



RANDOM\_STATE = 42



wherever applicable.



Use the same random state throughout.



Ensure results are reproducible.



Do NOT randomly change train/test splits between models.



7\. DATASET 1 — REGRESSION



Dataset:



Data/regression/SeoulBikeData.csv



Target:



Rented Bike Count



Before implementing anything:



Load the CSV.

Inspect the actual columns.

Verify the target exists.

Inspect datatypes.

Inspect missing values.

Inspect duplicates.

Inspect invalid values.

Inspect categorical columns.

Inspect numerical columns.

Inspect date/time columns.

Inspect the target distribution.



Do NOT blindly assume the dataset schema.



If the actual dataset differs from expectations, adapt to the actual CSV while preserving the project requirements.



8\. DATASET 2 — CLASSIFICATION



Dataset:



Data/classification/bank.csv



Before implementing anything:



Load the CSV.

Inspect actual columns.

Identify the target column from the actual dataset.

Verify the target is appropriate for classification.

Inspect datatypes.

Inspect missing values.

Inspect duplicates.

Inspect class distribution.

Inspect categorical columns.

Inspect numerical columns.



The expected Bank Marketing target is usually y, but DO NOT rely on this assumption without checking the actual file.



9\. REGRESSION NOTEBOOK — REQUIRED STRUCTURE



Create:



Notebooks/regression.ipynb



Use approximately this section structure:



1\. Project Overview

2\. Problem Definition

3\. Dataset Loading

4\. Dataset Audit

5\. Exploratory Data Analysis

6\. Data Cleaning

7\. Train/Test Split

8\. Feature Engineering

9\. Encoding and Preprocessing

10\. Model Training

11\. Model Evaluation

12\. Hyperparameter Tuning

13\. Cross-Validation

14\. Final Model Analysis

15\. Required Visualizations

16\. Final Comparison

17\. Conclusions



10\. REGRESSION — DATASET AUDIT



Show clearly:



Number of rows

Number of columns

Dataset shape

Column names

Datatypes

Missing-value counts

Duplicate count

Numerical summary statistics

Target summary statistics

Target distribution



Use readable Pandas tables.



Do not dump meaningless output unnecessarily.



Create a compact dataset audit table where practical.



11\. REGRESSION — EDA



The official rubric requires:



A. Distribution plots for each feature



For numerical features:



histograms / KDE where appropriate

readable titles

labelled axes



For categorical features:



count plots / frequency plots



Do NOT make plots unreadable because there are too many categories.



Handle high-cardinality categorical variables sensibly.



B. Correlation heatmap



Use numerical variables.



Clearly label the heatmap.



C. Target distribution



Show the distribution of:



Rented Bike Count

D. At least TWO feature-target scatter plots



Choose meaningful numerical features after inspecting the data.



Examples may include:



Temperature vs target

Humidity vs target

Solar Radiation vs target

Hour vs target



But select them based on actual data.



Do not blindly use example variables if the dataset differs.



12\. REGRESSION — EDA COMMENTARY



This is extremely important.



For every major visualization, add a Markdown cell immediately after it containing:



Observation



What the plot actually shows.



ML implication



What this means for preprocessing/model selection.



Example structure:



Observation:

...



ML implication:

...



Do NOT write generic statements.



The commentary must be based on the actual graph.



Examples of valid reasoning:



approximate linear/nonlinear relationship

skewness

possible outliers

strong/weak correlation

class/category imbalance where relevant

interaction patterns

possible nonlinear demand patterns

features that may need transformation/scaling



Do NOT claim causation from correlation.



13\. REGRESSION — DATA CLEANING



Perform and document:



Missing values



Check all columns.



If missing values exist:



choose a justified strategy

explain why

implement correctly



If there are no missing values:



explicitly report that the audit found none.



Do NOT artificially introduce missing values.



Duplicates



Check duplicates.



If duplicates exist, determine whether removal is justified.



Document the action.



Outliers



Check numerical features and target for potential outliers.



Use appropriate methods such as:



IQR analysis

boxplots

distribution analysis



Do NOT automatically delete every extreme value.



Distinguish between:



legitimate extreme observations

obvious data errors



If retaining legitimate outliers, explain why.



14\. REGRESSION — TRAIN/TEST SPLIT



Split before fitting preprocessing transformations.



Use approximately:



80% training

20% testing



with:



random\_state=42



The test set must remain untouched until final evaluation.



All ten regression models MUST use the exact same held-out test set.



Never create separate test sets for individual models.



15\. REGRESSION — FEATURE ENGINEERING



Create at least one meaningful engineered feature.



Prefer useful features derived from the existing dataset.



Because Seoul Bike contains temporal information, investigate useful date/time features such as:



Year

Month

Day

Day of week

Weekend indicator

Rush-hour indicator



Also consider cyclical representation where appropriate:



sin(hour)

cos(hour)



However:



DO NOT blindly create every possible feature.



Only create features with a clear justification.



For each engineered feature include:



Feature name

How it was created

Why it may help prediction

Potential limitations



Avoid data leakage.



Never create a feature using information that would not be available when making the prediction.



16\. REGRESSION — PREPROCESSING



Create a proper preprocessing pipeline.



Categorical variables:



Use appropriate encoding.

One-hot encoding is acceptable where appropriate.



Numerical variables:



Handle scaling where required.



CRITICAL:



All scalers and encoders MUST be fitted only on training data.



Never fit them on the full dataset before splitting.



Prefer sklearn Pipeline / ColumnTransformer so leakage is structurally prevented.



17\. REGRESSION — MODEL 1

Linear Regression



Implement as the baseline.



Generate:



predictions

R²

RMSE

MAE



Also interpret coefficients where meaningful.



If preprocessing causes coefficient interpretation to be non-trivial, explain that clearly.



18\. REGRESSION — MODEL 2

Ridge Regression



Implement Ridge.



Tune:



alpha



Use GridSearchCV or RandomizedSearchCV.



Record:



baseline result

tuned result

best parameters

improvement

19\. REGRESSION — MODEL 3

Lasso Regression



Implement Lasso.



Discuss:



L1 regularization

coefficient shrinkage

potential feature sparsity



Ensure convergence is handled properly.



20\. REGRESSION — MODEL 4

ElasticNet Regression



Implement ElasticNet.



Tune:



alpha

l1\_ratio



Record best parameters and results.



21\. REGRESSION — MODEL 5

Polynomial Regression



Use:



PolynomialFeatures



followed by:



Linear Regression



Compare multiple polynomial degrees.



For example:



degree = 1

degree = 2

degree = 3



Do NOT use unnecessarily large polynomial degrees because this can explode the feature space and computation.



Explain:



underfitting

nonlinear fit

overfitting risk



Select the degree based on validation performance, not aesthetics.



22\. REGRESSION — MODEL 6

Decision Tree Regressor



Train a Decision Tree Regressor.



Tune:



max\_depth



Show:



R²

RMSE

MAE



Also generate feature importance.



23\. REGRESSION — MODEL 7

Random Forest Regressor



Train Random Forest.



Tune at least:



n\_estimators



Preferably also consider:



max\_depth

min\_samples\_split

min\_samples\_leaf



Do not create an unnecessarily massive grid.



Record:



best parameters

test performance

comparison against untuned baseline

24\. REGRESSION — MODEL 8

Gradient Boosting Regressor



Use sklearn GradientBoostingRegressor unless another option is specifically needed.



Tune:



learning\_rate



Also consider:



n\_estimators

max\_depth



where appropriate.



Record:



best parameters

performance

improvement

25\. REGRESSION — MODEL 9

Support Vector Regressor



Use SVR.



IMPORTANT:



Features MUST be scaled before SVR.



Tune:



C

kernel

gamma

epsilon



Use a manageable search space.



Do NOT create a computationally insane GridSearch.



26\. REGRESSION — MODEL 10

K-Nearest Neighbors Regressor



Use KNeighborsRegressor.



Tune:



n\_neighbors



Discuss why scaling matters.



Use a reasonable search range.



27\. REGRESSION — REQUIRED EVALUATION METRICS



For ALL 10 regression algorithms calculate:



R²

RMSE

MAE



Also calculate:



5-fold cross-validated R²



for the TWO best-performing models as required by the project specification.



Use the same evaluation methodology.



Create ONE consolidated comparison table.



Required table structure:



Model

R2

RMSE

MAE



Sort/rank models by:



R² descending



Do NOT create ten separate tables.



One master comparison table must exist.



28\. REGRESSION — HYPERPARAMETER TUNING



At least TWO regression models MUST use:



GridSearchCV

or

RandomizedSearchCV



Prefer tuning:



Random Forest Regressor

SVR



You may tune additional models such as:



Ridge

ElasticNet

Gradient Boosting

Decision Tree

KNN



For every tuned model show:



Baseline metric

Best parameters

Tuned metric

Metric improvement



Do not claim improvement without calculating it.



29\. REGRESSION — TOP TWO CROSS-VALIDATION



After evaluating all models:



Identify the top two models based on the test comparison.

Run 5-fold cross-validation appropriately on the training data.

Report:

mean CV R²

standard deviation of CV R²

Compare this with the held-out test R².



Do not leak the test set into cross-validation.



Explain the difference between:



test performance

cross-validation performance

variability

30\. REGRESSION — REQUIRED VISUALIZATIONS



For the final/best model, generate:



A. Predicted vs Actual plot



Include:



actual values

predicted values

reference diagonal y=x where appropriate

title

x-axis

y-axis

legend where applicable

B. Residual plot



Calculate residuals:



actual - predicted



Plot appropriately.



Discuss whether residual patterns appear random or systematic.



C. Tree-based feature importance



Use at least one tree-based model.



Show feature importance clearly.



If preprocessing expands categorical features through one-hot encoding, correctly map transformed feature names to importance values.



31\. REGRESSION — FINAL CONCLUSION



The notebook must end with a concise evidence-based conclusion.



Include:



model comparison

important feature findings

preprocessing impact

tuning impact

CV findings

limitations

what could be improved later



Do NOT write:



"Random Forest is the best because it is powerful."



Instead, base the discussion strictly on measured results.



Do NOT present unsupported causal claims.



32\. CLASSIFICATION NOTEBOOK — REQUIRED STRUCTURE



Create:



Notebooks/classification.ipynb



Use:



1\. Project Overview

2\. Problem Definition

3\. Dataset Loading

4\. Dataset Audit

5\. Exploratory Data Analysis

6\. Data Cleaning

7\. Train/Test Split

8\. Feature Engineering

9\. Encoding and Preprocessing

10\. Part A Model Training

11\. Model Evaluation

12\. Hyperparameter Tuning where appropriate

13\. Comparative Evaluation

14\. Required Visualizations

15\. Conclusions

33\. CLASSIFICATION — DATASET AUDIT



Report:



shape

columns

datatypes

missing values

duplicates

class distribution

numerical summary statistics

categorical columns

target distribution



Create a clean audit table.



34\. CLASSIFICATION — EDA



Include:



Feature distributions



For relevant numerical and categorical features.



Target distribution



Show class counts and proportions.



Correlation heatmap



Use numerical variables.



Feature-target relationship visualizations



At least two meaningful visualizations.



Examples:



numerical feature by target class

categorical feature by target class



Choose based on actual data.



Avoid visual clutter.



35\. CLASSIFICATION — EDA COMMENTARY



After major visualizations provide:



Observation:

...



ML implication:

...



Base the text on the actual plots.



Do not fabricate insights.



Do not imply causality from correlations.



36\. CLASSIFICATION — CLEANING



Check:



missing values

duplicates

invalid values

suspicious categories

outliers where applicable

class imbalance



If missing values exist:



Choose and justify a strategy.



If the target classes are imbalanced:



Explicitly document the distribution.



Do NOT silently apply SMOTE or resampling unless there is a strong methodological reason and it is implemented correctly within training-only processing.



For Review 1, keep the pipeline simple and defensible.



37\. CLASSIFICATION — SPLIT



Use approximately:



80:20 train/test



with:



random\_state=42



IMPORTANT:



Use:



stratify=y



for the classification train/test split.



The test set must remain untouched until final evaluation.



ALL five Part-A algorithms must use the SAME test set.



38\. CLASSIFICATION — ENCODING AND SCALING



Use:



ColumnTransformer

Pipeline



where appropriate.



Categorical variables:



OneHotEncoder or another justified encoding.



Numerical variables:



appropriate scaling for distance/margin-based algorithms.



Scaling is particularly important for:



KNN

SVC

Logistic Regression



Gaussian Naive Bayes and tree models have different scaling requirements; do not apply transformations blindly.



Prevent leakage.



Fit preprocessing only on the training data.



39\. CLASSIFICATION — FEATURE ENGINEERING



Create at least one meaningful engineered feature.



Only create it if it makes sense for the Bank Marketing dataset.



Possible candidates may be derived from combinations of existing fields, but:



inspect the actual dataset first

avoid arbitrary feature creation

avoid leakage

justify every engineered feature



Document:



Feature

Construction

Reason

Potential benefit

Limitations

40\. CLASSIFICATION — PART A MODEL 1

Logistic Regression



Train LogisticRegression.



Tune useful parameters where appropriate.



Generate:



predictions

predicted probabilities if available



Evaluate using:



Accuracy

Precision

Recall

Weighted F1

ROC-AUC

Confusion Matrix



Use an appropriate binary ROC-AUC calculation.



41\. CLASSIFICATION — PART A MODEL 2

K-Nearest Neighbors Classifier



Tune:



n\_neighbors



Optionally investigate:



weights

metric



Scaling MUST be handled appropriately.



Evaluate:



Accuracy

Precision

Recall

Weighted F1

ROC-AUC where probability/score is available

Confusion Matrix

42\. CLASSIFICATION — PART A MODEL 3

Gaussian Naive Bayes



Train:



GaussianNB



Discuss:



conditional independence assumption

implications of that assumption for the dataset



Evaluate:



Accuracy

Precision

Recall

Weighted F1

ROC-AUC

Confusion Matrix

43\. CLASSIFICATION — PART A MODEL 4

Decision Tree Classifier



Train DecisionTreeClassifier.



Tune:



max\_depth



Potentially also:



min\_samples\_split

min\_samples\_leaf



Generate:



predictions

probabilities

confusion matrix

feature importance



Evaluate:



Accuracy

Precision

Recall

Weighted F1

ROC-AUC

44\. CLASSIFICATION — PART A MODEL 5

Support Vector Classifier



Train SVC.



IMPORTANT:



Scale features before SVC.



Use probability estimates or decision scores appropriately for ROC-AUC.



Tune:



C

kernel

gamma



Use a manageable search space.



Evaluate:



Accuracy

Precision

Recall

Weighted F1

ROC-AUC

Confusion Matrix

45\. CLASSIFICATION — REQUIRED METRICS



For EVERY Part-A algorithm report:



Accuracy

Precision

Recall

Weighted F1-score

ROC-AUC

Confusion Matrix



The official Review 1 rubric specifically requires:



Accuracy

weighted F1

confusion matrix



But report the additional classification metrics too because the overall project specification requires them.



For binary classification:



Use binary ROC-AUC appropriately.

Verify that the positive class is clearly defined.



Do NOT accidentally calculate an invalid ROC-AUC.



46\. CLASSIFICATION — COMPARISON TABLE



Create ONE preliminary comparison table.



Recommended structure:



Model

Accuracy

Precision

Recall

Weighted F1

ROC-AUC



Sort by a clearly stated metric.



Do not create arbitrary "overall scores".



Do not create a custom ranking score.



The purpose is comparison, not gaming the rubric.



47\. CONFUSION MATRICES



Create a confusion matrix visualization for EACH Part-A classifier.



Each must have:



title

actual axis label

predicted axis label

clear class labels

readable values



Do not produce meaningless raw arrays only.



The visualizations must be presentation-ready.



48\. MODEL COMPARISON



After all five classifiers:



Discuss the actual observed differences.



Examples of questions to answer:



Which algorithms perform similarly?

Which are sensitive to scaling?

Does class imbalance affect interpretation?

Are accuracy and weighted F1 telling a similar story?

Is ROC-AUC consistent with the other metrics?

What trade-offs appear between model complexity and performance?



Do not simply say:



"Model X is best."



Instead:



Explain the evidence.



49\. TUNING



The official Review 1 rubric specifically requires hyperparameter tuning for at least two regression models.



Classification tuning is useful but should remain computationally sensible.



For classification, tune at least the more appropriate models such as:



KNN

Decision Tree

SVC

Logistic Regression



when practical.



Whenever tuning is performed, report:



Baseline

Best Parameters

Tuned Result

Change in Metric

50\. DATA LEAKAGE RULES — ABSOLUTE



This is critical.



Never:



scaler.fit(X)



before splitting.



Never:



encoder.fit(X)



on the full dataset.



Never compute training-dependent statistics using the test set.



Never use the test set to select hyperparameters.



Never include the target in feature preprocessing.



Never engineer a feature using future/target information.



Use Pipelines and ColumnTransformers whenever practical.



51\. COMPUTATIONAL EFFICIENCY



The notebooks must actually finish running on a normal student laptop.



Do NOT create ridiculous parameter grids.



Use reasonable search spaces.



Prefer:



GridSearchCV with small/medium grids



or:



RandomizedSearchCV



when the space is larger.



Do not run unnecessary repeated training.



Cache preprocessing where useful.



52\. MODEL RESULT STORAGE



Create reusable Python structures such as dictionaries/lists/DataFrames to collect results.



Do NOT manually create comparison tables.



Example conceptual structure:



results = \[]

results.append(...)

results\_df = pd.DataFrame(results)



Every reported result must originate from actual model execution.



53\. NOTEBOOK OUTPUT REQUIREMENT



After implementation:



RUN THE ENTIRE NOTEBOOK FROM TOP TO BOTTOM.



Check:



no NameError

no ImportError

no missing variable

no incorrect path

no plotting errors

no shape mismatch

no preprocessing leakage

no failed model

no missing output

all required tables visible

all required visualizations visible



The notebook must be submitted with outputs visible.



Do NOT leave important cells unexecuted.



54\. VISUALIZATION QUALITY



Every plot must have:



title

labelled axes

legend where applicable

readable font sizes

no clipped labels



Use:



plt.tight\_layout()



where necessary.



Prefer a colourblind-friendly palette where appropriate.



Do NOT use decorative plots that add no analytical value.



55\. CODE QUALITY



Code should be:



modular

readable

logically ordered

reasonably commented

reusable

PEP8-ish where practical



Use helper functions for repetitive evaluation logic.



For example:



evaluate\_regression\_model(...)

evaluate\_classification\_model(...)



But do not over-engineer the notebook.



The lecturer should be able to understand the code quickly.



56\. README — KEEP ONLY ONE



We already have:



ReadME.md



Use this as the ONLY README.



Do not create more READMEs.



Update it with:



\# ML Capstone Project



\## Project Overview



\## Review 1 Scope



\## Datasets



\## Repository Structure



\## Regression Track



\## Classification Track Part A



\## Algorithms Implemented



\## Evaluation Metrics



\## Environment Setup



\## How to Run



\## Reproducibility



\## AI Assistance Disclosure



Do NOT put huge analysis results in the README.



The notebooks are where the detailed analysis belongs.



The README should remain clean and lecturer-friendly.



57\. REQUIREMENTS.TXT



Update:



Requirements.txt



with the actual libraries required by the notebooks.



At minimum investigate whether we need:



numpy

pandas

matplotlib

seaborn

scikit-learn

jupyter

notebook

openpyxl



Only include packages actually required.



Pin versions where practical after checking the installed environment.



Do not invent version numbers.



58\. FUTURE REVIEW 2 PREPARATION



Do NOT implement Review 2 now.



However, keep the repository architecture compatible with eventually adding:



Classification Part B:



Random Forest Classifier

AdaBoost Classifier

Gradient Boosting Classifier

Bagging Classifier

MLP Classifier



Clustering:



K-Means

Agglomerative Hierarchical Clustering



Do not train these now unless required by dependency/testing.



We will handle Review 2 later.



59\. GITHUB COMMIT STRATEGY



first of all, here's my https://github.com/Ne-Ran/Machine-Learning-Capstone-Project.git

I will give you authorizaition to access my github



This is extremely important.



We are intentionally making meaningful commits throughout the project.



Do NOT make one giant final commit.



After each meaningful milestone:



Check git status.

Review changed files.

Stage only appropriate files.

Commit using a meaningful commit message.

Verify the commit exists.



Do NOT commit random temporary files.



Do NOT commit:



.ipynb\_checkpoints/

\_\_pycache\_\_/

.env

credentials

API keys

temporary files

large generated files that are not required



Use .gitignore where appropriate.



60\. EXPECTED MEANINGFUL COMMITS



Use approximately this progression.



Commit 1

Initialize ML capstone repository structure



If the initial structure is already committed, DO NOT create a duplicate commit.



Commit 2

Complete regression dataset audit and EDA

Commit 3

Add regression preprocessing and feature engineering

Commit 4

Implement all regression models

Commit 5

Add regression evaluation and comparison

Commit 6

Add regression hyperparameter tuning and cross-validation

Commit 7

Complete regression analysis and visualizations

Commit 8

Complete classification dataset audit and EDA

Commit 9

Add classification preprocessing and feature engineering

Commit 10

Implement classification Part A algorithms

Commit 11

Add classification evaluation and confusion matrices

Commit 12

Finalize Review 1 notebooks and documentation



Do NOT create meaningless commits just to increase commit count.



Every commit must correspond to an actual project milestone.



61\. GIT SAFETY



Before every commit:



git status

git diff



Inspect the changes.



Never commit secrets.



Never commit generated caches.



Never commit broken notebooks.



Never commit half-finished notebook states.



After each commit verify:



git log --oneline



At the end verify the complete repository structure.



62\. FINAL REVIEW-1 RUBRIC CHECK



Before declaring the task complete, create a final checklist INSIDE THE NOTEBOOK or as a final Markdown section.



Verify:



Section A — Dataset \& EDA



\[ ] Dataset shape shown

\[ ] Datatypes shown

\[ ] Missing values shown

\[ ] Target/class distribution shown

\[ ] Distribution plots for features

\[ ] Correlation heatmap

\[ ] Target distribution

\[ ] At least 2 scatter/feature-target visualizations

\[ ] Major plots have written observations



Section B — Preprocessing \& Feature Engineering



\[ ] Missing values handled or explicitly confirmed absent

\[ ] Duplicates checked

\[ ] Outliers checked

\[ ] Categorical encoding

\[ ] Scaling

\[ ] Train/test split

\[ ] Stratified split for classification

\[ ] No preprocessing leakage

\[ ] At least 1 engineered feature

\[ ] Engineering justified



Section C — Regression



\[ ] Linear Regression

\[ ] Ridge

\[ ] Lasso

\[ ] ElasticNet

\[ ] Polynomial Regression

\[ ] Decision Tree Regressor

\[ ] Random Forest Regressor

\[ ] Gradient Boosting Regressor

\[ ] SVR

\[ ] KNN Regressor



\[ ] Predictions generated

\[ ] R² for all

\[ ] RMSE for all

\[ ] MAE for all

\[ ] Single comparison table

\[ ] Ranked by R²

\[ ] Hyperparameter tuning on >= 2 models

\[ ] Best parameters reported

\[ ] Improvement reported

\[ ] 5-fold CV R² for top 2 models

\[ ] Residual plot

\[ ] Predicted-vs-actual plot

\[ ] Tree feature importance



Section D — Classification Part A



\[ ] Logistic Regression

\[ ] KNN

\[ ] Gaussian NB

\[ ] Decision Tree

\[ ] SVC



\[ ] Accuracy

\[ ] Precision

\[ ] Recall

\[ ] Weighted F1

\[ ] ROC-AUC

\[ ] Confusion matrix for every model

\[ ] Preliminary comparison table



Presentation/Viva readiness



\[ ] Every notebook runs top-to-bottom

\[ ] No errors

\[ ] Results are visible

\[ ] All plots are readable

\[ ] Every major decision is explainable

\[ ] No fabricated interpretation

\[ ] No data leakage

\[ ] Random state 42

\[ ] Git history contains meaningful commits

\[ ] Root README updated

\[ ] Requirements.txt updated



63\. FINAL QUALITY CONTROL



After everything is implemented:



Step 1



Run:



regression.ipynb



from first cell to last.



Step 2



Run:



classification.ipynb



from first cell to last.



Step 3



Inspect every output.



Step 4



Verify every rubric requirement.



Step 5



Check for leakage.



Step 6



Check every metric.



Step 7



Check every plot.



Step 8



Check the repository structure.



Step 9



Check Git status.



Step 10



Commit final changes.



64\. IMPORTANT — DO NOT DO THESE THINGS



Do NOT:



change the datasets without permission

download random alternative datasets

create multiple README files

create unnecessary folders

create unnecessary scripts

fabricate results

fabricate interpretations

fabricate dataset statistics

hard-code model scores

use the test set for tuning

fit preprocessing on full data

silently remove data

silently remove outliers

silently drop columns

claim correlation means causation

use different test sets for models

skip required algorithms

skip required metrics

hide failed models

hide poor results

produce giant unreadable plots

leave cells unexecuted

leave errors in notebooks

use random\_state values inconsistently

create meaningless Git commits

commit secrets

over-engineer the project

65\. MOST IMPORTANT PRIORITY



Optimize for:



CORRECTNESS

↓

RUBRIC COVERAGE

↓

REPRODUCIBILITY

↓

CLEAR EXPLANATION

↓

PRESENTATION QUALITY



Do NOT optimize merely for highest model score.



A slightly lower score with a clean, leakage-free, explainable pipeline is preferable to an artificially inflated score.



We are being evaluated on the complete ML workflow, not just model accuracy/R².



66\. FINAL RESPONSE AFTER COMPLETION



When you finish implementation, do NOT simply say:



"Done."



Instead give me a concise project-status report containing:



Files created/modified

Final repository structure

Regression algorithms implemented

Classification Part-A algorithms implemented

Regression metrics obtained

Classification metrics obtained

Tuned models and best parameters

Top two regression models used for 5-fold CV

Feature-engineering decisions

Any warnings/errors encountered

Git commits created

Any remaining rubric gaps



Do not hide problems.



If something cannot be completed correctly, explicitly state the exact issue instead of pretending it is complete.



67\. START NOW



First:



Inspect the existing repository structure.

Inspect both CSV files.

Inspect the official capstone requirements.

Verify the actual schemas and targets.

Create the two notebooks.

Build the regression notebook completely.

Build the classification Part-A notebook completely.

Update README and Requirements.txt.

Run both notebooks from top to bottom.

Perform the final rubric audit.

Create meaningful Git commits at each milestone.

End with the final project-status report.



Do not ask me to manually create intermediate files.



Do not wait for approval between normal implementation steps.



Make technically sound decisions yourself, but never invent evidence or silently violate the project requirements.





