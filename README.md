# FoDS-Parkinson

# Parkinson's Disease Classification Using Voice Features

## Overview

This project was completed as part of the **Foundations of Data Science (FoDS) course at ETH Zurich (FS26)**.

The objective of this project is to investigate whether acoustic voice features can be used to distinguish individuals with Parkinson's disease (PD) from healthy controls using machine learning methods.

Our primary research question is:

> To what extent can vocal instability features, such as jitter and shimmer, be used as predictive characteristics for classifying individuals as Parkinson's patients or healthy controls?

In addition to jitter and shimmer, we investigate the predictive value of broader acoustic and nonlinear voice features and assess how preprocessing choices influence model performance.

---

# Dataset

We used the **UCI Parkinson's Disease Dataset** originally published by Little et al. (2009).

## Dataset Characteristics

- 195 voice recordings
- 32 subjects
- Approximately 75:25 class imbalance between Parkinson's disease and healthy controls
- 22 acoustic voice features
- Binary classification task (PD vs Healthy)

The dataset contains multiple recordings from each participant and includes:

- Fundamental frequency measures
- Jitter measures
- Shimmer measures
- Noise-related measures
- Nonlinear dynamical voice features

Because multiple recordings originate from the same individual, special care was taken to avoid subject-level data leakage during model evaluation.

---

# Data Preprocessing

## Missing Values

Two parallel preprocessing strategies were evaluated:

### Dataset A (Imputed)

- Missing values imputed using group-wise median imputation
- Medians computed separately within PD and healthy groups

### Dataset B (Dropped)

- Recordings containing missing feature values removed

---

## Feature Sets

Two feature subsets were compared:

### All Features

- All 22 acoustic voice features

### Jitter/Shimmer Features

- Reduced subset containing only jitter- and shimmer-related vocal instability measures

---

## Feature Transformation

Skewed features were log-transformed prior to modelling.

For Logistic Regression, Support Vector Machine, and Random Forest, features were subsequently standardized using StandardScaler.

Histogram-based Gradient Boosting does not require feature scaling and therefore used only the log-transformed features.

---

## Outlier Analysis

Potential outliers were identified using three complementary methods:

- Interquartile Range (IQR) rule
- Z-score thresholding
- PCA-based Mahalanobis distance

A total of 21 recordings were flagged as potential outliers.

Models were evaluated under two conditions:

- Full training dataset
- Training dataset with the 21 flagged recordings removed

---

# Experimental Design

Four supervised classifiers were trained and compared across a **2 × 2 × 2** experimental grid:

### Datasets

- Dataset A (imputed)
- Dataset B (dropped)

### Feature Sets

- All 22 features
- Jitter/shimmer subset

### Outlier Conditions

- Full training data
- Training data with flagged outliers removed

This resulted in a systematic comparison of preprocessing choices and feature selection strategies.

Class-balanced loss weighting was applied in all supervised models to compensate for the approximately 75:25 class imbalance.

---

# Machine Learning Models

Four supervised learning algorithms were evaluated.

## Logistic Regression (LR)

Logistic Regression used:

- Log-transformed features
- Standardized features
- L1 regularisation
- `liblinear` solver

Hyperparameter tuning searched over:

- Regularisation strength C
- 10 log-uniform values between 10⁻³ and 10³

---

## Support Vector Machine (SVM)

Support Vector Machine used:

- Log-transformed features
- Standardized features
- Radial Basis Function (RBF) kernel

Hyperparameter search:

- C ∈ {0.1, 1, 10, 100}
- γ ∈ {scale, auto}

---

## Random Forest (RF)

Random Forest used:

- Log-transformed features
- Standardized features

Hyperparameter search:

- n_estimators ∈ {100, 200}
- max_depth ∈ {None, 10, 20}
- min_samples_leaf ∈ {1, 5}

---

## Histogram-based Gradient Boosting (HGB)

Histogram-based Gradient Boosting builds trees sequentially, with each tree fitted to the negative gradient of the loss from the previous iteration.

Continuous feature values are discretised into histograms prior to tree construction, improving computational efficiency while maintaining strong predictive performance.

Hyperparameter search included:

- Learning rate
- Maximum number of iterations
- Maximum number of leaf nodes
- L2 regularisation strength
- Minimum samples per leaf

A total of 240 parameter combinations were evaluated.

Early stopping was applied within each fold using:

```python
n_iter_no_change = 20
```

---

# Evaluation Strategy

## Primary Evaluation: Leave-One-Subject-Out Cross-Validation (LOSO)

The primary evaluation method was Leave-One-Subject-Out Cross-Validation (LOSO).

Each of the 32 subjects was held out in turn:

1. One subject served as the test subject
2. The model was trained on the remaining 31 subjects
3. All recordings from the held-out subject were used for evaluation

This procedure ensures that recordings from the same participant never appear in both training and testing data.

LOSO was selected because the dataset contains multiple recordings per participant, making recording-level splits vulnerable to information leakage.

---

## Nested Hyperparameter Optimisation

All four models were evaluated within a nested LOSO framework.

### Outer Loop

- LeaveOneGroupOut (LOSO)
- Subject-level evaluation

### Inner Loop

- GroupKFold (k = 5)
- Hyperparameter optimisation using GridSearchCV

Balanced Accuracy was used as the optimisation metric.

For Logistic Regression and SVM, hyperparameters were re-optimised within each LOSO training fold.

For Random Forest and Histogram-based Gradient Boosting, fixed hyperparameters obtained from a full-data grid search were used during LOSO evaluation. Consequently, their LOSO estimates should be interpreted as approximate lower-bound performance estimates.

---

## Secondary Evaluation: Held-Out Test Set

A subject-level held-out test split was also reported as a secondary reference.

The split was stratified by diagnosis and ensured that all recordings from a given subject appeared exclusively in either training or testing data.

### Dataset Sizes

Dataset A:

- Training: 152 recordings
- Testing: 43 recordings

Dataset B:

- Training: 146 recordings
- Testing: 41 recordings

Because the test set contains only two healthy control subjects, a single misclassification can substantially influence Balanced Accuracy. Therefore, LOSO was treated as the primary performance estimate.

---

# Evaluation Metrics

## Primary Metric

Balanced Accuracy (BA)

Balanced Accuracy was chosen because of the approximately 75:25 class imbalance, where a majority-class baseline already achieves roughly 75% standard accuracy.

Balanced Accuracy is defined as:

BA = ½ × (Sensitivity + Specificity)

or

BA = ½ × [ TP/(TP+FN) + TN/(TN+FP) ]

---

## Secondary Metrics

- Macro F1-score
- Accuracy
- Precision
- Recall
- ROC-AUC
- Confusion Matrix

---

# Model Explainability

Model interpretability was assessed using **SHAP (SHapley Additive exPlanations)**.

SHAP values were computed for all four supervised models to identify which acoustic features contributed most strongly to predictions.

## Main Findings

- Strong agreement across all models regarding the most important predictors
- spread1 consistently ranked among the highest-impact features
- PPE emerged as one of the most influential nonlinear measures
- spread2, RPDE, and D2 also contributed substantially
- Jitter and shimmer features remained informative but were rarely the dominant predictors

These findings suggest that nonlinear vocal dynamics contain stronger discriminatory information than vocal instability measures alone.

---

# Unsupervised Learning

To investigate whether Parkinson's disease recordings form natural clusters in feature space, unsupervised learning methods were also applied.

Methods included:

- K-Means Clustering
- Hierarchical Clustering (Ward linkage)

Clustering performance was evaluated using:

- Silhouette Coefficient
- Davies-Bouldin Index
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Purity

The clustering analyses indicated substantial overlap between PD and healthy recordings and showed that disease status does not naturally separate into clearly distinct clusters in the original acoustic feature space.

---

# Key Findings

- Subject-level evaluation produced substantially more conservative performance estimates than recording-level evaluation.
- LOSO provided a realistic estimate of generalisation to previously unseen individuals.
- HGB and Random Forest achieved the strongest overall LOSO Balanced Accuracy.
- Models trained on all 22 features generally outperformed models trained only on jitter and shimmer features.
- Jitter and shimmer were informative but not sufficient as standalone dominant predictors.
- Nonlinear acoustic features such as spread1, spread2, PPE, RPDE, and D2 consistently emerged as the most important predictors.
- Results highlight the importance of subject-level validation when evaluating voice-based Parkinson's disease classifiers.

---

# Authors

- Yutong Liu
- Isabella Müller-Vogt
- Nataliia Stempkovska
- Defne Tekbulut
- Veronika Tretjakova
