# FoDS-Parkinson
# Parkinson's Disease Classification Using Voice Features

## Overview

This project was completed as part of the **Foundations of Data Science Course at ETH Zurich (FS26)**.

The goal of the project is to investigate whether acoustic voice features can be used to distinguish individuals with Parkinson's disease (PD) from healthy controls using machine learning methods.

Our research question is:

> To what extent can vocal instability features, such as jitter and shimmer, be used as predictive characteristics for classifying individuals as Parkinson’s patients or healthy controls?

---

## Dataset

We used the **UCI Parkinson's Disease Dataset** originally published by Little et al. (2009).

### Dataset Characteristics

- 195 voice recordings
- 31 subjects
- 23 Parkinson's disease patients
- 8 healthy controls
- 22 acoustic voice features
- Binary classification task (PD vs Healthy)

The features include:

- Frequency measures
- Jitter measures
- Shimmer measures
- Noise-related measures
- Nonlinear vocal dynamics

---

## Data Preprocessing

Several preprocessing steps were performed before model training:

### Missing Data

Two strategies were evaluated:

**Dataset A**
- Missing values imputed using group-wise median imputation

**Dataset B**
- Rows containing missing values removed

### Feature Transformation

- Log transformation of skewed variables
- Standardization using StandardScaler

### Outlier Analysis

Outliers were identified using:

- IQR rule
- Z-score thresholding
- PCA-based Mahalanobis distance

Models were trained both with and without detected outliers.

### Data Splitting

To avoid data leakage:

- Splitting was performed at the subject level
- Recordings from the same subject were never allowed in both training and test sets

---

## Models

The following supervised learning algorithms were evaluated:

### Logistic Regression (LR)

- L1 and L2 regularization
- Hyperparameter tuning using GridSearchCV

### Random Forest (RF)

- Ensemble tree-based classifier
- Hyperparameter tuning using GroupKFold cross-validation

### Support Vector Machine (SVM)

- RBF kernel
- Nested GroupKFold cross-validation

### Histogram-based Gradient Boosting (HGB)

- Histogram-based Gradient Boosting Classifier
- Nested GroupKFold cross-validation

---

## Evaluation

Primary evaluation metrics:

- Balanced Accuracy
- Macro F1 Score

Additional metrics:

- Accuracy
- Precision
- Recall
- ROC-AUC

Cross-validation was performed using subject-level splits to ensure realistic generalization to unseen individuals.

---

## Explainability

Model interpretation was performed using **SHAP (SHapley Additive exPlanations)**.

Key findings:

- spread1 and PPE consistently ranked among the most important features
- Strong agreement across all four models
- Nonlinear vocal dynamics were generally more influential than jitter and shimmer features

---

## Authors

- Yutong Liu
- Isabella Müller-Vogt
- Nataliia Stempkovska
- Defne Tekbulut
- Veronika Tretjakova
