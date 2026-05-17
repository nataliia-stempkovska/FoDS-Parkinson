from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold


# =========================================================
# PATHS
# =========================================================

print("SCRIPT STARTED")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "HGB" / "_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("ROOT:", ROOT)
print("DATA_DIR:", DATA_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)

TRAIN_A_PATH = DATA_DIR / "train_a.csv"
TEST_A_PATH = DATA_DIR / "test_a.csv"

TRAIN_B_PATH = DATA_DIR / "train_b.csv"
TEST_B_PATH = DATA_DIR / "test_b.csv"


# =========================================================
# SETTINGS
# =========================================================

TARGET_COL = "status"
DROP_COLS = ["name"]
RANDOM_STATE = 42

CLASS_NAMES = ["Healthy", "Parkinson"]


# =========================================================
# HELPER FUNCTIONS
# =========================================================


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def make_safe_name(text):
    return text.replace(":", "").replace(" ", "_").replace("→", "to").replace("|", "")


def load_split(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def prepare_xy(train_df, test_df, feature_mode="all"):
    train_df = train_df.copy()
    test_df = test_df.copy()

    y_train = train_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)

    X_train = train_df.drop(columns=[TARGET_COL] + DROP_COLS, errors="ignore")
    X_test = test_df.drop(columns=[TARGET_COL] + DROP_COLS, errors="ignore")

    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    if feature_mode == "jitter_shimmer":
        selected_cols = [
            col
            for col in X_train.columns
            if "jitter" in col.lower() or "shimmer" in col.lower()
        ]

        X_train = X_train[selected_cols]
        X_test = X_test[selected_cols]

    if X_train.shape[1] == 0:
        raise ValueError(f"No features found for feature_mode='{feature_mode}'")

    return X_train, X_test, y_train, y_test


def tune_hgb(X_train, y_train, split_name, feature_mode):
    print_section(f"HYPERPARAMETER TUNING: {split_name} | {feature_mode}")

    base_model = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
    )

    param_grid = {
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "max_iter": [100, 200, 300],
        "max_leaf_nodes": [5, 10, 15, 20, 31],
        "l2_regularization": [0.0, 0.01, 0.1, 1.0],
        "min_samples_leaf": [5, 10, 20],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    grid.fit(X_train, y_train)

    print("Best balanced CV accuracy:", round(grid.best_score_, 3))
    print("Best parameters:")
    print(grid.best_params_)

    safe_name = make_safe_name(f"{split_name}_{feature_mode}")

    cv_results_df = pd.DataFrame(grid.cv_results_)
    cv_results_path = OUTPUT_DIR / f"cv_results_{safe_name}.csv"
    cv_results_df.to_csv(cv_results_path, index=False)

    best_params_path = OUTPUT_DIR / f"best_params_{safe_name}.csv"
    pd.DataFrame([grid.best_params_]).to_csv(best_params_path, index=False)

    print(f"Saved CV results to: {cv_results_path}")
    print(f"Saved best parameters to: {best_params_path}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def evaluate_model(
    model, X_test, y_test, split_name, feature_mode, best_params, best_cv_score
):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print_section(f"TEST RESULTS: {split_name} | FEATURES: {feature_mode}")

    print(f"Accuracy:              {acc:.3f}")
    print(f"Balanced accuracy:     {bal_acc:.3f}")
    print(f"ROC-AUC:               {auc:.3f}")
    print(f"Best CV balanced acc:  {best_cv_score:.3f}")

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
    )
    print(report_text)

    safe_name = make_safe_name(f"{split_name}_{feature_mode}")

    cm_path = OUTPUT_DIR / f"confusion_matrix_{safe_name}.csv"
    pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in CLASS_NAMES],
        columns=[f"pred_{name}" for name in CLASS_NAMES],
    ).to_csv(cm_path)

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
    )
    report_path = OUTPUT_DIR / f"classification_report_{safe_name}.csv"
    pd.DataFrame(report_dict).transpose().to_csv(report_path)

    plot_path = OUTPUT_DIR / f"confusion_matrix_{safe_name}.png"

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )

    disp.plot(values_format="d")
    plt.title(f"{split_name} | {feature_mode}")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved confusion matrix CSV to: {cm_path}")
    print(f"Saved classification report to: {report_path}")
    print(f"Saved confusion matrix plot to: {plot_path}")

    return {
        "split": split_name,
        "feature_mode": feature_mode,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "roc_auc": auc,
        "best_cv_balanced_accuracy": best_cv_score,
        "best_params": best_params,
    }


def save_feature_importance(model, X_test, y_test, split_name, feature_mode, top_n=10):
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=RANDOM_STATE,
        scoring="balanced_accuracy",
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False)

    print_section(f"TOP {top_n} FEATURES: {split_name} | {feature_mode}")
    print(importance_df.head(top_n).to_string(index=False))

    safe_name = make_safe_name(f"{split_name}_{feature_mode}")
    importance_path = OUTPUT_DIR / f"feature_importance_{safe_name}.csv"
    importance_df.to_csv(importance_path, index=False)

    print(f"Saved feature importance to: {importance_path}")

    return importance_df


def run_experiment(split_name, train_path, test_path, feature_mode):
    train_df, test_df = load_split(train_path, test_path)

    X_train, X_test, y_train, y_test = prepare_xy(
        train_df,
        test_df,
        feature_mode=feature_mode,
    )

    print_section(f"DATA CHECK: {split_name} | {feature_mode}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")
    print(f"Train class distribution:\n{y_train.value_counts().sort_index()}")
    print(f"Test class distribution:\n{y_test.value_counts().sort_index()}")
    print(f"Features used:\n{list(X_train.columns)}")

    model, best_params, best_cv_score = tune_hgb(
        X_train,
        y_train,
        split_name,
        feature_mode,
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        split_name,
        feature_mode,
        best_params,
        best_cv_score,
    )

    save_feature_importance(
        model,
        X_test,
        y_test,
        split_name,
        feature_mode,
    )

    return metrics


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    all_results = []

    experiments = [
        ("Split A: train_a → test_a", TRAIN_A_PATH, TEST_A_PATH),
        ("Split B: train_b → test_b", TRAIN_B_PATH, TEST_B_PATH),
    ]

    feature_modes = [
        "all",
        "jitter_shimmer",
    ]

    for split_name, train_path, test_path in experiments:
        for feature_mode in feature_modes:
            metrics = run_experiment(
                split_name=split_name,
                train_path=train_path,
                test_path=test_path,
                feature_mode=feature_mode,
            )

            all_results.append(metrics)

    results_df = pd.DataFrame(all_results)

    print_section("FINAL COMPARISON")
    print(results_df.to_string(index=False))

    results_path = OUTPUT_DIR / "hgb_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\nSaved final comparison to: {results_path}")
    print(f"All outputs saved in: {OUTPUT_DIR}")
