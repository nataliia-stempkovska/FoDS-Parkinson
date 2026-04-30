from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


# =========================================================
# PATHS
# =========================================================

print("SCRIPT STARTED")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

print("ROOT:", ROOT)
print("DATA_DIR:", DATA_DIR)

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


# =========================================================
# HELPER FUNCTIONS
# =========================================================


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_split(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def prepare_xy(train_df, test_df, feature_mode="all"):
    """
    feature_mode:
    - 'all' = all numeric voice features
    - 'jitter_shimmer' = only jitter and shimmer-related features
    """

    train_df = train_df.copy()
    test_df = test_df.copy()

    # Separate target
    y_train = train_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)

    # Drop target and non-feature columns
    X_train = train_df.drop(columns=[TARGET_COL] + DROP_COLS, errors="ignore")
    X_test = test_df.drop(columns=[TARGET_COL] + DROP_COLS, errors="ignore")

    # Keep only numeric columns
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    # Make sure train and test have same columns
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

    return X_train, X_test, y_train, y_test


def train_hgb(X_train, y_train):
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, split_name, feature_mode):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print_section(f"RESULTS: {split_name} | FEATURES: {feature_mode}")

    print(f"Accuracy:          {acc:.3f}")
    print(f"Balanced accuracy: {bal_acc:.3f}")
    print(f"ROC-AUC:           {auc:.3f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson"]))

    return {
        "split": split_name,
        "feature_mode": feature_mode,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "roc_auc": auc,
    }


def show_feature_importance(model, X_test, y_test, split_name, feature_mode, top_n=10):
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
    )

    importance_df = importance_df.sort_values(by="importance_mean", ascending=False)

    print_section(f"TOP {top_n} FEATURES: {split_name} | {feature_mode}")
    print(importance_df.head(top_n).to_string(index=False))

    return importance_df


def run_experiment(split_name, train_path, test_path, feature_mode):
    train_df, test_df = load_split(train_path, test_path)

    X_train, X_test, y_train, y_test = prepare_xy(
        train_df, test_df, feature_mode=feature_mode
    )

    print_section(f"DATA CHECK: {split_name} | {feature_mode}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")
    print(f"Train class distribution:\n{y_train.value_counts().sort_index()}")
    print(f"Test class distribution:\n{y_test.value_counts().sort_index()}")
    print(f"Features used:\n{list(X_train.columns)}")

    model = train_hgb(X_train, y_train)

    metrics = evaluate_model(
        model, X_test, y_test, split_name=split_name, feature_mode=feature_mode
    )

    importance_df = show_feature_importance(
        model, X_test, y_test, split_name=split_name, feature_mode=feature_mode
    )

    return metrics, importance_df


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
            metrics, importance_df = run_experiment(
                split_name=split_name,
                train_path=train_path,
                test_path=test_path,
                feature_mode=feature_mode,
            )

            all_results.append(metrics)

    results_df = pd.DataFrame(all_results)

    print_section("FINAL COMPARISON")
    print(results_df.to_string(index=False))
