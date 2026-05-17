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


# =========================================================
# PATHS
# =========================================================

print("SCRIPT STARTED: HGB WITHOUT OUTLIERS")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

OUTPUT_BASE_DIR = ROOT / "HGB" / "_outputs"
BEST_PARAMS_DIR = OUTPUT_BASE_DIR

OUTPUT_DIR = OUTPUT_BASE_DIR / "without_outliers"
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


def remove_outliers_iqr(X_train, y_train, factor=1.5):
    """
    Removes outliers only from the training set using the IQR rule.
    The test set remains unchanged.
    """
    Q1 = X_train.quantile(0.25)
    Q3 = X_train.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR

    non_outlier_mask = ~((X_train < lower) | (X_train > upper)).any(axis=1)

    X_clean = X_train.loc[non_outlier_mask].copy()
    y_clean = y_train.loc[non_outlier_mask].copy()

    removed = len(X_train) - len(X_clean)

    return X_clean, y_clean, removed


def load_best_params(split_name, feature_mode):
    safe_name = make_safe_name(f"{split_name}_{feature_mode}")
    params_path = BEST_PARAMS_DIR / f"best_params_{safe_name}.csv"

    if not params_path.exists():
        raise FileNotFoundError(
            f"Could not find best params file:\n{params_path}\n"
            f"Run the tuned HGB pipeline first."
        )

    params_df = pd.read_csv(params_path)
    params = params_df.iloc[0].to_dict()

    # Convert possible float values back to int where needed
    int_params = ["max_iter", "max_leaf_nodes", "min_samples_leaf"]
    for param in int_params:
        if param in params:
            params[param] = int(params[param])

    return params


def train_hgb_with_best_params(X_train, y_train, best_params):
    model = HistGradientBoostingClassifier(
        **best_params,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model, X_test, y_test, split_name, feature_mode, best_params, removed
):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print_section(f"TEST RESULTS WITHOUT OUTLIERS: {split_name} | {feature_mode}")

    print(f"Removed training outliers: {removed}")
    print(f"Accuracy:                 {acc:.3f}")
    print(f"Balanced accuracy:        {bal_acc:.3f}")
    print(f"ROC-AUC:                  {auc:.3f}")

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    safe_name = make_safe_name(f"{split_name}_{feature_mode}_without_outliers")

    cm_path = OUTPUT_DIR / f"confusion_matrix_{safe_name}.csv"
    report_path = OUTPUT_DIR / f"classification_report_{safe_name}.csv"
    plot_path = OUTPUT_DIR / f"confusion_matrix_{safe_name}.png"

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
    pd.DataFrame(report_dict).transpose().to_csv(report_path)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )
    disp.plot(values_format="d")
    plt.title(f"{split_name} | {feature_mode} | without outliers")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved confusion matrix CSV to: {cm_path}")
    print(f"Saved classification report to: {report_path}")
    print(f"Saved confusion matrix plot to: {plot_path}")

    return {
        "split": split_name,
        "feature_mode": feature_mode,
        "outlier_mode": "without_outliers",
        "removed_training_outliers": removed,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "roc_auc": auc,
        "best_params_used": best_params,
    }


def save_feature_importance(model, X_test, y_test, split_name, feature_mode):
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

    print_section(f"FEATURE IMPORTANCE WITHOUT OUTLIERS: {split_name} | {feature_mode}")
    print(importance_df.head(10).to_string(index=False))

    safe_name = make_safe_name(f"{split_name}_{feature_mode}_without_outliers")
    importance_path = OUTPUT_DIR / f"feature_importance_{safe_name}.csv"

    importance_df.to_csv(importance_path, index=False)

    print(f"Saved feature importance to: {importance_path}")


def run_experiment(split_name, train_path, test_path, feature_mode):
    train_df, test_df = load_split(train_path, test_path)

    X_train, X_test, y_train, y_test = prepare_xy(
        train_df,
        test_df,
        feature_mode=feature_mode,
    )

    print_section(f"DATA CHECK WITHOUT OUTLIERS: {split_name} | {feature_mode}")
    print(f"Original train shape: {X_train.shape}")
    print(f"Test shape:           {X_test.shape}")

    X_train_clean, y_train_clean, removed = remove_outliers_iqr(X_train, y_train)

    print(f"Clean train shape:    {X_train_clean.shape}")
    print(f"Removed outliers:     {removed}")
    print(
        f"Train class distribution after outlier removal:\n{y_train_clean.value_counts().sort_index()}"
    )
    print(f"Test class distribution:\n{y_test.value_counts().sort_index()}")
    print(f"Features used:\n{list(X_train_clean.columns)}")

    best_params = load_best_params(split_name, feature_mode)

    print("\nUsing saved best parameters:")
    print(best_params)

    model = train_hgb_with_best_params(
        X_train_clean,
        y_train_clean,
        best_params,
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        split_name,
        feature_mode,
        best_params,
        removed,
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

    print_section("FINAL COMPARISON WITHOUT OUTLIERS")
    print(results_df.to_string(index=False))

    results_path = OUTPUT_DIR / "hgb_results_without_outliers.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\nSaved final comparison to: {results_path}")
    print(f"All without-outlier outputs saved in: {OUTPUT_DIR}")
