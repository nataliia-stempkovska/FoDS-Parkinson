from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold


warnings.filterwarnings("ignore")


# =========================================================
# PATHS
# =========================================================

print("SCRIPT STARTED")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# New clean folder for report outputs
OUTPUT_DIR = ROOT / "HGB" / "_outputs_report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("ROOT:", ROOT)
print("DATA_DIR:", DATA_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)


# =========================================================
# SETTINGS
# =========================================================

TARGET_COL = "status"
DROP_COLS = ["name"]
RANDOM_STATE = 42
CLASS_NAMES = ["Healthy", "Parkinson"]

SPLITS = {
    "split_a_imputed": {
        "train": DATA_DIR / "train_a.csv",
        "test": DATA_DIR / "test_a.csv",
        "description": "Split A: imputed missing values",
    },
    "split_b_deleted_missing": {
        "train": DATA_DIR / "train_b.csv",
        "test": DATA_DIR / "test_b.csv",
        "description": "Split B: rows with missing values deleted",
    },
}

FEATURE_MODES = [
    "all_features",
    "jitter_shimmer_only",
]


# =========================================================
# HELPER FUNCTIONS
# =========================================================


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_split(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def prepare_xy(train_df, test_df, feature_mode):
    y_train = train_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)

    X_train = train_df.drop(columns=[TARGET_COL] + DROP_COLS, errors="ignore")
    X_test = test_df.drop(columns=[TARGET_COL] + DROP_COLS, errors="ignore")

    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    if feature_mode == "jitter_shimmer_only":
        selected_cols = [
            col
            for col in X_train.columns
            if "jitter" in col.lower() or "shimmer" in col.lower()
        ]
        X_train = X_train[selected_cols]
        X_test = X_test[selected_cols]

    if X_train.shape[1] == 0:
        raise ValueError(f"No features found for feature_mode={feature_mode}")

    return X_train, X_test, y_train, y_test


def tune_hgb(X_train, y_train, split_name, feature_mode):
    print_section(f"TUNING HGB | {split_name} | {feature_mode}")

    model = HistGradientBoostingClassifier(
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
        estimator=model,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True,
    )

    grid.fit(X_train, y_train)

    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results.to_csv(
        OUTPUT_DIR / f"hgb_cv_results_{split_name}_{feature_mode}.csv",
        index=False,
    )

    best_params = grid.best_params_

    pd.DataFrame([best_params]).to_csv(
        OUTPUT_DIR / f"hgb_best_params_{split_name}_{feature_mode}.csv",
        index=False,
    )

    print("Best CV balanced accuracy:", round(grid.best_score_, 3))
    print("Best parameters:", best_params)

    return grid.best_estimator_, best_params, grid.best_score_


def get_metrics(y_true, y_pred, y_prob, prefix):
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}_roc_auc": roc_auc_score(y_true, y_prob),
        f"{prefix}_pr_auc": average_precision_score(y_true, y_prob),
    }


def save_confusion_matrices(y_test, y_pred, split_name, feature_mode):
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = confusion_matrix(y_test, y_pred, normalize="true") * 100

    pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in CLASS_NAMES],
        columns=[f"pred_{name}" for name in CLASS_NAMES],
    ).to_csv(OUTPUT_DIR / f"hgb_confusion_matrix_{split_name}_{feature_mode}.csv")

    pd.DataFrame(
        cm_percent,
        index=[f"true_{name}" for name in CLASS_NAMES],
        columns=[f"pred_{name}" for name in CLASS_NAMES],
    ).to_csv(
        OUTPUT_DIR / f"hgb_confusion_matrix_percent_{split_name}_{feature_mode}.csv"
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"HGB confusion matrix\n{split_name}, {feature_mode}")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"hgb_confusion_matrix_{split_name}_{feature_mode}.png",
        dpi=300,
    )
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_percent,
        display_labels=CLASS_NAMES,
    )
    disp.plot(ax=ax, cmap="Blues", values_format=".1f", colorbar=True)
    ax.set_title(f"HGB confusion matrix (%)\n{split_name}, {feature_mode}")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"hgb_confusion_matrix_percent_{split_name}_{feature_mode}.png",
        dpi=300,
    )
    plt.close()

    return cm


def save_curves(y_test, y_prob, split_name, feature_mode):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_test,
        y_prob,
        name="HGB",
        ax=ax,
    )
    ax.set_title(f"HGB ROC curve\n{split_name}, {feature_mode}")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"hgb_roc_curve_{split_name}_{feature_mode}.png",
        dpi=300,
    )
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(
        y_test,
        y_prob,
        name="HGB",
        ax=ax,
    )
    ax.set_title(f"HGB Precision-Recall curve\n{split_name}, {feature_mode}")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"hgb_pr_curve_{split_name}_{feature_mode}.png",
        dpi=300,
    )
    plt.close()


def save_classification_report(y_test, y_pred, split_name, feature_mode):
    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).T.to_csv(
        OUTPUT_DIR / f"hgb_classification_report_{split_name}_{feature_mode}.csv"
    )


def save_feature_importance(model, X_test, y_test, split_name, feature_mode, top_n=15):
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
    ).sort_values("importance_mean", ascending=False)

    importance_df.to_csv(
        OUTPUT_DIR / f"hgb_feature_importance_{split_name}_{feature_mode}.csv",
        index=False,
    )

    top = importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
    )
    ax.set_xlabel("Permutation importance")
    ax.set_title(f"HGB feature importance\n{split_name}, {feature_mode}")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"hgb_feature_importance_{split_name}_{feature_mode}.png",
        dpi=300,
    )
    plt.close()

    return importance_df


def run_experiment(split_name, split_info, feature_mode):
    train_df, test_df = load_split(
        split_info["train"],
        split_info["test"],
    )

    X_train, X_test, y_train, y_test = prepare_xy(
        train_df,
        test_df,
        feature_mode,
    )

    print_section(f"DATA CHECK | {split_name} | {feature_mode}")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Train class distribution:")
    print(y_train.value_counts().sort_index())
    print("Test class distribution:")
    print(y_test.value_counts().sort_index())
    print("Features used:")
    print(list(X_train.columns))

    model, best_params, best_cv_score = tune_hgb(
        X_train,
        y_train,
        split_name,
        feature_mode,
    )

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]

    train_metrics = get_metrics(
        y_train,
        y_train_pred,
        y_train_prob,
        "train",
    )

    test_metrics = get_metrics(
        y_test,
        y_test_pred,
        y_test_prob,
        "test",
    )

    cm = save_confusion_matrices(
        y_test,
        y_test_pred,
        split_name,
        feature_mode,
    )

    save_curves(
        y_test,
        y_test_prob,
        split_name,
        feature_mode,
    )

    save_classification_report(
        y_test,
        y_test_pred,
        split_name,
        feature_mode,
    )

    importance_df = save_feature_importance(
        model,
        X_test,
        y_test,
        split_name,
        feature_mode,
    )

    row = {
        "model": "HGB",
        "split": split_name,
        "split_description": split_info["description"],
        "feature_set": feature_mode,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
        "features_used": "; ".join(X_train.columns),
        "cv_best_balanced_accuracy": best_cv_score,
        "best_params": json.dumps(best_params),
        "true_healthy_pred_healthy": cm[0, 0],
        "true_healthy_pred_parkinson": cm[0, 1],
        "true_parkinson_pred_healthy": cm[1, 0],
        "true_parkinson_pred_parkinson": cm[1, 1],
        "top_1_feature": importance_df.iloc[0]["feature"],
        "top_1_importance": importance_df.iloc[0]["importance_mean"],
        "top_2_feature": importance_df.iloc[1]["feature"]
        if len(importance_df) > 1
        else None,
        "top_2_importance": importance_df.iloc[1]["importance_mean"]
        if len(importance_df) > 1
        else None,
        "top_3_feature": importance_df.iloc[2]["feature"]
        if len(importance_df) > 2
        else None,
        "top_3_importance": importance_df.iloc[2]["importance_mean"]
        if len(importance_df) > 2
        else None,
    }

    row.update(train_metrics)
    row.update(test_metrics)

    print_section(f"RESULTS | {split_name} | {feature_mode}")
    for key, value in row.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    return row


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    all_rows = []

    for split_name, split_info in SPLITS.items():
        for feature_mode in FEATURE_MODES:
            row = run_experiment(
                split_name,
                split_info,
                feature_mode,
            )
            all_rows.append(row)

    results_df = pd.DataFrame(all_rows)

    comparison_cols = [
        "model",
        "split",
        "feature_set",
        "n_train",
        "n_test",
        "n_features",
        "cv_best_balanced_accuracy",
        "train_accuracy",
        "test_accuracy",
        "train_balanced_accuracy",
        "test_balanced_accuracy",
        "train_precision",
        "test_precision",
        "train_recall",
        "test_recall",
        "train_f1",
        "test_f1",
        "train_roc_auc",
        "test_roc_auc",
        "train_pr_auc",
        "test_pr_auc",
        "true_healthy_pred_healthy",
        "true_healthy_pred_parkinson",
        "true_parkinson_pred_healthy",
        "true_parkinson_pred_parkinson",
        "top_1_feature",
        "top_1_importance",
        "top_2_feature",
        "top_2_importance",
        "top_3_feature",
        "top_3_importance",
        "best_params",
        "features_used",
    ]

    results_df = results_df[comparison_cols]

    main_path = OUTPUT_DIR / "hgb_results_for_final_model_comparison.csv"
    results_df.to_csv(main_path, index=False)

    short_path = OUTPUT_DIR / "hgb_results_summary_short.csv"
    results_df[
        [
            "model",
            "split",
            "feature_set",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_roc_auc",
            "test_pr_auc",
            "cv_best_balanced_accuracy",
            "top_1_feature",
            "top_2_feature",
            "top_3_feature",
        ]
    ].to_csv(short_path, index=False)

    print_section("FINAL HGB REPORT TABLE")
    print(results_df.to_string(index=False))

    print("\nSaved main comparison CSV:")
    print(main_path)

    print("\nSaved short summary CSV:")
    print(short_path)

    print("\nAll report-ready HGB files saved in:")
    print(OUTPUT_DIR)
