import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


def run_shap_feature_importance(
    model,
    model_name,
    X_train,
    X_test,
    output_dir="shap_outputs",
    feature_names=None,
    max_background=100,
    max_explain=100,
    random_state=42,
):
    """
    Runs SHAP feature importance for HGB, RF, SVM, and LR models.
    Saves:
    - SHAP bar plot
    - SHAP beeswarm plot
    - CSV with mean absolute SHAP values
    """

    model_name = model_name.upper()

    save_dir = os.path.join(output_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Convert to DataFrame if needed
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)

    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_names)

    background = shap.sample(
        X_train, min(max_background, len(X_train)), random_state=random_state
    )

    X_explain = shap.sample(
        X_test, min(max_explain, len(X_test)), random_state=random_state
    )

    print(f"Running SHAP for {model_name}...")

    if model_name in ["HGB", "RF"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)

    elif model_name == "LR":
        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(X_explain)

    elif model_name == "SVM":
        if hasattr(model, "predict_proba"):
            explainer = shap.KernelExplainer(model.predict_proba, background)
        else:
            explainer = shap.KernelExplainer(model.decision_function, background)

        shap_values = explainer.shap_values(X_explain)

    else:
        raise ValueError("model_name must be one of: HGB, RF, SVM, LR")

    # Binary classification: use positive class
    if isinstance(shap_values, list):
        shap_values_plot = shap_values[1]
    else:
        shap_values_plot = shap_values

    shap_values_plot = np.array(shap_values_plot)

    # If SHAP output is 3D, use class 1
    if shap_values_plot.ndim == 3:
        shap_values_plot = shap_values_plot[:, :, 1]

    # Bar plot
    plt.figure()
    shap.summary_plot(
        shap_values_plot,
        X_explain,
        feature_names=X_explain.columns,
        plot_type="bar",
        show=False,
    )
    plt.savefig(
        os.path.join(save_dir, f"{model_name}_shap_bar.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Beeswarm plot
    plt.figure()
    shap.summary_plot(
        shap_values_plot, X_explain, feature_names=X_explain.columns, show=False
    )
    plt.savefig(
        os.path.join(save_dir, f"{model_name}_shap_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # CSV feature importance
    mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)

    shap_importance = pd.DataFrame(
        {"feature": X_explain.columns, "mean_abs_shap": mean_abs_shap}
    ).sort_values("mean_abs_shap", ascending=False)

    shap_importance.to_csv(
        os.path.join(save_dir, f"{model_name}_shap_feature_importance.csv"), index=False
    )

    print(f"SHAP results saved in: {save_dir}")

    return shap_importance
