import numpy as np
import pandas as pd
import functools
import shap

def predict_wrapper(X, pipeline, feature_columns):
    """
    Guarantees the pipeline receives a DataFrame for correct feature handling,
    and the output is a stable 2D NumPy array for KernelExplainer.
    """
    # 1. Convert NumPy array (X from explainer) back to DataFrame
    X_df = pd.DataFrame(X, columns=feature_columns)
    
    # 2. Get raw predictions
    raw_preds = pipeline.predict_proba(X_df)
    positive_class_probs = np.atleast_2d(raw_preds)[:, 1].ravel()
    
    # 3. Reshape the vector to the strict (N, 1) 2D output required by KernelExplainer
    predictions = positive_class_probs.reshape(-1, 1)
    
    # Return the result as a list to satisfy the explainer's internal indexing logic
    return np.array(predictions, dtype=np.float32)



def get_shape_values(pipeline, X_train, y_train, background_sample, test_sample,):
    print("Fitting the pipeline")
    pipeline.fit(X_train, y_train)

    wrapper_with_args = functools.partial(
        predict_wrapper, 
        pipeline=pipeline, 
        feature_columns=X_train.columns
    )

    print("Creating KernelExplainer and computing SHAP values (This may be slow)...")

    # Use functools.partial for stable function passing
    explainer = shap.KernelExplainer(
        model=wrapper_with_args, 
        data=background_sample.values # Pass NumPy array for stability
    )
    shap_values = explainer.shap_values(test_sample.values)

    return np.squeeze(shap_values), explainer.expected_value

def plot_shap_summary(shap_values, X_test_sample, title = ""):
    """
    Generates and displays the SHAP plots (Bar, Beeswarm).

    Args:
        shap_values (np.ndarray): The calculated SHAP values for the positive class (N_samples, N_features).
        X_test_sample: The unscaled feature data used to calculate the SHAP values.

    """
    
    # Safely convert the expected_value (which might be a 1-element array) to a scalar
    # if isinstance(expected_value, np.ndarray) and expected_value.size == 1:
    #     base_value = expected_value.item()
    # else:
    #     base_value = expected_value
    
    
    ## 1. Global Feature Importance (Bar Plot)
    print(title)
    print("\n1. Global Feature Importance (Bar Plot)...")
    shap.summary_plot(
        shap_values,
        features=X_test_sample.values, 
        feature_names=X_test_sample.columns.tolist(),
        plot_type="bar" 
    )

    ## 2. Summary Plot (The SHAP Beeswarm)
    print(title)
    print("\n2. Summary Plot (Beeswarm)...")
    shap.summary_plot(
        shap_values,
        features=X_test_sample.values, 
        feature_names=X_test_sample.columns.tolist(), 
    )

def plot_shap_force(base_value, shap_values, test_sample, sample_index, title):
    """
    Generates and displays the Froce SHAP plot for a selected sample.

    Args:
        shap_values (np.ndarray), base_value (E[f(x)]) from the explainer, X_test,
        sample_index (int): The index of the sample to use for the individual Force Plot.
    """
    print(title)
    print(f"Force Plot (Individual Explanation) for Sample Index {sample_index}...")

    return shap.force_plot(
        base_value, 
        shap_values[sample_index, :], 
        test_sample.iloc[sample_index],
        matplotlib=True 
    )

