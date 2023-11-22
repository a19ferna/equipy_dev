import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fairness_density_plot(y, sensitive_features):
    """
    Visualizes the distribution of predictions based on different sensitive features using kernel density estimates (KDE).

    Parameters:
    - y (dict): A dictionary containing sequentally fair output datasets.
    - sensitive_features (array-like (shape (n_samples, n_sensitive_features)) 
                : The samples representing multiple sensitive attributes.
    
    Returns:
    None

    Raises:
    ValueError: If the input data is not in the expected format.

    Plotting Conventions:
    - The x-axis represents prediction values, and the y-axis represents density.

    Example:
    >>> y = {
            'Base model': [prediction_values],
            'sens_var_1': [prediction_values],
            'sens_var_2': [prediction_values],
            ...
        }
    >>> sensitive_features = [[sensitive_features_of_ind_1_values], [sensitive_feature_of_ind_2_values], ...]

    Usage:
    fairness_density_plot(y, sensitive_features)
    """

    plt.figure(figsize=(12, 9))
    n_a = len(sensitive_features.T)
    n_m = 1

    for key in y.keys():
        title = None
        df = pd.DataFrame()
        for i, sens in enumerate(sensitive_features.T):
            df[f"sensitive_feature_{i+1}"] = sens

        df['Prediction'] = y[key]
        if key == 'Base model':
            for i in range(len(sensitive_features.T)):
                title = key
                plt.subplot(n_a, n_m + 1, i * (n_m+1) + 1)
                modalities = df[f'sensitive_feature_{i+1}'].unique()
                for mod in modalities:
                    subset_data = df[df[f'sensitive_feature_{i+1}'] == mod]
                    sns.kdeplot(
                        subset_data['Prediction'], label=f'sensitive_feature_{i+1}: {mod}', fill=True, alpha=0.2)
                plt.legend()
                plt.title(title, fontsize=11)

        else:
            for i in range(len(sensitive_features.T)):
                if key == f'sensitive_feature_{i+1}':
                    title = key
                    plt.subplot(n_a, n_m + 1, i * (n_m+1) + 2)
                    modalities = df[f'sensitive_feature_{i+1}'].unique()
                    for mod in modalities:
                        subset_data = df[df[f'sensitive_feature_{i+1}'] == mod]
                        sns.kdeplot(
                            subset_data['Prediction'], label=f'sensitive_feature_{i+1}: {mod}', fill=True, alpha=0.2)
                        plt.legend()
                    plt.title(title, fontsize=11)

    plt.xlabel('Prediction')
    plt.ylabel('Density')
