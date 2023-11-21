import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fairness_density_plot(y_fair_test, x_sa_test):
    """
    Visualizes the distribution of predictions based on different sensitive features using kernel density estimates (KDE).

    Parameters:
    - y_fair_test (dict): A dictionary containing sequentally fair output datasets.
    - x_sa_test (array-like (shape (n_samples, n_sensitive_features)) 
                : The test samples representing multiple sensitive attributes.
    
    Returns:
    None

    Raises:
    ValueError: If the input data is not in the expected format.

    Plotting Conventions:
    - The x-axis represents prediction values, and the y-axis represents density.

    Example:
    >>> y_fair_test = {
            'Base model': [prediction_values],
            'sens_var_1': [prediction_values],
            'sens_var_2': [prediction_values],
            ...
        }
    >>> x_sa_test = [[sensitive_features_of_ind_1_values], [sensitive_feature_of_ind_2_values], ...]

    Usage:
    viz_fairness_distrib(y_fair_test, x_sa_test)
    """

    plt.figure(figsize=(12, 9))
    n_a = len(x_sa_test.T)
    n_m = 1

    for key in y_fair_test.keys():
        title = None
        df_test = pd.DataFrame()
        for i, sens in enumerate(x_sa_test.T):
            df_test[f"sensitive_feature_{i+1}"] = sens

        df_test['Prediction'] = y_fair_test[key]
        if key == 'Base model':
            for i in range(len(x_sa_test.T)):
                title = key
                plt.subplot(n_a, n_m + 1, i * (n_m+1) + 1)
                modalities = df_test[f'sensitive_feature_{i+1}'].unique()
                for mod in modalities:
                    subset_data = df_test[df_test[f'sensitive_feature_{i+1}'] == mod]
                    sns.kdeplot(
                        subset_data['Prediction'], label=f'sensitive_feature_{i+1}: {mod}', fill=True, alpha=0.2)
                plt.legend()
                plt.title(title, fontsize=11)

        else:
            for i in range(len(x_sa_test.T)):
                if key == f'sens_var_{i+1}':
                    title = key
                    plt.subplot(n_a, n_m + 1, i * (n_m+1) + 2)
                    modalities = df_test[f'sensitive_feature_{i+1}'].unique()
                    for mod in modalities:
                        subset_data = df_test[df_test[f'sensitive_feature_{i+1}'] == mod]
                        sns.kdeplot(
                            subset_data['Prediction'], label=f'sensitive_feature_{i+1}: {mod}', fill=True, alpha=0.2)
                        plt.legend()
                    plt.title(title, fontsize=11)

    plt.xlabel('Prediction')
    plt.ylabel('Density')
