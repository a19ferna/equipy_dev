import numpy as np
import itertools
from fairness._wasserstein import MultiWasserStein
from metrics._performance_metrics import performance_dict
from sklearn.metrics import mean_squared_error

## BASE FOR PERMUTATION CALCULATION ##

def permutations_cols(x_sa):
    """
    Generate permutations of columns in the input array x_sa.

    Parameters:
    - x_sa (array-like): Input array where each column represents a different sensitive feature.

    Returns:
    dict: A dictionary where keys are tuples representing permutations of column indices,
          and values are corresponding permuted arrays of sensitive features.

    Example:
    >>> x_sa = [[1, 2], [3, 4], [5, 6]]
    >>> permutations_cols(x_sa)
    {(0, 1): [[1, 2], [3, 4], [5, 6]], (1, 0): [[3, 4], [1, 2], [5, 6]]}

    Note:
    This function generates all possible permutations of columns and stores them in a dictionary.
    """
    n = len(x_sa[0])
    ind_cols = list(range(n))
    permut_cols = list(itertools.permutations(ind_cols))
    x_sa_with_ind = np.vstack((ind_cols, x_sa))

    dict_all_combs = {}
    for permutation in permut_cols:
        permuted_x_sa = x_sa_with_ind[:, permutation]
        # First row as the key (converted to tuple)
        key = tuple(permuted_x_sa[0])
        # Other rows as values (converted to list)
        values = permuted_x_sa[1:].tolist()
        dict_all_combs[key] = values

    return dict_all_combs

def calculate_perm_wst(y_calib, x_sa_calib, y_test, x_sa_test, epsilon=None):
    """
    Calculate Wasserstein distance for different permutations of sensitive features between calibration and test sets.
    
    Parameters:
    - y_calib (array-like): Calibration set predictions.
    - x_sa_calib (array-like): Calibration set sensitive features.
    - y_test (array-like): Test set predictions.
    - x_sa_test (array-like): Test set sensitive features.
    - epsilon (array-like or None, optional): Fairness constraints. Defaults to None.

    Returns:
    dict: A dictionary where keys are tuples representing permutations of column indices,
          and values are corresponding sequential fairness values for each permutation.

    Example:
    >>> y_calib = [1, 2, 3]
    >>> x_sa_calib = [[1, 2], [3, 4], [5, 6]]
    >>> y_test = [4, 5, 6]
    >>> x_sa_test = [[7, 8], [9, 10], [11, 12]]
    >>> calculate_perm_wst(y_calib, x_sa_calib, y_test, x_sa_test)
    {(0, 1): {'Base model': 0.5, 'sens_var_1': 0.2}, (1, 0): {'Base model': 0.3, 'sens_var_0': 0.6}}

    Note:
    This function calculates Wasserstein distance for different permutations of sensitive features
    between calibration and test sets and stores the sequential fairness values in a dictionary.
    """
    all_perm_calib = permutations_cols(x_sa_calib)
    all_perm_test = permutations_cols(x_sa_test)
    if epsilon != None:
        all_perm_epsilon = permutations_cols(np.array([np.array(epsilon).T]))
        for key in all_perm_epsilon.keys():
            all_perm_epsilon[key] = all_perm_epsilon[key][0]

    store_dict = {}
    for key in all_perm_calib:
        wst = MultiWasserStein()
        wst.fit(y_calib, np.array(all_perm_calib[key]))
        if epsilon == None:
            wst.transform(y_test, np.array(
                all_perm_test[key]))
        else :
            wst.transform(y_test, np.array(
                all_perm_test[key]), all_perm_epsilon[key])
        store_dict[key] = wst.get_sequential_fairness()
        old_keys = list(store_dict[key].keys())
        new_keys = ['Base model'] + [f'sens_var_{k}' for k in key]
        key_mapping = dict(zip(old_keys, new_keys))
        store_dict[key] = {key_mapping[old_key]: value for old_key, value in store_dict[key].items()}
    return store_dict

### USEFUL FOR PERFORMANCE ##
def performance_permutations(y_true, permut_y_fair_dict, metric=mean_squared_error):
    """
    Compute the performance values for multiple fair output datasets compared to the true labels, considering permutations.

    Parameters:
    y_true (array-like): True labels or ground truth values.
    permut_y_fair_dict (dict): A dictionary containing permutations of fair output datasets.
    metric (function, optional): The metric used to compute the performance, default=sklearn.metrics.mean_square_error

    Returns:
    list: A list of dictionaries containing performance values for each permutation of fair output datasets.

    Example:
    >>> y_true = np.array([15, 38, 68])
    >>> permut_y_fair_dict = {(1,2): {'Base model':np.array([19,39,65]), 'sens_var_1':np.array([22,40,50]), 'sens_var_2':np.array([28,39,42])},
                               (2,1): {'Base model':np.array([19,39,65]), 'sens_var_2':np.array([34,39,60]), 'sens_var_1':np.array([28,39,42])}}
    >>> performance_values = performance_permutations(y_true, permut_y_fair_dict)
    [{'Base model': 8.666666666666666, 'sens_var_1': 125.66666666666667, 'sens_var_2': 282.0}, 
        {'Base model': 8.666666666666666, 'sens_var_2': 142.0, 'sens_var_1': 282.0}]
    """
    performance_list = []
    for key in permut_y_fair_dict.keys():
        performance_list.append(performance_dict(y_true, permut_y_fair_dict[key], metric))
    return performance_list

