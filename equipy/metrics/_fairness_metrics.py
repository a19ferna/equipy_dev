import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

from scipy.interpolate import interp1d
import numpy as np

#WARNING:You cannot calculate the EQF function of a single value : this means that if only one individual 
# has a specific sensitive value, you cannot use the transform function. 

class EQF:
    """
    Empirical Quantile Function (EQF) Class.

    This class computes and encapsulates the Empirical Quantile Function for a given set of sample data.
    The EQF provides an interpolation of the cumulative distribution function (CDF) based on the input data.

    Parameters:
    sample_data (array-like): A 1-D array or list-like object containing the sample data.

    Attributes:
    interpolater (scipy.interpolate.interp1d): An interpolation function that maps quantiles to values.
    min_val (float): The minimum value in the sample data.
    max_val (float): The maximum value in the sample data.

    Methods:
    __init__(sample_data): Initializes the EQF object by calculating the interpolater, min_val, and max_val.
    _calculate_eqf(sample_data): Private method to calculate interpolater, min_val, and max_val.
    __call__(value_): Callable method to compute the interpolated value for a given quantile.

    Example usage:
    >>> sample_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> eqf = EQF(sample_data)
    >>> print(eqf(0.5))  # Interpolated value at quantile 0.5
    5.5

    Example usage:
    >>> sample_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> eqf = EQF(sample_data)
    >>> print(eqf([0.2, 0.5, 0.8]))  # Interpolated value at quantiles 0.2, 0.5 and 0.8
    [2.8 5.5 8.2]


    Raises:
    ValueError: If the input value_ is outside the range [0, 1].

    Note:
    - The EQF interpolates values within the range [0, 1] representing quantiles.
    - The input sample_data should be a list or array-like containing numerical values.

    """

    def __init__(self,
                 sample_data,
                 ):
        # définition self.interpoler, self.min_val, self.max_val
        self._calculate_eqf(sample_data)

    def _calculate_eqf(self, sample_data):
        """
        Calculate the Empirical Quantile Function for the given sample data.

        Parameters:
        sample_data (array-like): A 1-D array or list-like object containing the sample data.
        """
        sorted_data = np.sort(sample_data)
        linspace = np.linspace(0, 1, num=len(sample_data))
        # fonction d'interpolation
        self.interpolater = interp1d(linspace, sorted_data)
        self.min_val = sorted_data[0]
        self.max_val = sorted_data[-1]

    def __call__(self, value_):
        """
        Compute the interpolated value for a given quantile.

        Parameters:
        value_ (float): Quantile value between 0 and 1.

        Returns:
        float: Interpolated value corresponding to the input quantile.

        Raises:
        ValueError: If the input value_ is outside the range [0, 1].
        """
        try:
            return self.interpolater(value_)
        except ValueError:
            raise ValueError('Error with input value')

def diff_quantile(data1, data2):
    """
    Compute the unfairness between two populations based on their quantile functions.

    Parameters:
    data1 (array-like): The first set of data points.
    data2 (array-like): The second set of data points.

    Returns:
    float: The unfairness value between the two populations.

    Example:
    >>> data1 = np.array([5, 2, 4, 6, 1])
    >>> data2 = np.array([9, 6, 4, 7, 6])
    >>> diff = diff_quantile(data1, data2)
    >>> print(diff)
    3.9797979797979797
    """
    probs = np.linspace(0, 1, num=100)
    eqf1 = np.quantile(data1, probs)
    eqf2 = np.quantile(data2, probs)
    unfair_value = np.max(np.abs(eqf1-eqf2))
    return unfair_value


def unfairness(estimator, sensitive_features):
    """
    Compute the unfairness value for a given fair output and multiple sensitive attributes data contening several modalities.

    Parameters:
    estimator (array-like): Predicted (fair or not) output data.
    x_ssa_test (array-like): Sensitive attribute data.

    Returns:
    float: Unfairness value in the dataset.

    Example:
    >>> estimator = np.array([5, 0, 6, 7, 9])
    >>> sensitive_features = np.array([[1, 2, 1, 1, 2], [0, 1, 2, 1, 0]]).T
    >>> unf = unfairness(estimator, sensitive_features)
    >>> print(unf)
    6.0
    """
    new_list = []
    if len(np.shape(sensitive_features)) == 1:
        sens_val = list(set(sensitive_features))
        data1 = estimator
        lst_unfairness = []
        for mod in sens_val:
            data2 = estimator[sensitive_features == mod]
            lst_unfairness.append(diff_quantile(data1, data2))
        new_list.append(max(lst_unfairness))
    else :
        for sens in sensitive_features.T:
            sens_val = list(set(sens))
            data1 = estimator
            lst_unfairness = []
            for mod in sens_val:
                data2 = estimator[sens == mod]
                lst_unfairness.append(diff_quantile(data1, data2))
            new_list.append(max(lst_unfairness))
    return max(new_list)

def unfairness_multi(y_fair_dict, sensitive_features):
    """
    Compute unfairness values for multiple fair output datasets and multiple sensitive attributes datasets.

    Parameters:
    y_fair_dict (dict): A dictionary where keys represent sensitive features and values are arrays
            containing the fair predictions corresponding to each sensitive feature.
            Each sensitive feature's fairness adjustment is performed sequentially,
            ensuring that each feature is treated fairly relative to the previous ones.
    sensitive_features (array-like): Sensitive attribute data.

    Returns:
    dict: A dictionary containing unfairness values for each level of fairness.

    Example:
    >>> y_fair_dict = {'Base model':np.array([19,39,65]), 'sens_var_1':np.array([22,40,50]), 'sens_var_2':np.array([28,39,42])}
    >>> sensitive_features = np.array([['blue', 2], ['red', 9], ['green', 5]])
    >>> unfs_dict = unfairness_multi(y_fair_dict, sensitive_features)
    >>> print(unfs_dict)
    {'sens_var_0': 46.0, 'sens_var_1': 28.0, 'sens_var_2': 14.0}
    """
    unfairness_dict = {}
    for i, y_fair in enumerate(y_fair_dict.values()):
        result = unfairness(y_fair, sensitive_features)
        unfairness_dict[f'sens_var_{i}'] = result
    return unfairness_dict


def unfairness_multi_permutations(permut_y_fair_dict, all_combs_sensitive_features):
    """
    Compute unfairness values for multiple fair output datasets and multiple sensitive attribute datasets.

    Parameters:
    permut_y_fair_dict (dict): A dictionary containing permutations of fair output datasets.
    all_combs_sensitive_features (dict): A dictionary containing combinations of columns permutations for sensitive attribute datasets.

    Returns:
    list: A list of dictionaries containing unfairness values for each permutation of fair output datasets.

    Example:
    >>> permut_y_fair_dict = {(1,2): {'Base model':np.array([19,39,65]), 'sens_var_1':np.array([22,40,50]), 'sens_var_2':np.array([28,39,42])},
                               (2,1): {'Base model':np.array([19,39,65]), 'sens_var_2':np.array([34,39,60]), 'sens_var_1':np.array([28,39,42])}}
    >>> all_combs_sensitive_features = {(1,2): np.array([['blue', 2], ['red', 9], ['green', 5]]),
                               (2,1): np.array([[2, 'blue'], [9, 'red'], [5, 'green']])}
    >>> unfs_list = unfairness_multi_permutations(permut_y_fair_dict, all_combs_sensitive_features)
    >>> print(unfs_list)
    [{'sens_var_0': 46.0, 'sens_var_1': 28.0, 'sens_var_2': 14.0}, 
        {'sens_var_0': 46.0, 'sens_var_1': 26.0, 'sens_var_2': 14.0}]
    """
    unfs_list = []
    for key in permut_y_fair_dict.keys():
        unfs_list.append(unfairness_multi(
            permut_y_fair_dict[key], np.array(all_combs_sensitive_features[key])))
    return unfs_list