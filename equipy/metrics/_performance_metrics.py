from sklearn.metrics import mean_squared_error

def performance(y_true, y_fair_dict, metric=mean_squared_error):
    """
    Compute the performance values for multiple fair output datasets compared to the true labels.

    Parameters:
    y_true (array-like): True labels or ground truth values.
    y_fair_dict (dict): A dictionary containing sequentally fair output datasets.
    classif (bool, optional): If True, assumes classification task and computes accuracy. 
                              If False (default), assumes regression task and computes mean squared error.

    Returns:
    dict: A dictionary containing performance values for sequentally fair output datasets.

    Example:
    >>> y_true = np.array([15, 38, 68])
    >>> y_fair_dict = {'Base model':np.array([19,39,65]), 'sens_var_1':np.array([22,40,50]), 'sens_var_2':np.array([28,39,42])}
    >>> performance_values = performance_multi(y_true, y_fair_dict, classif=False)
    >>> print(performance_values)
    {'Base model': 8.666666666666666, 'sens_var_1': 125.66666666666667, 'sens_var_2': 282.0}
    """
    performance_dict = {}
    for key in y_fair_dict.keys():
        performance_dict[key] = metric(y_true, list(y_fair_dict[key]))
    return performance_dict
