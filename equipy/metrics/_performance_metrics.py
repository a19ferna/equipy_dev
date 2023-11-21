def risk(y_true, y_predict, classif=False):
    """
    Compute the risk value for predicted fair output compared to the true labels.

    Parameters:
    y_true (array-like): True labels or ground truth values.
    y_predict (array-like): Predicted (fair or not) output values.
    classif (bool, optional): If True, assumes classification task and computes accuracy. 
                              If False (default), assumes regression task and computes mean squared error.

    Returns:
    float: The calculated risk value.

    Example:
    >>> y_true = np.array([1, 0, 1, 1, 0])
    >>> y_predict = np.array([0, 1, 1, 1, 0])
    >>> classification_risk = risk(y_true, y_predict, classif=True)
    >>> print(classification_risk)
    0.6

    >>> y_true = [1.2, 2.5, 3.8, 4.0, 5.2]
    >>> y_predict = [1.0, 2.7, 3.5, 4.2, 5.0]
    >>> regression_risk = risk(y_true, y_predict)
    >>> print(regression_risk)
    0.05
    """
    if classif:
        return accuracy_score(y_true, y_predict)
    else:
        return mean_squared_error(y_true, y_predict)


def risk_multi(y_true, y_fair_dict, classif=False):
    """
    Compute the risk values for multiple fair output datasets compared to the true labels.

    Parameters:
    y_true (array-like): True labels or ground truth values.
    y_fair_dict (dict): A dictionary containing sequentally fair output datasets.
    classif (bool, optional): If True, assumes classification task and computes accuracy. 
                              If False (default), assumes regression task and computes mean squared error.

    Returns:
    dict: A dictionary containing risk values for sequentally fair output datasets.

    Example:
    >>> y_true = np.array([15, 38, 68])
    >>> y_fair_dict = {'Base model':np.array([19,39,65]), 'sens_var_1':np.array([22,40,50]), 'sens_var_2':np.array([28,39,42])}
    >>> risk_values = risk_multi(y_true, y_fair_dict, classif=False)
    >>> print(risk_values)
    {'Base model': 8.666666666666666, 'sens_var_1': 125.66666666666667, 'sens_var_2': 282.0}
    """
    risk_dict = {}
    for key in y_fair_dict.keys():
        risk_dict[key] = risk(y_true, list(y_fair_dict[key]), classif)
    return risk_dict


def risk_multi_permutations(y_true, permut_y_fair_dict, classif=False):
    """
    Compute the risk values for multiple fair output datasets compared to the true labels, considering permutations.

    Parameters:
    y_true (array-like): True labels or ground truth values.
    permut_y_fair_dict (dict): A dictionary containing permutations of fair output datasets.
    classif (bool, optional): If True, assumes classification task and computes accuracy. 
                              If False (default), assumes regression task and computes mean squared error.

    Returns:
    list: A list of dictionaries containing risk values for each permutation of fair output datasets.

    Example:
    >>> y_true = np.array([15, 38, 68])
    >>> permut_y_fair_dict = {(1,2): {'Base model':np.array([19,39,65]), 'sens_var_1':np.array([22,40,50]), 'sens_var_2':np.array([28,39,42])},
                               (2,1): {'Base model':np.array([19,39,65]), 'sens_var_2':np.array([34,39,60]), 'sens_var_1':np.array([28,39,42])}}
    >>> risk_values = risk_multi_permutations(y_true, permut_y_fair_dict, classif=False)
    [{'Base model': 8.666666666666666, 'sens_var_1': 125.66666666666667, 'sens_var_2': 282.0}, 
        {'Base model': 8.666666666666666, 'sens_var_2': 142.0, 'sens_var_1': 282.0}]
    """
    risk_list = []
    for key in permut_y_fair_dict.keys():
        risk_list.append(risk_multi(y_true, permut_y_fair_dict[key], classif))
    return risk_list