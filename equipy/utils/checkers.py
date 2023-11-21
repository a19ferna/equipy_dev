import numpy as np 

def _check_metric(y_true):
    if np.all(np.isin(y_true, [0,1])):
        raise Warning("You used mean squared error as metric but it looks like you are using classification scores")
    
def _check_shape(y, x_ssa):
    """
    Check the shape and data types of input arrays y and x_ssa.

    Parameters
    ----------
    y : array-like
        Target values of the data.
    x_ssa : array-like
        Input samples representing the sensitive attribute.

    Raises
    ------
    ValueError
        If the input arrays have incorrect shapes or data types.
    """
    if not isinstance(x_ssa, np.ndarray):
        raise ValueError('x_sa must be an array')

    if not isinstance(y, np.ndarray):
        raise ValueError('y must be an array')

    if len(x_ssa) != len(y):
        raise ValueError('x_sa and y should have the same length')

    for el in y:
        if not isinstance(el, float):
            raise ValueError('y should contain only float numbers')

def _check_mod(sens_val_calib, sens_val_test):
    """
    Check if modalities in test data are included in calibration data's modalities.

    Parameters
    ----------
    sens_val_calib : list
        Modalities from the calibration data.
    sens_val_test : list
        Modalities from the test data.

    Raises
    ------
    ValueError
        If modalities in test data are not present in calibration data.
    """
    if not all(elem in sens_val_calib for elem in sens_val_test):
        raise ValueError(
            'Modalities in x_ssa_test should be included in modalities of x_sa_calib')

def _check_epsilon(epsilon):
    """
    Check if epsilon (fairness parameter) is within the valid range [0, 1].

    Parameters
    ----------
    epsilon : float
        Fairness parameter controlling the trade-off between fairness and accuracy.

    Raises
    ------
    ValueError
        If epsilon is outside the valid range [0, 1].
    """
    if epsilon < 0 or epsilon > 1:
        raise ValueError(
            'epsilon must be between 0 and 1')
    
def _check_epsilon_size(self, epsilon, x_sa_test):
    """
    Check if the epsilon list matches the number of sensitive features.

    Parameters
    ----------
    epsilon : list, shape (n_sensitive_features,)
        Fairness parameters controlling the trade-off between fairness and accuracy for each sensitive feature.

    x_sa_test : array-like, shape (n_samples, n_sensitive_features)
        Test samples representing multiple sensitive attributes.

    Raises
    ------
    ValueError
        If the length of epsilon does not match the number of sensitive features.
    """

    if x_sa_test.ndim == 1:
        if len(epsilon) != 1:
            raise ValueError(
                'epsilon must have the same length than the number of sensitive features')
    else:
        if len(epsilon) != np.shape(x_sa_test)[1]:
            raise ValueError(
                    'epsilon must have the same length than the number of sensitive features')