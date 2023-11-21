import numpy as np 

def _check_metric(y_true):
    if np.all(np.isin(y_true, [0,1])):
        raise Warning("You used mean squared error as metric but it looks like you are using classification scores")