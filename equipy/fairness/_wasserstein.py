from  fairness._base import BaseHelper
from utils.checkers import _check_epsilon, _check_epsilon_size, _check_mod, _check_shape, _check_nb_observations
import numpy as np

class Wasserstein(BaseHelper):
    """
    Class implementing Wasserstein distance-based fairness adjustment for binary classification tasks.

    Parameters
    ----------
    sigma : float, optional (default=0.0001)
        Standard deviation of the random noise added during fairness adjustment.

    Attributes
    ----------
    sigma : float
        Standard deviation of the random noise added during fairness adjustment.
    sens_val_calib : dict
        Dictionary storing modality values obtained from calibration data.
    weights : dict
        Dictionary storing weights (probabilities) for each modality based on their occurrences in calibration data.
    ecdf : dict
        Dictionary storing ECDF (Empirical Cumulative Distribution Function) objects for each sensitive modality.
    eqf : dict
        Dictionary storing EQF (Empirical Quantile Function) objects for each sensitive modality.

    Methods
    -------
    fit(y_calib, x_ssa_calib)
        Fit the fairness adjustment model using calibration data.
    transform(y_test, x_ssa_test, epsilon=0)
        Transform test data to enforce fairness using Wasserstein distance.
    """

    def __init__(self, sigma=0.0001):
        super().__init__()
        self.sigma = sigma
        self.sens_val_calib = None
        self.weights = None

    def fit(self, y_calib, x_ssa_calib):
        """
        Perform fit on the calibration data and save the ECDF, EQF, and weights of the sensitive variable.

        Parameters
        ----------
        y_calib : array-like, shape (n_samples,)
            The calibration labels.

        x_ssa_calib : array-like, shape (n_samples,)
            The calibration samples representing one single sensitive attribute.

        Returns
        -------
        None

        Notes
        -----
        This method computes the ECDF (Empirical Cumulative Distribution Function),
        EQF (Empirical Quantile Function), and weights for the sensitive variable
        based on the provided calibration data. These computed values are used
        during the transformation process to ensure fairness in predictions.

        Examples
        --------
        >>> wasserstein = Wasserstein(sigma=0.001)
        >>> y_calib = np.array([0.0, 1.0, 1.0, 0.0])
        >>> x_ssa_calib = np.array([1, 2, 0, 2])
        >>> wasserstein.fit(y_calib, x_ssa_calib)
        """
        _check_shape(y_calib, x_ssa_calib)

        self.sens_val_calib = self._get_mod(self, x_ssa_calib)
        self.weights = self._get_weights(self, x_ssa_calib)
        self._estimate_ecdf_eqf(self, y_calib, x_ssa_calib, self.sigma)

    def transform(self, y_test, x_ssa_test, epsilon=0):
        """
        Transform the test data to enforce fairness using Wasserstein distance.

        Parameters
        ----------
        y_test : array-like, shape (n_samples,)
            The target values of the test data.

        x_ssa_test : array-like, shape (n_samples,)
            The test samples representing a single sensitive attribute.

        epsilon : float, optional (default=0)
            The fairness parameter controlling the trade-off between fairness and accuracy.
            It represents the fraction of the original predictions retained after fairness adjustment.
            Epsilon should be a value between 0 and 1, where 0 means full fairness and 1 means no fairness constraint.

        Returns
        -------
        y_fair : array-like, shape (n_samples,)
            Fair predictions for the test data after enforcing fairness constraints.

        Notes
        -----
        This method applies Wasserstein distance-based fairness adjustment to the test data
        using the precomputed ECDF (Empirical Cumulative Distribution Function),
        EQF (Empirical Quantile Function), and weights obtained from the calibration data.
        Random noise within the range of [-sigma, sigma] is added to the test data to ensure fairness.
        The parameter epsilon controls the trade-off between fairness and accuracy,
        with 0 enforcing full fairness and 1 retaining the original predictions.

        Examples
        --------
        >>> y_calib = np.array([0.05, 0.08, 0.9, 0.9, 0.01, 0.88])
        >>> x_ssa_calib = np.array([1, 3, 2, 3, 1, 2])
        >>> wasserstein = Wasserstein(sigma=0.001)
        >>> wasserstein.fit(y_calib, x_ssa_calib)
        >>> y_test = np.array([0.01, 0.99, 0.98, 0.04])
        >>> x_ssa_test = np.array([3, 1, 2, 3])
        >>> print(wasserstein.transform(y_test, x_ssa_test, epsilon=0.2))
        [0.26063673 0.69140959 0.68940959 0.26663673]
        """

        _check_epsilon(epsilon)
        _check_shape(y_test, x_ssa_test)
        sens_val_test = self._get_mod(self, x_ssa_test)
        _check_mod(self.sens_val_calib, sens_val_test)

        sens_loc = self._get_loc(self, x_ssa_test)
        y_fair = np.zeros_like(y_test)
        eps = np.random.uniform(-self.sigma, self.sigma, len(y_test))
        for mod1 in sens_val_test:
            for mod2 in sens_val_test:
                y_fair[sens_loc[mod1]] += self.weights[mod2] * \
                    self.eqf[mod2](self.ecdf[mod1](
                        y_test[sens_loc[mod1]]+eps[sens_loc[mod1]]))

        return (1-epsilon)*y_fair + epsilon*y_test
""

class MultiWasserStein():
    """
    Class extending Wasserstein for multi-sensitive attribute fairness adjustment.

    Parameters
    ----------
    sigma : float, optional (default=0.0001)
        Standard deviation of the random noise added during fairness adjustment.

    Attributes
    ----------
    sigma : float
        Standard deviation of the random noise added during fairness adjustment.
    y_fair_test : dict
        Dictionary storing fair predictions for each sensitive feature.
    sens_val_calib_all : dict
        Dictionary storing modality values obtained from calibration data for all sensitive features.
    weights_all : dict
        Dictionary storing weights (probabilities) for each modality based on their occurrences in calibration data
        for all sensitive features.
    ecdf_all : dict
        Dictionary storing ECDF (Empirical Cumulative Distribution Function) objects for each sensitive modality
        for all sensitive features.
    eqf_all : dict
        Dictionary storing EQF (Empirical Quantile Function) objects for each sensitive modality
        for all sensitive features.

    Methods
    -------
    fit(y_calib, x_sa_calib)
        Fit the multi-sensitive attribute fairness adjustment model using calibration data.
    transform(y_test, x_sa_test, epsilon=None)
        Transform test data to enforce fairness using Wasserstein distance for multiple sensitive attributes.
    get_sequential_fairness()
        Get fair predictions for each sensitive feature, applied step by step.
    """

    def __init__(self, sigma=0.0001):
        """
        Initialize the MultiWasserStein instance.

        Parameters
        ----------
        sigma : float, optional (default=0.0001)
            The standard deviation of the random noise added to the data during transformation.

        Returns
        -------
        None
        """

        self.y_fair_test = {}

        self.sens_val_calib_all = {}
        self.weights_all = {}

        self.eqf_all = {}
        self.ecdf_all = {}


    def fit(self, y_calib, x_sa_calib):
        """
        Perform fit on the calibration data and save the ECDF, EQF, and weights for each sensitive variable.

        Parameters
        ----------
        y_calib : array-like, shape (n_samples,)
            The calibration labels.

        x_sa_calib : array-like, shape (n_samples, n_sensitive_features)
            The calibration samples representing multiple sensitive attributes.

        Returns
        -------
        None

        Notes
        -----
        This method computes the ECDF (Empirical Cumulative Distribution Function),
        EQF (Empirical Quantile Function), and weights for each sensitive variable
        based on the provided calibration data. These computed values are used
        during the transformation process to ensure fairness in predictions.
        """
        _check_nb_observations(x_sa_calib)

        for i, sens in enumerate(x_sa_calib.T):
            wasserstein_instance = Wasserstein(sigma=self.sigma)
            if i == 0:
                y_calib_inter = y_calib
            
            wasserstein_instance.fit(y_calib_inter, sens)
            self.sens_val_calib_all[f'sens_var_{i+1}'] = wasserstein_instance.sens_val_calib
            self.weights_all[f'sens_var_{i+1}'] = wasserstein_instance.weights
            self.eqf_all[f'sens_var_{i+1}'] = wasserstein_instance.eqf
            self.ecdf_all[f'sens_var_{i+1}'] = wasserstein_instance.ecdf
            y_calib_inter = wasserstein_instance.transform(y_calib_inter, sens)

    def transform(self, y_test, x_sa_test, epsilon=None):
        """
        Transform the test data to enforce fairness using Wasserstein distance.

        Parameters
        ----------
        y_test : array-like, shape (n_samples,)
            The target values of the test data.

        x_sa_test : array-like, shape (n_samples, n_sensitive_features)
            The test samples representing multiple sensitive attributes.

        epsilon : list, shape (n_sensitive_features,), optional (default=None)
            The fairness parameters controlling the trade-off between fairness and accuracy
            for each sensitive feature. If None, no fairness constraints are applied.

        Returns
        -------
        y_fair : array-like, shape (n_samples,)
            Fair predictions for the test data after enforcing fairness constraints.

        Notes
        -----
        This method applies Wasserstein distance-based fairness adjustment to the test data
        using the precomputed ECDF (Empirical Cumulative Distribution Function),
        EQF (Empirical Quantile Function), and weights obtained from the calibration data.
        Random noise within the range of [-sigma, sigma] is added to the test data to ensure fairness.
        The parameter epsilon is a list, where each element controls the trade-off between fairness and accuracy
        for the corresponding sensitive feature.

        Examples
        --------
        >>> wasserstein = MultiWasserStein(sigma=0.001)
        >>> y_calib = np.array([0.6, 0.43, 0.32, 0.8])
        >>> x_sa_calib = np.array([['blue', 5], ['blue', 9], ['green', 5], ['green', 9]])
        >>> wasserstein.fit(y_calib, x_sa_calib)
        >>> y_test = [0.8, 0.35, 0.23, 0.2]
        >>> x_sa_test = np.array([['blue', 9], ['blue', 5], ['blue', 5], ['green', 9]])
        >>> epsilon = [0.1, 0.2] 
        >>> fair_predictions = wasserstein.transform(y_test, x_sa_test, epsilon=epsilon)
        >>> print(fair_predictions)
        [0.7015008  0.37444565 0.37204565 0.37144565]
        """
        if epsilon == None:
            if x_sa_test.ndim == 1:
                epsilon = [0]
            else:
                epsilon = [0]*np.shape(x_sa_test)[1]
        _check_epsilon_size(epsilon, x_sa_test)

        self.y_fair_test['Base model'] = y_test

        for i, sens in enumerate(x_sa_test.T):
            wasserstein_instance = Wasserstein(sigma=self.sigma)
            if i == 0:
                y_test_inter = y_test
            wasserstein_instance.sens_val_calib = self.sens_val_calib_all[
                f'sens_var_{i+1}']
            wasserstein_instance.weights = self.weights_all[f'sens_var_{i+1}']
            wasserstein_instance.eqf = self.eqf_all[f'sens_var_{i+1}']
            wasserstein_instance.ecdf = self.ecdf_all[f'sens_var_{i+1}']
            y_test_inter = wasserstein_instance.transform(
                y_test_inter, sens, epsilon[i])
            self.y_fair_test[f'sens_var_{i+1}'] = y_test_inter
        return self.y_fair_test[f'sens_var_{i+1}']

    def get_sequential_fairness(self):
        """
        Get the dictionary of fair predictions for each sensitive feature, applied step by step.

        Returns
        -------
        dict
            A dictionary where keys represent sensitive features and values are arrays
            containing the fair predictions corresponding to each sensitive feature.
            Each sensitive feature's fairness adjustment is performed sequentially,
            ensuring that each feature is treated fairly relative to the previous ones.

        Notes
        -----
        This method returns fair predictions for each sensitive feature, applying fairness constraints
        sequentially. The first sensitive feature is adjusted for fairness, and then subsequent features
        are adjusted in sequence, ensuring that each feature is treated fairly relative to the previous ones.

        Examples
        --------
        >>> wasserstein = MultiWasserStein(sigma=0.001)
        >>> y_calib = np.array([0.6, 0.43, 0.32, 0.8])
        >>> x_sa_calib = np.array([['blue', 5], ['blue', 9], ['green', 5], ['green', 9]])
        >>> wasserstein.fit(y_calib, x_sa_calib)
        >>> y_test = np.array([0.8, 0.35, 0.23, 0.2])
        >>> x_sa_test = np.array([['blue', 9], ['blue', 5], ['blue', 5], ['green', 9]])
        >>> epsilon = [0.1, 0.2]  
        >>> fair_predictions = wasserstein.transform(y_test, x_sa_test, epsilon=epsilon)
        >>> sequential_fairness = wasserstein.get_sequential_fairness()
        >>> print(sequential_fairness)
        {'Base model': array([0.8 , 0.35, 0.23, 0.2 ]), 
            'sens_var_1': array([0.71026749, 0.37278694, 0.36078694, 0.35778694]), 
            'sens_var_2': array([0.63767235, 0.44038334, 0.43798334, 0.43738334])}
        """
        return self.y_fair_test