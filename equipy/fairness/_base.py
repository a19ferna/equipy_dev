from statsmodels.distributions.empirical_distribution import ECDF
from equipy.metrics._fairness_metrics import EQF
import numpy as np


class BaseHelper():
    """
    Base class providing helper methods for Wasserstein distance-based fairness adjustment.

    Attributes
    ----------
    ecdf : dict
        Dictionary storing ECDF (Empirical Cumulative Distribution Function) objects for each sensitive modality.
    eqf : dict
        Dictionary storing EQF (Empirical Quantile Function) objects for each sensitive modality.

    Methods
    -------
    _check_shape(y, x_ssa)
        Check the shape and data types of input arrays y and x_ssa.
    _check_mod(sens_val_calib, sens_val_test)
        Check if modalities in test data are included in calibration data's modalities.
    _check_epsilon(epsilon)
        Check if epsilon (fairness parameter) is within the valid range [0, 1].
    _get_mod(x_ssa)
        Get unique modalities from the input sensitive attribute array.
    _get_loc(x_ssa)
        Get the indices of occurrences for each modality in the input sensitive attribute array.
    _get_weights(x_ssa)
        Calculate weights (probabilities) for each modality based on their occurrences.
    _estimate_ecdf_eqf(y, x_ssa, sigma)
        Estimate ECDF and EQF for each modality, incorporating random noise within [-sigma, sigma].

    Notes
    -----
    This base class provides essential methods for Wasserstein distance-based fairness adjustment. It includes
    methods for shape validation, modality checks, epsilon validation, modality extraction, localization of
    modalities in the input data, weight calculation, and ECDF/EQF estimation with random noise.
    """

    def __init__(self):
        self.ecdf = {}
        self.eqf = {}

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

    def _get_mod(self, x_ssa):
        """
        Get unique modalities from the input sensitive attribute array.

        Parameters
        ----------
        x_ssa : array-like, shape (n_samples,)
            Input samples representing the sensitive attributes.

        Returns
        -------
        list
            List of unique modalities present in the input sensitive attribute array.
        """
        return list(set(x_ssa))

    def _get_loc(self, x_ssa):
        """
        Get the indices of occurrences for each modality in the input sensitive attribute array.

        Parameters
        ----------
        x_ssa : array-like, shape (n_samples,)
            Input sample representing the sensitive attribute.

        Returns
        -------
        dict
            Dictionary where keys are modalities and values are arrays containing their indices.
        """
        sens_loc = {}
        for mod in self._get_mod(x_ssa):
            sens_loc[mod] = np.where(x_ssa == mod)[0]
        return sens_loc

    def _get_weights(self, x_ssa):
        """
        Calculate weights (probabilities) for each modality based on their occurrences.

        Parameters
        ----------
        x_ssa : array-like, shape (n_samples,)
            Input samples representing the sensitive attribute.

        Returns
        -------
        dict
            Dictionary where keys are modalities and values are their corresponding weights.
        """
        sens_loc = self._get_loc(x_ssa)
        weights = {}
        for mod in self._get_mod(x_ssa):
            # Calculate probabilities
            weights[mod] = len(sens_loc[mod])/len(x_ssa)
        return weights

    def _estimate_ecdf_eqf(self, y, x_ssa, sigma):
        """
        Estimate ECDF and EQF for each modality, incorporating random noise within [-sigma, sigma].

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target values corresponding to the sensitive attribute array.
        x_ssa : array-like, shape (n_samples,)
            Input samples representing the sensitive attribute.
        sigma : float
            Standard deviation of the random noise added to the data.

        Returns
        -------
        None
        """
        sens_loc = self._get_loc(x_ssa)
        eps = np.random.uniform(-sigma, sigma, len(y))
        for mod in self._get_mod(x_ssa):
            # Fit the ecdf and eqf objects
            self.ecdf[mod] = ECDF(y[sens_loc[mod]] +
                                  eps[sens_loc[mod]])
            self.eqf[mod] = EQF(y[sens_loc[mod]]+eps[sens_loc[mod]])