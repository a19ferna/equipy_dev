from statsmodels.distributions.empirical_distribution import ECDF
from metrics._fairness_metrics import EQF
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
    methods for modality extraction, localization of modalities in the input data, weight calculation, and ECDF/EQF 
    estimation with random noise.
    """

    def __init__(self):
        self.ecdf = {}
        self.eqf = {}

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