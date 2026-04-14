"""Abstract base class for regime detection model adapters."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class RegimeModelAdapter(ABC):
    """Interface for pluggable regime detection models."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Fit the model on training data."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict hidden states for given data."""

    @abstractmethod
    def score(self, X: np.ndarray) -> float:
        """Return log-likelihood of the data under the fitted model."""

    @property
    @abstractmethod
    def transition_matrix(self) -> np.ndarray | None:
        """Return the transition probability matrix if available."""

    @property
    @abstractmethod
    def n_states(self) -> int:
        """Number of hidden states."""

    @abstractmethod
    def get_state_means(self) -> np.ndarray:
        """Return per-state mean vectors."""

    @abstractmethod
    def get_state_covariances(self) -> np.ndarray:
        """Return per-state covariance matrices."""

    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        """Return model metadata for serialization."""

    def aic(self, X: np.ndarray) -> float:
        """Akaike Information Criterion."""
        ll = self.score(X)
        k = self._n_params()
        return float(2 * k - 2 * ll)

    def bic(self, X: np.ndarray) -> float:
        """Bayesian Information Criterion."""
        ll = self.score(X)
        n = X.shape[0]
        k = self._n_params()
        return float(k * np.log(n) - 2 * ll)

    @abstractmethod
    def _n_params(self) -> int:
        """Number of free parameters in the model."""
