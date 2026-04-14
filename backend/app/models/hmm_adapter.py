"""Gaussian HMM adapter using hmmlearn."""

from typing import Any

import numpy as np
from hmmlearn.hmm import GaussianHMM

from backend.app.models.base import RegimeModelAdapter


class HMMAdapter(RegimeModelAdapter):
    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "full",
        n_iter: int = 200,
        tol: float = 1e-4,
        random_seed: int = 42,
    ):
        self._n = n_states
        self._cov_type = covariance_type
        self._n_iter = n_iter
        self._tol = tol
        self._seed = random_seed
        self._model: GaussianHMM | None = None
        self._n_features: int = 0

    def fit(self, X: np.ndarray) -> None:
        self._n_features = X.shape[1] if X.ndim > 1 else 1
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._model = GaussianHMM(
            n_components=self._n,
            covariance_type=self._cov_type,
            n_iter=self._n_iter,
            tol=self._tol,
            random_state=self._seed,
        )
        self._model.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self._model.predict(X)

    def score(self, X: np.ndarray) -> float:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return float(self._model.score(X))

    @property
    def transition_matrix(self) -> np.ndarray | None:
        if self._model is None:
            return None
        return self._model.transmat_

    @property
    def n_states(self) -> int:
        return self._n

    def get_state_means(self) -> np.ndarray:
        return self._model.means_

    def get_state_covariances(self) -> np.ndarray:
        return self._model.covars_

    def get_info(self) -> dict[str, Any]:
        return {
            "model_type": "GaussianHMM",
            "n_states": self._n,
            "covariance_type": self._cov_type,
            "n_iter": self._n_iter,
            "tol": self._tol,
            "random_seed": self._seed,
            "converged": getattr(self._model, "monitor_", None) is not None,
        }

    def _n_params(self) -> int:
        k = self._n_features
        n = self._n
        start_probs = n - 1
        trans = n * (n - 1)
        means = n * k
        if self._cov_type == "full":
            covs = n * k * (k + 1) // 2
        elif self._cov_type == "diag":
            covs = n * k
        elif self._cov_type == "spherical":
            covs = n
        else:
            covs = k * (k + 1) // 2
        return start_probs + trans + means + covs
