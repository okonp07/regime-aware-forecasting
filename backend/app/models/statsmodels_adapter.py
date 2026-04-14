"""Markov Switching adapter using statsmodels (optional secondary model)."""

from typing import Any

import numpy as np

from backend.app.models.base import RegimeModelAdapter

try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    HAS_MARKOV = True
except ImportError:
    HAS_MARKOV = False


class MarkovRegressionAdapter(RegimeModelAdapter):
    """Wraps statsmodels MarkovRegression for regime detection.

    This model operates on a single endogenous series (e.g. log returns)
    with switching mean and variance.
    """

    def __init__(
        self,
        n_states: int = 3,
        random_seed: int = 42,
        n_iter: int = 200,
    ):
        if not HAS_MARKOV:
            raise ImportError(
                "statsmodels MarkovRegression not available. "
                "Install statsmodels>=0.14 to use this adapter."
            )
        self._n = n_states
        self._seed = random_seed
        self._n_iter = n_iter
        self._model = None
        self._result = None
        self._n_features = 1

    def fit(self, X: np.ndarray) -> None:
        series = X[:, 0] if X.ndim > 1 else X
        np.random.seed(self._seed)
        self._model = MarkovRegression(
            series,
            k_regimes=self._n,
            switching_variance=True,
        )
        self._result = self._model.fit(maxiter=self._n_iter, disp=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("Model not fitted")
        probs = self._result.smoothed_marginal_probabilities
        n_obs = X.shape[0] if X.ndim > 1 else len(X)
        if len(probs) >= n_obs:
            return np.argmax(probs.values[-n_obs:], axis=1)
        return np.argmax(probs.values, axis=1)

    def score(self, X: np.ndarray) -> float:
        if self._result is None:
            return float("-inf")
        return float(self._result.llf)

    @property
    def transition_matrix(self) -> np.ndarray | None:
        if self._result is None:
            return None
        n = self._n
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                key = f"p[{i}->{j}]"
                if key in self._result.params.index:
                    mat[i, j] = self._result.params[key]
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return mat / row_sums

    @property
    def n_states(self) -> int:
        return self._n

    def get_state_means(self) -> np.ndarray:
        if self._result is None:
            return np.zeros((self._n, 1))
        means = []
        for i in range(self._n):
            key = f"const[{i}]"
            if key in self._result.params.index:
                means.append(self._result.params[key])
            else:
                means.append(0.0)
        return np.array(means).reshape(-1, 1)

    def get_state_covariances(self) -> np.ndarray:
        if self._result is None:
            return np.ones((self._n, 1, 1))
        covs = []
        for i in range(self._n):
            key = f"sigma2[{i}]"
            if key in self._result.params.index:
                covs.append(self._result.params[key])
            else:
                covs.append(1.0)
        return np.array(covs).reshape(-1, 1, 1)

    def get_info(self) -> dict[str, Any]:
        return {
            "model_type": "MarkovRegression",
            "n_states": self._n,
            "random_seed": self._seed,
            "n_iter": self._n_iter,
            "aic": float(self._result.aic) if self._result else None,
            "bic": float(self._result.bic) if self._result else None,
        }

    def _n_params(self) -> int:
        if self._result is not None:
            return int(self._result.df_model)
        return self._n * 3
