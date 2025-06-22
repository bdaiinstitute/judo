# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from typing import TypedDict, Union

import numpy as np


class MinMaxNormalizerKwargs(TypedDict, total=False):
    """Kwargs for MinMaxNormalizer."""

    min: np.ndarray
    max: np.ndarray
    eps: float


class RunningMeanStdNormalizerKwargs(TypedDict, total=False):
    """Kwargs for RunningMeanStdNormalizer."""

    momentum: float
    init_std: float
    min_std: float
    max_std: float
    eps: float


class IdentityNormalizerKwargs(TypedDict, total=False):
    """Kwargs for IdentityNormalizer."""

    pass


NormalizerKwargs = Union[IdentityNormalizerKwargs, MinMaxNormalizerKwargs, RunningMeanStdNormalizerKwargs]


class Normalizer:
    """Base class for normalizers."""

    def __init__(self, dim: int) -> None:
        """Initialize the normalizer.

        Args:
            dim: Dimension of the data.
        """
        self.dim = dim

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize the data.

        Args:
            x: The data to normalize. Shape=(..., dim).
        """
        raise NotImplementedError

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Denormalize the data.

        Args:
            x: The data to denormalize. Shape=(..., dim).
        """
        raise NotImplementedError

    def update(self, x: np.ndarray) -> None:
        """Update the normalizer.

        Args:
            x: The data to update the normalizer with. Shape=(batch_size, dim).
        """
        pass


class IdentityNormalizer(Normalizer):
    """Normalizer that does nothing."""

    def __init__(self, dim: int) -> None:
        """Initialize the normalizer."""
        super().__init__(dim)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Return the data as is."""
        return x

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Return the data as is."""
        return x


class MinMaxNormalizer(Normalizer):
    """Normalizer that uses min and max values to scale the data to the range [-1, 1]."""

    def __init__(self, dim: int, min: np.ndarray, max: np.ndarray, eps: float = 1e-8) -> None:
        """Initialize the normalizer.

        Args:
            dim: Dimension of the data.
            min: The minimum values of the data.
            max: The maximum values of the data.
            eps: Small value to prevent division by zero.
        """
        super().__init__(dim)
        self.min = min
        self.max = max
        self.eps = eps

        # Only normalize the dimensions that are not -inf or inf
        self.norm_dims = np.where((self.min != -np.inf) & (self.max != np.inf))[0]

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize the data."""
        x_normalized = x.copy()
        x_normalized[..., self.norm_dims] = (
            2
            * (x[..., self.norm_dims] - self.min[self.norm_dims])
            / (self.max[self.norm_dims] - self.min[self.norm_dims])
            - 1
        )
        return x_normalized

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Denormalize the data."""
        x_denormalized = x.copy()
        x_denormalized[..., self.norm_dims] = (x[..., self.norm_dims] + 1) * (
            self.max[self.norm_dims] - self.min[self.norm_dims]
        ) / 2 + self.min[self.norm_dims]
        return x_denormalized


class RunningMeanStdNormalizer(Normalizer):
    """Normalizer that uses running statistics (mean and std) to scale the data."""

    def __init__(
        self,
        dim: int,
        momentum: float = 0.99,
        init_std: float = 1.0,
        min_std: float = 1e-6,
        max_std: float = 1e6,
        eps: float = 1e-8,
    ) -> None:
        """Initialize the normalizer.

        Args:
            dim: Dimension of the data.
            momentum: The momentum for exponential moving average.
            init_std: The initial standard deviation for the running statistics.
            min_std: The minimum standard deviation.
            max_std: The maximum standard deviation.
            eps: Small value to prevent division by zero.
        """
        super().__init__(dim)
        self.eps = eps
        self.momentum = momentum
        self.min_std = min_std
        self.max_std = max_std

        # Running statistics
        self.count = 0
        self.mean = np.zeros(dim)
        self.std = np.ones(dim) * init_std

        # For Welford's online algorithm. M2: sum of squares of differences from the current mean
        self.M2 = np.zeros(dim)

    def update(self, x: np.ndarray) -> None:
        """Update the running statistics.

        Args:
            x: The data to update the running statistics with. Shape=(batch_size, dim).
        """
        assert x.ndim == 2, f"Expected 2D array (batch_size, dim), but got {x.ndim}D array {x.shape}"
        assert x.shape[1] == self.dim, f"Expected dimension {self.dim}, but got {x.shape[1]}"

        batch_size = x.shape[0]
        self.count += batch_size

        # new values - old mean
        delta = x - self.mean

        # Update mean
        self.mean += np.sum(delta, axis=0) / self.count

        # new values - new mean
        delta2 = x - self.mean

        # Update M2
        self.M2 += np.sum(delta * delta2, axis=0)
        self.M2 = np.maximum(self.M2, 0)

        # Update std
        self.std = np.sqrt(self.M2 / self.count)
        self.std = np.clip(self.std, self.min_std, self.max_std)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize the data."""
        return (x - self.mean) / (self.std + self.eps)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Denormalize the data."""
        return x * self.std + self.mean


normalizer_registry = {
    "identity": IdentityNormalizer,
    "min_max": MinMaxNormalizer,
    "running_mean_std": RunningMeanStdNormalizer,
}


def make_normalizer(normalizer_type: str, dim: int, **kwargs: NormalizerKwargs) -> Normalizer:
    """Make a normalizer from a string."""
    if normalizer_type not in normalizer_registry:
        raise ValueError(f"Invalid normalizer type: {normalizer_type}")
    return normalizer_registry[normalizer_type](dim, **kwargs)
