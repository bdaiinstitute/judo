# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np

NormalizerType = Literal["none", "min_max", "running"]


@dataclass
class MinMaxNormalizerConfig:
    """Config for MinMaxNormalizer."""

    min: np.ndarray
    max: np.ndarray
    eps: float


@dataclass
class RunningMeanStdNormalizerConfig:
    """Config for RunningMeanStdNormalizer."""

    init_std: float
    min_std: float
    max_std: float
    eps: float


@dataclass
class IdentityNormalizerConfig:
    """Config for IdentityNormalizer."""


NormalizerConfig = Union[IdentityNormalizerConfig, MinMaxNormalizerConfig, RunningMeanStdNormalizerConfig]


class Normalizer(ABC):
    """Base class for normalizers."""

    def __init__(self, dim: int) -> None:
        """Initialize the normalizer.

        Args:
            dim: Dimension of the data.
        """
        self.dim = dim

    @abstractmethod
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize the data.

        Args:
            x: The data to normalize. Shape=(..., dim).
        """
        raise NotImplementedError

    @abstractmethod
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
        return None


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

    def __init__(self, dim: int, min: np.ndarray, max: np.ndarray, eps: float = 1e-6) -> None:
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
        if len(self.norm_dims) != dim:
            excluded_dims = np.where((self.min == -np.inf) | (self.max == np.inf))[0]
            warnings.warn(
                f"MinMaxNormalizer: {len(excluded_dims)} action dimensions ({excluded_dims.tolist()}) have infinite range "
                f"and will not be normalized. Please check your model description for proper ctrlrange. "
                f"Consider using RunningMeanStdNormalizer instead for automatic range estimation.",
                UserWarning,
                stacklevel=2,
            )

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize the data."""
        x_normalized = x.copy()
        min_vals = self.min[self.norm_dims]
        max_vals = self.max[self.norm_dims]
        x_normalized[..., self.norm_dims] = 2 * (x[..., self.norm_dims] - min_vals) / (max_vals - min_vals) - 1
        return x_normalized

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Denormalize the data."""
        x_denormalized = x.copy()
        min_vals = self.min[self.norm_dims]
        max_vals = self.max[self.norm_dims]
        x_denormalized[..., self.norm_dims] = (x[..., self.norm_dims] + 1) * (max_vals - min_vals) / 2 + min_vals
        return x_denormalized


class RunningMeanStdNormalizer(Normalizer):
    """Normalizer that uses running statistics (mean and std) to scale the data elementwise.

    Each dimension is normalized using its own statistics independently, without using a full covariance matrix.
    """

    def __init__(
        self,
        dim: int,
        init_std: float = 1.0,
        min_std: float = 1e-5,
        max_std: float = 1e3,
        eps: float = 1e-6,
    ) -> None:
        """Initialize the normalizer.

        Args:
            dim: Dimension of the data.
            init_std: The initial standard deviation for the running statistics.
            min_std: The minimum standard deviation.
            max_std: The maximum standard deviation.
            eps: Small value to prevent division by zero.
        """
        super().__init__(dim)
        self.eps = eps
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
        assert x.shape[-1] == self.dim, f"Expected dimension {self.dim}, but got {x.shape[-1]}"

        batch_dims = x.shape[:-1]
        batch_axis = tuple(range(len(batch_dims)))
        batch_size = np.prod(batch_dims)
        self.count += batch_size

        # Welford's online algorithm
        delta = x - self.mean
        self.mean += np.sum(delta, axis=batch_axis) / self.count
        delta2 = x - self.mean

        # Update M2
        self.M2 += np.sum(delta * delta2, axis=batch_axis)
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
    "none": IdentityNormalizer,
    "min_max": MinMaxNormalizer,
    "running": RunningMeanStdNormalizer,
}


def make_normalizer(normalizer_type: NormalizerType, dim: int, **kwargs: NormalizerConfig) -> Normalizer:
    """Make a normalizer from a string."""
    if normalizer_type not in normalizer_registry:
        raise ValueError(f"Invalid normalizer type: {normalizer_type}")
    return normalizer_registry[normalizer_type](dim, **kwargs)
