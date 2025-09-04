"""Core type definitions and configuration for Bayesian search.

The parameter spec dataclasses are intentionally frozen (immutable) to make
search spaces declarative and safe to share across code without accidental
mutation.
"""

from dataclasses import dataclass
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union


@dataclass(frozen=True)
class Real:
    """A continuous parameter in [low, high].

    When ``log`` is True, sampling and encoding are carried out in log-space.
    """

    low: float
    high: float
    log: bool = False  # if True, sampled/log-encoded on a log scale


@dataclass(frozen=True)
class Integer:
    """An integer parameter in [low, high].

    When ``log`` is True, sampling and encoding are carried out in log-space
    and values are rounded to the nearest integer and clipped to bounds.
    """

    low: int
    high: int
    log: bool = False  # if True, sampled/log-encoded on a log scale


@dataclass(frozen=True)
class Categorical:
    """A categorical parameter with a fixed set of choices."""

    choices: Tuple[Any, ...]


ParamSpec = Union[Real, Integer, Categorical]


@dataclass
class BayesSearchConfig:
    """Configuration for the high-level Bayesian optimization loop.

    Attributes mirror common GP + EI options and iteration controls.
    """

    n_init: int = 10
    n_iter: int = 40
    candidate_pool: int = 2048
    lengthscale: float = 1.0
    variance: float = 1.0
    noise: float = 1e-6
    xi: float = 0.01
    seed: Optional[int] = None
    # Optional progress callback hooks; kept out of tests unless provided.
    # Signature: on_event(event: str, state: dict) -> None
    # Events: "start", "init", "iter", "end"
    callback: Optional[Any] = None
