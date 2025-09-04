import math
import random
from dataclasses import fields
from dataclasses import is_dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np

from bayesian_search.bo_types import Categorical
from bayesian_search.bo_types import Integer
from bayesian_search.bo_types import ParamSpec
from bayesian_search.bo_types import Real


class SpaceEncoder:
    """Encode/decode a spec dataclass into a continuous search vector.

    - Real/Integer map to 1 dimension each (optionally in log-space).
    - Categorical maps to k dimensions via one-hot encoding.
    """

    def __init__(self, spec_cls: type):
        if not is_dataclass(spec_cls):
            raise TypeError("spec_cls must be a @dataclass.")
        self.spec_cls = spec_cls
        self._fields = fields(spec_cls)
        self._segments: List[Tuple[str, ParamSpec, int]] = []
        self._cat_maps: Dict[str, Dict[Any, int]] = {}
        dim = 0
        for f in self._fields:
            spec = getattr(spec_cls, f.name, None)
            if not isinstance(spec, (Real, Integer, Categorical)):
                raise TypeError(
                    f"Field '{f.name}' must be Real/Integer/Categorical (as class attribute)."
                )
            if isinstance(spec, Categorical):
                k = len(spec.choices)
                self._segments.append((f.name, spec, k))
                self._cat_maps[f.name] = {
                    c: i for i, c in enumerate(spec.choices)
                }
                dim += k
            else:
                self._segments.append((f.name, spec, 1))
                dim += 1
        self.dim = dim

    def sample(self, rng: random.Random) -> Any:
        """Draw a random spec instance respecting bounds and log-scales."""
        kwargs: Dict[str, Any] = {}
        for name, spec, _ in self._segments:
            if isinstance(spec, Real):
                if spec.log:
                    x = math.exp(
                        rng.uniform(math.log(spec.low), math.log(spec.high))
                    )
                else:
                    x = rng.uniform(spec.low, spec.high)
                kwargs[name] = x
            elif isinstance(spec, Integer):
                if spec.log:
                    x = int(
                        round(
                            math.exp(
                                rng.uniform(
                                    math.log(spec.low), math.log(spec.high)
                                )
                            )
                        )
                    )
                else:
                    x = rng.randint(spec.low, spec.high)
                kwargs[name] = int(np.clip(x, spec.low, spec.high))
            else:  # Categorical
                kwargs[name] = rng.choice(spec.choices)
        return self.spec_cls(**kwargs)

    def encode(self, obj: Any) -> np.ndarray:
        """Dataclass instance -> dense vector (categoricals as one-hot)."""
        vec: List[float] = []
        for name, spec, k in self._segments:
            val = getattr(obj, name)
            if isinstance(spec, Real):
                x = float(val)
                if spec.log:
                    x = math.log(max(x, 1e-12))
                vec.append(x)
            elif isinstance(spec, Integer):
                x = int(val)
                if spec.log:
                    x = math.log(max(x, 1))
                vec.append(float(x))
            else:
                one = [0.0] * k
                try:
                    idx = self._cat_maps[name][val]
                except KeyError as e:
                    raise KeyError(
                        f"Unknown categorical value '{val}' for field '{name}'."
                    ) from e
                one[idx] = 1.0
                vec.extend(one)
        return np.array(vec, dtype=float)

    def decode(self, vec: np.ndarray) -> Any:
        """Vector -> dataclass instance (categoricals via argmax)."""
        kwargs: Dict[str, Any] = {}
        i = 0
        for name, spec, k in self._segments:
            if isinstance(spec, Categorical):
                segment = vec[i : i + k]
                idx = int(np.argmax(segment))
                kwargs[name] = spec.choices[idx]
                i += k
            else:
                x = float(vec[i])
                if isinstance(spec, Real) and spec.log:
                    x = float(np.exp(x))
                if isinstance(spec, Integer):
                    if spec.log:
                        x = float(np.exp(x))
                    x = int(round(x))
                    x = int(np.clip(x, spec.low, spec.high))
                else:
                    x = float(np.clip(x, spec.low, spec.high))
                kwargs[name] = x
                i += 1
        return self.spec_cls(**kwargs)
