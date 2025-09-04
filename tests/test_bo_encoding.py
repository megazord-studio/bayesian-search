import math
import random
from dataclasses import dataclass

import numpy as np

from bayesian_search.bo_encoding import SpaceEncoder
from bayesian_search.bo_types import Categorical
from bayesian_search.bo_types import Integer
from bayesian_search.bo_types import Real


@dataclass
class Spec:
    a: Real = Real(1e-3, 1e-1, log=True)
    b: Integer = Integer(1, 10, log=False)
    c: Categorical = Categorical(("x", "y", "z"))


# Story: We define a search space with a log-real, an integer, and a categorical.
# We sample a point, encode it to a vector, and decode back—ensuring bounds,
# shapes, and round-trip behavior make sense.
def test_space_encoder_roundtrip_and_dims():
    enc = SpaceEncoder(Spec)
    assert enc.dim == 1 + 1 + 3

    rng = random.Random(123)
    sample = enc.sample(rng)

    # bounds respected
    assert 1e-3 <= float(sample.a) <= 1e-1
    assert 1 <= int(sample.b) <= 10
    assert sample.c in ("x", "y", "z")

    vec = enc.encode(sample)
    assert vec.shape == (enc.dim,)

    dec = enc.decode(vec)
    # Categorical should round-trip exactly via argmax
    assert dec.c == sample.c

    # Real log encoded -> encoding stores log, decode applies exp
    # We check that decode respects original bounds and is close to original
    assert 1e-3 <= float(dec.a) <= 1e-1
    assert abs(math.log(float(sample.a)) - math.log(float(dec.a))) < 1e-6

    # Integer without log should decode to same value after clipping/rounding
    assert int(dec.b) == int(sample.b)


# Story: For log-integer parameters, encoding stores log(k), decoding applies
# exp and rounds back. A sampled point should round-trip exactly within bounds.
def test_integer_log_encoding_roundtrip():
    @dataclass
    class Spec2:
        k: Integer = Integer(2, 100, log=True)

    enc = SpaceEncoder(Spec2)
    rng = random.Random(7)
    s = enc.sample(rng)
    v = enc.encode(s)
    d = enc.decode(v)

    assert 2 <= int(d.k) <= 100
    assert int(d.k) == int(s.k)


# Story: One-hot encoding should place each categorical label in a unique
# position. We verify the argmax index toggles across all labels.
def test_categorical_one_hot_encoding_positions():
    @dataclass
    class Spec3:
        x: Categorical = Categorical(("a", "b", "c", "d"))

    enc = SpaceEncoder(Spec3)
    from collections import Counter

    counts = Counter()
    for label in ("a", "b", "c", "d"):
        obj = Spec3(x=label)  # type: ignore[arg-type]
        vec = enc.encode(obj)
        idx = int(np.argmax(vec))
        counts[idx] += 1
    # All positions should be used exactly once
    assert sum(counts.values()) == 4
    assert len(counts) == 4
