import math
from dataclasses import dataclass

import numpy as np
import pytest

from bayesian_search.bo_encoding import SpaceEncoder
from bayesian_search.bo_types import Categorical
from bayesian_search.bo_types import Integer
from bayesian_search.bo_types import Real


# Story: The encoder should only accept proper dataclass specs. If we hand it
# a plain class, it should refuse and explain via TypeError.
def test_space_encoder_requires_dataclass():
    class NotADataclass:
        a = Real(0.0, 1.0)

    with pytest.raises(TypeError):
        SpaceEncoder(NotADataclass)


# Story: Even if a class is a dataclass, its class attributes must be
# parameter specs (Real/Integer/Categorical). Here we deliberately set a plain
# float to show the encoder catches schema mismatches.
def test_space_encoder_requires_param_specs_as_class_attributes():
    @dataclass
    class Bad:
        # Field exists but class attribute missing or wrong type
        a: float = 0.5  # not a Real/Integer/Categorical spec

    with pytest.raises(TypeError):
        SpaceEncoder(Bad)


# Story: When decoding, out-of-bounds numbers should be gracefully clipped
# back into the declared [low, high] ranges for both Reals and Integers.
# We craft vectors far outside to make the clipping obvious.
def test_decode_clips_real_and_integer_bounds():
    @dataclass
    class S:
        r: Real = Real(-1.0, 1.0, log=False)
        i: Integer = Integer(1, 5, log=False)

    enc = SpaceEncoder(S)

    # Build a vector that is out of bounds for both
    vec = np.array([10.0, -100.0], dtype=float)
    dec = enc.decode(vec)
    assert math.isclose(float(dec.r), 1.0)
    assert int(dec.i) == 1

    vec2 = np.array([-5.0, 99.0], dtype=float)
    dec2 = enc.decode(vec2)
    assert math.isclose(float(dec2.r), -1.0)
    assert int(dec2.i) == 5


# Story: Integer with log=True means the encoded number lives in log-space.
# Decoding applies exp, then rounds and clips. We test extreme lows/highs
# and a mid value to verify rounding behavior.
def test_decode_integer_log_rounds_and_clips():
    @dataclass
    class S2:
        k: Integer = Integer(2, 9, log=True)

    enc = SpaceEncoder(S2)
    # Provide a vector where the encoded value corresponds to exp(x)
    # Negative large -> small -> should clip to low bound 2
    dec_low = enc.decode(np.array([-100.0]))
    assert int(dec_low.k) == 2

    # Large -> very big -> clip to high bound 9
    dec_high = enc.decode(np.array([100.0]))
    assert int(dec_high.k) == 9

    # Mid value => exp(ln(5.4)) ~ 5.4, should round to 5
    dec_mid = enc.decode(np.array([math.log(5.4)]))
    assert int(dec_mid.k) == 5


# Story: For categoricals, only declared choices are encodable. Feeding an
# unknown label should raise a clear error so users notice misconfigurations.
def test_encode_raises_on_unknown_categorical_value():
    @dataclass
    class S3:
        c: Categorical = Categorical(("a", "b"))

    enc = SpaceEncoder(S3)
    obj = S3(c="z")  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        enc.encode(obj)
