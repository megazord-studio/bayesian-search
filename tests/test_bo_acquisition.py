import numpy as np

from bayesian_search.bo_acquisition import expected_improvement


# Story: Expected Improvement (EI) balances how much better we might do
# (the improvement) with our uncertainty. We narrate three facts:
# 1) EI is never negative, 2) zero variance means zero EI, 3) higher mean
# with the same variance yields at least as much EI.
def test_expected_improvement_basic_properties():
    mu = np.array([0.0, 1.0, 2.0])
    var = np.array([1.0, 0.5, 0.0])
    best = 0.5

    ei = expected_improvement(mu, var, best, xi=0.0)

    # Non-negative
    assert np.all(ei >= -1e-12)

    # Zero variance should yield 0 EI
    assert ei[2] == 0.0

    # Higher mean with the same var should not decrease EI
    mu2 = np.array([0.0, 1.0, 2.0])
    var2 = np.array([1.0, 1.0, 1.0])
    ei2 = expected_improvement(mu2, var2, best, xi=0.0)
    assert ei2[1] >= ei2[0]
    assert ei2[2] >= ei2[1]
