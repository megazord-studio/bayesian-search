import numpy as np

from bayesian_search.bo_acquisition import expected_improvement


# Story: When all candidates look worse than the current best (negative
# improvement), EI should be tiny but not negative. This keeps the search
# numerically well-behaved.
def test_ei_negative_improvement_near_zero():
    # When mu is far below best, EI should be small but non-negative
    mu = np.array([-5.0, -1.0])
    var = np.array([1.0, 0.1])
    best = 0.0
    ei = expected_improvement(mu, var, best, xi=0.0)
    assert np.all(ei >= -1e-12)
    assert np.all(ei < 1e-3)


# Story: The xi parameter is a pessimism penalty—raising it should make EI
# smaller or equal, nudging exploration away from mediocre-looking points.
def test_ei_effect_of_xi_penalty():
    mu = np.array([0.2, 0.5, 1.0])
    var = np.array([0.5, 0.5, 0.5])
    best = 0.4
    ei0 = expected_improvement(mu, var, best, xi=0.0)
    ei_big = expected_improvement(mu, var, best, xi=0.5)
    # Larger xi should not increase EI
    assert np.all(ei_big <= ei0 + 1e-12)


# Story: With zero posterior variance everywhere, there is no uncertainty to
# exploit and EI must be exactly zero—there’s nothing new to learn.
def test_ei_all_zero_variance():
    mu = np.array([0.0, 1.0, -1.0])
    var = np.array([0.0, 0.0, 0.0])
    best = 0.5
    ei = expected_improvement(mu, var, best, xi=0.0)
    assert np.all(ei == 0.0)
