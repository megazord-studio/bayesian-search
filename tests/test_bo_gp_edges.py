import numpy as np
import pytest

from bayesian_search.bo_gp import GP


# Story: A GP must be fit before predicting. We enforce that misuse raises
# an informative assertion error.
def test_gp_predict_before_fit_raises():
    gp = GP()
    with pytest.raises(AssertionError):
        gp.predict(np.zeros((1, 1)))


# Story: With an extremely small lengthscale, the kernel becomes nearly
# diagonal. A tiny noise term should keep the Cholesky stable and predictions
# sane.
def test_gp_extreme_lengthscale_and_noise_stability():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    y = rng.normal(size=20)

    # Very small lengthscale -> near diagonal kernel; ensure cholesky works due to noise
    gp = GP(lengthscale=1e-6, variance=1.0, noise=1e-6)
    gp.fit(X, y)

    Xs = rng.normal(size=(5, 3))
    mu, var = gp.predict(Xs)
    assert mu.shape == (5,)
    assert var.shape == (5,)
    assert np.all(var >= 0.0)
