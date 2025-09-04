import numpy as np

from bayesian_search.bo_gp import GP
from bayesian_search.bo_gp import rbf_kernel


# Story: The RBF kernel should be symmetric: K(X,Y) == K(Y,X).T, and have
# expected shapes. This sanity-checks our distance math.
def test_rbf_kernel_symmetry_and_shape():
    X = np.array([[0.0], [1.0], [2.0]])
    Y = np.array([[0.5], [1.5]])
    Kxy = rbf_kernel(X, Y, lengthscale=1.0, variance=2.0)
    Kyx = rbf_kernel(Y, X, lengthscale=1.0, variance=2.0)
    assert Kxy.shape == (3, 2)
    assert Kyx.shape == (2, 3)
    # Symmetry property: K(X,Y) == K(Y,X).T
    assert np.allclose(Kxy, Kyx.T, atol=1e-12)


# Story: After fitting a GP on noisy data, predictions on new points should
# return well-shaped mean/variance arrays and non-negative variances.
def test_gp_fit_predict_shapes_and_variance():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(10, 2))
    y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=10)

    gp = GP(lengthscale=0.5, variance=1.5, noise=1e-6)
    gp.fit(X, y)

    Xs = rng.uniform(-1, 1, size=(5, 2))
    mu, var = gp.predict(Xs)
    assert mu.shape == (5,)
    assert var.shape == (5,)
    assert np.all(var >= 0.0)


# Story: With near-zero noise, querying the training inputs should reproduce
# the training targets—this checks our Cholesky solves and kernel math.
def test_gp_interpolation_low_noise():
    # With near-zero noise and querying training points, mean should match y
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, -1.0, 0.5])
    gp = GP(lengthscale=1.0, variance=1.0, noise=1e-10)
    gp.fit(X, y)
    mu, var = gp.predict(X)
    assert np.allclose(mu, y, atol=1e-6)
    assert np.all(var >= 0.0)
