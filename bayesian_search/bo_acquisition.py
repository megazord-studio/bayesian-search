import math

import numpy as np


def erf(x: np.ndarray) -> np.ndarray:
    """Vectorized error-function approximation.

    Uses Abramowitz/Stegun 7.1.26 to avoid importing scipy.
    """
    sign = np.sign(x)
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    p = 0.3275911
    t = 1.0 / (1.0 + p * np.abs(x))
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(
        -x * x
    )
    return sign * y


def expected_improvement(
    mu: np.ndarray, var: np.ndarray, best: float, xi: float = 0.01
) -> np.ndarray:
    """Expected Improvement acquisition for maximization.

    Parameters
    - mu: posterior mean at candidate points
    - var: posterior variance at candidate points
    - best: current best observed objective value
    - xi: exploration bias (larger -> more conservative improvement)
    """
    # Standard EI for maximization with robust handling of zero variance
    eps = 1e-12
    sigma = np.sqrt(np.maximum(var, 0.0))
    imp = mu - best - xi

    # Avoid divide-by-zero by using a safe sigma for Z; we'll zero EI afterward
    zero_mask = sigma < eps
    safe_sigma = sigma.copy()
    safe_sigma[zero_mask] = 1.0

    Z = imp / safe_sigma
    cdf = 0.5 * (1.0 + erf(Z / math.sqrt(2)))
    pdf = (1.0 / math.sqrt(2 * math.pi)) * np.exp(-0.5 * Z**2)

    ei = imp * cdf + sigma * pdf
    ei[zero_mask] = 0.0
    return ei
