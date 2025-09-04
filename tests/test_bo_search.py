from dataclasses import dataclass

from bayesian_search.bo_search import bayesian_search
from bayesian_search.bo_types import BayesSearchConfig
from bayesian_search.bo_types import Real


# Simple, smooth objective in 1D: maximize -(x-0.2)^2 with x in [0,1]
@dataclass
class P:
    x: Real = Real(0.0, 1.0, log=False)


def objective(p: P) -> float:
    x = float(p.x)
    return -((x - 0.2) ** 2)


# Story: On a simple 1D bowl-shaped function with maximum at x=0.2, our
# Bayesian search should quickly hone in near the optimum and keep a proper
# history.
def test_bayesian_search_finds_near_optimum():
    cfg = BayesSearchConfig(
        n_init=5,
        n_iter=10,
        candidate_pool=256,
        lengthscale=0.2,
        variance=1.0,
        noise=1e-6,
        xi=0.0,
        seed=123,
    )
    res = bayesian_search(P, evaluate=objective, config=cfg, verbose=False)

    assert "best_params" in res and "best_score" in res and "history" in res
    best_x = float(res["best_params"].x)
    best_score = float(res["best_score"])

    # Best should be close to 0.2 and score near 0
    assert abs(best_x - 0.2) < 0.15
    assert best_score <= 1e-6  # close to zero from below

    # History length equals n_init + n_iter
    assert len(res["history"]) == cfg.n_init + cfg.n_iter


# Story: Given the same RNG seed and configuration, Bayesian search should be
# deterministic—same proposals, same scores, same best.
def test_bayesian_search_is_deterministic_with_seed():
    cfg = BayesSearchConfig(
        n_init=4,
        n_iter=6,
        candidate_pool=128,
        seed=777,
    )
    res1 = bayesian_search(P, evaluate=objective, config=cfg, verbose=False)
    res2 = bayesian_search(P, evaluate=objective, config=cfg, verbose=False)

    assert res1["best_score"] == res2["best_score"]
    assert float(res1["best_params"].x) == float(res2["best_params"].x)
    # Also, history scores identical
    h1 = [h["score"] for h in res1["history"]]
    h2 = [h["score"] for h in res2["history"]]
    assert h1 == h2
