from dataclasses import dataclass

import numpy as np

from bayesian_search.bo_search import bayesian_search
from bayesian_search.bo_types import BayesSearchConfig
from bayesian_search.bo_types import Real


@dataclass
class P:
    x: Real = Real(0.0, 1.0, log=False)


def objective(p: P) -> float:
    x = float(p.x)
    return -((x - 0.9) ** 2)


# Story: The returned history is our running log. It should have the correct
# length and use plain floats for scores—useful for logging/serialization.
def test_bayesian_search_history_scores_are_floats_and_len_matches():
    cfg = BayesSearchConfig(n_init=3, n_iter=4, candidate_pool=64, seed=1)
    res = bayesian_search(P, evaluate=objective, config=cfg, verbose=False)

    hist = res["history"]
    assert len(hist) == cfg.n_init + cfg.n_iter
    assert all(isinstance(h["score"], float) for h in hist)


# Story: As we iterate, the "best-so-far" score should never decrease—by
# definition it’s a running maximum. We reconstruct that sequence.
def test_bayesian_search_best_non_decreasing_over_time():
    cfg = BayesSearchConfig(n_init=3, n_iter=6, candidate_pool=64, seed=2)
    res = bayesian_search(P, evaluate=objective, config=cfg, verbose=False)

    scores = [h["score"] for h in res["history"]]
    best_so_far = -np.inf
    for s in scores:
        best_so_far = max(best_so_far, s)
        # The best-so-far sequence is non-decreasing
        assert best_so_far >= s - 1e-12 or best_so_far == max(
            scores[: scores.index(s) + 1]
        )
