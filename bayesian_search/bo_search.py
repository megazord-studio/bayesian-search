import random
from typing import Any
from typing import Callable
from typing import Dict
from typing import List

import numpy as np

from bayesian_search.bo_acquisition import expected_improvement
from bayesian_search.bo_encoding import SpaceEncoder
from bayesian_search.bo_gp import GP
from bayesian_search.bo_types import BayesSearchConfig


def bayesian_search(
    spec_cls: type,
    evaluate: Callable[[Any], float],
    config: BayesSearchConfig = BayesSearchConfig(),
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a simple Bayesian optimization loop.

    Parameters
    - spec_cls: dataclass describing the search space using Real/Integer/Categorical
    - evaluate: callable that maps a spec instance to a scalar score (to maximize)
    - config: BayesSearchConfig controlling iterations and GP hyperparameters
    - verbose: print per-iteration progress

    Returns a dict with best_params (dataclass), best_score (float), and
    history (list of dicts with 'score' per evaluation in order).
    """
    rng = random.Random(config.seed)
    enc = SpaceEncoder(spec_cls)

    X_vectors: List[np.ndarray] = []
    y_scores: List[float] = []

    # optional callback helper
    def _cb(event: str, **state: Any) -> None:
        if getattr(config, "callback", None) is not None:
            try:
                config.callback(event, state)  # type: ignore[misc]
            except Exception:
                # Swallow callback errors to avoid breaking the search loop
                pass

    _cb("start")

    # initial random samples
    for i in range(config.n_init):
        p = enc.sample(rng)
        score = float(evaluate(p))
        X_vectors.append(enc.encode(p))
        y_scores.append(score)
        if verbose:
            print(
                f"[init {i + 1}/{config.n_init}] score={score:.6f} params={p}"
            )
        _cb(
            "init",
            i=i,
            total=config.n_init,
            params=p,
            score=score,
            best=float(max(y_scores)) if y_scores else float("nan"),
            history=y_scores.copy(),
        )

    X_train = np.vstack(X_vectors)
    y_train = np.array(y_scores, dtype=float)

    # BO loop
    for t in range(config.n_iter):
        gp = GP(
            lengthscale=config.lengthscale,
            variance=config.variance,
            noise=config.noise,
        )
        gp.fit(X_train, y_train)

        candidates = [
            enc.encode(enc.sample(rng)) for _ in range(config.candidate_pool)
        ]
        C = np.vstack(candidates)
        mu, var = gp.predict(C)
        best_so_far = float(np.max(y_train))
        ei = expected_improvement(mu, var, best_so_far, xi=config.xi)
        idx = int(np.argmax(ei))
        x_next = C[idx]
        p_next = enc.decode(x_next)

        score_next = float(evaluate(p_next))
        X_train = np.vstack([X_train, x_next])
        y_train = np.append(y_train, score_next)

        if verbose:
            print(
                f"[iter {t + 1}/{config.n_iter}] best={best_so_far:.6f} -> new={score_next:.6f} params={p_next}"
            )
        _cb(
            "iter",
            t=t,
            total=config.n_iter,
            params=p_next,
            score=score_next,
            best=float(np.max(y_train)),
            history=y_train.tolist(),
        )

    best_idx = int(np.argmax(y_train))
    best_params = enc.decode(X_train[best_idx])
    result = {
        "best_params": best_params,
        "best_score": float(y_train[best_idx]),
        "history": [{"score": float(s)} for s in y_train],
    }
    _cb("end", **result)
    return result
