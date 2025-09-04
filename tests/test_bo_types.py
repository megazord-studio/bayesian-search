import dataclasses

from bayesian_search.bo_types import BayesSearchConfig
from bayesian_search.bo_types import Categorical
from bayesian_search.bo_types import Integer
from bayesian_search.bo_types import Real


# Story: Our parameter specs are dataclasses and should be immutable (frozen),
# so users can't accidentally mutate search definitions at runtime.
def test_real_integer_categorical_dataclasses():
    r = Real(0.1, 1.0, log=True)
    i = Integer(1, 5, log=False)
    c = Categorical(("a", "b"))

    assert dataclasses.is_dataclass(r)
    assert dataclasses.is_dataclass(i)
    assert dataclasses.is_dataclass(c)

    # frozen dataclasses are immutable
    try:
        dataclasses.replace(r, low=0.2)
    except Exception:
        # replace is allowed even for frozen; direct assignment should fail
        pass
    try:
        r.low = 0.2  # type: ignore[attr-defined]
        mutated = True
    except Exception:
        mutated = False
    assert not mutated


# Story: The BayesSearchConfig establishes sensible defaults. We lock these
# in with a test so changes are deliberate and visible to users.
def test_bayes_search_config_defaults():
    cfg = BayesSearchConfig()
    assert cfg.n_init == 10
    assert cfg.n_iter == 40
    assert cfg.candidate_pool == 2048
    assert cfg.lengthscale == 1.0
    assert cfg.variance == 1.0
    assert cfg.noise == 1e-6
    assert cfg.xi == 0.01
    assert cfg.seed is None
