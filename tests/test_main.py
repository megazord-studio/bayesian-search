import main as app
from bayesian_search.bo_search import bayesian_search
from bayesian_search.bo_types import BayesSearchConfig


# Story: The scikit-learn example should be deterministic for the same inputs
# because we fix train/val split and classifier random_state.
def test_example_evaluate_function_deterministic_without_seed():
    p = app.MyParams(
        n_estimators=120,
        max_depth=8,
        max_features="sqrt",  # type: ignore[arg-type]
        min_samples_split=3,
        min_samples_leaf=1,
        bootstrap=True,
    )
    v1 = app.evaluate_my_model(p)
    v2 = app.evaluate_my_model(p)
    assert v1 == v2


# Story: A tiny end-to-end run should wire up the whole pipeline without
# printing too much. It’s a smoke test for configuration and plumbing.
def test_main_like_run_smoke():
    cfg = BayesSearchConfig(n_init=2, n_iter=2, candidate_pool=32, seed=0)
    res = bayesian_search(
        app.MyParams,
        evaluate=app.evaluate_my_model,
        config=cfg,
        verbose=False,
    )
    assert "best_score" in res and "best_params" in res
    assert len(res["history"]) == cfg.n_init + cfg.n_iter
