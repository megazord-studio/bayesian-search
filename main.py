from dataclasses import dataclass
from typing import cast

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from bayesian_search.bo_search import bayesian_search
from bayesian_search.bo_types import BayesSearchConfig
from bayesian_search.bo_types import Categorical
from bayesian_search.bo_types import Integer

# Load dataset once at module import to avoid repeated IO
_DATASET = load_breast_cancer()
_X = _DATASET.data
_y = _DATASET.target

# Deterministic split for evaluation across the whole run
_X_train, _X_val, _y_train, _y_val = train_test_split(
    _X, _y, test_size=0.2, random_state=123, stratify=_y
)


@dataclass
class MyParams:
    """Search space for RandomForest hyperparameters.

    Note: We keep types simple and discrete to match our lightweight encoder.
    """

    n_estimators: Integer = Integer(50, 150)
    max_depth: Integer = Integer(2, 20)
    max_features: Categorical = Categorical(("sqrt", "log2"))
    min_samples_split: Integer = Integer(2, 10)
    min_samples_leaf: Integer = Integer(1, 5)
    bootstrap: Categorical = Categorical((True, False))


def evaluate_my_model(p: MyParams) -> float:
    """Train a RandomForest on a scikit-learn dataset and return val accuracy.

    The search maximizes this score. A fixed train/validation split and
    random_state are used for determinism under the same seed.
    """
    n_estimators = cast(int, p.n_estimators)
    max_depth = cast(int, p.max_depth)
    max_features = cast(str, p.max_features)
    min_samples_split = cast(int, p.min_samples_split)
    min_samples_leaf = cast(int, p.min_samples_leaf)
    bootstrap = cast(bool, p.bootstrap)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        n_jobs=-1,
        random_state=123,
    )
    clf.fit(_X_train, _y_train)
    preds = clf.predict(_X_val)
    acc = accuracy_score(_y_val, preds)
    return float(acc)


if __name__ == "__main__":
    # Optional live plotting of progress if matplotlib is available
    callback = None
    try:
        from bayesian_search.bo_viz import LivePlotter

        callback = LivePlotter(
            title="RandomForest BO Progress", ylabel="Val accuracy"
        )
    except Exception:
        callback = None

    result = bayesian_search(
        MyParams,
        evaluate=evaluate_my_model,
        config=BayesSearchConfig(
            n_init=6, n_iter=100, candidate_pool=512, seed=42, callback=callback
        ),
        verbose=True,
    )
    print("\nBest result:")
    print("Validation accuracy:", result["best_score"])
    print("Best hyperparameters:", result["best_params"])
