# Generic Bayesian Search

A tiny, dependency-light Bayesian optimization helper built around:
- A declarative parameter space using frozen dataclasses (Real, Integer, Categorical)
- A minimal Gaussian Process (RBF kernel) implemented with NumPy
- The Expected Improvement (EI) acquisition function

It’s designed for small experiments, educational purposes, and lightweight tuning loops without pulling in heavy dependencies.

## Quick start

Install dependencies (managed via `uv`):

```bash
uv sync
```

A complete, runnable example is provided in main.py. See that file for a scikit-learn RandomForest tuning demo and run it with:

```bash
uv run python main.py
```

## What you get

- Declarative spaces via dataclasses
  - Real(low, high, log=False)
  - Integer(low, high, log=False)  # rounded and clipped on decode
  - Categorical((choices, ...))     # one-hot-encoded during search
- SpaceEncoder to sample/encode/decode between dataclass instances and dense vectors
- A simple GP (RBF kernel) with Cholesky factorization
- Expected Improvement acquisition with robust zero-variance handling
- A high-level `bayesian_search` loop that orchestrates sampling, GP fit, and EI-based candidate selection

## Design notes

- No SciPy dependency: we include a numerically stable error function approximation for EI.
- Frozen parameter spec dataclasses (Real, Integer, Categorical) make search spaces immutable and safe to share.
- Clean, documented modules:
  - `bayesian_search/bo_types.py` – parameter specs and `BayesSearchConfig`
  - `bayesian_search/bo_encoding.py` – `SpaceEncoder` for encode/decode/sample
  - `bayesian_search/bo_gp.py` – `rbf_kernel` and `GP`
  - `bayesian_search/bo_acquisition.py` – `expected_improvement`
  - `bayesian_search/bo_search.py` – `bayesian_search` loop

## API sketch

- `SpaceEncoder(spec_cls)`
  - `sample(rng)` -> spec instance
  - `encode(obj)` -> np.ndarray (continuous vector; categoricals one-hot)
  - `decode(vec)` -> spec instance (integers rounded/clipped; log handled)
- `GP(lengthscale=1.0, variance=1.0, noise=1e-6)`
  - `fit(X, y)`
  - `predict(Xstar)` -> (mu, var)
- `expected_improvement(mu, var, best, xi=0.01)` -> np.ndarray
- `bayesian_search(spec_cls, evaluate, config=BayesSearchConfig(), verbose=True)` -> dict
  - returns `{ "best_params": spec_instance, "best_score": float, "history": [{"score": float}, ...] }`

## Configuration tips

- `n_init` and `n_iter` control total evaluations: total = n_init + n_iter
- `candidate_pool` trades speed for search breadth per iteration (more is slower but can improve results)
- `lengthscale`, `variance`, `noise` are GP hyperparameters; reasonable defaults are provided
- `xi` controls improvement margin in EI (larger xi -> more conservative improvement)
- `seed` makes the whole run deterministic (sampling, candidate generation)

## Limitations

- Intended for low-dimensional, small-scale problems; not a replacement for mature BO libraries
- Uses a simple RBF GP with fixed hyperparameters (no automatic tuning)
- Categorical variables are handled via one-hot encoding; interactions are not modeled explicitly

## Development

Common tasks via Poe/uv:

```bash
# Install dependencies
uv sync

# Format
uv run poe format

# Typecheck
uv run poe types

# Test
uv run poe test
```

The test suite provides both happy-path and edge-case coverage and serves as living documentation:
- See `tests/test_bo_search.py` and `tests/test_main.py` for end-to-end usage.
- See `tests/test_bo_encoding*.py`, `tests/test_bo_gp*.py`, and `tests/test_bo_acquisition*.py` for focused unit tests.
