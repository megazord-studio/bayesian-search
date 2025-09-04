import sys

import pytest


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("matplotlib") is None,
    reason="matplotlib not available",
)
def test_liveplotter_updates_and_does_not_block(monkeypatch):
    # Ensure a non-interactive backend
    import matplotlib

    matplotlib.use("Agg", force=True)

    from bayesian_search.bo_viz import LivePlotter

    viz = LivePlotter(title="Test", ylabel="Score", keep_open=False)

    # Simulate start
    viz("start", {})

    # Simulate some history on init
    history = [0.1, 0.2, 0.15]
    viz("init", {"history": history, "best": max(history), "total": 3})

    # After update, internal buffers should reflect history
    assert viz._scores == [pytest.approx(x) for x in history]
    assert viz._bests == [
        pytest.approx(0.1),
        pytest.approx(0.2),
        pytest.approx(0.2),
    ]

    # Simulate an iteration with improvement
    history2 = history + [0.25]
    viz("iter", {"history": history2, "best": max(history2), "total": 1})
    assert viz._scores[-1] == pytest.approx(0.25)
    assert viz._bests[-1] == pytest.approx(0.25)

    # Figures and axes should exist
    assert (
        hasattr(viz, "fig")
        and hasattr(viz, "ax_top")
        and hasattr(viz, "ax_bottom")
    )
    # Improvements should reflect new best events
    assert viz._improvements == [
        0.0,
        pytest.approx(0.1),
        0.0,
        pytest.approx(0.05),
    ]

    # Finalize should not block when keep_open=False
    viz._finalize()


def test_liveplotter_importerror_when_matplotlib_missing(monkeypatch):
    # Simulate matplotlib not installed by shadowing import
    mod_base = "matplotlib"
    plt_mod = "matplotlib.pyplot"
    saved_base = sys.modules.get(mod_base, None)
    saved_plt = sys.modules.get(plt_mod, None)
    sys.modules[mod_base] = None  # type: ignore
    sys.modules[plt_mod] = None  # type: ignore
    try:
        from importlib import reload

        import bayesian_search.bo_viz as bo_viz

        reload(bo_viz)
        with pytest.raises(ImportError):
            bo_viz.LivePlotter()
    finally:
        # restore
        if saved_base is not None:
            sys.modules[mod_base] = saved_base
        else:
            sys.modules.pop(mod_base, None)
        if saved_plt is not None:
            sys.modules[plt_mod] = saved_plt
        else:
            sys.modules.pop(plt_mod, None)
