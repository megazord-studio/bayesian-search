"""Optional visualization helpers for the Bayesian search loop.

This module is dependency-light and only uses matplotlib if available.
Nothing is imported unless you construct LivePlotter. If matplotlib is not
installed, LivePlotter will raise a clear ImportError.
"""

from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Sequence


class LivePlotter:
    """A tiny live plot for best score over evaluations with convergence cues.

    Usage:
        viz = LivePlotter(title="Bayesian Search Progress")
        config = BayesSearchConfig(callback=viz)  # it is callable

    The bayesian_search() will call this object as a callback with
    events: "start", "init", "iter", "end". We update the live plots.

    What the figure shows:
    - Top panel: each evaluation's score (faded) and the running best score (bold).
      Red dots mark when a new best was found.
    - Bottom panel: the improvement amount (Δ best) achieved at each evaluation,
      helping you see convergence slowdowns.
    The window stays open at the end so you can inspect it; it won't call plt.close().
    """

    def __init__(
        self,
        title: str = "Bayesian Search Progress",
        ylabel: str = "Score",
        keep_open: bool = True,
    ):
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "matplotlib is required for LivePlotter. Install it to enable plotting."
            ) from e

        # Lazy initialize fig/ax to support environments without display until used
        self._initialized = False
        self.title = title
        self.ylabel = ylabel
        self.keep_open = keep_open
        self._scores: list[float] = []
        self._bests: list[float] = []
        self._best_indices: list[int] = []
        self._improvements: list[
            float
        ] = []  # delta best improvements per step
        self._start_time: Optional[float] = None
        self._last_time: Optional[float] = None
        self._total_expected: Optional[int] = None
        self._stats_text = None  # matplotlib text handle

    # The config.callback is called like callback(event, state)
    def __call__(
        self, event: str, state: dict[str, Any]
    ) -> None:  # pragma: no cover
        import time

        if event == "start":
            self._start_time = time.time()
            self._last_time = self._start_time
            self._total_expected = None
            self._init_plot()
        elif event in ("init", "iter"):
            scores = self._normalize_scores(state.get("history"))
            if self._total_expected is None:
                # Prefer total from iter stage; otherwise from init
                total = state.get("total")
                if isinstance(total, int):
                    # total for that phase; overall total unknown; we can still show counts
                    self._total_expected = None
            now = time.time()
            if scores is not None:
                self._scores = scores
                self._bests = [
                    max(scores[: i + 1]) for i in range(len(scores))
                ]
                # track where a new best occurred (index of best updates)
                self._best_indices = []
                best_running = float("-inf")
                self._improvements = []
                for i, s in enumerate(self._scores):
                    if s > best_running:
                        self._best_indices.append(i)
                        imp = (
                            s - best_running
                            if best_running != float("-inf")
                            else 0.0
                        )
                        self._improvements.append(max(imp, 0.0))
                        best_running = s
                    else:
                        self._improvements.append(0.0)
                self._update(now)
        elif event == "end":
            self._finalize()

    def _normalize_scores(
        self, history: Optional[Sequence[float]]
    ) -> Optional[list[float]]:
        if history is None:
            return None
        try:
            return [float(s) for s in history]
        except Exception:
            return None

    def _init_plot(self) -> None:
        if self._initialized:
            return
        import matplotlib
        import matplotlib.pyplot as plt

        backend = matplotlib.get_backend().lower()
        if backend not in {"agg", "svg", "pdf", "ps", "cairo"}:
            plt.ion()
        # Two-row layout: top scores/best; bottom improvement bars
        self.fig, (self.ax_top, self.ax_bottom) = plt.subplots(
            2, 1, figsize=(7, 6), sharex=True, height_ratios=[3, 1]
        )
        # Top lines
        (self.line_all,) = self.ax_top.plot([], [], label="scores", alpha=0.35)
        (self.line_best,) = self.ax_top.plot(
            [], [], label="best so far", linewidth=2
        )
        # Best marker
        self.scatter_best = self.ax_top.scatter(
            [], [], s=36, color="#d62728", zorder=3, label="new best"
        )
        # Description text (top-left inside)
        desc = "Scores per evaluation (faded) and running best (bold).\nRed dots mark new bests. Bottom shows Δ best to visualize convergence."
        self.ax_top.text(
            0.02,
            0.98,
            desc,
            transform=self.ax_top.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9
            ),
        )
        # Stats text (top-right inside)
        self._stats_text = self.ax_top.text(
            0.98,
            0.02,
            "",
            transform=self.ax_top.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.8
            ),
        )
        self.ax_top.set_title(self.title)
        self.ax_top.set_ylabel(self.ylabel)
        self.ax_top.legend(loc="best")
        self.ax_top.grid(True, alpha=0.3)

        # Bottom: improvements bars
        self.bar_improve = self.ax_bottom.bar([], [])
        self.ax_bottom.set_ylabel("Δ best")
        self.ax_bottom.grid(True, axis="y", alpha=0.3)
        self.ax_bottom.set_xlabel("Evaluation")

        self.fig.tight_layout()
        self._initialized = True
        self._draw()

    def _update(self, now: Optional[float] = None) -> None:
        if not self._initialized:
            self._init_plot()

        x = list(range(1, len(self._scores) + 1))
        # Update top lines
        self.line_all.set_data(x, self._scores)
        self.line_best.set_data(x, self._bests)
        # Update new-best scatter points
        bx = [i + 1 for i in self._best_indices]
        by = (
            [self._scores[i] for i in self._best_indices]
            if self._scores
            else []
        )
        # Re-create scatter to update data robustly
        self.scatter_best.remove() if hasattr(
            self, "scatter_best"
        ) and self.scatter_best is not None else None
        self.scatter_best = self.ax_top.scatter(
            bx, by, s=36, color="#d62728", zorder=3, label="new best"
        )

        # Rescale top axes with margins
        if self._scores:
            ymin = min(self._scores)
            ymax = max(self._scores)
            if ymin == ymax:
                ymin -= 0.1
                ymax += 0.1
            self.ax_top.set_xlim(1, max(2, len(self._scores)))
            self.ax_top.set_ylim(
                ymin - 0.02 * abs(ymin), ymax + 0.02 * abs(ymax)
            )

        # Bottom improvements as bars
        self.ax_bottom.cla()
        self.ax_bottom.grid(True, axis="y", alpha=0.3)
        self.ax_bottom.set_ylabel("Δ best")
        self.ax_bottom.set_xlabel("Evaluation")
        if self._improvements:
            self.ax_bottom.bar(
                x, self._improvements, color="#1f77b4", alpha=0.6
            )
            self.ax_bottom.set_xlim(1, max(2, len(self._improvements)))
            # Make y-scale non-negative and tight
            yb_max = (
                max(self._improvements) if any(self._improvements) else 1.0
            )
            self.ax_bottom.set_ylim(0, yb_max * 1.1 if yb_max > 0 else 1.0)

        # Update stats box
        if self._stats_text is not None:
            n = len(self._scores)
            best = self._bests[-1] if self._bests else float("nan")
            last = self._scores[-1] if self._scores else float("nan")
            if (
                now is not None
                and self._last_time is not None
                and self._start_time is not None
            ):
                dt_total = now - self._start_time
                dt_last = now - self._last_time
                self._last_time = now
                avg = dt_total / max(1, n)
                time_str = f"last {dt_last:.2f}s | avg {avg:.2f}s"
            else:
                time_str = ""
            self._stats_text.set_text(
                f"evals: {n}\nbest: {best:.5f}\nlast: {last:.5f}\n{time_str}"
            )

        self.fig.tight_layout()
        self._draw()

    def _finalize(self) -> None:
        # Keep the window open for inspection at the end if requested
        self._draw()
        try:
            import matplotlib
            import matplotlib.pyplot as plt

            backend = matplotlib.get_backend().lower()
            is_interactive = backend not in {
                "agg",
                "svg",
                "pdf",
                "ps",
                "cairo",
            }
            if self.keep_open and is_interactive:
                plt.ioff()
                # Block here so the figure doesn't disappear when the script exits
                plt.show()
        except Exception:
            # In headless environments this may fail; ignore to avoid breaking runs
            pass

    def _draw(self) -> None:
        try:
            import matplotlib
            import matplotlib.pyplot as plt

            backend = matplotlib.get_backend().lower()
            # Always draw to update figure content
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            # Only call pause on interactive backends to avoid warnings on Agg
            if backend not in {"agg", "svg", "pdf", "ps", "cairo"}:
                plt.pause(0.001)
        except Exception:
            # Silently ignore draw issues in non-interactive/headless contexts
            pass
