"""Opt-in Newton observer integration for the axisymmetric viscoelastic problem.

The ordinary problem class is used unchanged when observation is disabled.  This
adapter never changes the Newton tolerance or submits an additional solve.
"""

from __future__ import annotations

import sys
from pathlib import Path

from taylor_culick_fem.axisymmetric_viscoelastic import AxisymmetricViscoelasticProblem
from taylor_culick_fem.newton_observer import NewtonObserver


class ObservedAxisymmetricViscoelasticProblem(AxisymmetricViscoelasticProblem):
    """Add bounded observation to the existing Newton callbacks."""

    def __init__(self, output: Path):
        super().__init__()
        self._newton_observer = NewtonObserver(output)
        self._observer_time = None
        self._observer_history = ()

    def _observe(self, callback, *args):
        observer = self._newton_observer
        if observer is None:
            return
        try:
            callback(*args)
        except Exception as exc:
            # Instrumentation must never turn a successful solver step into a failure.
            try:
                observer.disable(exc)
            except Exception:
                pass
            self._newton_observer = None
            print("Newton observer disabled after an instrumentation error", file=sys.stderr)
        else:
            if not observer.enabled and self._newton_observer is observer:
                self._newton_observer = None
                print(f"Newton observer disabled: {observer.failure}", file=sys.stderr)

    def _time(self):
        return float(self.get_current_time(as_float=True))

    def _labels(self):
        indexes, names = self.get_dof_description()
        return [str(names[int(index)]) if 0 <= int(index) < len(names) else "unknown"
                for index in indexes]

    def actions_before_newton_solve(self):
        # The inherited method applies the predictor and sets the effective tolerance.
        super().actions_before_newton_solve()
        observer = self._newton_observer
        if observer is None:
            return

        def begin():
            if observer.has_active_solve:
                observer.end_solve(self._observer_history, outcome="unknown")
            now = self._time()
            dt = float(self.time_pt().dt(0))
            floor_dt = dt if self.newton_floor_max_dt is None else min(dt, self.newton_floor_max_dt)
            scale = float(self.window_position())
            if self.newton_scale_uses_window:
                scale += float(self.window_length)
            labels = self._labels() if observer.wants_details else ()
            observer.start_solve(now, dt, float(self.newton_solver_tolerance), scale,
                                 float(self.newton_tolerance_at_unit_radius), floor_dt, labels)
            self._observer_time = now
            self._observer_history = ()

        self._observe(begin)

    def actions_before_newton_step(self):
        super().actions_before_newton_step()
        observer = self._newton_observer
        if observer is None:
            return

        def before():
            history = tuple(self.get_last_residual_convergence())
            self._observer_history = history
            values = self.get_current_dofs()[0] if observer.wants_details else None
            observer.before_step(values, history)

        self._observe(before)

    def actions_after_newton_step(self):
        super().actions_after_newton_step()
        observer = self._newton_observer
        if observer is None or not observer.has_active_solve:
            return

        def after():
            if self._observer_time is not None and self._time() != self._observer_time:
                # oomph-lib also calls this hook after restoring a rejected step.
                observer.rollback()
                history = tuple(self.get_last_residual_convergence())
                observer.end_solve(history, outcome="temporal_rejected")
                self._observer_history = ()
                return
            values = self.get_current_dofs()[0] if observer.wants_details else None
            observer.after_step(values)

        self._observe(after)

    def actions_after_newton_solve(self):
        super().actions_after_newton_solve()
        if self._newton_observer is not None:
            self._observe(lambda: setattr(self, "_observer_history",
                                          tuple(self.get_last_residual_convergence())))

    def actions_after_transient_solve(self):
        super().actions_after_transient_solve()
        observer = self._newton_observer
        if observer is not None and observer.has_active_solve:
            self._observe(observer.end_solve, self._observer_history, "temporal_accepted")
            self._observer_history = ()

    def __exit__(self, exc_type, exc, traceback):
        observer = self._newton_observer
        if observer is not None:
            try:
                if observer.has_active_solve:
                    observer.end_solve(self._observer_history, outcome="unknown")
                observer.close()
            except Exception:
                print("Newton observer failed to close", file=sys.stderr)
        return super().__exit__(exc_type, exc, traceback)
