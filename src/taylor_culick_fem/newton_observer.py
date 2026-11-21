"""Bounded, passive recording of Newton iterations supplied by a caller.

This module does not register solver hooks, assemble residuals or determine
whether a timestep was accepted.  The caller must supply the *native* residual
history at the phases documented below, and distinguish rollback callbacks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


class NewtonObserver:
    """Record a bounded sample of solves without modifying supplied arrays.

    ``before_step`` precedes the native residual assembly and pairs with the
    immediately following ``after_step``.  In the pinned solver the history
    available there ends at the *previous* update.  ``end_solve`` receives the
    final native history; an unpaired ``after_step`` is recorded as rollback,
    not as a Newton update.  A caller must explicitly supply the temporal
    outcome; the default is unknown.

    Shadow decisions require at least one Newton update and use strict ``<``.
    The recorded shadow distances are observed distances to the last iterate,
    not bounds on the error or on a hypothetical further Newton correction.
    """

    FACTORS = (30, 60, 1000)

    def __init__(
        self,
        path: str | Path,
        *,
        max_full_solves: int = 32,
        max_bytes: int = 1_000_000,
        max_snapshots: int = 12,
        max_snapshot_bytes: int = 32_000_000,
        max_iterations: int = 40,
        max_labels: int = 256,
        max_residual_samples: int = 8,
    ) -> None:
        self.enabled = False
        self.failure: str | None = None
        self._file = None
        self._active = None
        self._serial = 0
        self._bytes = 0
        self._residual_samples = 0
        try:
            limits = (max_full_solves, max_bytes, max_snapshots, max_snapshot_bytes,
                      max_iterations, max_labels, max_residual_samples)
            if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in limits):
                raise ValueError("observer limits must be nonnegative integers")
            self.max_full_solves, self.max_bytes, self.max_snapshots = limits[:3]
            (self.max_snapshot_bytes, self.max_iterations, self.max_labels,
             self.max_residual_samples) = limits[3:]
            self._file = Path(path).open("x", encoding="utf-8")
            self.enabled = True
        except Exception as exc:
            self._disable(exc)

    def _disable(self, exc: Exception) -> None:
        if self.failure is None:
            self.failure = f"{type(exc).__name__}: {exc}"
        self.enabled = False
        self._active = None
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    @property
    def wants_details(self) -> bool:
        """Whether the active (or next) solve needs labels and snapshots."""
        if not self.enabled:
            return False
        if self._active is not None:
            return self._active["full"]
        return self._serial < self.max_full_solves

    @property
    def has_active_solve(self) -> bool:
        return self.enabled and self._active is not None

    @property
    def wants_residual_sample(self) -> bool:
        """Permit at most one explicitly phased sample per detailed solve."""
        return bool(self.has_active_solve and self._active["full"]
                    and self._active["residual_sample"] is None
                    and self._residual_samples < self.max_residual_samples)

    def disable(self, exc: Exception) -> None:
        """Record the first instrumentation failure and stop observing."""
        self._disable(exc)

    def _write(self, record: dict) -> None:
        if not self.enabled:
            return
        encoded = (json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        if self._bytes + len(encoded) > self.max_bytes:
            self._disable(RuntimeError("observer output byte limit reached"))
            return
        assert self._file is not None
        self._file.write(encoded.decode("utf-8"))
        self._file.flush()
        self._bytes += len(encoded)

    @staticmethod
    def _finite(value: float, name: str, *, positive: bool = False) -> float:
        number = float(value)
        if not math.isfinite(number) or (positive and number <= 0):
            raise ValueError(f"invalid {name}")
        return number

    def start_solve(self, time, dt, tolerance, scale, base, floor_dt, labels) -> None:
        if not self.enabled:
            return
        try:
            if self._active is not None:
                # Do not silently merge or reinterpret overlapping attempts.
                self.end_solve([], outcome="incomplete")
                if not self.enabled:
                    return
            time = self._finite(time, "time")
            dt = self._finite(dt, "dt", positive=True)
            tolerance = self._finite(tolerance, "tolerance", positive=True)
            scale = self._finite(scale, "scale")
            base = self._finite(base, "base", positive=True)
            floor_dt = self._finite(floor_dt, "floor_dt", positive=True)
            if not isinstance(labels, (list, tuple)) or not all(isinstance(x, str) for x in labels):
                raise ValueError("labels must be a list or tuple of names")
            if len(labels) * 8 > self.max_snapshot_bytes:
                raise ValueError("dof label count exceeds snapshot budget")
            if len(set(labels)) > self.max_labels:
                raise ValueError("field label limit reached")
            self._serial += 1
            full = self._serial <= self.max_full_solves
            grouped = {}
            if full:
                for index, label in enumerate(labels):
                    grouped.setdefault(label, []).append(index)
                grouped = {label: np.asarray(indexes, dtype=np.intp) for label, indexes in grouped.items()}
            self._active = {
                "id": self._serial, "time": time, "dt": dt, "tolerance": tolerance,
                "scale": scale, "base": base, "floor_dt": floor_dt,
                "labels": tuple(labels) if full else (), "groups": grouped, "full": full,
                "iterations": [], "iteration_count": 0, "rollbacks": 0,
                "pending": None, "last": None, "shadow": {}, "history": [],
                "shadow_unavailable": {}, "residual_sample": None,
            }
        except Exception as exc:
            self._disable(exc)

    def _array(self, values):
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or len(array) != len(self._active["labels"]):
            raise ValueError("dof values and labels do not match")
        if not np.isfinite(array).all():
            raise ValueError("non-finite dof values")
        if array.nbytes * self.max_snapshots > self.max_snapshot_bytes:
            raise ValueError("snapshot byte limit reached")
        return array.copy()

    def _classify_shadow(self, values, history) -> None:
        state = self._active
        # Index zero is the residual before update one.  Only update k may
        # shadow-accept at residual index k; the actual solver forces one step.
        k = state["iteration_count"]
        if k < 1 or len(history) <= k:
            return
        residual = float(history[k])
        if not math.isfinite(residual):
            return
        for factor in self.FACTORS:
            if factor in state["shadow"] or factor in state["shadow_unavailable"]:
                continue
            limit = max(state["base"], factor * np.finfo(float).eps * state["scale"] ** 2 / state["floor_dt"])
            if residual < limit:
                # before, after and delta coexist with the shadow snapshots.
                if 3 + len(state["shadow"]) >= self.max_snapshots:
                    state["shadow_unavailable"][factor] = "snapshot limit"
                else:
                    state["shadow"][factor] = {"iteration": k, "residual": residual,
                                                "tolerance": limit, "values": values.copy()}

    def before_step(self, values, residual_history) -> None:
        if not self.enabled or self._active is None:
            return
        try:
            state = self._active
            if state["pending"] is not None:
                raise ValueError("unpaired before-step callback")
            if not state["full"]:
                state["pending"] = True
                return
            if state["iteration_count"] >= self.max_iterations:
                raise ValueError("iteration sample limit reached")
            if self.max_snapshots < 2:
                raise ValueError("snapshot count limit too small")
            vector = self._array(values)
            self._classify_shadow(vector, residual_history)
            state["pending"] = vector
        except Exception as exc:
            self._disable(exc)

    def after_step(self, values) -> None:
        if not self.enabled or self._active is None:
            return
        try:
            state = self._active
            before = state["pending"]
            if before is None:
                # Native rollback performs an after callback without a before.
                state["rollbacks"] += 1
                return
            state["pending"] = None
            state["iteration_count"] += 1
            if not state["full"]:
                return
            # The old last snapshot is redundant once a new update is paired.
            state["last"] = None
            after = self._array(values)
            delta = after - before
            absolute = np.abs(delta)
            local_scale = np.finfo(float).eps * np.maximum(1.0, np.maximum(np.abs(before), np.abs(after)))
            relative = absolute / local_scale
            fields = {label: float(np.max(absolute[indexes])) for label, indexes in state["groups"].items()}
            scaled = {label: float(np.max(relative[indexes])) for label, indexes in state["groups"].items()}
            state["iterations"].append({"iteration": state["iteration_count"],
                                        "applied_abs_max_by_field": fields,
                                        "applied_max_over_eps_scale_by_field": scaled})
            state["last"] = after
        except Exception as exc:
            self._disable(exc)

    def rollback(self) -> None:
        """Discard any unpaired snapshot when the native solver restores a step."""
        if not self.enabled or self._active is None:
            return
        self._active["pending"] = None
        self._active["rollbacks"] += 1

    def sample_residual(self, info: dict) -> None:
        """Record caller-supplied sampled data, never assemble a residual.

        A phase label is mandatory: these are not native acceptance residuals.
        Invalid, oversized or non-finite samples disable the observer.
        """
        if not self.wants_residual_sample:
            return
        try:
            if not isinstance(info, dict) or not isinstance(info.get("phase"), str) or not info["phase"]:
                raise ValueError("sampled residual needs an explicit phase")
            encoded = json.dumps(info, allow_nan=False, separators=(",", ":"))
            if len(encoded.encode("utf-8")) > 4096:
                raise ValueError("sampled residual metadata limit reached")
            self._active["residual_sample"] = json.loads(encoded)
            self._residual_samples += 1
        except Exception as exc:
            self._disable(exc)

    def end_solve(self, residual_history, outcome="unknown") -> None:
        if not self.enabled or self._active is None:
            return
        try:
            state = self._active
            history = [self._finite(x, "native residual") for x in residual_history]
            if len(history) > self.max_iterations + 1:
                raise ValueError("native history limit reached")
            if state["full"] and state["last"] is not None:
                self._classify_shadow(state["last"], history)
            result = {
                "schema": "newton-observer-v1", "solve_id": state["id"],
                "time": state["time"], "dt": state["dt"],
                "effective_tolerance": state["tolerance"], "scale": state["scale"],
                "base": state["base"], "floor_dt": state["floor_dt"],
                "sampling": "full" if state["full"] else "summary",
                "iteration_count": state["iteration_count"], "rollback_callbacks": state["rollbacks"],
                "unpaired_before": state["pending"] is not None,
                "final_native_below_tolerance": (history[-1] < state["tolerance"] if history else None),
                "attempt_outcome": str(outcome),
                "temporal_acceptance": (
                    "accepted" if outcome == "temporal_accepted" else
                    "rejected" if outcome == "temporal_rejected" else "unknown"
                ),
                "native_residuals": history,
                "residual_row_location": "unavailable_without_extra_assembly",
                "relative_acceptance_shadow": "unavailable_no_rule_specified",
            }
            if state["full"]:
                result["iterations"] = state["iterations"]
                result["sampled_residual"] = state["residual_sample"]
                result["shadow"] = {}
                for factor, shadow in state["shadow"].items():
                    delta = state["last"] - shadow["values"]
                    absolute = np.abs(delta)
                    fields = {label: float(np.max(absolute[indexes]))
                              for label, indexes in state["groups"].items()}
                    signed = {}
                    for name in ("X", "U"):
                        matches = [i for i, label in enumerate(state["labels"])
                                   if label == name or label.endswith("/" + name)]
                        if len(matches) == 1:
                            signed[name] = float(delta[matches[0]])
                    result["shadow"][str(factor)] = {
                        "iteration": shadow["iteration"], "residual": shadow["residual"],
                        "tolerance": shadow["tolerance"], "observed_distance_abs_max_by_field": fields,
                        "signed_final_minus_shadow": signed,
                    }
                result["shadow_unavailable"] = {str(k): v for k, v in state["shadow_unavailable"].items()}
            self._active = None
            self._write(result)
        except Exception as exc:
            self._disable(exc)

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception as exc:
                self._disable(exc)
        self._file = None
        self.enabled = False
