"""
behavior.py - BEHAVIOUR of the line: the "data file" half of the architecture.

Cycle-time distributions, defect probabilities, operator variability, failure and
repair data, component weights, bin stocking, replenishment, load-cell and camera
characteristics and pick-detection parameters.

Everything is keyed by object ID (process, operator, station, component) - never
by position - so the same data file stays valid after a reconfiguration: a process
that moves to another station keeps its own time distribution.
"""
from __future__ import annotations

import copy
import math
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .config_model import LineStructure, Report
except ImportError:  # files placed side by side without the line_sim folder
    from config_model import LineStructure, Report

DEFAULTS = {
    "simulation": {"horizon_s": 3600.0, "warmup_s": 300.0, "seed": 42},
    "process_time": {"dist": "lognormal", "mean": 20.0, "cv": 0.2},
    "pick_window": 0.6,
    "operator": {"speed_factor": 1.0, "cv_factor": 1.0},
    "unit_weight_g": 10.0,
    "bin": {"capacity_units": 40, "reorder_point_units": 8, "tare_g": 350.0},
    "lead_time": {"dist": "triangular", "min": 60, "mode": 120, "max": 240},
    "load_cells": {"sample_rate_hz": 1.0, "noise_sd_g": 0.3, "drift_g_per_h": 0.0, "resolution_g": 0.1,
                   "spike_prob": 0.003, "spike_g": 25.0},
    "camera": {"dropout": 0.10, "confidence_a": 8.0, "confidence_b": 2.0},
    "pick_detection": {"threshold_fraction": 0.5, "noise_k_sigma": 4.0, "persistence_samples": 2,
                       "match_tolerance_s": 3.0},
}

DIST_TYPES = ["lognormal", "normal", "gamma", "exponential", "triangular", "uniform", "constant", "empirical"]


# =============================================================================
# Random streams and distributions
# =============================================================================


def make_rng(seed: int, *parts) -> np.random.Generator:
    """Independent, reproducible stream per purpose/object (common random numbers:
    process P3 draws the same sequence of times wherever it is performed)."""
    words = [int(seed) & 0xFFFFFFFF] + [zlib.crc32(str(p).encode("utf-8")) for p in parts]
    return np.random.default_rng(words)


def _num(spec: dict, key: str, default=None) -> Optional[float]:
    v = spec.get(key, default)
    if v is None:
        return None
    return float(v)


def _kind(spec: dict) -> str:
    return str(spec.get("dist", "lognormal")).lower().strip()


def dist_mean(spec: dict) -> float:
    d = _kind(spec)
    if d in ("constant", "fixed", "deterministic"):
        v = spec.get("value", spec.get("mean", 0.0))
        return float(v)
    if d in ("lognormal", "normal", "gamma", "exponential"):
        return float(spec.get("mean", 0.0))
    if d == "triangular":
        return (float(spec["min"]) + float(spec["mode"]) + float(spec["max"])) / 3.0
    if d == "uniform":
        return (float(spec["min"]) + float(spec["max"])) / 2.0
    if d == "empirical":
        s = spec.get("samples") or []
        return float(np.mean(s)) if len(s) else 0.0
    raise ValueError(f"Unknown distribution '{d}'.")


def dist_cv(spec: dict) -> float:
    d = _kind(spec)
    m = dist_mean(spec)
    if m <= 0:
        return 0.0
    if d in ("constant", "fixed", "deterministic"):
        return 0.0
    if d in ("lognormal", "normal", "gamma"):
        if spec.get("cv") is not None:
            return float(spec["cv"])
        return float(spec.get("sd", 0.0)) / m
    if d == "exponential":
        return 1.0
    if d == "triangular":
        a, c, b = float(spec["min"]), float(spec["mode"]), float(spec["max"])
        return math.sqrt((a * a + b * b + c * c - a * b - a * c - b * c) / 18.0) / m
    if d == "uniform":
        return (float(spec["max"]) - float(spec["min"])) / math.sqrt(12.0) / m
    if d == "empirical":
        s = np.asarray(spec.get("samples") or [], dtype=float)
        return float(s.std(ddof=1) / m) if len(s) > 1 else 0.0
    raise ValueError(f"Unknown distribution '{d}'.")


def sample(spec: dict, rng: np.random.Generator, scale: float = 1.0, cv_factor: float = 1.0) -> float:
    """Draw one value. `scale` multiplies the result (operator speed); `cv_factor`
    widens/narrows the spread (operator consistency)."""
    d = _kind(spec)
    if d in ("constant", "fixed", "deterministic"):
        x = dist_mean(spec)
    elif d == "lognormal":
        m = float(spec["mean"])
        cv = dist_cv(spec) * cv_factor
        if cv <= 0 or m <= 0:
            x = m
        else:
            s2 = math.log(1.0 + cv * cv)
            x = rng.lognormal(math.log(m) - 0.5 * s2, math.sqrt(s2))
    elif d == "normal":
        m = float(spec["mean"])
        sd = dist_cv(spec) * m * cv_factor
        x = max(0.05 * m, rng.normal(m, sd)) if sd > 0 else m
    elif d == "gamma":
        m = float(spec["mean"])
        cv = dist_cv(spec) * cv_factor
        if cv <= 0:
            x = m
        else:
            k = 1.0 / (cv * cv)
            x = rng.gamma(k, m / k)
    elif d == "exponential":
        x = rng.exponential(float(spec["mean"]))
    elif d == "triangular":
        a, c, b = float(spec["min"]), float(spec["mode"]), float(spec["max"])
        if cv_factor != 1.0:
            a, b = c - (c - a) * cv_factor, c + (b - c) * cv_factor
        x = c if b <= a else rng.triangular(a, min(max(c, a), b), b)
    elif d == "uniform":
        a, b = float(spec["min"]), float(spec["max"])
        if cv_factor != 1.0:
            mid = 0.5 * (a + b)
            a, b = mid - (mid - a) * cv_factor, mid + (b - mid) * cv_factor
        x = rng.uniform(a, b) if b > a else a
    elif d == "empirical":
        s = spec.get("samples") or [0.0]
        x = float(s[int(rng.integers(len(s)))])
    else:
        raise ValueError(f"Unknown distribution '{d}'.")
    return max(0.0, float(x) * scale)


def describe_dist(spec: dict) -> str:
    try:
        return f"{_kind(spec)} (mean {dist_mean(spec):.1f} s, cv {dist_cv(spec):.2f})"
    except Exception as exc:  # pragma: no cover - display only
        return f"invalid ({exc})"


# =============================================================================
# Behaviour accessor
# =============================================================================


class Behavior:
    """Look-ups into a data dict with defaults for anything missing."""

    def __init__(self, data: Optional[dict]):
        self.data = data or {}

    def _sec(self, key: str) -> dict:
        v = self.data.get(key)
        return v if isinstance(v, dict) else {}

    # ---- simulation --------------------------------------------------------------
    def sim_settings(self) -> dict:
        s = {**DEFAULTS["simulation"], **self._sec("simulation")}
        return {"horizon_s": float(s["horizon_s"]), "warmup_s": float(s["warmup_s"]), "seed": int(s["seed"])}

    # ---- processes -----------------------------------------------------------------
    def has_process_time(self, pid: str) -> bool:
        return pid in self._sec("process_times")

    def process_time_spec(self, pid: str) -> dict:
        pt = self._sec("process_times")
        spec = pt.get(pid) or pt.get("default") or DEFAULTS["process_time"]
        return spec

    def sample_process_time(self, pid: str, rng: np.random.Generator, operator: dict) -> float:
        return sample(self.process_time_spec(pid), rng, scale=operator["speed_factor"],
                      cv_factor=operator["cv_factor"])

    def defect_probability(self, pid: str) -> float:
        dp = self._sec("defect_probability")
        return float(dp.get(pid, dp.get("default", 0.0)) or 0.0)

    def pick_window(self) -> float:
        return min(1.0, max(0.0, float(self.data.get("pick_window", DEFAULTS["pick_window"]))))

    # ---- operators -----------------------------------------------------------------
    def operator(self, oid: str) -> dict:
        ops = self._sec("operators")
        spec = {**DEFAULTS["operator"], **(ops.get("default") or {}), **(ops.get(oid) or {})}
        return {"speed_factor": float(spec["speed_factor"]), "cv_factor": float(spec["cv_factor"])}

    # ---- failures ------------------------------------------------------------------
    def failure(self, sid: str) -> Optional[dict]:
        f = self._sec("failures")
        spec = {**(f.get("default") or {}), **(f.get(sid) or {})}
        mtbf = spec.get("mtbf_s")
        if mtbf in (None, 0, "null"):
            return None
        rt = spec.get("repair_time")
        if not isinstance(rt, dict):
            rt = {"dist": "exponential", "mean": float(spec.get("mttr_s", 120.0))}
        return {"mtbf_s": float(mtbf), "repair_time": rt}

    def availability(self, sid: str) -> float:
        f = self.failure(sid)
        if not f:
            return 1.0
        mttr = dist_mean(f["repair_time"])
        return f["mtbf_s"] / (f["mtbf_s"] + mttr)

    # ---- components, bins, replenishment -------------------------------------------
    def unit_weight(self, cid: Optional[str]) -> float:
        if cid is None:
            return 0.0
        comps = self._sec("components")
        spec = {**(comps.get("default") or {}), **(comps.get(cid) or {})}
        return float(spec.get("unit_weight_g", DEFAULTS["unit_weight_g"]))

    def bin_params(self, cid: Optional[str]) -> dict:
        b = self._sec("bins")
        spec = {**DEFAULTS["bin"], **(b.get("default") or {}), **((b.get(cid) or {}) if cid else {})}
        return {"capacity_units": int(spec["capacity_units"]),
                "reorder_point_units": int(spec["reorder_point_units"]),
                "tare_g": float(spec["tare_g"])}

    def lead_time_spec(self) -> dict:
        r = self._sec("replenishment")
        return r.get("lead_time") or DEFAULTS["lead_time"]

    # ---- sensors -------------------------------------------------------------------
    def load_cell_params(self) -> dict:
        return {**DEFAULTS["load_cells"], **self._sec("load_cells")}

    def camera_params(self) -> dict:
        c = {**DEFAULTS["camera"], **self._sec("camera")}
        if isinstance(c.get("confidence"), dict):  # also accept {confidence: {a:, b:}}
            c["confidence_a"] = c["confidence"].get("a", c["confidence_a"])
            c["confidence_b"] = c["confidence"].get("b", c["confidence_b"])
        return c

    def detection_params(self) -> dict:
        return {**DEFAULTS["pick_detection"], **self._sec("pick_detection")}

    # ---- derived -------------------------------------------------------------------
    def expected_station_time(self, L: LineStructure, sid: str) -> float:
        op = self.operator(str(L.stations.get(sid, {}).get("operator")))
        return sum(dist_mean(self.process_time_spec(p)) for p in L.station_processes(sid)) * op["speed_factor"]


# =============================================================================
# Analytic capacity check (no simulation) and data/structure consistency
# =============================================================================


def theoretical_capacity(cfg: dict, data: dict) -> Tuple[pd.DataFrame, float, str]:
    """Upper bound on throughput from mean times, parallelism, shared operators and
    availability. Ignores variability, blocking and material shortages."""
    L = LineStructure(cfg)
    B = Behavior(data)
    rows = []
    for i, stage in enumerate(L.stages):
        caps, works = [], []
        for s in stage:
            t = B.expected_station_time(L, s)
            works.append(t)
            caps.append(3600.0 * B.availability(s) / t if t > 0 else math.inf)
        rows.append({"resource": " | ".join(stage), "kind": "stage", "work_per_unit_s": float(np.mean(works)),
                     "parallel": len(stage), "availability": float(np.mean([B.availability(s) for s in stage])),
                     "capacity_uph": float(sum(caps))})
    op_st: Dict[str, List[str]] = {}
    for s in L.routed_stations:
        op_st.setdefault(str(L.stations.get(s, {}).get("operator")), []).append(s)
    for op, ss in op_st.items():
        if len(ss) < 2:
            continue
        work = sum(B.expected_station_time(L, s) / len(L.stages[L.stage_index[s]]) for s in ss)
        rows.append({"resource": f"operator {op} ({', '.join(ss)})", "kind": "shared operator",
                     "work_per_unit_s": work, "parallel": 1, "availability": 1.0,
                     "capacity_uph": 3600.0 / work if work > 0 else math.inf})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, float("nan"), ""
    i = int(df["capacity_uph"].idxmin())
    return df, float(df.loc[i, "capacity_uph"]), str(df.loc[i, "resource"])


def check_data(data: dict, cfg: dict) -> Report:
    """Does the data file cover the objects in the structure file?"""
    R = Report()
    L = LineStructure(cfg)
    B = Behavior(data)
    pt = B._sec("process_times")
    for p in L.processes:
        if p not in pt:
            R.warnings.append(f"No time data for {p} - using the default {describe_dist(B.process_time_spec(p))}.")
    for p in pt:
        if p != "default" and p not in L.processes:
            R.info.append(f"Time data for {p} is not used by this structure (kept for other configurations).")
    for p in list(pt):
        try:
            m = dist_mean(pt[p])
            if m <= 0:
                R.errors.append(f"Process time of {p} has a non-positive mean.")
        except Exception as exc:
            R.errors.append(f"Process time of {p} is invalid: {exc}")
    comps = B._sec("components")
    used = {L.bin_component(b) for b in L.bins} - {None}
    for c in sorted(used):
        if c not in comps:
            R.warnings.append(f"No unit weight for {c} - load cells assume {B.unit_weight(c):.1f} g.")
    ops = B._sec("operators")
    for o in L.operators:
        if o not in ops:
            R.info.append(f"Operator {o} uses default variability (speed 1.0, cv factor 1.0).")
    return R


def scale_process_times(data: dict, factors: Dict[str, float]) -> dict:
    """Return a copy of `data` with process means multiplied by per-process factors."""
    new = copy.deepcopy(data)
    pt = new.setdefault("process_times", {})
    for p, f in factors.items():
        spec = dict(pt.get(p) or pt.get("default") or DEFAULTS["process_time"])
        d = _kind(spec)
        if d in ("lognormal", "normal", "gamma", "exponential"):
            spec["mean"] = float(spec["mean"]) * f
        elif d in ("constant", "fixed", "deterministic"):
            spec["value"] = dist_mean(spec) * f
            spec.pop("mean", None)
        elif d in ("triangular", "uniform"):
            for k in ("min", "mode", "max"):
                if k in spec:
                    spec[k] = float(spec[k]) * f
        elif d == "empirical":
            spec["samples"] = [float(x) * f for x in spec.get("samples", [])]
        pt[p] = spec
    return new
