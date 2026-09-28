"""
simulator.py - configuration-driven discrete-event simulation of the line.

The model is BUILT from the two files, it is not hard-coded:
  * routing / stages / parallel stations / buffers ........ from the config file
  * which processes each station performs, in which order .. from the config file
  * which bin holds which component, which load cell reads it from the config file
  * which operator works where (shared operators allowed) .. from the config file
  * process-time distributions, defects, failures, weights,
    bin stocking, replenishment, camera behaviour ........... from the data file

What happens to one unit at one station
  1. the station needs an input unit (stage 1: always available; otherwise from the
     upstream buffer or directly from a blocked upstream station), its operator
     free, and no pending failure;
  2. for every process assigned to the station (in order) a process time T is drawn;
     the operator picks each component input from its bin (first pick at t=0, the
     others spread over the first `pick_window` of T) -> bin weight drops, the load
     cell sees it; an empty bin makes the station wait for material;
  3. at the end the camera reports "unit_complete" and the unit moves on; if the
     downstream buffer is full the station is BLOCKED holding the unit.
Bins that reach their reorder point are refilled after a random lead time.

Station states tracked:  busy, wait_material, down, wait_operator, blocked, starved.
"""
from __future__ import annotations

import heapq
import math
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .behavior import Behavior, make_rng, sample, theoretical_capacity
    from .config_model import ConfigError, LineStructure, validate
except ImportError:  # files placed side by side without the line_sim folder
    from behavior import Behavior, make_rng, sample, theoretical_capacity
    from config_model import ConfigError, LineStructure, validate

STATES = ["busy", "wait_material", "down", "wait_operator", "blocked", "starved"]
STATE_LABELS = {
    "busy": "Working",
    "wait_material": "Waiting for material",
    "down": "Down (failure)",
    "wait_operator": "Waiting for operator",
    "blocked": "Blocked (downstream full)",
    "starved": "Starved (no input)",
}
STATE_COLORS = {
    "busy": "#2E7D32",
    "wait_material": "#FDD835",
    "down": "#C62828",
    "wait_operator": "#6A1B9A",
    "blocked": "#EF6C00",
    "starved": "#90A4AE",
}


# =============================================================================
# Internal state objects
# =============================================================================


@dataclass
class _Unit:
    uid: int
    t_release: float
    defect: bool = False
    defect_process: Optional[str] = None


class _Bin:
    def __init__(self, bid, station, component, sensor, capacity, reorder, unit_w, tare):
        self.bid, self.station, self.component, self.sensor = bid, station, component, sensor
        self.capacity, self.reorder, self.unit_w, self.tare = capacity, reorder, unit_w, tare
        self.units = capacity if component else 0
        self.min_units = self.units
        self.refill_pending = False
        self.waiters: set = set()
        self.picks = 0
        self.parts = 0
        self.refills = 0
        self.stockouts = 0

    @property
    def weight(self) -> float:
        return self.tare + self.units * self.unit_w


class _Station:
    def __init__(self, sid, stage, operator, processes, camera):
        self.sid, self.stage, self.operator, self.processes, self.camera = sid, stage, operator, processes, camera
        self.state = "starved"
        self.since = 0.0
        self.unit: Optional[_Unit] = None
        self.finished = False           # holding a finished unit that could not move on
        self.blocked_since = 0.0
        self.steps: list = []
        self.step_i = 0
        self.token = 0
        self.job_start = 0.0
        self.tis: Dict[str, float] = defaultdict(float)   # time in state, inside KPI window
        self.busy_total = 0.0                             # all busy time (failure clock)
        self.next_fail_busy = math.inf
        self.failures = 0
        self.material_waits = 0
        self.timeline: List[Tuple[float, str]] = [(0.0, "starved")]


class _Operator:
    def __init__(self, oid):
        self.oid = oid
        self.busy_with: Optional[str] = None
        self.since = 0.0
        self.busy_time = 0.0
        self.jobs = 0
        self.stations: List[str] = []


class _Buffer:
    def __init__(self, bid, k, capacity):
        self.bid, self.k, self.cap = bid, k, capacity
        self.q: deque = deque()
        self.last_t = 0.0
        self.area = 0.0
        self.full_time = 0.0
        self.max_level = 0


@dataclass
class SimResult:
    kpis: dict
    stations: pd.DataFrame
    operators: pd.DataFrame
    buffers: pd.DataFrame
    bins: pd.DataFrame
    units: pd.DataFrame
    jobs: pd.DataFrame
    picks: pd.DataFrame
    bin_levels: pd.DataFrame
    camera: pd.DataFrame
    events: pd.DataFrame
    wip_series: pd.DataFrame
    buffer_series: pd.DataFrame
    state_timeline: pd.DataFrame
    capacity: pd.DataFrame
    config: dict
    data: dict
    horizon_s: float
    warmup_s: float
    seed: int
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])


def _overlap(a: float, b: float, lo: float, hi: float) -> float:
    return max(0.0, min(b, hi) - max(a, lo))


# =============================================================================
# The simulator
# =============================================================================


class LineSimulator:
    def __init__(self, cfg: dict, data: dict, horizon_s: Optional[float] = None,
                 warmup_s: Optional[float] = None, seed: Optional[int] = None,
                 record_events: bool = True, record_sensors: bool = True):
        report = validate(cfg)
        if not report.ok:
            raise ConfigError(report)
        self.cfg, self.data = cfg, data or {}
        self.L = LineStructure(cfg)
        self.B = Behavior(self.data)
        sims = self.B.sim_settings()
        self.H = float(horizon_s if horizon_s is not None else sims["horizon_s"])
        self.W = float(warmup_s if warmup_s is not None else sims["warmup_s"])
        self.W = min(max(0.0, self.W), 0.9 * self.H)
        self.seed = int(seed if seed is not None else sims["seed"])
        self.record_events, self.record_sensors = record_events, record_sensors
        self._build()

    # ---------------------------------------------------------------- building
    def _build(self):
        L, B, seed = self.L, self.B, self.seed
        self.now = 0.0
        self._heap: list = []
        self._seq = 0
        self.stages = [list(st) for st in L.stages]
        self.stations: Dict[str, _Station] = {}
        for i, stage in enumerate(self.stages):
            for sid in stage:
                spec = L.stations[sid]
                self.stations[sid] = _Station(sid, i, str(spec["operator"]), L.station_processes(sid),
                                              spec.get("camera"))
        self.operators: Dict[str, _Operator] = {}
        for st in self.stations.values():
            self.operators.setdefault(st.operator, _Operator(st.operator)).stations.append(st.sid)
        for op in self.operators.values():  # downstream stations get the operator first
            op.stations.sort(key=lambda s: -self.stations[s].stage)
        self.buffers: List[_Buffer] = []
        for k in range(len(self.stages) - 1):
            bid, cap = L.buffer_between(k)
            self.buffers.append(_Buffer(bid or f"direct {'|'.join(self.stages[k])}->{'|'.join(self.stages[k + 1])}",
                                        k, cap))
        self.recipes = {p: L.component_inputs(p) for p in L.processes}

        # logs
        self.log_events: list = []
        self.log_picks: list = []
        self.log_bins: list = []
        self.log_camera: list = []
        self.log_jobs: list = []
        self.log_units: list = []
        self.wip = 0
        self.wip_area = 0.0
        self.wip_last = 0.0
        self.wip_series = [(0.0, 0)]
        self.buffer_series = [(0.0, b.bid, 0) for b in self.buffers]
        self.unit_counter = 0

        # bins
        self.bins: Dict[str, _Bin] = {}
        self.bins_by_sc: Dict[Tuple[str, str], List[_Bin]] = defaultdict(list)
        for bid in sorted(L.bins):
            st = str(L.bins[bid].get("station"))
            if st not in self.stations:
                continue
            comp = L.bin_component(bid)
            bp = B.bin_params(comp)
            b = _Bin(bid, st, comp, L.sensor_of_bin.get(bid), bp["capacity_units"], bp["reorder_point_units"],
                     B.unit_weight(comp), bp["tare_g"])
            self.bins[bid] = b
            if comp:
                self.bins_by_sc[(st, comp)].append(b)
            self._log_bin(b, "init")

        # random streams (one per purpose/object -> common random numbers across configurations)
        self.rng_proc = {p: make_rng(seed, "ptime", p) for p in L.processes}
        self.rng_defect = {p: make_rng(seed, "defect", p) for p in L.processes}
        self.rng_fail = {s: make_rng(seed, "fail", s) for s in self.stations}
        self.rng_cam = {s: make_rng(seed, "camera", s) for s in self.stations}
        self.rng_repl = make_rng(seed, "replenish")
        self.cam_params = B.camera_params()
        self.op_params = {o: B.operator(o) for o in self.operators}
        self.pick_window = B.pick_window()
        for st in self.stations.values():
            f = B.failure(st.sid)
            if f:
                st.next_fail_busy = self.rng_fail[st.sid].exponential(f["mtbf_s"])

    # ---------------------------------------------------------------- engine
    def _schedule(self, delay: float, kind: str, payload=None):
        self._seq += 1
        heapq.heappush(self._heap, (self.now + max(0.0, delay), self._seq, kind, payload))

    def _wake(self, sid: str):
        self._schedule(0.0, "wake", sid)

    def run(self) -> SimResult:
        for sid in self.stages[0]:
            self._wake(sid)
        handlers = {"wake": self._on_wake, "step": self._on_step, "refill": self._on_refill,
                    "repair": self._on_repair, "resume": self._on_resume}
        while self._heap:
            t, _, kind, payload = heapq.heappop(self._heap)
            if t > self.H:
                break
            self.now = t
            handlers[kind](payload)
        self.now = self.H
        return self._finalize()

    # ---------------------------------------------------------------- logging helpers
    def _log(self, event: str, station: Optional[str] = None, **kw):
        if self.record_events:
            self.log_events.append({"t": self.now, "event": event, "station": station, **kw})

    def _log_bin(self, b: _Bin, kind: str):
        if self.record_sensors:
            self.log_bins.append({"t": self.now, "bin": b.bid, "sensor": b.sensor, "station": b.station,
                                  "component": b.component, "units": b.units, "weight_g": b.weight, "kind": kind})

    def _camera(self, st: _Station, action: str, unit: Optional[int], bin_id: Optional[str] = None):
        if not st.camera or not self.record_sensors:
            return
        rng = self.rng_cam[st.sid]
        has = bool(rng.random() >= float(self.cam_params["dropout"]))
        conf = float(rng.beta(float(self.cam_params["confidence_a"]), float(self.cam_params["confidence_b"]))) \
            if has else float("nan")
        self.log_camera.append({"t": self.now, "camera": st.camera, "station": st.sid, "action": action,
                                "unit": unit, "bin": bin_id, "has_frame": has, "confidence": conf})

    def _set_state(self, st: _Station, new: str):
        if st.state == new:
            return
        st.tis[st.state] += _overlap(st.since, self.now, self.W, self.H)
        if st.state == "busy":
            st.busy_total += self.now - st.since
        st.state, st.since = new, self.now
        st.timeline.append((self.now, new))

    def _wip_change(self, delta: int):
        self.wip_area += self.wip * _overlap(self.wip_last, self.now, self.W, self.H)
        self.wip_last = self.now
        self.wip += delta
        self.wip_series.append((self.now, self.wip))

    def _buffer_changed(self, buf: _Buffer, old_level: int):
        dt = _overlap(buf.last_t, self.now, self.W, self.H)
        buf.area += old_level * dt
        if buf.cap > 0 and old_level >= buf.cap:
            buf.full_time += dt
        buf.last_t = self.now
        buf.max_level = max(buf.max_level, len(buf.q))
        self.buffer_series.append((self.now, buf.bid, len(buf.q)))

    def _buf_push(self, buf: _Buffer, unit: _Unit):
        old = len(buf.q)
        buf.q.append(unit)
        self._buffer_changed(buf, old)

    def _buf_pop(self, buf: _Buffer) -> _Unit:
        old = len(buf.q)
        u = buf.q.popleft()
        self._buffer_changed(buf, old)
        return u

    # ---------------------------------------------------------------- handlers
    def _on_wake(self, sid: str):
        self._try_start(self.stations[sid])

    def _on_step(self, payload):
        sid, token = payload
        st = self.stations[sid]
        if st.token == token and st.unit is not None and not st.finished:
            self._advance(st)

    def _on_resume(self, sid: str):
        st = self.stations[sid]
        if st.state == "wait_material" and st.unit is not None and not st.finished:
            self._advance(st)

    def _on_refill(self, bid: str):
        b = self.bins[bid]
        b.units = b.capacity
        b.refill_pending = False
        b.refills += 1
        self._log_bin(b, "refill")
        self._log("refill", b.station, bin=bid, component=b.component)
        for sid in sorted(b.waiters):
            self._schedule(0.0, "resume", sid)
        b.waiters.clear()

    def _on_repair(self, sid: str):
        st = self.stations[sid]
        f = self.B.failure(sid)
        st.next_fail_busy = st.busy_total + (self.rng_fail[sid].exponential(f["mtbf_s"]) if f else math.inf)
        self._set_state(st, "starved")
        self._log("repaired", sid)
        self._try_start(st)

    # ---------------------------------------------------------------- flow logic
    def _input_source(self, st: _Station):
        if st.stage == 0:
            return ("new", None)
        buf = self.buffers[st.stage - 1]
        if buf.q:
            return ("buffer", buf)
        blocked = [self.stations[u] for u in self.stages[st.stage - 1] if self.stations[u].finished]
        if blocked:
            return ("handoff", min(blocked, key=lambda s: s.blocked_since))
        return None

    def _try_start(self, st: _Station):
        if st.unit is not None or st.state == "down":
            return
        src = self._input_source(st)
        if src is None:
            self._set_state(st, "starved")
            return
        op = self.operators[st.operator]
        if op.busy_with is not None:
            self._set_state(st, "wait_operator")
            return
        f = self.B.failure(st.sid)
        if f and st.busy_total >= st.next_fail_busy:
            ttr = sample(f["repair_time"], self.rng_fail[st.sid])
            st.failures += 1
            self._set_state(st, "down")
            self._log("failure", st.sid, duration_s=ttr)
            self._schedule(ttr, "repair", st.sid)
            return
        unit = self._take_input(st, src)
        self._start_job(st, unit)

    def _take_input(self, st: _Station, src) -> _Unit:
        kind, obj = src
        if kind == "new":
            self.unit_counter += 1
            unit = _Unit(self.unit_counter, self.now)
            self._wip_change(+1)
            self._log("release", st.sid, unit=unit.uid)
            return unit
        if kind == "buffer":
            unit = self._buf_pop(obj)
            self._release_blocked(obj.k)
            return unit
        up: _Station = obj  # direct hand-off from a blocked upstream station
        unit = up.unit
        up.unit, up.finished = None, False
        self._set_state(up, "starved")
        self._wake(up.sid)
        return unit

    def _release_blocked(self, k: int):
        """Space freed in buffer k: move finished units of blocked upstream stations into it."""
        buf = self.buffers[k]
        while len(buf.q) < buf.cap:
            blocked = [self.stations[u] for u in self.stages[k] if self.stations[u].finished]
            if not blocked:
                break
            up = min(blocked, key=lambda s: s.blocked_since)
            self._buf_push(buf, up.unit)
            up.unit, up.finished = None, False
            self._set_state(up, "starved")
            self._wake(up.sid)

    def _start_job(self, st: _Station, unit: _Unit):
        op = self.operators[st.operator]
        op.busy_with, op.since = st.sid, self.now
        opp = self.op_params[st.operator]
        st.unit, st.finished = unit, False
        st.token += 1
        st.job_start = self.now
        st.step_i = 0
        steps = []
        for p in st.processes:
            T = self.B.sample_process_time(p, self.rng_proc[p], opp)
            comps = self.recipes.get(p, [])
            n = len(comps)
            offs = [0.0] if n == 1 else [self.pick_window * T * i / (n - 1) for i in range(n)]
            prev = 0.0
            for (c, q), off in zip(comps, offs):
                if off - prev > 1e-9:
                    steps.append(("work", off - prev, p))
                steps.append(("pick", c, q, p))
                prev = off
            steps.append(("work", max(0.0, T - prev), p))
            steps.append(("done", p, T))
            if self.rng_defect[p].random() < self.B.defect_probability(p) and not unit.defect:
                unit.defect, unit.defect_process = True, p
        st.steps = steps
        self._set_state(st, "busy")
        self._log("start", st.sid, unit=unit.uid, operator=st.operator)
        self._camera(st, "start_work", unit.uid)
        self._advance(st)

    def _advance(self, st: _Station):
        while st.step_i < len(st.steps):
            step = st.steps[st.step_i]
            kind = step[0]
            if kind == "work":
                st.step_i += 1
                if step[1] > 0:
                    self._schedule(step[1], "step", (st.sid, st.token))
                    return
                continue
            if kind == "pick":
                _, comp, qty, p = step
                if not self._pick(st, comp, qty, p):
                    if st.state != "wait_material":
                        st.material_waits += 1
                        self._set_state(st, "wait_material")
                        self._log("material_wait", st.sid, unit=st.unit.uid, component=comp)
                    return
                if st.state != "busy":
                    self._set_state(st, "busy")
                st.step_i += 1
                continue
            if kind == "done":
                self._log("process_done", st.sid, unit=st.unit.uid, process=step[1], time_s=step[2])
                st.step_i += 1
                continue
        self._finish_job(st)

    def _pick(self, st: _Station, comp: str, qty: int, process: str) -> bool:
        bins = self.bins_by_sc.get((st.sid, comp), [])
        if sum(b.units for b in bins) < qty:
            for b in bins:
                b.waiters.add(st.sid)
                b.stockouts += 1
                self._request_refill(b)
            return False
        remaining = qty
        for b in bins:
            take = min(b.units, remaining)
            if take <= 0:
                continue
            b.units -= take
            b.min_units = min(b.min_units, b.units)
            b.picks += 1
            b.parts += take
            remaining -= take
            if self.record_sensors:
                self.log_picks.append({"t": self.now, "station": st.sid, "bin": b.bid, "sensor": b.sensor,
                                       "component": comp, "qty": take, "unit": st.unit.uid, "process": process})
            self._log_bin(b, "pick")
            self._camera(st, "pick", st.unit.uid, b.bid)
            if b.units <= b.reorder:
                self._request_refill(b)
            if remaining == 0:
                break
        return True

    def _request_refill(self, b: _Bin):
        if b.refill_pending or b.component is None:
            return
        b.refill_pending = True
        lead = sample(self.B.lead_time_spec(), self.rng_repl)
        self._log("refill_request", b.station, bin=b.bid, component=b.component, lead_time_s=lead)
        self._schedule(lead, "refill", b.bid)

    def _finish_job(self, st: _Station):
        unit = st.unit
        self.log_jobs.append({"station": st.sid, "stage": st.stage + 1, "operator": st.operator, "unit": unit.uid,
                              "t_start": st.job_start, "t_end": self.now, "service_s": self.now - st.job_start})
        op = self.operators[st.operator]
        op.busy_time += _overlap(op.since, self.now, self.W, self.H)
        op.busy_with = None
        op.jobs += 1
        self._camera(st, "unit_complete", unit.uid)
        self._log("finish", st.sid, unit=unit.uid)
        st.finished = True
        for sid in op.stations:  # operator is free again
            if sid != st.sid:
                self._wake(sid)
        self._try_push(st)

    def _try_push(self, st: _Station):
        unit = st.unit
        last = len(self.stages) - 1
        if st.stage == last:
            self._complete(unit)
            st.unit, st.finished = None, False
            self._set_state(st, "starved")
            self._wake(st.sid)
            return
        buf = self.buffers[st.stage]
        if len(buf.q) < buf.cap:
            self._buf_push(buf, unit)
            st.unit, st.finished = None, False
            self._set_state(st, "starved")
            for d in self.stages[st.stage + 1]:
                self._wake(d)
            self._wake(st.sid)
            return
        st.blocked_since = self.now
        self._set_state(st, "blocked")
        for d in self.stages[st.stage + 1]:  # a free downstream station can take it directly
            self._wake(d)

    def _complete(self, unit: _Unit):
        self._wip_change(-1)
        self.log_units.append({"unit": unit.uid, "t_release": unit.t_release, "t_complete": self.now,
                               "lead_time_s": self.now - unit.t_release, "defect": unit.defect,
                               "defect_process": unit.defect_process})
        self._log("complete", None, unit=unit.uid, defect=unit.defect)

    # ---------------------------------------------------------------- results
    def _finalize(self) -> SimResult:
        H, W = self.H, self.W
        window = max(1e-9, H - W)
        for st in self.stations.values():
            st.tis[st.state] += _overlap(st.since, H, W, H)
        for op in self.operators.values():
            if op.busy_with is not None:
                op.busy_time += _overlap(op.since, H, W, H)
        self.wip_area += self.wip * _overlap(self.wip_last, H, W, H)
        for buf in self.buffers:
            lvl = len(buf.q)
            dt = _overlap(buf.last_t, H, W, H)
            buf.area += lvl * dt
            if buf.cap > 0 and lvl >= buf.cap:
                buf.full_time += dt

        units = pd.DataFrame(self.log_units, columns=["unit", "t_release", "t_complete", "lead_time_s", "defect",
                                                      "defect_process"])
        done = units[units["t_complete"] >= W]
        n_done = int(len(done))
        good = int((~done["defect"].astype(bool)).sum()) if n_done else 0
        jobs = pd.DataFrame(self.log_jobs, columns=["station", "stage", "operator", "unit", "t_start", "t_end",
                                                    "service_s"])
        cap_df, cap_max, cap_bneck = theoretical_capacity(self.cfg, self.data)

        # stations
        rows = []
        for st in self.stations.values():
            j = jobs[(jobs["station"] == st.sid) & (jobs["t_end"] >= W)]
            row = {"station": st.sid, "stage": st.stage + 1, "operator": st.operator,
                   "processes": ", ".join(st.processes), "jobs": int(len(j)),
                   "mean_service_s": float(j["service_s"].mean()) if len(j) else float("nan"),
                   "model_time_s": self.B.expected_station_time(self.L, st.sid),
                   "failures": st.failures, "material_waits": st.material_waits}
            for s in STATES:
                row[s] = st.tis.get(s, 0.0) / window
            row["active"] = row["busy"] + row["down"] + row["wait_material"]
            rows.append(row)
        stations = pd.DataFrame(rows)
        bi = int(stations["active"].idxmax())
        bottleneck = str(stations.loc[bi, "station"])

        jw = jobs[jobs["t_end"] >= W]
        operators = pd.DataFrame([{"operator": o.oid, "stations": ", ".join(sorted(o.stations)),
                                   "utilization": o.busy_time / window,
                                   "jobs": int((jw["operator"] == o.oid).sum())}
                                  for o in self.operators.values()])
        buffers = pd.DataFrame([{"buffer": b.bid, "between": f"{'|'.join(self.stages[b.k])} -> "
                                                            f"{'|'.join(self.stages[b.k + 1])}",
                                 "capacity": b.cap, "avg_level": b.area / window, "max_level": b.max_level,
                                 "time_full": (b.full_time / window) if b.cap > 0 else float("nan")}
                                for b in self.buffers],
                               columns=["buffer", "between", "capacity", "avg_level", "max_level", "time_full"])
        bins = pd.DataFrame([{"bin": b.bid, "sensor": b.sensor, "station": b.station, "component": b.component,
                              "unit_weight_g": b.unit_w, "capacity": b.capacity, "reorder_point": b.reorder,
                              "picks": b.picks, "parts": b.parts, "refills": b.refills, "stockouts": b.stockouts,
                              "min_units": b.min_units, "end_units": b.units}
                             for b in self.bins.values()])

        timeline = []
        for st in self.stations.values():
            tl = st.timeline
            for i, (t0, s) in enumerate(tl):
                t1 = tl[i + 1][0] if i + 1 < len(tl) else H
                if t1 > t0:
                    timeline.append((st.sid, t0, t1, s))
        state_timeline = pd.DataFrame(timeline, columns=["station", "t0", "t1", "state"])

        lt = done["lead_time_s"] if n_done else pd.Series(dtype=float)
        th = n_done / window * 3600.0
        wip_avg = self.wip_area / window
        kpis = {
            "throughput_uph": th,
            "completed": n_done,
            "good": good,
            "fpy": good / n_done if n_done else float("nan"),
            "lead_time_mean_s": float(lt.mean()) if n_done else float("nan"),
            "lead_time_p50_s": float(lt.median()) if n_done else float("nan"),
            "lead_time_p90_s": float(lt.quantile(0.9)) if n_done else float("nan"),
            "wip_avg": wip_avg,
            "littles_law_wip": (th / 3600.0) * float(lt.mean()) if n_done else float("nan"),
            "bottleneck": bottleneck,
            "bottleneck_active": float(stations.loc[bi, "active"]),
            "theoretical_max_uph": cap_max,
            "theoretical_bottleneck": cap_bneck,
            "released": self.unit_counter,
            "in_process_end": self.wip,
            "failures": int(sum(s.failures for s in self.stations.values())),
            "material_waits": int(sum(s.material_waits for s in self.stations.values())),
            "avg_utilization": float(stations["busy"].mean()),
            "n_stations": len(self.stations),
            "n_operators": len(self.operators),
            "horizon_s": H,
            "warmup_s": W,
            "seed": self.seed,
        }
        cam = pd.DataFrame(self.log_camera, columns=["t", "camera", "station", "action", "unit", "bin", "has_frame",
                                                     "confidence"])
        if len(cam):
            kpis["camera_frame_rate"] = float(cam["has_frame"].mean())
        return SimResult(
            kpis=kpis, stations=stations, operators=operators, buffers=buffers, bins=bins, units=units, jobs=jobs,
            picks=pd.DataFrame(self.log_picks, columns=["t", "station", "bin", "sensor", "component", "qty", "unit",
                                                        "process"]),
            bin_levels=pd.DataFrame(self.log_bins, columns=["t", "bin", "sensor", "station", "component", "units",
                                                            "weight_g", "kind"]),
            camera=cam, events=pd.DataFrame(self.log_events),
            wip_series=pd.DataFrame(self.wip_series, columns=["t", "wip"]),
            buffer_series=pd.DataFrame(self.buffer_series, columns=["t", "buffer", "level"]),
            state_timeline=state_timeline, capacity=cap_df, config=self.cfg, data=self.data,
            horizon_s=H, warmup_s=W, seed=self.seed,
        )


def simulate(cfg: dict, data: dict, **kw) -> SimResult:
    return LineSimulator(cfg, data, **kw).run()


# =============================================================================
# Replications and comparison of configurations
# =============================================================================

_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}

SUMMARY_METRICS = ["throughput_uph", "lead_time_mean_s", "wip_avg", "fpy", "avg_utilization", "bottleneck_active",
                   "material_waits", "failures"]


def t975(dof: int) -> float:
    if dof <= 0:
        return float("nan")
    keys = [k for k in sorted(_T975) if k <= dof]
    return _T975[keys[-1]] if dof <= 30 else 1.96


def replicate(cfg: dict, data: dict, reps: int = 5, horizon_s: Optional[float] = None,
              warmup_s: Optional[float] = None, base_seed: Optional[int] = None,
              progress: Optional[Callable[[int], None]] = None) -> pd.DataFrame:
    """Independent replications. Replication r uses seed base_seed + r in every
    configuration (common random numbers -> fairer comparisons)."""
    base = int(base_seed if base_seed is not None else Behavior(data).sim_settings()["seed"])
    rows = []
    for r in range(int(reps)):
        res = LineSimulator(cfg, data, horizon_s=horizon_s, warmup_s=warmup_s, seed=base + r,
                            record_events=False, record_sensors=False).run()
        rows.append({"replication": r + 1, **{k: v for k, v in res.kpis.items()
                                              if isinstance(v, (int, float, str, np.floating, np.integer))}})
        if progress:
            progress(r + 1)
    return pd.DataFrame(rows)


def summarize_replications(df: pd.DataFrame, group: str = "scenario") -> pd.DataFrame:
    rows = []
    for name, g in df.groupby(group, sort=False):
        n = len(g)
        row = {group: name, "replications": n}
        for m in SUMMARY_METRICS:
            if m not in g:
                continue
            x = pd.to_numeric(g[m], errors="coerce").dropna()
            row[f"{m}_mean"] = float(x.mean()) if len(x) else float("nan")
            row[f"{m}_ci95"] = float(t975(len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")
        row["bottleneck"] = g["bottleneck"].mode().iat[0] if "bottleneck" in g and len(g) else ""
        row["bottleneck_share"] = float((g["bottleneck"] == row["bottleneck"]).mean()) if "bottleneck" in g else float("nan")
        for c in ("theoretical_max_uph", "theoretical_bottleneck", "n_stations", "n_operators"):
            if c in g:
                row[c] = g[c].iat[0]
        rows.append(row)
    return pd.DataFrame(rows)


def compare_scenarios(scenarios: Dict[str, dict], data: dict, reps: int = 5, horizon_s: Optional[float] = None,
                      warmup_s: Optional[float] = None, base_seed: Optional[int] = None,
                      progress: Optional[Callable[[float, str], None]] = None):
    """Run every configuration with the same data file and seeds.
    Returns (per-replication DataFrame, summary DataFrame, {skipped name: errors})."""
    frames, skipped = [], {}
    total = max(1, len(scenarios) * int(reps))
    done = 0
    for name, cfg in scenarios.items():
        rep = validate(cfg)
        if not rep.ok:
            skipped[name] = rep.errors
            done += int(reps)
            continue

        def _p(r, _name=name, _start=done):
            if progress:
                progress((_start + r) / total, f"{_name}: replication {r}/{reps}")

        df = replicate(cfg, data, reps, horizon_s, warmup_s, base_seed, progress=_p)
        df.insert(0, "scenario", name)
        frames.append(df)
        done += int(reps)
    per_rep = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = summarize_replications(per_rep) if len(per_rep) else pd.DataFrame()
    return per_rep, summary, skipped
