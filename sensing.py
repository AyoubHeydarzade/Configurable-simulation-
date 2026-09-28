"""
sensing.py - the sensor side of the twin, in the three phases of the architecture:

  Phase 1  Connectivity   raw load-cell weight per sensor (synthesised from the
                          simulation, or imported from the real testbed as CSV)
  Phase 2  Pick detection a meaningful, persistent weight drop -> discrete
                          "part removed" event (sensor, bin, component, parts, time)
  Phase 3  Timing         repeated, well-defined events -> cycle times and their
                          variability -> calibrated data file for the simulation

Sensor identity vs component identity: signals are keyed by the stable sensor ID
(e.g. LC_35). What the sensor means (bin B35 -> component C10, 22 g per part) is
looked up from the CURRENT configuration + data files, so a reconfiguration only
changes the look-up, not this code.
"""
from __future__ import annotations

import bisect
import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .behavior import Behavior, dist_cv, dist_mean, make_rng
    from .config_model import LineStructure
except ImportError:  # files placed side by side without the line_sim folder
    from behavior import Behavior, dist_cv, dist_mean, make_rng
    from config_model import LineStructure

DETECTION_COLUMNS = ["sensor", "bin", "station", "component", "t", "kind", "parts", "delta_g", "level_g"]


# =============================================================================
# Mapping sensor -> bin -> component -> unit weight (from the configuration)
# =============================================================================


def sensor_map(cfg: dict, data: dict) -> pd.DataFrame:
    L = LineStructure(cfg)
    B = Behavior(data)
    rows = []
    for lc in L.load_cells:
        b = L.sensors[lc].get("measures")
        b = str(b) if b is not None else None
        comp = L.bin_component(b) if b else None
        rows.append({"sensor": lc, "bin": b, "station": L.bins.get(b, {}).get("station") if b else None,
                     "component": comp, "component_name": L.name_of(comp) if comp else "",
                     "unit_weight_g": B.unit_weight(comp) if comp else float("nan"),
                     "label": L.sensor_label(lc)})
    return pd.DataFrame(rows)


# =============================================================================
# Phase 1 - raw signals
# =============================================================================


def synthesize_load_cell_signals(result, rate_hz: Optional[float] = None, noise_sd_g: Optional[float] = None,
                                 drift_g_per_h: Optional[float] = None, spike_prob: Optional[float] = None,
                                 spike_g: Optional[float] = None, resolution_g: Optional[float] = None,
                                 sensors: Optional[List[str]] = None, t0: float = 0.0, t1: Optional[float] = None,
                                 seed: Optional[int] = None) -> pd.DataFrame:
    """Sample the true bin weights of a simulation run like a real load cell would:
    fixed sample rate + Gaussian noise + slow drift + occasional hand-contact spikes."""
    p = Behavior(result.data).load_cell_params()
    rate = float(rate_hz if rate_hz is not None else p["sample_rate_hz"])
    noise = float(noise_sd_g if noise_sd_g is not None else p["noise_sd_g"])
    drift = float(drift_g_per_h if drift_g_per_h is not None else p["drift_g_per_h"])
    sp = float(spike_prob if spike_prob is not None else p["spike_prob"])
    sg = float(spike_g if spike_g is not None else p["spike_g"])
    res = float(resolution_g if resolution_g is not None else p["resolution_g"])
    seed = result.seed if seed is None else seed
    t1 = result.horizon_s if t1 is None else t1
    bl = result.bin_levels
    if bl is None or bl.empty:
        return pd.DataFrame(columns=["t", "sensor", "weight_g"])
    bl = bl[bl["sensor"].notna()]
    grid = np.arange(float(t0), float(t1), 1.0 / rate)
    frames = []
    for sensor, g in bl.groupby("sensor", sort=True):
        if sensors is not None and sensor not in sensors:
            continue
        g = g.sort_values("t", kind="stable")
        times = g["t"].to_numpy(float)
        weights = g["weight_g"].to_numpy(float)
        idx = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(times) - 1)
        w = weights[idx].copy()
        rng = make_rng(seed, "loadcell", sensor)
        if noise > 0:
            w += rng.normal(0.0, noise, len(w))
        if drift:
            w += drift * grid / 3600.0
        if sp > 0 and sg:
            m = rng.random(len(w)) < sp
            w[m] += rng.choice([-1.0, 1.0], m.sum()) * sg * rng.uniform(0.5, 1.0, m.sum())
        if res > 0:
            w = np.round(w / res) * res
        frames.append(pd.DataFrame({"t": grid, "sensor": sensor, "weight_g": w}))
    if not frames:
        return pd.DataFrame(columns=["t", "sensor", "weight_g"])
    return pd.concat(frames, ignore_index=True)


def _to_seconds(col: pd.Series) -> pd.Series:
    """Numeric times are kept; date-times become absolute epoch seconds, so load-cell and
    camera files share one clock (re-zero them together with `align_time`)."""
    if pd.api.types.is_numeric_dtype(col):
        return pd.to_numeric(col, errors="coerce").astype(float)
    ts = pd.to_datetime(col, errors="coerce", utc=True)
    return (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()


def align_time(*frames: Optional[pd.DataFrame]) -> float:
    """Shift all frames (in place) so the earliest time in any of them is 0; returns the origin."""
    mins = [f["t"].min() for f in frames if f is not None and len(f)]
    origin = float(min(mins)) if mins else 0.0
    for f in frames:
        if f is not None and len(f):
            f["t"] = f["t"] - origin
    return origin


def read_signal_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise an uploaded load-cell CSV to columns t (s), sensor, weight_g.
    Accepts e.g. timestamp/time/t/t_s, sensor/sensor_id/load_cell, weight_g/weight/value."""
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        raise ValueError(f"CSV needs one of the columns {names}; found {list(df.columns)}")

    tc = pick("t", "t_s", "time_s", "time", "timestamp", "datetime")
    sc = pick("sensor", "sensor_id", "load_cell", "loadcell", "lc")
    wc = pick("weight_g", "weight", "value", "grams", "reading")
    t = _to_seconds(df[tc])
    out = pd.DataFrame({"t": t, "sensor": df[sc].astype(str),
                        "weight_g": pd.to_numeric(df[wc], errors="coerce")}).dropna()
    return out.sort_values(["sensor", "t"], kind="stable").reset_index(drop=True)


def read_camera_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise an uploaded camera/action CSV to t, station, action, has_frame."""
    cols = {c.lower().strip(): c for c in df.columns}
    tc = next(cols[c] for c in ("t", "t_s", "time", "timestamp") if c in cols)
    t = _to_seconds(df[tc])
    out = pd.DataFrame({"t": t,
                        "station": df[cols.get("station", list(df.columns)[1])].astype(str),
                        "action": df[cols["action"]].astype(str) if "action" in cols else "unit_complete"})
    out["has_frame"] = df[cols["has_frame"]].astype(bool) if "has_frame" in cols else True
    return out.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)


# =============================================================================
# Phase 2 - pick detection
# =============================================================================


def detect_steps(t, w, unit_weight_g: Optional[float], noise_sd_g: float, threshold_fraction: float = 0.5,
                 noise_k_sigma: float = 4.0, persistence_samples: int = 2, ref_alpha: float = 0.05) -> List[dict]:
    """Debounced step detector.

    A change is accepted when |w - reference| exceeds the threshold for
    `persistence_samples` consecutive, mutually consistent samples. The size of
    the step divided by the part weight gives the number of parts; negative steps
    are picks, positive steps are refills. Single-sample spikes are rejected and
    slow drift is tracked by the reference level."""
    t = np.asarray(t, dtype=float)
    w = np.asarray(w, dtype=float)
    n = len(w)
    if n == 0:
        return []
    uw = float(unit_weight_g) if unit_weight_g and not math.isnan(unit_weight_g) else None
    thr = max((threshold_fraction * uw) if uw else 0.0, noise_k_sigma * noise_sd_g, 1e-6)
    persist = max(1, int(persistence_samples))
    ref = float(np.median(w[: min(n, max(3, persist))]))
    wl, tl = w.tolist(), t.tolist()
    out: List[dict] = []
    cand: List[int] = []
    for i in range(n):
        x = wl[i]
        if abs(x - ref) > thr:
            if cand and abs(x - wl[cand[0]]) > thr:
                cand = [i]  # the level moved again -> restart the candidate
            else:
                cand.append(i)
            if len(cand) >= persist:
                level = sum(wl[j] for j in cand) / len(cand)
                delta = level - ref
                parts = int(math.floor(abs(delta) / uw + 0.5)) if uw else 0
                if uw is None:
                    kind = "change"
                elif parts == 0:
                    kind = "unexplained"
                else:
                    kind = "pick" if delta < 0 else "refill"
                out.append({"t": tl[cand[0]], "kind": kind, "parts": parts, "delta_g": delta, "level_g": level})
                ref = level
                cand = []
        else:
            cand = []
            ref += ref_alpha * (x - ref)
    return out


def detect_all(signals: pd.DataFrame, cfg: dict, data: dict, noise_sd_g: Optional[float] = None,
               params: Optional[dict] = None) -> pd.DataFrame:
    """Run pick detection on every sensor, interpreting each sensor through the configuration."""
    if signals is None or signals.empty:
        return pd.DataFrame(columns=DETECTION_COLUMNS)
    B = Behavior(data)
    dp = {**B.detection_params(), **(params or {})}
    noise = float(noise_sd_g if noise_sd_g is not None else B.load_cell_params()["noise_sd_g"])
    smap = sensor_map(cfg, data).set_index("sensor")
    rows = []
    for sensor, g in signals.groupby("sensor", sort=True):
        info = smap.loc[sensor] if sensor in smap.index else None
        uw = float(info["unit_weight_g"]) if info is not None else float("nan")
        g = g.sort_values("t", kind="stable")
        ev = detect_steps(g["t"].to_numpy(), g["weight_g"].to_numpy(), uw, noise,
                          float(dp["threshold_fraction"]), float(dp["noise_k_sigma"]), int(dp["persistence_samples"]))
        for e in ev:
            rows.append({"sensor": sensor, "bin": info["bin"] if info is not None else None,
                         "station": info["station"] if info is not None else None,
                         "component": info["component"] if info is not None else None, **e})
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


def _match(true_t: List[float], det_t: List[float], tol: float) -> List[Tuple[int, int]]:
    used = [False] * len(true_t)
    pairs = []
    for j, d in enumerate(det_t):
        lo = bisect.bisect_left(true_t, d - tol)
        hi = bisect.bisect_right(true_t, d + tol)
        best, bd = None, None
        for i in range(lo, hi):
            if not used[i] and (bd is None or abs(true_t[i] - d) < bd):
                best, bd = i, abs(true_t[i] - d)
        if best is not None:
            used[best] = True
            pairs.append((best, j))
    return pairs


def score_detection(detections: pd.DataFrame, truth_picks: pd.DataFrame, cfg: dict, data: dict,
                    tol_s: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare detected picks with the simulation's ground truth, per sensor."""
    tol = float(tol_s if tol_s is not None else Behavior(data).detection_params()["match_tolerance_s"])
    smap = sensor_map(cfg, data).set_index("sensor")
    truth = truth_picks[truth_picks["sensor"].notna()] if truth_picks is not None else pd.DataFrame()
    det = detections[detections["kind"] == "pick"] if len(detections) else detections
    sensors = sorted(set(truth["sensor"]) | set(det["sensor"])) if len(truth) or len(det) else []
    rows, matched = [], []
    for s in sensors:
        tr = truth[truth["sensor"] == s].sort_values("t")
        dt = det[det["sensor"] == s].sort_values("t")
        tt, dd = tr["t"].tolist(), dt["t"].tolist()
        pairs = _match(tt, dd, tol)
        exact = 0
        lags = []
        for i, j in pairs:
            q_true = int(tr["qty"].iloc[i])
            q_det = int(dt["parts"].iloc[j])
            exact += int(q_true == q_det)
            lags.append(dd[j] - tt[i])
            matched.append({"sensor": s, "t_true": tt[i], "t_detected": dd[j], "lag_s": dd[j] - tt[i],
                            "qty_true": q_true, "parts_detected": q_det})
        tp = len(pairs)
        rows.append({
            "sensor": s, "bin": smap.loc[s, "bin"] if s in smap.index else None,
            "component": smap.loc[s, "component"] if s in smap.index else None,
            "unit_weight_g": smap.loc[s, "unit_weight_g"] if s in smap.index else float("nan"),
            "true_picks": len(tt), "detected": len(dd), "matched": tp,
            "precision": tp / len(dd) if dd else float("nan"),
            "recall": tp / len(tt) if tt else float("nan"),
            "count_exact": exact / tp if tp else float("nan"),
            "mean_lag_s": float(np.mean(lags)) if lags else float("nan"),
        })
    return pd.DataFrame(rows), pd.DataFrame(matched)


# =============================================================================
# Phase 3 - timing
# =============================================================================


def _anchor(L: LineStructure, sid: str):
    """Component picked at t=0 of every job at `sid` (first input of the first process),
    and the load cells of the bins holding it. None -> fall back to the camera."""
    procs = L.station_processes(sid)
    if not procs:
        return None, []
    comps = L.component_inputs(procs[0])
    if not comps:
        return None, []
    c = comps[0][0]
    sensors = [L.sensor_of_bin[b] for b in L.bins_at.get(sid, []) if L.bin_component(b) == c and b in L.sensor_of_bin]
    return c, sensors


def estimate_cycle_times(detections: pd.DataFrame, camera: Optional[pd.DataFrame], cfg: dict, model_data: dict,
                         rate_hz: float = 1.0, jobs: Optional[pd.DataFrame] = None,
                         lag_correction: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per station:
         start of a unit = detected pick on the anchor load cell (fallback: camera start_work)
         end of a unit   = camera 'unit_complete' (needs a frame; dropped frames reduce the sample)
         service time    = end - start (+ half a sample period to correct the detection lag)
       Also the pick-to-pick interval of the anchor (load cell only), which includes waiting.
       `jobs` (simulation ground truth) is optional and only used for comparison."""
    L = LineStructure(cfg)
    B = Behavior(model_data)
    corr = 0.5 / rate_hz if lag_correction and rate_hz > 0 else 0.0
    cam = camera if camera is not None else pd.DataFrame(columns=["t", "station", "action", "has_frame"])
    rows, samples = [], []
    for sid in L.routed_stations:
        comp, anchor_sensors = _anchor(L, sid)
        if anchor_sensors and len(detections):
            d = detections[(detections["sensor"].isin(anchor_sensors)) & (detections["kind"] == "pick")]
            starts = sorted(d["t"].tolist())
            method = f"load cell {'/'.join(anchor_sensors)} ({comp}) + camera"
        else:
            c0 = cam[(cam["station"] == sid) & (cam["action"] == "start_work") & (cam["has_frame"].astype(bool))]
            starts = sorted(c0["t"].tolist())
            method = "camera start + camera end"
        ends = sorted(cam[(cam["station"] == sid) & (cam["action"] == "unit_complete") &
                          (cam["has_frame"].astype(bool))]["t"].tolist())
        est = []
        for i, a in enumerate(starts):
            nxt = starts[i + 1] if i + 1 < len(starts) else math.inf
            k = bisect.bisect_right(ends, a)
            if k < len(ends) and ends[k] < nxt:
                v = ends[k] - a + corr
                est.append(v)
                samples.append({"station": sid, "t_start": a, "t_end": ends[k], "service_est_s": v})
        intervals = np.diff(starts) if len(starts) > 1 else np.array([])
        est_a = np.asarray(est)
        true_mean = true_cv = float("nan")
        n_jobs = None
        if jobs is not None and len(jobs):
            j = jobs[jobs["station"] == sid]["service_s"]
            n_jobs = len(j)
            if len(j):
                true_mean = float(j.mean())
                true_cv = float(j.std(ddof=1) / j.mean()) if len(j) > 1 and j.mean() > 0 else float("nan")
        model = B.expected_station_time(L, sid)
        est_mean = float(est_a.mean()) if len(est_a) else float("nan")
        rows.append({
            "station": sid, "processes": ", ".join(L.station_processes(sid)), "method": method,
            "starts_detected": len(starts), "estimates": len(est_a),
            "coverage": (len(est_a) / n_jobs) if n_jobs else float("nan"),
            "est_mean_s": est_mean,
            "est_cv": float(est_a.std(ddof=1) / est_a.mean()) if len(est_a) > 1 and est_a.mean() > 0 else float("nan"),
            "est_p50_s": float(np.median(est_a)) if len(est_a) else float("nan"),
            "model_mean_s": model,
            "true_mean_s": true_mean, "true_cv": true_cv,
            "interval_mean_s": float(intervals.mean()) if len(intervals) else float("nan"),
            "est_vs_model_pct": 100.0 * (est_mean / model - 1.0) if model > 0 and not math.isnan(est_mean) else float("nan"),
        })
    return pd.DataFrame(rows), pd.DataFrame(samples, columns=["station", "t_start", "t_end", "service_est_s"])


def calibrate_data(model_data: dict, cfg: dict, process_estimates: pd.DataFrame,
                   min_samples: int = 10) -> Tuple[dict, pd.DataFrame]:
    """Update process-time distributions in a copy of the data file from sensor estimates
    (output of `estimate_process_times`).

    * segment with ONE process: mean and CV are replaced by the measured values
      (divided by the operator's speed/cv factors, so the data stays operator-neutral);
    * segment with several processes (no sensor event separates them): every process
      is scaled by measured/model mean and keeps its CV.
    Parallel stations are pooled (weighted by the number of samples)."""
    L = LineStructure(cfg)
    B = Behavior(model_data)
    new = copy.deepcopy(model_data)
    pt = new.setdefault("process_times", {})
    acc: Dict[str, list] = {}
    for _, r in process_estimates.iterrows():
        n = int(r["samples"])
        if n < min_samples or not (r["model_mean_s"] > 0) or math.isnan(r["est_mean_s"]):
            continue
        procs = str(r["processes"]).split("+")
        op = B.operator(str(L.stations.get(r["station"], {}).get("operator")))
        single = len(procs) == 1
        for p in procs:
            acc.setdefault(p, []).append({
                "n": n, "factor": r["est_mean_s"] / r["model_mean_s"],
                "mean": (r["est_mean_s"] / op["speed_factor"]) if single else None,
                "cv": (r["est_cv"] / op["cv_factor"]) if single and not math.isnan(r["est_cv"]) else None})
    notes = []
    for p, lst in acc.items():
        w = sum(x["n"] for x in lst)
        old = B.process_time_spec(p)
        old_mean, old_cv = dist_mean(old), dist_cv(old)
        singles = [x for x in lst if x["mean"] is not None]
        if singles:
            ws = sum(x["n"] for x in singles)
            new_mean = sum(x["mean"] * x["n"] for x in singles) / ws
            cvs = [x["cv"] for x in singles if x["cv"] is not None]
            new_cv = float(np.mean(cvs)) if cvs else old_cv
            source = "measured directly"
        else:
            new_mean = old_mean * sum(x["factor"] * x["n"] for x in lst) / w
            new_cv = old_cv
            source = "scaled (shares a segment)"
        pt[p] = {"dist": "lognormal", "mean": round(float(new_mean), 2), "cv": round(float(max(0.01, new_cv)), 3)}
        notes.append({"process": p, "old_mean_s": old_mean, "new_mean_s": pt[p]["mean"],
                      "change_pct": 100.0 * (pt[p]["mean"] / old_mean - 1.0) if old_mean > 0 else float("nan"),
                      "old_cv": old_cv, "new_cv": pt[p]["cv"], "samples": w, "source": source})
    return new, pd.DataFrame(notes)


def estimate_process_times(detections: pd.DataFrame, camera: Optional[pd.DataFrame], cfg: dict, model_data: dict,
                           rate_hz: float = 1.0, events: Optional[pd.DataFrame] = None,
                           lag_correction: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split each station's job into process segments using well-defined sensor events:
    every process starts with the pick of its first component (a load-cell event);
    the last process ends with the camera's 'unit_complete'. Processes without a
    component input cannot be separated and are merged with the previous one
    (segment 'P3+P4'). `events` (simulation truth, process_done) is optional."""
    L = LineStructure(cfg)
    B = Behavior(model_data)
    corr = 0.5 / rate_hz if lag_correction and rate_hz > 0 else 0.0
    cam = camera if camera is not None else pd.DataFrame(columns=["t", "station", "action", "has_frame"])
    picks = detections[detections["kind"] == "pick"] if len(detections) else detections
    truth = None
    if events is not None and len(events) and "process" in events:
        truth = events[events["event"] == "process_done"]
    rows, samples = [], []
    for sid in L.routed_stations:
        procs = L.station_processes(sid)
        if not procs:
            continue
        # segments: [ [processes], anchor sensors ]
        segs: List[Tuple[List[str], List[str]]] = []
        used_sensors = set()
        for p in procs:
            comps = L.component_inputs(p)
            sens = []
            if comps:
                c = comps[0][0]
                sens = [L.sensor_of_bin[b] for b in L.bins_at.get(sid, [])
                        if L.bin_component(b) == c and b in L.sensor_of_bin]
            if sens and not (set(sens) & used_sensors):
                segs.append(([p], sens))
                used_sensors |= set(sens)
            elif segs:
                segs[-1][0].append(p)
            else:
                segs.append(([p], []))
        if not segs[0][1]:
            continue  # first process has no sensed pick: no load-cell start for this station
        times = {tuple(s): sorted(picks[picks["sensor"].isin(s)]["t"].tolist()) for _, s in segs}
        starts = times[tuple(segs[0][1])]
        ends = sorted(cam[(cam["station"] == sid) & (cam["action"] == "unit_complete") &
                          (cam["has_frame"].astype(bool))]["t"].tolist())
        est: Dict[int, List[float]] = {i: [] for i in range(len(segs))}
        for u, a in enumerate(starts):
            nxt = starts[u + 1] if u + 1 < len(starts) else math.inf
            bounds = [a]
            for _, s in segs[1:]:
                tt = times[tuple(s)]
                k = bisect.bisect_right(tt, bounds[-1])
                bounds.append(tt[k] if k < len(tt) and tt[k] < nxt else None)
            k = bisect.bisect_right(ends, a)
            end = ends[k] + corr if k < len(ends) and ends[k] < nxt else None
            bounds.append(end)
            for i in range(len(segs)):
                b0, b1 = bounds[i], bounds[i + 1]
                if b0 is not None and b1 is not None and b1 > b0:
                    est[i].append(b1 - b0)
                    samples.append({"station": sid, "segment": "+".join(segs[i][0]), "t_start": b0, "duration_s": b1 - b0})
        op = B.operator(str(L.stations[sid].get("operator")))
        for i, (ps, s) in enumerate(segs):
            x = np.asarray(est[i])
            model = sum(dist_mean(B.process_time_spec(p)) for p in ps) * op["speed_factor"]
            true_mean = float("nan")
            if truth is not None:
                tr = truth[(truth["station"] == sid) & (truth["process"].isin(ps))]
                if len(tr):
                    true_mean = float(tr.groupby("unit")["time_s"].sum().mean())
            rows.append({
                "station": sid, "processes": "+".join(ps),
                "start_event": f"pick on {'/'.join(s)}" if s else "-",
                "end_event": "camera unit_complete" if i == len(segs) - 1 else
                f"pick on {'/'.join(segs[i + 1][1])}",
                "samples": len(x),
                "est_mean_s": float(x.mean()) if len(x) else float("nan"),
                "est_cv": float(x.std(ddof=1) / x.mean()) if len(x) > 1 and x.mean() > 0 else float("nan"),
                "model_mean_s": model, "true_mean_s": true_mean,
            })
    return pd.DataFrame(rows), pd.DataFrame(samples, columns=["station", "segment", "t_start", "duration_s"])
