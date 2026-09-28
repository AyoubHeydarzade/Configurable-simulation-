"""
config_model.py - STRUCTURE of the line: objects + relationships.

This is the "configuration file" half of the architecture in
"Reconfigurable Assembly Line and Sensor Architecture":

    Objects        Station, Process, Operator, Component, Subassembly/Product,
                   Bin, Sensor (load cell / camera), Buffer
    Relationships  Process   performed_at  Station     stations.<S>.processes
                   Operator  operates      Station     stations.<S>.operator
                   Camera    observes      Station     stations.<S>.camera
                   Bin       located_at    Station     bins.<B>.station
                   Bin       contains      Component   bins.<B>.component
                   LoadCell  measures      Bin         sensors.<LC>.measures
                   Process   consumes      Comp./SA    processes.<P>.inputs
                   Process   produces      SA/Product  processes.<P>.output
                   Buffer    connects      S -> S      buffers.<BUF>.from/to
                   routing = order of stations (a list = parallel stations)

Nothing here knows about times or probabilities (that is the data file, see
behavior.py). A reconfiguration is an edit of relationships; the helpers at the
bottom perform the common ones on a copy of the config and report what changed.
"""
from __future__ import annotations

import copy
import html
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml

# =============================================================================
# YAML helpers
# =============================================================================


class _IndentDumper(yaml.SafeDumper):
    """Indent block sequences under their parent key (more readable YAML)."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


_IndentDumper.add_representer(
    type(None), lambda d, _: d.represent_scalar("tag:yaml.org,2002:null", "null")
)


def to_builtin(obj):
    """Convert numpy scalars, tuples, ... into plain Python types (for YAML)."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def parse_yaml(text: str) -> dict:
    data = yaml.safe_load(text) if text and text.strip() else {}
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("The top level of the YAML document must be a mapping (key: value).")
    return data


def load_yaml_file(path) -> dict:
    return parse_yaml(Path(path).read_text(encoding="utf-8"))


_BLOCK_SECTIONS = {"line", "simulation", "load_cells", "camera", "pick_detection"}


def _flow(v) -> str:
    """One-line flow-style YAML for a value."""
    s = yaml.dump(to_builtin(v), Dumper=_IndentDumper, default_flow_style=True, sort_keys=False,
                  width=10**9, allow_unicode=True).strip()
    if s.endswith("..."):
        s = s[:-3].strip()
    return s


def dump_yaml(data: dict, header: Optional[str] = None) -> str:
    """Readable YAML: one line per object (e.g. `B35: {station: S3, component: C10}`)."""
    parts: List[str] = []
    for key, val in to_builtin(data).items():
        if isinstance(val, dict) and val and key not in _BLOCK_SECTIONS:
            lines = [f"{_flow(key)}:"] + [f"  {_flow(k)}: {_flow(v)}" for k, v in val.items()]
            parts.append("\n".join(lines))
        elif isinstance(val, list):
            parts.append(f"{_flow(key)}: {_flow(val)}")
        else:
            parts.append(yaml.dump({key: val}, Dumper=_IndentDumper, sort_keys=False, default_flow_style=False,
                                   allow_unicode=True, width=100).rstrip())
    text = "\n\n".join(parts) + "\n"
    if header:
        hdr = "\n".join(("# " + h) if h else "#" for h in header.strip("\n").splitlines())
        text = hdr + "\n\n" + text
    return text


# =============================================================================
# Structured view of a configuration
# =============================================================================


def as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x if v is not None]
    return [str(x)]


def _section(cfg: dict, key: str) -> Dict[str, dict]:
    sec = cfg.get(key) or {}
    if not isinstance(sec, dict):
        return {}
    return {str(k): (v if isinstance(v, dict) else {}) for k, v in sec.items()}


def stages_of(cfg: dict) -> List[List[str]]:
    routing = cfg.get("routing") or []
    if isinstance(routing, str):
        return parse_routing_text(routing)
    return [as_list(e) for e in routing]


def parse_routing_text(text: str) -> List[List[str]]:
    """'S1 > S2 > S3|S3B > S4'  ->  [[S1], [S2], [S3, S3B], [S4]]"""
    stages = []
    for part in str(text).replace("->", ">").replace(",", ">").split(">"):
        names = [p.strip() for p in part.split("|") if p.strip()]
        if names:
            stages.append(names)
    return stages


def routing_text(cfg: dict) -> str:
    return " > ".join("|".join(st) for st in stages_of(cfg))


def routing_from_stages(stages: Sequence[Sequence[str]]) -> list:
    return [list(st) if len(st) > 1 else st[0] for st in stages if st]


class LineStructure:
    """Read-only, indexed view of a structure config dict."""

    def __init__(self, cfg: dict):
        c = cfg or {}
        self.cfg = c
        self.meta = c.get("line") or {}
        self.stages = stages_of(c)
        self.stage_index: Dict[str, int] = {}
        for i, stage in enumerate(self.stages):
            for s in stage:
                self.stage_index.setdefault(s, i)
        self.stations = _section(c, "stations")
        self.operators = _section(c, "operators")
        self.processes = _section(c, "processes")
        self.components = _section(c, "components")
        self.subassemblies = _section(c, "subassemblies")
        self.products = _section(c, "products")
        self.bins = _section(c, "bins")
        self.sensors = _section(c, "sensors")
        self.buffers = _section(c, "buffers")
        try:
            self.bins_per_station = int(self.meta.get("bins_per_station", 5) or 5)
        except (TypeError, ValueError):
            self.bins_per_station = 5

        self.bins_at: Dict[str, List[str]] = defaultdict(list)
        for b in sorted(self.bins):
            self.bins_at[str(self.bins[b].get("station"))].append(b)

        self.sensor_of_bin: Dict[str, str] = {}
        self.load_cells: List[str] = []
        self.cameras: List[str] = []
        for sid in sorted(self.sensors):
            s = self.sensors[sid]
            if s.get("type") == "load_cell":
                self.load_cells.append(sid)
                if s.get("measures") is not None:
                    self.sensor_of_bin.setdefault(str(s["measures"]), sid)
            elif s.get("type") == "camera":
                self.cameras.append(sid)

        self.producer: Dict[str, str] = {}
        for p, spec in self.processes.items():
            if spec.get("output") is not None:
                self.producer.setdefault(str(spec["output"]), p)

    # ---- basic queries ------------------------------------------------------
    @property
    def routed_stations(self) -> List[str]:
        return [s for stage in self.stages for s in stage]

    def station_processes(self, s: str) -> List[str]:
        return as_list(self.stations.get(s, {}).get("processes"))

    def stations_of_process(self, p: str) -> List[str]:
        return [s for s in self.routed_stations if p in self.station_processes(s)]

    def process_inputs(self, p: str) -> List[Tuple[str, object]]:
        inp = self.processes.get(p, {}).get("inputs") or {}
        if isinstance(inp, dict):
            return [(str(k), v) for k, v in inp.items()]
        return [(str(k), 1) for k in as_list(inp)]

    def component_inputs(self, p: str) -> List[Tuple[str, int]]:
        out = []
        for item, q in self.process_inputs(p):
            if item in self.components:
                try:
                    out.append((item, max(1, int(q))))
                except (TypeError, ValueError):
                    out.append((item, 1))
        return out

    def bin_component(self, b: str) -> Optional[str]:
        v = self.bins.get(b, {}).get("component")
        return None if v in (None, "", "null", "None", "(empty)") else str(v)

    def name_of(self, obj_id: Optional[str]) -> str:
        if obj_id is None:
            return ""
        for sec in (self.components, self.subassemblies, self.products, self.processes,
                    self.stations, self.operators):
            if obj_id in sec:
                return str(sec[obj_id].get("name", ""))
        return ""

    def buffer_between(self, k: int) -> Tuple[Optional[str], int]:
        """Buffer between stage k and k+1 -> (buffer id, capacity); (None, 0) = direct hand-off."""
        if k < 0 or k + 1 >= len(self.stages):
            return None, 0
        for bid, spec in self.buffers.items():
            f, t = str(spec.get("from")), str(spec.get("to"))
            if self.stage_index.get(f) == k and self.stage_index.get(t) == k + 1:
                try:
                    return bid, max(0, int(spec.get("capacity", 0) or 0))
                except (TypeError, ValueError):
                    return bid, 0
        return None, 0

    def topo_order(self) -> Dict[str, int]:
        """Process order implied by subassembly dependencies (ties: routing order)."""
        route_pos = {}
        for i, s in enumerate(self.routed_stations):
            for j, p in enumerate(self.station_processes(s)):
                route_pos.setdefault(p, (i, j))
        succ = defaultdict(set)
        indeg = {p: 0 for p in self.processes}
        for p in self.processes:
            for item, _ in self.process_inputs(p):
                q = self.producer.get(item)
                if q and q != p and q in indeg and p not in succ[q]:
                    succ[q].add(p)
                    indeg[p] += 1
        key = lambda p: (route_pos.get(p, (10**6, 0)), p)
        ready = sorted([p for p, d in indeg.items() if d == 0], key=key)
        order: List[str] = []
        while ready:
            p = ready.pop(0)
            order.append(p)
            for q in sorted(succ[p], key=key):
                indeg[q] -= 1
                if indeg[q] == 0:
                    ready.append(q)
            ready.sort(key=key)
        for p in sorted(self.processes, key=key):  # cycles: append the rest
            if p not in order:
                order.append(p)
        return {p: i for i, p in enumerate(order)}

    def sensor_label(self, sensor: str) -> str:
        spec = self.sensors.get(sensor, {})
        if spec.get("type") == "camera":
            return f"{sensor} (camera @ {spec.get('station', '?')})"
        b = str(spec.get("measures"))
        comp = self.bin_component(b)
        st = self.bins.get(b, {}).get("station", "?")
        comp_txt = f"{comp} {self.name_of(comp)}" if comp else "empty"
        return f"{sensor} -> {b} -> {comp_txt} @ {st}"


# =============================================================================
# Validation
# =============================================================================


@dataclass
class Report:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        return f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"

    def as_frame(self) -> pd.DataFrame:
        rows = [("error", m) for m in self.errors] + [("warning", m) for m in self.warnings] + \
               [("info", m) for m in self.info]
        return pd.DataFrame(rows, columns=["level", "message"])


class ConfigError(ValueError):
    def __init__(self, report: Report):
        self.report = report
        super().__init__("Invalid line configuration:\n- " + "\n- ".join(report.errors))


def validate(cfg: dict) -> Report:
    """Check objects and relationships for consistency. Errors block simulation."""
    R = Report()
    if not isinstance(cfg, dict):
        R.errors.append("Configuration is not a mapping.")
        return R
    for key in ("routing", "stations", "processes"):
        if not cfg.get(key):
            R.errors.append(f"Missing or empty section '{key}'.")
    if R.errors:
        return R
    L = LineStructure(cfg)

    # ---- routing -------------------------------------------------------------
    seen = set()
    for i, stage in enumerate(L.stages):
        if not stage:
            R.errors.append(f"Routing stage {i + 1} is empty.")
        for s in stage:
            if s not in L.stations:
                R.errors.append(f"Routing stage {i + 1} references unknown station '{s}'.")
            if s in seen:
                R.errors.append(f"Station {s} appears more than once in the routing.")
            seen.add(s)
    for s in L.stations:
        if s not in seen:
            R.warnings.append(f"Station {s} is defined but not in the routing - it is ignored by the simulation.")
    routed = [s for s in L.routed_stations if s in L.stations]

    # ---- stations: operator, camera, processes ------------------------------
    owners: Dict[str, List[str]] = defaultdict(list)
    for s in routed:
        spec = L.stations[s]
        op = spec.get("operator")
        if not op:
            R.errors.append(f"Station {s} has no operator.")
        elif str(op) not in L.operators:
            R.errors.append(f"Station {s}: operator '{op}' is not declared under operators.")
        cam = spec.get("camera")
        if not cam:
            R.warnings.append(f"Station {s} has no camera - camera-based timing is unavailable there.")
        elif str(cam) not in L.sensors or L.sensors[str(cam)].get("type") != "camera":
            R.errors.append(f"Station {s}: camera '{cam}' is not declared as a camera sensor.")
        elif str(L.sensors[str(cam)].get("station", s)) != s:
            R.warnings.append(f"Camera {cam} is linked to station {s} but its sensor entry says station "
                              f"{L.sensors[str(cam)].get('station')}.")
        procs = L.station_processes(s)
        if not procs:
            R.warnings.append(f"Station {s} has no process - it acts as a pass-through.")
        if len(procs) != len(set(procs)):
            R.errors.append(f"Station {s} lists the same process twice.")
        for p in procs:
            if p not in L.processes:
                R.errors.append(f"Station {s} performs unknown process '{p}'.")
            owners[p].append(s)

    for stage in L.stages:
        if len(stage) > 1 and all(s in L.stations for s in stage):
            ref = L.station_processes(stage[0])
            for s in stage[1:]:
                if L.station_processes(s) != ref:
                    R.errors.append(f"Parallel stations {stage[0]} and {s} must perform the same processes "
                                    f"({ref} vs {L.station_processes(s)}).")

    for p in L.processes:
        own = owners.get(p, [])
        if not own:
            R.errors.append(f"Process {p} ({L.name_of(p)}) is not performed at any station in the routing.")
        elif len({L.stage_index[o] for o in own}) > 1:
            R.errors.append(f"Process {p} is assigned to stations in different stages: {', '.join(own)}.")

    # ---- process definitions -------------------------------------------------
    produced_by: Dict[str, List[str]] = defaultdict(list)
    consumed_by: Dict[str, List[str]] = defaultdict(list)
    for p, spec in L.processes.items():
        out = spec.get("output")
        if out is None:
            R.errors.append(f"Process {p} has no output.")
        elif str(out) not in L.subassemblies and str(out) not in L.products:
            R.errors.append(f"Output '{out}' of {p} is not declared under subassemblies or products.")
        else:
            produced_by[str(out)].append(p)
        sa_inputs = []
        for item, q in L.process_inputs(p):
            if item in L.components:
                pass
            elif item in L.subassemblies or item in L.products:
                sa_inputs.append(item)
                consumed_by[item].append(p)
            else:
                R.errors.append(f"Process {p} consumes unknown item '{item}' (not a component or subassembly).")
            try:
                ok_q = int(q) >= 1 and float(q) == int(q)
            except (TypeError, ValueError):
                ok_q = False
            if not ok_q:
                R.errors.append(f"Process {p}: quantity of {item} must be a positive integer (got {q!r}).")
        if len(sa_inputs) > 1:
            R.warnings.append(f"Process {p} merges several subassemblies {sa_inputs}; the simulator models "
                              f"single-piece flow, so only the carrier unit is tracked.")
    for item, ps in produced_by.items():
        if len(ps) > 1:
            R.errors.append(f"{item} is produced by more than one process: {', '.join(ps)}.")
    for item, ps in consumed_by.items():
        if item not in produced_by:
            R.errors.append(f"{item} is consumed by {', '.join(ps)} but no process produces it.")
        if len(ps) > 1:
            R.warnings.append(f"{item} is consumed by several processes ({', '.join(ps)}).")
    for sa in L.subassemblies:
        if sa not in produced_by and sa not in consumed_by:
            R.info.append(f"Subassembly {sa} is declared but not used.")
    finals = [o for o in produced_by if o not in consumed_by]
    if not finals and produced_by:
        R.errors.append("No final product: every output is consumed again (circular recipe?).")
    for o in finals:
        if o in L.subassemblies:
            R.warnings.append(f"Final output {o} is declared as a subassembly; consider listing it under products.")
    if len(finals) > 1:
        R.warnings.append(f"Several outputs are never consumed: {', '.join(sorted(finals))}.")

    # ---- material availability and precedence -------------------------------
    pos: Dict[str, Tuple[int, int]] = {}
    for s in routed:
        for j, p in enumerate(L.station_processes(s)):
            pos.setdefault(p, (L.stage_index[s], j))
    for s in routed:
        for p in L.station_processes(s):
            if p not in L.processes:
                continue
            for item, _ in L.process_inputs(p):
                if item in L.components:
                    here = [b for b in L.bins_at.get(s, []) if L.bin_component(b) == item]
                    if not here:
                        R.errors.append(f"{p} at {s} consumes {item} ({L.name_of(item)}) but no bin at {s} "
                                        f"contains {item}.")
                    elif not any(b in L.sensor_of_bin for b in here):
                        R.warnings.append(f"{item} at {s} is in a bin without a load cell - its picks are not sensed.")
                elif item in produced_by:
                    q = produced_by[item][0]
                    if q in pos and p in pos and pos[q] >= pos[p]:
                        where = ", ".join(L.stations_of_process(q)) or "?"
                        R.errors.append(f"Precedence: {p} at {s} needs {item}, but it is produced by {q} at "
                                        f"{where}, which comes later (or at the same step) in the flow.")

    # ---- bins ------------------------------------------------------------------
    for b, spec in L.bins.items():
        st = spec.get("station")
        if st is None or str(st) not in L.stations:
            R.errors.append(f"Bin {b} is located at unknown station '{st}'.")
        comp = L.bin_component(b)
        if comp is not None and comp not in L.components:
            R.errors.append(f"Bin {b} contains unknown component '{comp}'.")
    for s in routed:
        n = len(L.bins_at.get(s, []))
        if n > L.bins_per_station:
            R.warnings.append(f"Station {s} has {n} bins but only {L.bins_per_station} physical slots.")
        used = {c for p in L.station_processes(s) for c, _ in L.component_inputs(p)}
        for b in L.bins_at.get(s, []):
            c = L.bin_component(b)
            if c is not None and c not in used:
                R.info.append(f"Bin {b} at {s} holds {c} which no process at {s} consumes (spare stock).")

    # ---- sensors -------------------------------------------------------------
    measured: Dict[str, List[str]] = defaultdict(list)
    for sid, spec in L.sensors.items():
        t = spec.get("type")
        if t == "load_cell":
            m = spec.get("measures")
            if m is None:
                R.warnings.append(f"Load cell {sid} is not attached to any bin.")
            elif str(m) not in L.bins:
                R.errors.append(f"Load cell {sid} measures unknown bin '{m}'.")
            else:
                measured[str(m)].append(sid)
        elif t == "camera":
            st = spec.get("station")
            if st is not None and str(st) not in L.stations:
                R.errors.append(f"Camera {sid} observes unknown station '{st}'.")
        else:
            R.warnings.append(f"Sensor {sid} has unknown type '{t}' (expected load_cell or camera).")
    for b, ss in measured.items():
        if len(ss) > 1:
            R.errors.append(f"Bin {b} is measured by several load cells: {', '.join(ss)}.")
    for b in L.bins:
        if b not in measured:
            R.warnings.append(f"Bin {b} has no load cell.")

    # ---- buffers ---------------------------------------------------------------
    boundaries: Dict[int, str] = {}
    for bid, spec in L.buffers.items():
        f, t = str(spec.get("from")), str(spec.get("to"))
        if f not in L.stage_index or t not in L.stage_index:
            R.errors.append(f"Buffer {bid} connects {f} -> {t}, but both must be stations in the routing.")
            continue
        k = L.stage_index[f]
        if L.stage_index[t] != k + 1:
            R.errors.append(f"Buffer {bid} must connect neighbouring stages ({f} is stage {k + 1}, "
                            f"{t} is stage {L.stage_index[t] + 1}).")
            continue
        if k in boundaries:
            R.errors.append(f"Buffers {boundaries[k]} and {bid} both sit between stage {k + 1} and {k + 2}.")
        boundaries[k] = bid
        try:
            cap = int(spec.get("capacity", 0))
            if cap < 0:
                raise ValueError
        except (TypeError, ValueError):
            R.errors.append(f"Buffer {bid}: capacity must be a non-negative integer.")

    # ---- operators ----------------------------------------------------------------
    op_stations: Dict[str, List[str]] = defaultdict(list)
    for s in routed:
        op = L.stations[s].get("operator")
        if op:
            op_stations[str(op)].append(s)
    for op, ss in op_stations.items():
        if len(ss) > 1:
            R.info.append(f"Operator {op} is shared by {', '.join(ss)} - these stations cannot work at the same time.")
    for op in L.operators:
        if op not in op_stations:
            R.info.append(f"Operator {op} is not assigned to any station.")
    return R


# =============================================================================
# Relationships (the "objects + relationships" view) and diffs
# =============================================================================

REL_COLUMNS = ["subject", "relation", "object", "detail"]


def relationships(cfg: dict) -> pd.DataFrame:
    L = LineStructure(cfg)
    rows = []

    def add(subject, relation, obj, detail=""):
        rows.append((str(subject), relation, str(obj), str(detail)))

    for i, stage in enumerate(L.stages):
        for s in stage:
            add(s, "in_stage", i + 1, "parallel" if len(stage) > 1 else "")
    for s, spec in L.stations.items():
        for j, p in enumerate(L.station_processes(s)):
            add(p, "performed_at", s, f"step {j + 1}")
        if spec.get("operator"):
            add(spec["operator"], "operates", s)
        if spec.get("camera"):
            add(spec["camera"], "observes", s)
    for b in L.bins:
        add(b, "located_at", L.bins[b].get("station"))
        add(b, "contains", L.bin_component(b) or "(empty)")
    for sid, spec in L.sensors.items():
        if spec.get("type") == "load_cell":
            add(sid, "measures", spec.get("measures"))
    for p, spec in L.processes.items():
        for item, q in L.process_inputs(p):
            add(p, "consumes", item, f"x{q}")
        if spec.get("output") is not None:
            add(p, "produces", spec["output"])
    for bid, spec in L.buffers.items():
        add(bid, "connects", f"{spec.get('from')} -> {spec.get('to')}", f"capacity {spec.get('capacity', 0)}")
    return pd.DataFrame(rows, columns=REL_COLUMNS)


def diff_relationships(before: dict, after: dict) -> pd.DataFrame:
    """Relationship-level diff: a reconfiguration is mainly a change in these rows."""
    ra = set(map(tuple, relationships(before).values.tolist()))
    rb = set(map(tuple, relationships(after).values.tolist()))
    removed, added = ra - rb, rb - ra
    keys = sorted({(r[0], r[1]) for r in removed | added})

    def fmt(rows):
        return "; ".join(o if not d else f"{o} ({d})" for (_, _, o, d) in sorted(rows))

    out = []
    for s, r in keys:
        b = [x for x in removed if (x[0], x[1]) == (s, r)]
        a = [x for x in added if (x[0], x[1]) == (s, r)]
        change = "changed" if a and b else ("added" if a else "removed")
        out.append({"subject": s, "relation": r, "before": fmt(b) or "-", "after": fmt(a) or "-", "change": change})
    return pd.DataFrame(out, columns=["subject", "relation", "before", "after", "change"])


# =============================================================================
# Tables for display
# =============================================================================


def station_table(cfg: dict) -> pd.DataFrame:
    L = LineStructure(cfg)
    rows = []
    for s in L.routed_stations + [x for x in L.stations if x not in L.stage_index]:
        spec = L.stations.get(s, {})
        k = L.stage_index.get(s)
        rows.append({
            "station": s,
            "name": spec.get("name", ""),
            "stage": (k + 1) if k is not None else None,
            "parallel_with": ", ".join(x for x in L.stages[k] if x != s) if k is not None else "",
            "operator": spec.get("operator", ""),
            "camera": spec.get("camera", ""),
            "processes": ", ".join(L.station_processes(s)),
            "bins": len(L.bins_at.get(s, [])),
            "load_cells": sum(1 for b in L.bins_at.get(s, []) if b in L.sensor_of_bin),
        })
    return pd.DataFrame(rows)


def bin_table(cfg: dict) -> pd.DataFrame:
    L = LineStructure(cfg)
    rows = []
    for s in list(L.bins_at):
        used = defaultdict(list)
        for p in L.station_processes(s):
            for c, q in L.component_inputs(p):
                used[c].append(f"{p} x{q}")
        for b in L.bins_at[s]:
            c = L.bin_component(b)
            rows.append({
                "station": s, "bin": b, "load_cell": L.sensor_of_bin.get(b, ""),
                "component": c or "", "component_name": L.name_of(c) if c else "(empty)",
                "used_by": ", ".join(used.get(c, [])) if c else "",
            })
    return pd.DataFrame(rows)


def process_table(cfg: dict) -> pd.DataFrame:
    L = LineStructure(cfg)
    order = L.topo_order()
    rows = []
    for p in sorted(L.processes, key=lambda x: order.get(x, 0)):
        spec = L.processes[p]
        rows.append({
            "process": p, "name": spec.get("name", ""),
            "station": ", ".join(L.stations_of_process(p)) or "(unassigned)",
            "inputs": ", ".join(f"{i}x{q}" for i, q in L.process_inputs(p)),
            "output": spec.get("output", ""),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Graphviz diagram
# =============================================================================


def _h(x) -> str:
    return html.escape(str(x), quote=True)


def _station_label(L: LineStructure, s: str, detail: bool) -> str:
    spec = L.stations.get(s, {})
    rows = [f'<tr><td colspan="3" bgcolor="#1F4E79"><font color="white"><b>{_h(s)}</b>  {_h(spec.get("name", ""))}'
            f'</font></td></tr>',
            f'<tr><td colspan="3" align="left">Operator {_h(spec.get("operator", "-"))}  |  '
            f'Camera {_h(spec.get("camera", "-"))}</td></tr>']
    procs = L.station_processes(s)
    ptxt = "<br/>".join(f"{_h(p)} {_h(L.name_of(p))}" for p in procs) or "<i>pass-through</i>"
    rows.append(f'<tr><td colspan="3" align="left" bgcolor="#E3EEF9">{ptxt}</td></tr>')
    if detail:
        for b in L.bins_at.get(s, []):
            c = L.bin_component(b)
            lc = L.sensor_of_bin.get(b, "-")
            ctxt = f"{_h(c)} {_h(L.name_of(c))}" if c else '<font color="#999999">empty</font>'
            rows.append(f'<tr><td>{_h(lc)}</td><td>{_h(b)}</td><td align="left">{ctxt}</td></tr>')
    return ('<table border="0" cellborder="1" cellspacing="0" cellpadding="3" bgcolor="white">'
            + "".join(rows) + "</table>")


def to_dot(cfg: dict, detail: bool = True) -> str:
    L = LineStructure(cfg)
    out = ['digraph line {', 'rankdir=LR; nodesep=0.3; ranksep=0.45; bgcolor="transparent";',
           'node [shape=plaintext, fontname="Helvetica", fontsize=10];',
           'edge [color="#777777", arrowsize=0.7, fontname="Helvetica"];',
           '"__in" [shape=circle, style=filled, fillcolor="#DDDDDD", label="in", fontsize=8, width=0.35];',
           '"__out" [shape=doublecircle, style=filled, fillcolor="#CFE8CF", label="out", fontsize=8, width=0.35];']
    for stage in L.stages:
        for s in stage:
            if s in L.stations:
                out.append(f'"{s}" [label=<{_station_label(L, s, detail)}>];')
            else:
                out.append(f'"{s}" [shape=box, color=red, fontcolor=red, label="{_h(s)} (undefined)"];')
        if len(stage) > 1:
            out.append("{rank=same; " + " ".join(f'"{s}";' for s in stage) + "}")
    if L.stages:
        for s in L.stages[0]:
            out.append(f'"__in" -> "{s}";')
        for s in L.stages[-1]:
            out.append(f'"{s}" -> "__out";')
    for k in range(len(L.stages) - 1):
        bid, cap = L.buffer_between(k)
        if bid is not None and cap > 0:
            node = f"__buf{k}"
            out.append(f'"{node}" [shape=box, style="rounded,filled", fillcolor="#FFF4D6", color="#C9A227", '
                       f'fontsize=9, label="{_h(bid)}\\ncap {cap}"];')
            for s in L.stages[k]:
                out.append(f'"{s}" -> "{node}";')
            for s in L.stages[k + 1]:
                out.append(f'"{node}" -> "{s}";')
        else:
            for s in L.stages[k]:
                for t in L.stages[k + 1]:
                    out.append(f'"{s}" -> "{t}" [label="direct", fontsize=8, fontcolor="#999999"];')
    out.append("}")
    return "\n".join(out)


# =============================================================================
# Reconfiguration operations. Each returns (new_cfg, [messages]); input untouched.
# =============================================================================


def _next_id(existing, prefix: str, start: int = 1) -> str:
    i = start
    while f"{prefix}{i}" in existing:
        i += 1
    return f"{prefix}{i}"


def _bins(cfg: dict) -> dict:
    cfg.setdefault("bins", {})
    for b, v in list(cfg["bins"].items()):
        if not isinstance(v, dict):
            cfg["bins"][b] = {}
    return cfg["bins"]


def move_process(cfg: dict, process: str, to_station: str, move_bins: bool = True) -> Tuple[dict, List[str]]:
    """Perform `process` at `to_station` (and its parallel twins) instead of where it is now.
    With move_bins, bins for its components are freed at the old station and free
    bins at the new station are re-pointed to those components (sensors keep their IDs)."""
    new = copy.deepcopy(cfg)
    L = LineStructure(new)
    if process not in L.processes:
        raise ValueError(f"Unknown process '{process}'.")
    if to_station not in L.stations:
        raise ValueError(f"Unknown station '{to_station}'.")
    from_st = [s for s in L.stations if process in L.station_processes(s)]
    k = L.stage_index.get(to_station)
    targets = L.stages[k] if k is not None else [to_station]
    if set(targets) <= set(from_st):
        return new, [f"{process} is already performed at {to_station}."]
    order = L.topo_order()
    for s in from_st:
        new["stations"][s]["processes"] = [p for p in L.station_processes(s) if p != process]
    for t in targets:
        procs = [p for p in as_list(new["stations"][t].get("processes")) if p != process] + [process]
        procs.sort(key=lambda p: order.get(p, 10**6))
        new["stations"][t]["processes"] = procs
    msgs = [f"{process} ({L.name_of(process)}) is now performed at {', '.join(targets)} "
            f"(was {', '.join(from_st) or 'unassigned'})."]
    if move_bins:
        bins = _bins(new)
        comps = [c for c, _ in L.component_inputs(process)]
        L2 = LineStructure(new)
        for s in from_st:
            if s in targets:
                continue
            still = {c for p in L2.station_processes(s) for c, _ in L2.component_inputs(p)}
            for c in comps:
                if c in still:
                    continue
                for b in L2.bins_at.get(s, []):
                    if L2.bin_component(b) == c:
                        bins[b]["component"] = None
                        msgs.append(f"{b} (sensor {L2.sensor_of_bin.get(b, '-')}) at {s} freed - was {c}.")
        for t in targets:
            for c in comps:
                L3 = LineStructure(new)
                if any(L3.bin_component(b) == c for b in L3.bins_at.get(t, [])):
                    continue
                free = [b for b in L3.bins_at.get(t, []) if L3.bin_component(b) is None]
                if not free:
                    msgs.append(f"WARNING: no free bin at {t} for {c} - free or add a bin, then assign {c}.")
                    continue
                bins[free[0]]["component"] = c
                msgs.append(f"{free[0]} (sensor {L3.sensor_of_bin.get(free[0], '-')}) at {t} now contains {c} "
                            f"({L3.name_of(c)}).")
    return new, msgs


def set_bin_component(cfg: dict, bin_id: str, component: Optional[str]) -> Tuple[dict, List[str]]:
    new = copy.deepcopy(cfg)
    L = LineStructure(new)
    if bin_id not in L.bins:
        raise ValueError(f"Unknown bin '{bin_id}'.")
    old = L.bin_component(bin_id)
    comp = component if component not in ("", "(empty)", None) else None
    _bins(new)[bin_id]["component"] = comp
    lc = L.sensor_of_bin.get(bin_id, "no load cell")
    return new, [f"{lc} -> {bin_id} -> {old or 'empty'}  became  {lc} -> {bin_id} -> {comp or 'empty'} "
                 f"(the sensor keeps its ID; only the relationship changed)."]


def set_sensor_bin(cfg: dict, sensor: str, bin_id: Optional[str]) -> Tuple[dict, List[str]]:
    new = copy.deepcopy(cfg)
    L = LineStructure(new)
    if sensor not in L.sensors or L.sensors[sensor].get("type") != "load_cell":
        raise ValueError(f"'{sensor}' is not a load cell.")
    if bin_id is not None and bin_id not in L.bins:
        raise ValueError(f"Unknown bin '{bin_id}'.")
    old = L.sensors[sensor].get("measures")
    new["sensors"][sensor]["measures"] = bin_id
    msgs = [f"{sensor} now measures {bin_id or 'nothing'} (was {old or 'nothing'})."]
    if bin_id is not None:
        others = [s for s in L.load_cells if s != sensor and L.sensors[s].get("measures") == bin_id]
        if others:
            msgs.append(f"WARNING: {bin_id} is also measured by {', '.join(others)} - re-point one of them.")
    return new, msgs


def set_station_operator(cfg: dict, station: str, operator: str) -> Tuple[dict, List[str]]:
    new = copy.deepcopy(cfg)
    if station not in (new.get("stations") or {}):
        raise ValueError(f"Unknown station '{station}'.")
    new.setdefault("operators", {})
    msgs = []
    if operator not in new["operators"]:
        new["operators"][operator] = {"name": f"Operator {operator}"}
        msgs.append(f"New operator {operator} added.")
    old = new["stations"][station].get("operator")
    new["stations"][station]["operator"] = operator
    msgs.append(f"{operator} now operates {station} (was {old}).")
    return new, msgs


def set_buffer_capacity(cfg: dict, stage_k: int, capacity: int) -> Tuple[dict, List[str]]:
    """Set capacity of the buffer between stage k and k+1 (0-based); creates it if missing."""
    new = copy.deepcopy(cfg)
    L = LineStructure(new)
    if not 0 <= stage_k < len(L.stages) - 1:
        raise ValueError("Stage index out of range.")
    bid, old = L.buffer_between(stage_k)
    new.setdefault("buffers", {})
    if bid is None:
        bid = _next_id(new["buffers"], "BUF")
        new["buffers"][bid] = {"from": L.stages[stage_k][0], "to": L.stages[stage_k + 1][0], "capacity": int(capacity)}
        return new, [f"New buffer {bid} between {'|'.join(L.stages[stage_k])} and "
                     f"{'|'.join(L.stages[stage_k + 1])} with capacity {capacity}."]
    new["buffers"][bid]["capacity"] = int(capacity)
    return new, [f"{bid} capacity {old} -> {capacity}."] if old != capacity else []


def add_parallel_station(cfg: dict, station: str, new_id: Optional[str] = None,
                         operator: Optional[str] = None) -> Tuple[dict, List[str]]:
    """Clone `station` (processes, bins, load cells, camera) as a parallel twin with its own operator."""
    new = copy.deepcopy(cfg)
    L = LineStructure(new)
    if station not in L.stations or station not in L.stage_index:
        raise ValueError(f"Station '{station}' is not in the routing.")
    stations = new["stations"]
    if not new_id:
        new_id = next(f"{station}{ch}" for ch in "BCDEFGHIJ" if f"{station}{ch}" not in stations)
    if new_id in stations:
        raise ValueError(f"Station '{new_id}' already exists.")
    ops = new.setdefault("operators", {})
    operator = operator or _next_id(ops, "O", start=len(ops) + 1)
    ops.setdefault(operator, {"name": f"Operator {operator}"})
    sensors = new.setdefault("sensors", {})
    bins = _bins(new)
    tag = new_id[1:] if new_id.startswith("S") else new_id
    cam_id = f"CAM_{new_id}"
    sensors[cam_id] = {"type": "camera", "station": new_id}
    src = stations[station]
    stations[new_id] = {"name": f"{src.get('name', station)} (parallel)", "operator": operator,
                        "camera": cam_id, "processes": list(L.station_processes(station))}
    msgs = [f"Parallel station {new_id} added next to {station}, operated by {operator}, observed by {cam_id}."]
    for k, b in enumerate(L.bins_at.get(station, []), start=1):
        nb = f"B{tag}{k}"
        while nb in bins:
            nb += "_"
        bins[nb] = {"station": new_id, "component": L.bin_component(b)}
        if b in L.sensor_of_bin:
            lc = f"LC_{tag}{k}"
            while lc in sensors:
                lc += "_"
            sensors[lc] = {"type": "load_cell", "measures": nb}
        msgs.append(f"  {nb} (sensor {f'LC_{tag}{k}' if b in L.sensor_of_bin else '-'}) mirrors {b}: "
                    f"{L.bin_component(b) or 'empty'}.")
    k = L.stage_index[station]
    stages = [list(st) for st in L.stages]
    stages[k].append(new_id)
    new["routing"] = routing_from_stages(stages)
    return new, msgs


def add_station(cfg: dict, after_station: str, new_id: Optional[str] = None, n_bins: Optional[int] = None,
                operator: Optional[str] = None) -> Tuple[dict, List[str]]:
    """Insert an empty station (pass-through, empty bins with load cells, camera) after `after_station`."""
    new = copy.deepcopy(cfg)
    L = LineStructure(new)
    if after_station not in L.stage_index:
        raise ValueError(f"Station '{after_station}' is not in the routing.")
    stations = new["stations"]
    new_id = new_id or _next_id(stations, "S", start=len(stations) + 1)
    if new_id in stations:
        raise ValueError(f"Station '{new_id}' already exists.")
    n_bins = L.bins_per_station if n_bins is None else int(n_bins)
    ops = new.setdefault("operators", {})
    operator = operator or _next_id(ops, "O", start=len(ops) + 1)
    ops.setdefault(operator, {"name": f"Operator {operator}"})
    sensors = new.setdefault("sensors", {})
    bins = _bins(new)
    tag = new_id[1:] if new_id.startswith("S") else new_id
    cam_id = f"CAM_{new_id}"
    sensors[cam_id] = {"type": "camera", "station": new_id}
    stations[new_id] = {"name": f"Station {tag}", "operator": operator, "camera": cam_id, "processes": []}
    for k in range(1, n_bins + 1):
        nb, lc = f"B{tag}{k}", f"LC_{tag}{k}"
        bins[nb] = {"station": new_id, "component": None}
        sensors[lc] = {"type": "load_cell", "measures": nb}
    k = L.stage_index[after_station]
    stages = [list(st) for st in L.stages]
    stages.insert(k + 1, [new_id])
    new["routing"] = routing_from_stages(stages)
    buffers = new.setdefault("buffers", {})
    msgs = [f"Station {new_id} inserted after {after_station} (operator {operator}, camera {cam_id}, "
            f"{n_bins} empty bins with load cells). Move processes to it to use it."]
    bid, cap = L.buffer_between(k)
    if bid is not None:  # old buffer now feeds the station after the new one
        buffers[bid]["from"] = new_id
        msgs.append(f"{bid} now connects {new_id} -> {buffers[bid]['to']}.")
    nb_id = _next_id(buffers, "BUF")
    buffers[nb_id] = {"from": after_station, "to": new_id, "capacity": 1}
    msgs.append(f"New buffer {nb_id}: {after_station} -> {new_id}, capacity 1.")
    return new, msgs


def remove_station(cfg: dict, station: str, move_processes_to: Optional[str] = None) -> Tuple[dict, List[str]]:
    """Remove a station and its bins, load cells and camera. Its processes are moved first
    (unless a parallel twin still performs them)."""
    new = copy.deepcopy(cfg)
    L = LineStructure(new)
    if station not in L.stations:
        raise ValueError(f"Unknown station '{station}'.")
    msgs: List[str] = []
    k = L.stage_index.get(station)
    parallel = k is not None and len(L.stages[k]) > 1
    procs = L.station_processes(station)
    if procs and not parallel:
        if not move_processes_to:
            raise ValueError(f"{station} performs {procs}; choose a station to move them to.")
        for p in procs:
            new, m = move_process(new, p, move_processes_to, move_bins=True)
            msgs += m
    L = LineStructure(new)
    for b in list(L.bins_at.get(station, [])):
        new["bins"].pop(b, None)
        lc = L.sensor_of_bin.get(b)
        if lc:
            new["sensors"].pop(lc, None)
    cam = L.stations[station].get("camera")
    if cam:
        (new.get("sensors") or {}).pop(cam, None)
    new["stations"].pop(station)
    stages = [[s for s in st if s != station] for st in L.stages]
    buffers = new.get("buffers") or {}
    if parallel:
        mate = [s for s in L.stages[k] if s != station][0]
        for spec in buffers.values():
            if spec.get("from") == station:
                spec["from"] = mate
            if spec.get("to") == station:
                spec["to"] = mate
    elif k is not None:
        b_in, cap_in = L.buffer_between(k - 1)
        b_out, cap_out = L.buffer_between(k)
        for b in (b_in, b_out):
            if b:
                buffers.pop(b, None)
        if 0 < k < len(L.stages) - 1:
            bid = b_in or b_out or _next_id(buffers, "BUF")
            buffers[bid] = {"from": L.stages[k - 1][0], "to": L.stages[k + 1][0], "capacity": max(cap_in, cap_out)}
            msgs.append(f"{bid} now connects {L.stages[k - 1][0]} -> {L.stages[k + 1][0]} "
                        f"(capacity {max(cap_in, cap_out)}).")
    new["routing"] = routing_from_stages([st for st in stages if st])
    msgs.append(f"Station {station} removed together with its bins, load cells and camera.")
    return new, msgs


def replace_process_input(cfg: dict, process: str, old_item: str, new_item: str) -> Tuple[dict, List[str]]:
    """Swap one input of a process recipe (e.g. C10 -> C21, alternate supplier)."""
    new = copy.deepcopy(cfg)
    L = LineStructure(new)
    if process not in L.processes:
        raise ValueError(f"Unknown process '{process}'.")
    items = L.process_inputs(process)
    if old_item not in [i for i, _ in items]:
        raise ValueError(f"{process} does not consume {old_item}.")
    new["processes"][process]["inputs"] = {(new_item if i == old_item else i): q for i, q in items}
    return new, [f"{process} now consumes {new_item} instead of {old_item}."]


def set_routing(cfg: dict, stages: Sequence[Sequence[str]]) -> Tuple[dict, List[str]]:
    new = copy.deepcopy(cfg)
    new["routing"] = routing_from_stages(stages)
    return new, [f"Routing set to {' > '.join('|'.join(s) for s in stages)}."]


def set_version(cfg: dict, name: str, description: Optional[str] = None) -> dict:
    new = copy.deepcopy(cfg)
    new.setdefault("line", {})
    new["line"]["version"] = name
    if description:
        new["line"]["description"] = description
    return new
