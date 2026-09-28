"""
app.py - Streamlit dashboard for the reconfigurable manual assembly line.

    streamlit run app.py

One machine-readable description of the line drives everything on this page:
  configs/line_config.yaml  STRUCTURE  objects + relationships (stations, processes, bins,
                                       load cells, cameras, operators, buffers, routing)
  configs/line_data.yaml    BEHAVIOUR  times, distributions, weights, failures, sensor specs
The simulation, the sensor pipeline and the diagrams are all built from those two files,
so a reconfiguration is an edit of relationships - not a rewrite of the model.
"""
from __future__ import annotations

import copy
import inspect
import io
import json
import math
import sys
import time
import zipfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:  # make `line_sim` importable however the app is launched
    sys.path.insert(0, str(ROOT))

from line_sim import behavior as bh  # noqa: E402
from line_sim import config_model as cm  # noqa: E402
from line_sim import sensing as sn  # noqa: E402
from line_sim.simulator import STATE_COLORS, STATE_LABELS, STATES, LineSimulator, compare_scenarios  # noqa: E402

CONFIG_DIR = ROOT / "configs"
VARIANT_DIR = CONFIG_DIR / "variants"
WORKING = "(working configuration)"

st.set_page_config(page_title="Reconfigurable Assembly Line - Twin & Simulator", page_icon="🏭", layout="wide")
alt.data_transformers.disable_max_rows()


def _stretch(fn) -> dict:
    """Full-width keyword that works on old (use_container_width) and new (width) Streamlit."""
    try:
        p = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if "width" in p and "use_container_width" in p and p["use_container_width"].default is None:
        return {"width": "stretch"}
    return {"use_container_width": True} if "use_container_width" in p else {}


W_DF, W_ED, W_CH = _stretch(st.dataframe), _stretch(st.data_editor), _stretch(st.altair_chart)
W_GV, W_BTN, W_SUB, W_DL = (_stretch(st.graphviz_chart), _stretch(st.button), _stretch(st.form_submit_button),
                            _stretch(st.download_button))
STATE_SCALE = alt.Scale(domain=[STATE_LABELS[s] for s in STATES], range=[STATE_COLORS[s] for s in STATES])
ss = st.session_state


# =============================================================================
# State
# =============================================================================


def load_file_scenarios() -> dict:
    out = {"baseline": cm.load_yaml_file(CONFIG_DIR / "line_config.yaml")}
    if VARIANT_DIR.exists():
        for p in sorted(VARIANT_DIR.glob("*.y*ml")):
            try:
                cfg = cm.load_yaml_file(p)
            except Exception as exc:  # pragma: no cover
                st.warning(f"Could not read {p.name}: {exc}")
                continue
            out[str((cfg.get("line") or {}).get("version") or p.stem)] = cfg
    return out


def init_state():
    if ss.get("initialized"):
        return
    ss.scenarios = load_file_scenarios()
    ss.working_name = "baseline"
    ss.working_cfg = copy.deepcopy(ss.scenarios["baseline"])
    ss.data = cm.load_yaml_file(CONFIG_DIR / "line_data.yaml")
    ss.ver = 0
    ss.change_log = []
    ss.flash = []
    ss.result = None
    ss.result_meta = {}
    ss.compare = None
    ss.testbed = None
    ss.sensing_cache = {}
    ss.uploads_seen = set()
    ss.initialized = True


def log(msgs):
    stamp = time.strftime("%H:%M:%S")
    ss.change_log = (ss.change_log + [f"{stamp}  {m}" for m in msgs])[-80:]
    ss.flash = list(msgs)[:4]


def set_working(cfg: dict, name: str | None = None, msgs=None, rerun: bool = True):
    ss.working_cfg = cfg
    if name:
        ss.working_name = name
    elif not ss.working_name.endswith(" (edited)"):
        ss.working_name += " (edited)"
    ss.ver += 1
    if msgs:
        log(msgs)
    if rerun:
        st.rerun()


def set_data(data: dict, msgs=None, rerun: bool = True):
    ss.data = data
    ss.ver += 1
    if msgs:
        log(msgs)
    if rerun:
        st.rerun()


def apply_op(fn, *args, **kw):
    """Run a reconfiguration operation on the working config and show what changed."""
    try:
        new, msgs = fn(ss.working_cfg, *args, **kw)
    except ValueError as exc:
        st.error(str(exc))
        return
    set_working(new, msgs=msgs or ["No change."])


# =============================================================================
# Small helpers
# =============================================================================


def fmt(x, spec=".1f", suffix=""):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "-"
        return f"{x:{spec}}{suffix}"
    except (TypeError, ValueError):
        return str(x)


def render_report(rep: cm.Report, ok_text="Configuration is consistent.", expanded=False):
    if rep.errors:
        st.error("**Errors (simulation blocked)**\n\n" + "\n".join(f"- {e}" for e in rep.errors))
    elif ok_text:
        st.success(ok_text)
    if rep.warnings:
        with st.expander(f"⚠️ {len(rep.warnings)} warning(s)", expanded=expanded):
            st.markdown("\n".join(f"- {w}" for w in rep.warnings))
    if rep.info:
        with st.expander(f"ℹ️ {len(rep.info)} note(s)"):
            st.markdown("\n".join(f"- {w}" for w in rep.info))


def yaml_download(label, data, file_name, header=None, key=None):
    st.download_button(label, cm.dump_yaml(data, header=header).encode("utf-8"), file_name=file_name,
                       mime="text/yaml", key=key, **W_DL)


def window_start(container, max_start: int, default: int, step: int, key: str) -> int:
    if max_start <= 0:
        return 0
    return int(container.slider("Window start (s)", 0, max_start, min(default, max_start), step=step, key=key))


def comp_options(L: cm.LineStructure):
    return ["(empty)"] + list(L.components)


def comp_label(L: cm.LineStructure, c):
    return "(empty)" if c in (None, "(empty)") else f"{c} - {L.name_of(c)}"


# =============================================================================
# Sidebar
# =============================================================================


def sidebar():
    with st.sidebar:
        st.header("🏭 Working configuration")
        rep = cm.validate(ss.working_cfg)
        st.markdown(f"**{ss.working_name}**")
        if rep.ok:
            st.caption(f"✅ valid · {rep.summary()}")
        else:
            st.caption(f"❌ {rep.summary()} - see *Line structure*")

        names = list(ss.scenarios)
        default = names.index(ss.working_name) if ss.working_name in names else 0
        pick = st.selectbox("Saved configurations", names, index=default, key=f"sb_pick_{ss.ver}")
        if st.button("Load into working configuration", **W_BTN):
            set_working(copy.deepcopy(ss.scenarios[pick]), name=pick, msgs=[f"Loaded configuration '{pick}'."])

        with st.expander("📤 Upload YAML files"):
            up = st.file_uploader("Structure file (line_config)", type=["yaml", "yml"], key="up_cfg")
            if up is not None and ("cfg", up.name, up.size) not in ss.uploads_seen:
                ss.uploads_seen.add(("cfg", up.name, up.size))
                try:
                    cfg = cm.parse_yaml(up.getvalue().decode("utf-8"))
                    name = str((cfg.get("line") or {}).get("version") or Path(up.name).stem)
                    ss.scenarios[name] = cfg
                    set_working(copy.deepcopy(cfg), name=name, msgs=[f"Uploaded structure '{name}'."])
                except Exception as exc:
                    st.error(f"Could not read structure file: {exc}")
            upd = st.file_uploader("Behaviour file (line_data)", type=["yaml", "yml"], key="up_data")
            if upd is not None and ("data", upd.name, upd.size) not in ss.uploads_seen:
                ss.uploads_seen.add(("data", upd.name, upd.size))
                try:
                    set_data(cm.parse_yaml(upd.getvalue().decode("utf-8")), msgs=[f"Uploaded behaviour data {upd.name}."])
                except Exception as exc:
                    st.error(f"Could not read data file: {exc}")

        st.divider()
        st.header("▶️ Simulation")
        sims = bh.Behavior(ss.data).sim_settings()
        c1, c2 = st.columns(2)
        horizon_min = c1.number_input("Horizon (min)", 5, 24 * 60, int(round(sims["horizon_s"] / 60)), step=5,
                                      key="sim_h")
        warm_min = c2.number_input("Warm-up (min)", 0, 240, int(round(sims["warmup_s"] / 60)), step=1,
                                   key="sim_w", help="Excluded from KPIs - the line starts empty.")
        seed = st.number_input("Random seed", 0, 10**6, int(sims["seed"]), key="sim_seed")
        if st.button("▶️ Run simulation", type="primary", disabled=not rep.ok, **W_BTN):
            try:
                t0 = time.time()
                with st.spinner("Simulating ..."):
                    res = LineSimulator(ss.working_cfg, ss.data, horizon_s=horizon_min * 60.0,
                                        warmup_s=min(warm_min, horizon_min - 1) * 60.0, seed=int(seed)).run()
                ss.result = res
                ss.result_meta = {"config": ss.working_name, "wall_s": time.time() - t0}
                ss.sensing_cache = {}
                ss.flash = [f"Simulation done in {time.time() - t0:.2f} s - {res.kpis['throughput_uph']:.1f} units/h"]
                st.rerun()
            except cm.ConfigError as exc:
                st.error(str(exc))

        st.divider()
        st.caption("Download the files that define this line:")
        yaml_download("⬇️ Structure (line_config.yaml)", ss.working_cfg, "line_config.yaml",
                      header=f"Structure of the line - configuration '{ss.working_name}'", key="dl_cfg_sb")
        yaml_download("⬇️ Behaviour (line_data.yaml)", ss.data, "line_data.yaml",
                      header="Behaviour of the line - times, distributions, measurements", key="dl_data_sb")


def architecture_strip():
    L = cm.LineStructure(ss.working_cfg)
    rep_d = bh.check_data(ss.data, ss.working_cfg)
    res = ss.result
    c = st.columns(4)
    with c[0].container(border=True):
        st.markdown("**① Configuration file** · objects + relationships")
        st.caption(f"{len(L.routed_stations)} stations in {len(L.stages)} stages · {len(L.processes)} processes · "
                   f"{len(L.bins)} bins · {len(L.load_cells)} load cells · {len(L.cameras)} cameras · "
                   f"{len(set(str(L.stations[s].get('operator')) for s in L.routed_stations if s in L.stations))} "
                   f"operators")
    with c[1].container(border=True):
        st.markdown("**② Data file** · times + distributions")
        covered = sum(1 for p in L.processes if bh.Behavior(ss.data).has_process_time(p))
        st.caption(f"time data for {covered}/{len(L.processes)} processes · {len(rep_d.warnings)} gap(s) · "
                   f"pick window {bh.Behavior(ss.data).pick_window():.0%}")
    with c[2].container(border=True):
        st.markdown("**③ Simulation** · built from ① + ②")
        st.caption("not run yet" if res is None else
                   f"'{ss.result_meta.get('config')}' · {res.horizon_s / 60:.0f} min · seed {res.seed} · "
                   f"{ss.result_meta.get('wall_s', 0):.2f} s wall")
    with c[3].container(border=True):
        st.markdown("**④ Results** · KPIs")
        st.caption("-" if res is None else
                   f"{res.kpis['throughput_uph']:.1f} units/h · bottleneck {res.kpis['bottleneck']} · "
                   f"lead time {res.kpis['lead_time_mean_s'] / 60:.1f} min")


# =============================================================================
# Tab 1 - Line structure
# =============================================================================


def station_cards(L: cm.LineStructure, per_row: int = 3):
    stations = [s for s in L.routed_stations if s in L.stations]
    for i in range(0, len(stations), per_row):
        cols = st.columns(per_row)
        for col, s in zip(cols, stations[i:i + per_row]):
            spec = L.stations[s]
            k = L.stage_index[s]
            par = [x for x in L.stages[k] if x != s]
            with col.container(border=True):
                st.markdown(f"**{s}** · {spec.get('name', '')} · stage {k + 1}"
                            + (f" (parallel with {', '.join(par)})" if par else ""))
                st.caption(f"👤 {spec.get('operator', '-')}  ·  📷 {spec.get('camera', '-')}")
                procs = L.station_processes(s)
                st.markdown(" → ".join(f"`{p}` {L.name_of(p)}" for p in procs) if procs else "_pass-through_")
                used = {c for p in procs for c, _ in L.component_inputs(p)}
                rows = ["| Load cell | Bin | Component |", "|---|---|---|"]
                for b in L.bins_at.get(s, []):
                    c = L.bin_component(b)
                    ctxt = f"{c} {L.name_of(c)}" + ("" if c in used else " *(spare)*") if c else "*empty*"
                    rows.append(f"| {L.sensor_of_bin.get(b, '-')} | {b} | {ctxt} |")
                st.markdown("\n".join(rows))


def flow_html(L: cm.LineStructure, highlight: str | None = None) -> str:
    """Readable, wrapping flow diagram: stages (parallel stations stacked), buffers between them."""
    import html as _h

    e = lambda x: _h.escape(str(x))
    box = ("background:#FFFFFF;color:#1B1B1B;border:1px solid #9FB3C8;border-radius:8px;width:190px;"
           "font-size:12.5px;line-height:1.35;box-shadow:0 1px 2px rgba(0,0,0,.08);overflow:hidden")
    parts = []
    for k, stage in enumerate(L.stages):
        cards = []
        for s in stage:
            spec = L.stations.get(s)
            if spec is None:
                cards.append(f'<div style="{box};border-color:#C62828;padding:8px;color:#C62828">{e(s)} (undefined)</div>')
                continue
            hl = "border:2px solid #C62828;" if s == highlight else ""
            procs = "<br>".join(f"<b>{e(p)}</b> {e(L.name_of(p))}" for p in L.station_processes(s)) or "<i>pass-through</i>"
            n_bins = len(L.bins_at.get(s, []))
            n_used = sum(1 for b in L.bins_at.get(s, []) if L.bin_component(b))
            cards.append(
                f'<div style="{box};{hl}"><div style="background:#1F4E79;color:#fff;padding:4px 8px">'
                f'<b>{e(s)}</b> {e(spec.get("name", ""))}</div>'
                f'<div style="padding:4px 8px;color:#444">👤 {e(spec.get("operator", "-"))} · 📷 '
                f'{e(spec.get("camera", "-"))}</div>'
                f'<div style="padding:4px 8px;background:#E3EEF9">{procs}</div>'
                f'<div style="padding:3px 8px;color:#555;font-size:11.5px">{n_used}/{n_bins} bins stocked · '
                f'{sum(1 for b in L.bins_at.get(s, []) if b in L.sensor_of_bin)} load cells</div></div>')
        parts.append('<div style="display:flex;flex-direction:column;gap:6px">' + "".join(cards) + "</div>")
        if k < len(L.stages) - 1:
            bid, cap = L.buffer_between(k)
            chip = (f'<div style="background:#FFF4D6;border:1px solid #C9A227;color:#5C4700;border-radius:12px;'
                    f'padding:2px 8px;font-size:11.5px;text-align:center">{e(bid)}<br>cap {cap}</div>'
                    if bid and cap > 0 else
                    '<div style="color:#888;font-size:11px;text-align:center">direct<br>hand-off</div>')
            parts.append(f'<div style="display:flex;align-items:center;gap:4px;color:#607D8B;font-size:18px">'
                         f'→{chip}→</div>')
    return ('<div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px;row-gap:14px;padding:4px 0 10px">'
            '<div style="color:#607D8B;font-size:12px">parts<br>in →</div>' + "".join(parts) +
            '<div style="color:#2E7D32;font-size:12px">→ finished<br>units</div></div>')


def tab_structure():
    cfg = ss.working_cfg
    L = cm.LineStructure(cfg)
    rep = cm.validate(cfg)
    st.subheader("Objects and relationships of the current line")
    st.caption("Everything below is derived from the configuration file - nothing is hard-coded. "
               "Each station box lists its operator, camera, the processes it performs and, per bin slot, "
               "the stable load-cell ID → bin → component relationship.")
    st.markdown(flow_html(L), unsafe_allow_html=True)
    with st.expander("Detailed diagram with every load cell → bin → component (Graphviz)"):
        st.caption("Use the full-screen button of the diagram to read it comfortably.")
        try:
            st.graphviz_chart(cm.to_dot(cfg, detail=True), **W_GV)
        except Exception as exc:  # pragma: no cover
            st.warning(f"Diagram unavailable: {exc}")
    render_report(rep)
    st.markdown("#### Stations: operator, camera, processes and bin slots")
    station_cards(L)

    t1, t2, t3, t4, t5 = st.tabs(["Stations", "Bins & sensors", "Processes", "Relationships", "YAML"])
    with t1:
        st.dataframe(cm.station_table(cfg), hide_index=True, **W_DF)
    with t2:
        st.dataframe(cm.bin_table(cfg), hide_index=True, **W_DF)
        st.caption("A load cell keeps its ID forever; reconfiguring changes which bin it measures and which "
                   "component that bin contains.")
    with t3:
        st.dataframe(cm.process_table(cfg), hide_index=True, **W_DF)
    with t4:
        rel = cm.relationships(cfg)
        kinds = sorted(rel["relation"].unique())
        sel = st.multiselect("Relations", kinds, default=[k for k in kinds if k not in ("in_stage",)], key="rel_kinds")
        st.dataframe(rel[rel["relation"].isin(sel)], hide_index=True, height=420, **W_DF)
    with t5:
        st.code(cm.dump_yaml(cfg), language="yaml")


# =============================================================================
# Tab 2 - Reconfigure
# =============================================================================


def tab_reconfigure():
    cfg = ss.working_cfg
    L = cm.LineStructure(cfg)
    v = ss.ver
    st.subheader("Reconfigure the line by changing relationships")
    st.caption("Each action edits the configuration file only. The simulation, sensor interpretation and "
               "diagrams follow automatically. Invalid combinations are reported, not silently accepted.")
    st.markdown(flow_html(L), unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("**Move a process to another station**")
        with st.form(f"f_move_{v}"):
            procs = list(L.processes)
            p = st.selectbox("Process", procs, format_func=lambda x: f"{x} - {L.name_of(x)} "
                             f"(at {', '.join(L.stations_of_process(x)) or 'none'})")
            s = st.selectbox("New station", L.routed_stations)
            mb = st.checkbox("Also move its component bins (re-point free bins)", value=True)
            if st.form_submit_button("Move process", **W_SUB):
                apply_op(cm.move_process, p, s, move_bins=mb)
    with c2.container(border=True):
        st.markdown("**Change what a bin contains** (e.g. LC_35 → B35 → C21)")
        with st.form(f"f_bin_{v}"):
            bins = list(L.bins)
            b = st.selectbox("Bin", bins, format_func=lambda x: f"{x} · {L.sensor_of_bin.get(x, 'no LC')} · "
                             f"{L.bins[x].get('station')} · {comp_label(L, L.bin_component(x))}")
            c = st.selectbox("Component", comp_options(L), format_func=lambda x: comp_label(L, x))
            if st.form_submit_button("Set bin content", **W_SUB):
                apply_op(cm.set_bin_component, b, None if c == "(empty)" else c)
    with c3.container(border=True):
        st.markdown("**Re-point a load cell to another bin**")
        with st.form(f"f_lc_{v}"):
            lc = st.selectbox("Load cell", L.load_cells, format_func=L.sensor_label)
            b2 = st.selectbox("Measures bin", ["(none)"] + list(L.bins))
            if st.form_submit_button("Re-point load cell", **W_SUB):
                apply_op(cm.set_sensor_bin, lc, None if b2 == "(none)" else b2)

    c4, c5, c6 = st.columns(3)
    with c4.container(border=True):
        st.markdown("**Operator assignment**")
        with st.form(f"f_op_{v}"):
            s = st.selectbox("Station", L.routed_stations, key=f"op_st_{v}")
            new_op = cm._next_id(L.operators, "O", start=len(L.operators) + 1)
            o = st.selectbox("Operator", list(L.operators) + [f"{new_op} (new)"])
            if st.form_submit_button("Assign operator", **W_SUB):
                apply_op(cm.set_station_operator, s, o.replace(" (new)", ""))
            st.caption("Give two stations the same operator to test sharing one person.")
    with c5.container(border=True):
        st.markdown("**Add / remove stations**")
        with st.form(f"f_par_{v}"):
            s = st.selectbox("Station", L.routed_stations, key=f"par_st_{v}")
            action = st.radio("Action", ["Add a parallel twin", "Insert a new empty station after it",
                                         "Remove it"], key=f"par_act_{v}")
            tgt = st.selectbox("If removing: move its processes to", ["-"] + L.routed_stations, key=f"par_tgt_{v}")
            if st.form_submit_button("Apply", **W_SUB):
                if action.startswith("Add"):
                    apply_op(cm.add_parallel_station, s)
                elif action.startswith("Insert"):
                    apply_op(cm.add_station, s)
                else:
                    apply_op(cm.remove_station, s, None if tgt in ("-", s) else tgt)
    with c6.container(border=True):
        st.markdown("**Swap a component in a process recipe**")
        with st.form(f"f_rec_{v}"):
            p = st.selectbox("Process", list(L.processes), key=f"rec_p_{v}",
                             format_func=lambda x: f"{x} - {L.name_of(x)}")
            olds = sorted({i for q in L.processes for i, _ in L.process_inputs(q)})
            old = st.selectbox("Replace input", olds, key=f"rec_old_{v}")
            newc = st.selectbox("With", list(L.components) + list(L.subassemblies), key=f"rec_new_{v}",
                                format_func=lambda x: f"{x} - {L.name_of(x)}")
            if st.form_submit_button("Swap input", **W_SUB):
                apply_op(cm.replace_process_input, p, old, newc)

    with st.container(border=True):
        st.markdown("**Buffers** (0 = direct hand-off, the upstream station blocks until the next one is free)")
        rows = []
        for k in range(len(L.stages) - 1):
            bid, cap = L.buffer_between(k)
            rows.append({"between": f"{'|'.join(L.stages[k])} → {'|'.join(L.stages[k + 1])}", "buffer": bid or "(none)",
                         "capacity": cap})
        with st.form(f"f_buf_{v}"):
            ed = st.data_editor(pd.DataFrame(rows), hide_index=True, disabled=["between", "buffer"], key=f"buf_ed_{v}",
                                column_config={"capacity": st.column_config.NumberColumn(min_value=0, max_value=100,
                                                                                         step=1)}, **W_ED)
            if st.form_submit_button("Apply buffer capacities", **W_SUB):
                new, msgs = ss.working_cfg, []
                for k, r in ed.iterrows():
                    cap = int(r["capacity"]) if not pd.isna(r["capacity"]) else 0
                    if cap != rows[k]["capacity"] or (rows[k]["buffer"] == "(none)" and cap > 0):
                        new, m = cm.set_buffer_capacity(new, int(k), cap)
                        msgs += m
                if not msgs:
                    log(["Buffer capacities unchanged."])
                    st.rerun()
                set_working(new, msgs=msgs)

    with st.expander("🧮 Edit all relationships as tables"):
        advanced_tables(L, v)
    with st.expander("📝 Edit the configuration file (YAML) directly"):
        txt = st.text_area("line_config.yaml", cm.dump_yaml(ss.working_cfg), height=420, key=f"yaml_cfg_{v}")
        if st.button("Apply YAML", key=f"yaml_apply_{v}"):
            try:
                set_working(cm.parse_yaml(txt), msgs=["Configuration replaced from YAML editor."])
            except Exception as exc:
                st.error(f"YAML error: {exc}")

    st.markdown("#### What changed")
    names = list(ss.scenarios)
    ref = st.selectbox("Compare the working configuration with", names, index=0, key=f"diff_ref_{v}")
    diff = cm.diff_relationships(ss.scenarios[ref], ss.working_cfg)
    if diff.empty:
        st.info(f"No relationship differs from '{ref}'.")
    else:
        st.caption(f"{len(diff)} relationship(s) differ from '{ref}' - a reconfiguration is mainly a change in "
                   f"these rows.")
        st.dataframe(diff, hide_index=True, **W_DF)
    render_report(cm.validate(ss.working_cfg), ok_text="The reconfigured line is consistent and can be simulated.")

    cA, cB = st.columns([2, 1])
    with cA:
        with st.form(f"f_save_{v}"):
            name = st.text_input("Save the working configuration as", value=ss.working_name.replace(" (edited)", "_v2"))
            desc = st.text_input("Short description (optional)")
            if st.form_submit_button("💾 Save as configuration"):
                name = name.strip()
                if not name or name == WORKING:
                    st.error("Please choose a name.")
                else:
                    ss.scenarios[name] = cm.set_version(ss.working_cfg, name, desc or None)
                    set_working(copy.deepcopy(ss.scenarios[name]), name=name,
                                msgs=[f"Saved configuration '{name}' - available in Compare."])
    with cB:
        st.markdown("&nbsp;")
        yaml_download("⬇️ Download working configuration", ss.working_cfg, f"{ss.working_name.split()[0]}.yaml",
                      header=f"Configuration '{ss.working_name}'", key="dl_cfg_rc")
    if ss.change_log:
        with st.expander(f"Change log ({len(ss.change_log)})"):
            st.code("\n".join(reversed(ss.change_log)), language=None)


def _parse_inputs(text: str) -> dict:
    out = {}
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        for sep in ("x", "*", ":", "×"):
            if sep in part[1:]:
                item, q = part.rsplit(sep, 1)
                try:
                    out[item.strip()] = int(q)
                    break
                except ValueError:
                    pass
        else:
            out[part] = 1
    return out


def advanced_tables(L: cm.LineStructure, v: int):
    cfg = ss.working_cfg
    with st.form(f"f_tables_{v}"):
        st.markdown("**Routing** - stages separated by `>`, parallel stations by `|` (e.g. `S1 > S2 > S3|S3B > S4`)")
        rt = st.text_input("Routing", cm.routing_text(cfg), key=f"rt_{v}")
        st.markdown("**Stations** (processes in execution order, comma separated)")
        sdf = pd.DataFrame([{"station": s, "operator": str(spec.get("operator", "")), "camera": str(spec.get("camera", "")),
                             "processes": ", ".join(L.station_processes(s))} for s, spec in L.stations.items()])
        sed = st.data_editor(sdf, hide_index=True, disabled=["station"], key=f"st_ed_{v}", **W_ED)
        st.markdown("**Processes** (inputs like `SA1, C04x2, C05x6`)")
        pdf = pd.DataFrame([{"process": p, "name": spec.get("name", ""),
                             "inputs": ", ".join(f"{i}x{q}" for i, q in L.process_inputs(p)),
                             "output": str(spec.get("output", ""))} for p, spec in L.processes.items()])
        ped = st.data_editor(pdf, hide_index=True, disabled=["process"], key=f"p_ed_{v}", **W_ED)
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Bins** (bin → station, bin → component)")
            bdf = pd.DataFrame([{"bin": b, "station": str(spec.get("station", "")),
                                 "component": L.bin_component(b) or "(empty)"} for b, spec in L.bins.items()])
            bed = st.data_editor(bdf, hide_index=True, disabled=["bin"], key=f"b_ed_{v}", height=380,
                                 column_config={
                                     "station": st.column_config.SelectboxColumn(options=list(L.stations)),
                                     "component": st.column_config.SelectboxColumn(options=comp_options(L))}, **W_ED)
        with cc2:
            st.markdown("**Load cells** (sensor → bin)")
            ldf = pd.DataFrame([{"load_cell": s, "measures": str(L.sensors[s].get("measures") or "(none)")}
                                for s in L.load_cells])
            led = st.data_editor(ldf, hide_index=True, disabled=["load_cell"], key=f"l_ed_{v}", height=380,
                                 column_config={"measures": st.column_config.SelectboxColumn(
                                     options=["(none)"] + list(L.bins))}, **W_ED)
        if st.form_submit_button("Apply table edits", type="primary"):
            new = copy.deepcopy(cfg)
            new["routing"] = cm.routing_from_stages(cm.parse_routing_text(rt))
            for _, r in sed.iterrows():
                spec = new["stations"][r["station"]]
                spec["operator"] = str(r["operator"]).strip()
                spec["camera"] = str(r["camera"]).strip() or None
                spec["processes"] = [x.strip() for x in str(r["processes"]).split(",") if x.strip()]
                if spec["operator"] and spec["operator"] not in new.setdefault("operators", {}):
                    new["operators"][spec["operator"]] = {"name": f"Operator {spec['operator']}"}
            for _, r in ped.iterrows():
                spec = new["processes"][r["process"]]
                spec["name"] = r["name"]
                spec["inputs"] = _parse_inputs(r["inputs"])
                spec["output"] = str(r["output"]).strip()
            for _, r in bed.iterrows():
                new["bins"][r["bin"]] = {"station": r["station"],
                                         "component": None if r["component"] in (None, "(empty)") else r["component"]}
            for _, r in led.iterrows():
                new["sensors"][r["load_cell"]]["measures"] = None if r["measures"] in (None, "(none)") else r["measures"]
            if new == cfg:
                log(["Table edits: nothing changed."])
                st.rerun()
            diff = cm.diff_relationships(cfg, new)
            set_working(new, msgs=[f"Table edits applied: {len(diff)} relationship(s) changed."])


# =============================================================================
# Tab 3 - Behaviour data
# =============================================================================


def tab_data():
    data = ss.data
    cfg = ss.working_cfg
    L = cm.LineStructure(cfg)
    B = bh.Behavior(data)
    v = ss.ver
    st.subheader("Behaviour data - times, distributions, measurements")
    st.caption("Keyed by process / operator / station / component ID, so it stays valid when the structure "
               "changes: a process that moves keeps its own time distribution.")
    render_report(bh.check_data(data, cfg), ok_text="The data file covers every object of the working configuration.")

    cap, mx, bn = bh.theoretical_capacity(cfg, data)
    if len(cap):
        st.markdown(f"**Analytic check** (mean times only, no variability): expected bottleneck **{bn}** → at most "
                    f"**{mx:.1f} units/h**.")
        st.dataframe(cap.round(3), hide_index=True, **W_DF)

    pt = data.get("process_times") or {}
    dp = data.get("defect_probability") or {}
    prows = []
    for p in list(L.processes) + [x for x in pt if x not in L.processes and x != "default"]:
        spec = B.process_time_spec(p)
        kind = str(spec.get("dist", "lognormal"))
        try:
            mean, cv = bh.dist_mean(spec), bh.dist_cv(spec)
        except Exception:
            mean, cv = float("nan"), float("nan")
        prows.append({"process": p, "name": L.name_of(p), "station": ", ".join(L.stations_of_process(p)) or "(unused)",
                      "dist": kind, "mean": mean, "cv": cv, "min": spec.get("min"), "mode": spec.get("mode"),
                      "max": spec.get("max"), "defect_prob": float(dp.get(p, dp.get("default", 0.0)) or 0.0),
                      "has_data": p in pt})
    orows = [{"operator": o, "stations": ", ".join(s for s in L.routed_stations
                                                    if str(L.stations[s].get("operator")) == o),
              **B.operator(o)} for o in L.operators]
    frows = []
    for s in L.routed_stations:
        f = B.failure(s)
        frows.append({"station": s, "mtbf_s": f["mtbf_s"] if f else None,
                      "mean_repair_s": bh.dist_mean(f["repair_time"]) if f else None})
    crows = []
    used = {L.bin_component(b) for b in L.bins} - {None}
    for c in L.components:
        bp = B.bin_params(c)
        crows.append({"component": c, "name": L.name_of(c), "in_a_bin": c in used, "unit_weight_g": B.unit_weight(c),
                      "capacity_units": bp["capacity_units"], "reorder_point_units": bp["reorder_point_units"]})

    with st.form(f"f_data_{v}"):
        st.markdown("**Process times** (seconds at operator speed 1.0) and defect probability")
        ped = st.data_editor(pd.DataFrame(prows), hide_index=True, key=f"pt_ed_{v}",
                             disabled=["process", "name", "station", "has_data"],
                             column_config={"dist": st.column_config.SelectboxColumn(options=bh.DIST_TYPES),
                                            "mean": st.column_config.NumberColumn(format="%.2f"),
                                            "cv": st.column_config.NumberColumn(format="%.3f"),
                                            "defect_prob": st.column_config.NumberColumn(format="%.4f", min_value=0.0,
                                                                                         max_value=1.0)}, **W_ED)
        st.caption("lognormal/normal/gamma use mean + cv · exponential uses mean · triangular uses min/mode/max · "
                   "uniform uses min/max · constant uses mean · empirical keeps its samples (edit in YAML).")
        a, b = st.columns(2)
        with a:
            st.markdown("**Operators** (speed_factor 1.1 = 10 % slower)")
            oed = st.data_editor(pd.DataFrame(orows), hide_index=True, disabled=["operator", "stations"],
                                 key=f"op_ed_{v}", **W_ED)
            st.markdown("**Station failures** (empty MTBF = never fails; MTBF counted in busy time)")
            fed = st.data_editor(pd.DataFrame(frows), hide_index=True, disabled=["station"], key=f"fl_ed_{v}", **W_ED)
        with b:
            st.markdown("**Components** - part weight seen by the load cell, bin stocking")
            ced = st.data_editor(pd.DataFrame(crows), hide_index=True, disabled=["component", "name", "in_a_bin"],
                                 key=f"cp_ed_{v}", height=420, **W_ED)
        st.markdown("**Sensors, material handling and pick behaviour**")
        lc, cam, rp, det = B.load_cell_params(), B.camera_params(), B.lead_time_spec(), B.detection_params()
        k = st.columns(6)
        rate = k[0].number_input("Load-cell rate (Hz)", 0.1, 50.0, float(lc["sample_rate_hz"]), step=0.5)
        noise = k[1].number_input("Noise σ (g)", 0.0, 20.0, float(lc["noise_sd_g"]), step=0.1)
        spike = k[2].number_input("Spike prob / sample", 0.0, 0.2, float(lc["spike_prob"]), step=0.001, format="%.3f")
        drop = k[3].number_input("Camera dropout", 0.0, 0.95, float(cam["dropout"]), step=0.05)
        pw = k[4].number_input("Pick window (fraction)", 0.0, 1.0, float(B.pick_window()), step=0.05)
        persist = k[5].number_input("Detection persistence", 1, 10, int(det["persistence_samples"]))
        k2 = st.columns(3)
        lt_min = k2[0].number_input("Refill lead time min (s)", 0.0, 3600.0, float(rp.get("min", 60)), step=10.0)
        lt_mode = k2[1].number_input("mode (s)", 0.0, 3600.0, float(rp.get("mode", 120)), step=10.0)
        lt_max = k2[2].number_input("max (s)", 0.0, 7200.0, float(rp.get("max", 240)), step=10.0)
        submitted = st.form_submit_button("Apply behaviour data", type="primary")
    if submitted:
        new = copy.deepcopy(data)
        npt = new.setdefault("process_times", {})
        ndp = new.setdefault("defect_probability", {})
        issues = []
        for _, r in ped.iterrows():
            p, d = r["process"], str(r["dist"])
            num = lambda x: None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)
            mean, cv, mn, mo, mxv = num(r["mean"]), num(r["cv"]), num(r["min"]), num(r["mode"]), num(r["max"])
            if d in ("lognormal", "normal", "gamma"):
                spec = {"dist": d, "mean": mean, "cv": cv if cv is not None else 0.0}
            elif d == "exponential":
                spec = {"dist": d, "mean": mean}
            elif d == "triangular":
                spec = {"dist": d, "min": mn, "mode": mo, "max": mxv}
            elif d == "uniform":
                spec = {"dist": d, "min": mn, "max": mxv}
            elif d == "constant":
                spec = {"dist": d, "value": mean}
            else:
                spec = copy.deepcopy(B.process_time_spec(p))
            if any(val is None for key, val in spec.items() if key != "dist"):
                issues.append(f"{p}: missing values for a {d} distribution - kept the previous one.")
                continue
            npt[p] = spec
            ndp[p] = float(r["defect_prob"] or 0.0)
        nop = new.setdefault("operators", {})
        for _, r in oed.iterrows():
            nop[r["operator"]] = {"speed_factor": float(r["speed_factor"]), "cv_factor": float(r["cv_factor"])}
        nfl = new.setdefault("failures", {})
        for _, r in fed.iterrows():
            m = r["mtbf_s"]
            if m is None or pd.isna(m) or float(m) <= 0:
                nfl[r["station"]] = {"mtbf_s": None}
            else:
                rep_t = r["mean_repair_s"] if not pd.isna(r["mean_repair_s"]) else 120.0
                nfl[r["station"]] = {"mtbf_s": float(m), "repair_time": {"dist": "exponential", "mean": float(rep_t)}}
        ncp = new.setdefault("components", {})
        nbn = new.setdefault("bins", {})
        for _, r in ced.iterrows():
            ncp.setdefault(r["component"], {})["unit_weight_g"] = float(r["unit_weight_g"])
            nbn.setdefault(r["component"], {}).update({"capacity_units": int(r["capacity_units"]),
                                                       "reorder_point_units": int(r["reorder_point_units"])})
        new.setdefault("load_cells", {}).update({"sample_rate_hz": rate, "noise_sd_g": noise, "spike_prob": spike})
        new.setdefault("camera", {})["dropout"] = drop
        new["pick_window"] = pw
        new.setdefault("pick_detection", {})["persistence_samples"] = int(persist)
        new.setdefault("replenishment", {})["lead_time"] = {"dist": "triangular", "min": lt_min,
                                                            "mode": min(max(lt_mode, lt_min), lt_max), "max": lt_max}
        for i in issues:
            st.warning(i)
        set_data(new, msgs=["Behaviour data updated."] + issues, rerun=not issues)

    with st.expander("📝 Edit the data file (YAML) directly"):
        txt = st.text_area("line_data.yaml", cm.dump_yaml(ss.data), height=420, key=f"yaml_data_{v}")
        if st.button("Apply data YAML", key=f"yaml_data_apply_{v}"):
            try:
                set_data(cm.parse_yaml(txt), msgs=["Data file replaced from YAML editor."])
            except Exception as exc:
                st.error(f"YAML error: {exc}")


# =============================================================================
# Tab 4 - Simulation results
# =============================================================================


def insights(res) -> list:
    k, sdf = res.kpis, res.stations
    out = []
    b = sdf.loc[sdf["station"] == k["bottleneck"]].iloc[0]
    out.append(f"**{k['bottleneck']}** is the bottleneck - active (working, down or waiting for material) "
               f"{b['active']:.0%} of the time. The analytic check predicted **{k['theoretical_bottleneck']}** "
               f"at ≤ {k['theoretical_max_uph']:.1f} units/h; simulated {k['throughput_uph']:.1f} units/h "
               f"({k['throughput_uph'] / k['theoretical_max_uph']:.0%} of that bound).")
    blocked = sdf[sdf["blocked"] > 0.15]
    if len(blocked):
        out.append("Blocked upstream of the bottleneck: " + ", ".join(
            f"{r.station} {r.blocked:.0%}" for r in blocked.itertuples()) +
            ". Bigger buffers there only add WIP unless the bottleneck gets faster.")
    starved = sdf[(sdf["starved"] > 0.35) & (sdf["stage"] > 1)]
    if len(starved):
        out.append("Starved (idle, waiting for work): " + ", ".join(
            f"{r.station} {r.starved:.0%}" for r in starved.itertuples()) + " - spare capacity downstream.")
    wo = sdf[sdf["wait_operator"] > 0.02]
    if len(wo):
        out.append("Waiting for a shared operator: " + ", ".join(f"{r.station} {r.wait_operator:.0%}"
                                                                 for r in wo.itertuples()) + ".")
    wm = sdf[sdf["wait_material"] > 0.01]
    if len(wm):
        so = res.bins[res.bins["stockouts"] > 0]
        parts = []
        for r in wm.itertuples():
            sb = so[so["station"] == r.station]
            bins_txt = ", ".join(f"{b.bin}/{b.component}" for b in sb.itertuples())
            parts.append(f"{r.station} {r.wait_material:.1%}" + (f" (bin {bins_txt})" if bins_txt else ""))
        out.append("Waiting for material: " + ", ".join(parts) +
                   " - check bin capacity, reorder point or refill lead time in the data file.")
    out.append(f"Little's law check: average WIP {k['wip_avg']:.2f} vs throughput × lead time "
               f"{k['littles_law_wip']:.2f} units.")
    return out


def tab_results():
    res = ss.result
    if res is None:
        st.info("Run a simulation from the sidebar to see results.")
        return
    k = res.kpis
    st.caption(f"Configuration **{ss.result_meta.get('config')}** · horizon {res.horizon_s / 60:.0f} min · "
               f"warm-up {res.warmup_s / 60:.0f} min (excluded) · seed {res.seed}")
    m = st.columns(6)
    m[0].metric("Throughput", f"{k['throughput_uph']:.1f} u/h")
    m[1].metric("Completed (after warm-up)", k["completed"])
    m[2].metric("First-pass yield", fmt(100 * k["fpy"], ".1f", " %"))
    m[3].metric("Mean lead time", fmt(k["lead_time_mean_s"] / 60, ".2f", " min"))
    m[4].metric("Average WIP", f"{k['wip_avg']:.2f}")
    m[5].metric("Bottleneck", k["bottleneck"])
    m2 = st.columns(6)
    m2[0].metric("Analytic max", f"{k['theoretical_max_uph']:.1f} u/h")
    m2[1].metric("Operators", k["n_operators"])
    m2[2].metric("Units / operator-hour", f"{k['throughput_uph'] / max(1, k['n_operators']):.1f}")
    m2[3].metric("Failures", k["failures"])
    m2[4].metric("Material waits", k["material_waits"])
    m2[5].metric("Camera frames OK", fmt(100 * k.get("camera_frame_rate", float("nan")), ".1f", " %"))
    with st.container(border=True):
        st.markdown("\n".join(f"- {x}" for x in insights(res)))

    st.markdown("#### Where each station spends its time")
    sdf = res.stations
    long = sdf.melt(id_vars=["station"], value_vars=STATES, var_name="state", value_name="share")
    long["state_label"] = long["state"].map(STATE_LABELS)
    long["order"] = long["state"].map({s: i for i, s in enumerate(STATES)})
    ch = alt.Chart(long).mark_bar().encode(
        y=alt.Y("station:N", sort=list(sdf["station"]), title=None, axis=alt.Axis(labelOverlap=False)),
        x=alt.X("share:Q", stack="normalize", axis=alt.Axis(format="%"), title="share of time after warm-up"),
        color=alt.Color("state_label:N", scale=STATE_SCALE, legend=alt.Legend(orient="bottom", title=None, columns=3)),
        order=alt.Order("order:Q"),
        tooltip=["station", alt.Tooltip("state_label:N", title="state"), alt.Tooltip("share:Q", format=".1%")],
    ).properties(height=alt.Step(30))
    st.altair_chart(ch, **W_CH)
    show = sdf.copy()
    for s in STATES + ["active"]:
        show[s] = 100 * show[s]
    st.dataframe(show, hide_index=True, **W_DF, column_config={
        **{s: st.column_config.NumberColumn(STATE_LABELS.get(s, "Active (bottleneck metric)"), format="%.1f%%")
           for s in STATES + ["active"]},
        "mean_service_s": st.column_config.NumberColumn("mean service (s)", format="%.1f"),
        "model_time_s": st.column_config.NumberColumn("model time (s)", format="%.1f")})

    st.markdown("#### Station timeline")
    H = res.horizon_s
    c1, c2 = st.columns([3, 1])
    span = c2.number_input("Window length (s)", 60, int(H), min(900, int(H)), step=60, key="gantt_len")
    start = window_start(c1, int(H - span), int(res.warmup_s), 30, "gantt_start")
    tl = res.state_timeline
    tl = tl[(tl["t1"] > start) & (tl["t0"] < start + span)].copy()
    tl["t0"] = tl["t0"].clip(lower=start)
    tl["t1"] = tl["t1"].clip(upper=start + span)
    tl["state_label"] = tl["state"].map(STATE_LABELS)
    g = alt.Chart(tl).mark_bar().encode(
        x=alt.X("t0:Q", title="time (s)", scale=alt.Scale(domain=[start, start + span])), x2="t1:Q",
        y=alt.Y("station:N", sort=list(sdf["station"]), title=None),
        color=alt.Color("state_label:N", scale=STATE_SCALE, legend=None),
        tooltip=["station", "state_label", alt.Tooltip("t0:Q", format=".1f"), alt.Tooltip("t1:Q", format=".1f")],
    ).properties(height=30 * len(sdf) + 40)
    st.altair_chart(g, **W_CH)

    a, b = st.columns(2)
    with a:
        st.markdown("#### Operators")
        op = res.operators.copy()
        st.altair_chart(alt.Chart(op).mark_bar(color="#1F77B4").encode(
            x=alt.X("utilization:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("operator:N", title=None), tooltip=["operator", "stations", alt.Tooltip("utilization", format=".1%"),
                                                        "jobs"]).properties(height=30 * len(op) + 40), **W_CH)
        st.markdown("#### Buffers")
        st.dataframe(res.buffers.round(3), hide_index=True, **W_DF,
                     column_config={"time_full": st.column_config.NumberColumn("time full", format="%.3f")})
    with b:
        st.markdown("#### Buffer occupancy over time")
        bs = res.buffer_series.sort_values(["buffer", "t"], kind="stable").copy()
        if len(bs):
            bs["t1"] = bs.groupby("buffer")["t"].shift(-1).fillna(res.horizon_s)
            bs = bs[bs["t1"] > bs["t"]]
            caps = dict(zip(res.buffers["buffer"], res.buffers["capacity"]))
            bs["fill"] = [lv / caps[b] if caps.get(b) else 0.0 for b, lv in zip(bs["buffer"], bs["level"])]
            st.altair_chart(alt.Chart(bs).mark_rect().encode(
                x=alt.X("t:Q", title="time (s)", scale=alt.Scale(domain=[0, res.horizon_s], nice=False)), x2="t1:Q",
                y=alt.Y("buffer:N", title=None, sort=list(res.buffers["buffer"])),
                color=alt.Color("fill:Q", title="fill", scale=alt.Scale(domain=[0, 1], scheme="orangered"),
                                legend=alt.Legend(format="%")),
                tooltip=["buffer", alt.Tooltip("t:Q", format=".0f"), "level"]).properties(height=alt.Step(30)), **W_CH)
            st.caption("Dark = buffer full (upstream station about to block), white = empty.")

    a, b = st.columns(2)
    with a:
        st.markdown("#### Work in process")
        st.altair_chart(alt.Chart(res.wip_series).mark_line(interpolate="step-after", color="#6A1B9A").encode(
            x=alt.X("t:Q", title="time (s)", scale=alt.Scale(domain=[0, res.horizon_s], nice=False)),
            y=alt.Y("wip:Q", title="units in the line")).properties(height=240), **W_CH)
    with b:
        st.markdown("#### Lead time")
        u = res.units[res.units["t_complete"] >= res.warmup_s].copy()
        u["lead_time_min"] = u["lead_time_s"] / 60
        if len(u):
            st.altair_chart(alt.Chart(u).mark_bar(color="#2E7D32").encode(
                x=alt.X("lead_time_min:Q", bin=alt.Bin(maxbins=30), title="lead time (min)"),
                y=alt.Y("count():Q", title="units")).properties(height=240), **W_CH)

    st.markdown("#### Bins and replenishment")
    bins = res.bins[res.bins["component"].notna()]
    st.dataframe(bins, hide_index=True, **W_DF)
    sel = st.selectbox("Bin level over time", list(bins["bin"]), key="bin_level_sel",
                       format_func=lambda x: f"{x} ({bins.set_index('bin').loc[x, 'component']}, "
                                             f"{bins.set_index('bin').loc[x, 'sensor']})")
    bl = res.bin_levels[res.bin_levels["bin"] == sel]
    if len(bl):
        base = alt.Chart(bl).encode(x=alt.X("t:Q", title="time (s)"))
        st.altair_chart((base.mark_line(interpolate="step-after").encode(y=alt.Y("units:Q", title="parts in bin")) +
                         base.transform_filter("datum.kind == 'refill'").mark_point(color="#C62828", size=70,
                                                                                     filled=True).encode(y="units:Q")
                         ).properties(height=220), **W_CH)
        st.caption("Red dots = refills by the material handler.")
    with st.expander("Analytic capacity per resource"):
        st.dataframe(res.capacity.round(3), hide_index=True, **W_DF)


# =============================================================================
# Tab 5 - Sensors & calibration
# =============================================================================


def sensing_bundle(res, rate, noise, spike_p, det_params, model_data):
    key = (res.run_id, rate, noise, spike_p, tuple(sorted(det_params.items())), json.dumps(model_data, sort_keys=True,
                                                                                            default=str))
    cache = ss.sensing_cache
    if key not in cache:
        sig = sn.synthesize_load_cell_signals(res, rate_hz=rate, noise_sd_g=noise, spike_prob=spike_p)
        det = sn.detect_all(sig, res.config, model_data, noise_sd_g=noise, params=det_params)
        metrics, matched = sn.score_detection(det, res.picks, res.config, model_data)
        cyc, _ = sn.estimate_cycle_times(det, res.camera, res.config, model_data, rate_hz=rate, jobs=res.jobs)
        pe, ps = sn.estimate_process_times(det, res.camera, res.config, model_data, rate_hz=rate, events=res.events)
        cache.clear()
        cache[key] = dict(signals=sig, det=det, metrics=metrics, matched=matched, cyc=cyc, pe=pe, ps=ps)
    return cache[key]


def signal_chart(sig, sensor, t0, t1, det=None, truth_levels=None):
    s = sig[(sig["sensor"] == sensor) & (sig["t"] >= t0) & (sig["t"] <= t1)]
    if len(s) > 5000:
        s = s.iloc[:: int(math.ceil(len(s) / 5000))]
    layers = [alt.Chart(s).mark_line(color="#4C78A8", strokeWidth=1).encode(
        x=alt.X("t:Q", title="time (s)"), y=alt.Y("weight_g:Q", title="weight (g)", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("t:Q", format=".1f"), alt.Tooltip("weight_g:Q", format=".1f")])]
    marks = []
    if truth_levels is not None and len(truth_levels):
        tv = truth_levels[(truth_levels["sensor"] == sensor) & (truth_levels["t"].between(t0, t1)) &
                          (truth_levels["kind"].isin(["pick", "refill"]))]
        marks.append(pd.DataFrame({"t": tv["t"], "weight_g": tv["weight_g"], "series": "true " + tv["kind"]}))
    if det is not None and len(det):
        dv = det[(det["sensor"] == sensor) & (det["t"].between(t0, t1))]
        marks.append(pd.DataFrame({"t": dv["t"], "weight_g": dv["level_g"], "series": "detected " + dv["kind"],
                                   "parts": dv["parts"]}))
    if marks:
        mk = pd.concat(marks, ignore_index=True)
        dom = ["true pick", "true refill", "detected pick", "detected refill", "detected unexplained"]
        layers.append(alt.Chart(mk).mark_point(size=80, filled=True, opacity=0.85).encode(
            x="t:Q", y="weight_g:Q",
            color=alt.Color("series:N", scale=alt.Scale(domain=dom, range=["#2E7D32", "#1565C0", "#E65100", "#6A1B9A",
                                                                           "#C62828"]),
                            legend=alt.Legend(orient="bottom", title=None)),
            shape=alt.Shape("series:N", scale=alt.Scale(domain=dom, range=["circle", "circle", "triangle-down",
                                                                           "triangle-up", "cross"]), legend=None),
            tooltip=["series", alt.Tooltip("t:Q", format=".1f"), alt.Tooltip("weight_g:Q", format=".1f"), "parts"]))
    return alt.layer(*layers).properties(height=300)


def tab_sensors():
    st.subheader("Sensors: connectivity → pick detection → timing → calibration")
    st.caption("Signals are keyed by stable sensor IDs. Their meaning (bin, component, part weight) comes from the "
               "current configuration + data files, so a reconfiguration needs no code change here.")
    src = st.radio("Data source", ["Last simulation run", "Emulated physical testbed (hidden deviations)",
                                   "Uploaded CSV (real testbed)"], horizontal=True, key="sens_src")
    if src.startswith("Uploaded"):
        return sensors_uploaded()
    hidden = None
    if src.startswith("Emulated"):
        with st.container(border=True):
            st.markdown("The testbed emulator runs the **working configuration** with process times that secretly "
                        "deviate from the data file. The sensors then have to discover those deviations.")
            c = st.columns(4)
            dev = c[0].slider("Max deviation (±%)", 0, 50, 20, key="tb_dev")
            tseed = c[1].number_input("Hidden seed", 0, 10**6, 7, key="tb_seed")
            hours = c[2].number_input("Observation (h)", 1, 12, 4, key="tb_h")
            c[3].markdown("&nbsp;")
            if c[3].button("Run testbed emulation", type="primary", **W_BTN):
                if not cm.validate(ss.working_cfg).ok:
                    st.error("The working configuration has errors.")
                else:
                    rng = np.random.default_rng(int(tseed))
                    L = cm.LineStructure(ss.working_cfg)
                    fac = {p: float(rng.uniform(1 - dev / 100, 1 + dev / 100)) for p in L.processes}
                    hidden_data = bh.scale_process_times(ss.data, fac)
                    with st.spinner("Emulating the testbed ..."):
                        r = LineSimulator(ss.working_cfg, hidden_data, horizon_s=hours * 3600.0,
                                          seed=int(tseed) + 1000).run()
                    ss.testbed = {"result": r, "factors": fac, "hidden_data": hidden_data, "config": ss.working_name}
                    ss.sensing_cache = {}
        if ss.testbed is None:
            st.info("Run the testbed emulation to continue.")
            return
        res = ss.testbed["result"]
        hidden = ss.testbed
        st.caption(f"Testbed emulation of '{ss.testbed['config']}' · {res.horizon_s / 3600:.0f} h observed")
    else:
        res = ss.result
        if res is None:
            st.info("Run a simulation from the sidebar first (or use the testbed emulator).")
            return

    B = bh.Behavior(ss.data)
    lc, dp = B.load_cell_params(), B.detection_params()
    c = st.columns(6)
    rate = c[0].number_input("Sample rate (Hz)", 0.2, 20.0, float(lc["sample_rate_hz"]), step=0.5, key="sx_rate")
    noise = c[1].number_input("Noise σ (g)", 0.0, 20.0, float(lc["noise_sd_g"]), step=0.1, key="sx_noise")
    spike = c[2].number_input("Spike prob", 0.0, 0.2, float(lc["spike_prob"]), step=0.001, format="%.3f", key="sx_sp")
    thr = c[3].number_input("Threshold × part", 0.1, 1.0, float(dp["threshold_fraction"]), step=0.05, key="sx_thr")
    ksig = c[4].number_input("Threshold × σ", 1.0, 10.0, float(dp["noise_k_sigma"]), step=0.5, key="sx_k")
    pers = c[5].number_input("Persistence (samples)", 1, 10, int(dp["persistence_samples"]), key="sx_p")
    n_samples = res.horizon_s * rate * len(cm.LineStructure(res.config).load_cells)
    if n_samples > 6e6:
        st.warning(f"That is {n_samples / 1e6:.1f} M load-cell samples - lower the rate or the horizon.")
        return
    det_params = {"threshold_fraction": thr, "noise_k_sigma": ksig, "persistence_samples": int(pers)}
    with st.spinner("Sampling load cells and detecting picks ..."):
        bnd = sensing_bundle(res, rate, noise, spike, det_params, ss.data)

    # ---- Phase 1
    st.markdown("### Phase 1 · Connectivity - raw load-cell readings")
    smap = sn.sensor_map(res.config, ss.data)
    active = smap[smap["component"].notna()]
    opts = list(active["sensor"]) + [s for s in smap["sensor"] if s not in set(active["sensor"])]
    lab = dict(zip(smap["sensor"], smap["label"]))
    a, b, c3 = st.columns([2, 2, 1])
    sensor = a.selectbox("Sensor", opts, format_func=lambda s: lab.get(s, s), key="sx_sensor")
    span = c3.number_input("Window (s)", 30, int(res.horizon_s), min(600, int(res.horizon_s)), step=30, key="sx_span")
    t0 = window_start(b, int(res.horizon_s - span), int(res.warmup_s), 10, "sx_t0")
    st.altair_chart(signal_chart(bnd["signals"], sensor, t0, t0 + span, bnd["det"], res.bin_levels), **W_CH)
    st.caption("Line = sampled weight (noise, spikes). Green = true picks from the simulation, orange triangles = "
               "picks found by the detector. A drop of n × part weight is counted as n parts.")

    # ---- Phase 2
    st.markdown("### Phase 2 · Pick detection - weight drop → pick event")
    mt = bnd["metrics"]
    if len(mt):
        tp, tr, dt = mt["matched"].sum(), mt["true_picks"].sum(), mt["detected"].sum()
        exact = (mt["count_exact"] * mt["matched"]).sum() / max(1, tp)
        k = st.columns(4)
        k[0].metric("Recall (picks found)", fmt(100 * tp / tr if tr else float("nan"), ".1f", " %"))
        k[1].metric("Precision (no false picks)", fmt(100 * tp / dt if dt else float("nan"), ".1f", " %"))
        k[2].metric("Part count exact", fmt(100 * exact, ".1f", " %"))
        k[3].metric("Mean detection lag", fmt(mt["mean_lag_s"].mean(), ".2f", " s"))
        st.dataframe(mt.round(3), hide_index=True, **W_DF)
    with st.expander(f"All detected events ({len(bnd['det'])})"):
        st.dataframe(bnd["det"].round(2), hide_index=True, height=300, **W_DF)

    # ---- Phase 3
    timing_and_calibration(bnd["pe"], bnd["cyc"], hidden, ss.data, res.config, key="sim")


def timing_and_calibration(pe, cyc, hidden, model_data, cfg, key):
    st.markdown("### Phase 3 · Timing - events → process times → calibration")
    st.caption("Each process starts with the pick of its first component (load-cell event); the last process "
               "ends with the camera's *unit complete*. Dropped camera frames reduce the sample, they do not "
               "bias it. Estimates include any waiting for material inside the job.")
    if pe is None or pe.empty:
        st.info("No timing estimates (need load-cell picks and camera 'unit_complete' events).")
        return
    show = pe.copy()
    if hidden is not None:
        show["hidden_true_mean_s"] = [
            sum(bh.dist_mean(hidden["hidden_data"]["process_times"][p]) for p in str(ps).split("+"))
            * bh.Behavior(model_data).operator(str(cm.LineStructure(cfg).stations[s].get("operator")))["speed_factor"]
            for s, ps in zip(show["station"], show["processes"])]
    st.dataframe(show.round(3), hide_index=True, **W_DF)
    long = show.melt(id_vars=["station", "processes"],
                     value_vars=[c for c in ["model_mean_s", "est_mean_s", "true_mean_s", "hidden_true_mean_s"]
                                 if c in show and show[c].notna().any()],
                     var_name="source", value_name="seconds")
    long["segment"] = long["station"] + " · " + long["processes"]
    long["source"] = long["source"].map({"model_mean_s": "data file (model)", "est_mean_s": "estimated from sensors",
                                         "true_mean_s": "simulation truth", "hidden_true_mean_s": "hidden truth"})
    src_dom = ["data file (model)", "estimated from sensors", "simulation truth", "hidden truth"]
    st.altair_chart(alt.Chart(long).mark_bar().encode(
        x=alt.X("segment:N", title=None, sort=list(dict.fromkeys(long["segment"])), axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("source:N", sort=src_dom), y=alt.Y("seconds:Q", title="mean time (s)"),
        color=alt.Color("source:N", scale=alt.Scale(domain=src_dom, range=["#90A4AE", "#1E88E5", "#2E7D32", "#C62828"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["segment", "source", alt.Tooltip("seconds:Q", format=".2f")]).properties(height=280), **W_CH)
    if cyc is not None and len(cyc):
        with st.expander("Station-level view (start pick → unit complete, and pick-to-pick interval)"):
            st.dataframe(cyc.round(3), hide_index=True, **W_DF)
            st.caption("The pick-to-pick interval uses load cells only; it equals the station's effective cycle "
                       "(service + waiting), i.e. 3600 / throughput.")

    st.markdown("#### Calibrate the data file")
    min_n = st.number_input("Minimum samples per process", 3, 500, 10, key=f"cal_min_{key}")
    new_data, notes = sn.calibrate_data(model_data, cfg, pe, min_samples=int(min_n))
    if notes.empty:
        st.info("Not enough samples to calibrate.")
        return
    if hidden is not None:
        hd = hidden["hidden_data"]["process_times"]
        notes["hidden_true_mean_s"] = notes["process"].map(lambda p: bh.dist_mean(hd[p]) if p in hd else float("nan"))
        notes["error_before_pct"] = 100 * (notes["old_mean_s"] / notes["hidden_true_mean_s"] - 1)
        notes["error_after_pct"] = 100 * (notes["new_mean_s"] / notes["hidden_true_mean_s"] - 1)
        k = st.columns(2)
        k[0].metric("Mean |error| before calibration", f"{notes['error_before_pct'].abs().mean():.1f} %")
        k[1].metric("Mean |error| after calibration", f"{notes['error_after_pct'].abs().mean():.1f} %")
    st.dataframe(notes.round(3), hide_index=True, **W_DF)
    c1, c2 = st.columns(2)
    if c1.button("✅ Apply calibrated process times to the working data file", key=f"cal_apply_{key}", **W_BTN):
        set_data(new_data, msgs=[f"Calibrated {len(notes)} process time(s) from sensor data."])
    with c2:
        yaml_download("⬇️ Download calibrated line_data.yaml", new_data, "line_data_calibrated.yaml",
                      header="Behaviour data calibrated from load-cell + camera events", key=f"cal_dl_{key}")


def sensors_uploaded():
    st.markdown("Upload **load-cell readings** (columns: `t` or `timestamp`, `sensor`, `weight_g`) and optionally "
                "**camera events** (`t`, `station`, `action` = start_work / pick / unit_complete, `has_frame`). "
                "Sensors are interpreted through the working configuration.")
    if ss.result is not None:
        demo = sn.synthesize_load_cell_signals(ss.result, t1=min(ss.result.horizon_s, 1800))
        st.download_button("Example load-cell CSV (from the last run)", demo.to_csv(index=False).encode(),
                           "loadcells_example.csv", "text/csv")
        st.download_button("Example camera CSV (from the last run)",
                           ss.result.camera[["t", "station", "action", "has_frame"]].to_csv(index=False).encode(),
                           "camera_example.csv", "text/csv")
    a, b = st.columns(2)
    f1 = a.file_uploader("Load-cell CSV", type=["csv"], key="lc_csv")
    f2 = b.file_uploader("Camera CSV (optional)", type=["csv"], key="cam_csv")
    if f1 is None:
        return
    try:
        sig = sn.read_signal_csv(pd.read_csv(f1))
    except Exception as exc:
        st.error(f"Could not read load-cell CSV: {exc}")
        return
    cam = None
    if f2 is not None:
        try:
            cam = sn.read_camera_csv(pd.read_csv(f2))
        except Exception as exc:
            st.error(f"Could not read camera CSV: {exc}")
    sn.align_time(sig, cam)
    cfg = ss.working_cfg
    smap = sn.sensor_map(cfg, ss.data)
    unknown = sorted(set(sig["sensor"]) - set(smap["sensor"]))
    if unknown:
        st.warning(f"Sensors not in the configuration (ignored for interpretation): {', '.join(unknown)}")
    dt = sig.groupby("sensor")["t"].diff().median()
    rate = 1.0 / dt if dt and dt > 0 else 1.0
    noise = float(sig.groupby("sensor")["weight_g"].apply(lambda w: (w.diff().abs().median() or 0) / 0.954).median())
    noise = noise if noise > 0 else bh.Behavior(ss.data).load_cell_params()["noise_sd_g"]
    st.caption(f"{len(sig):,} readings · {sig['sensor'].nunique()} sensors · ≈ {rate:.2f} Hz · noise σ ≈ {noise:.2f} g "
               f"(estimated)")
    det = sn.detect_all(sig, cfg, ss.data, noise_sd_g=noise)
    lab = dict(zip(smap["sensor"], smap["label"]))
    sensor = st.selectbox("Sensor", sorted(sig["sensor"].unique()), format_func=lambda s: lab.get(s, s), key="up_sensor")
    tmin, tmax = float(sig["t"].min()), float(sig["t"].max())
    st.altair_chart(signal_chart(sig, sensor, tmin, min(tmax, tmin + 900), det), **W_CH)
    st.markdown("**Detected events per sensor**")
    if len(det):
        st.dataframe(det.groupby(["sensor", "bin", "component", "kind"]).agg(events=("t", "count"),
                                                                            parts=("parts", "sum")).reset_index(),
                     hide_index=True, **W_DF)
        st.download_button("⬇️ Detected pick events (CSV)", det.to_csv(index=False).encode(), "pick_events.csv",
                           "text/csv")
    if cam is not None:
        pe, _ = sn.estimate_process_times(det, cam, cfg, ss.data, rate_hz=rate)
        cyc, _ = sn.estimate_cycle_times(det, cam, cfg, ss.data, rate_hz=rate)
        timing_and_calibration(pe, cyc, None, ss.data, cfg, key="upl")
    else:
        st.info("Add a camera CSV to estimate process times (the camera marks the end of each unit).")


# =============================================================================
# Tab 6 - Compare configurations
# =============================================================================


def tab_compare():
    st.subheader("Compare configurations")
    st.caption("Every configuration runs with the same behaviour data and the same random seeds (common random "
               "numbers), so differences come from the structure, not from luck.")
    names = list(ss.scenarios) + [WORKING]
    sel = st.multiselect("Configurations", names, default=list(ss.scenarios), key="cmp_sel")
    c = st.columns(4)
    reps = c[0].number_input("Replications", 2, 50, 5, key="cmp_reps")
    hmin = c[1].number_input("Horizon (min)", 10, 24 * 60, 60, step=10, key="cmp_h")
    wmin = c[2].number_input("Warm-up (min)", 0, 120, 5, key="cmp_w")
    seed = c[3].number_input("Base seed", 0, 10**6, 100, key="cmp_seed")
    if st.button("▶️ Run comparison", type="primary", disabled=not sel):
        scen = {n: (ss.working_cfg if n == WORKING else ss.scenarios[n]) for n in sel}
        bar = st.progress(0.0, text="Starting ...")
        t0 = time.time()
        per, summ, skipped = compare_scenarios(scen, ss.data, int(reps), hmin * 60.0, min(wmin, hmin - 1) * 60.0,
                                               int(seed), progress=lambda f, m: bar.progress(min(1.0, f), text=m))
        bar.progress(1.0, text=f"Done in {time.time() - t0:.1f} s")
        base = ss.scenarios.get("baseline")
        changes = {n: (len(cm.diff_relationships(base, scen[n])) if base is not None else None) for n in scen}
        ss.compare = {"per": per, "summ": summ, "skipped": skipped, "changes": changes}
    cmpd = ss.compare
    if not cmpd:
        return
    for n, errs in cmpd["skipped"].items():
        st.error(f"'{n}' skipped - invalid configuration: " + "; ".join(errs[:3]))
    s = cmpd["summ"]
    if s.empty:
        return
    tbl = pd.DataFrame({
        "configuration": s["scenario"],
        "throughput (u/h)": [f"{m:.1f} ± {c:.1f}" for m, c in zip(s["throughput_uph_mean"], s["throughput_uph_ci95"])],
        "lead time (min)": (s["lead_time_mean_s_mean"] / 60).round(2),
        "avg WIP": s["wip_avg_mean"].round(2),
        "FPY %": (100 * s["fpy_mean"]).round(2),
        "bottleneck": [f"{b} ({sh:.0%})" for b, sh in zip(s["bottleneck"], s["bottleneck_share"])],
        "operators": s["n_operators"],
        "u/h per operator": (s["throughput_uph_mean"] / s["n_operators"]).round(1),
        "analytic max (u/h)": s["theoretical_max_uph"].round(1),
        "relationship changes vs baseline": [cmpd["changes"].get(n) for n in s["scenario"]],
    })
    st.dataframe(tbl, hide_index=True, **W_DF)
    d = s[["scenario", "throughput_uph_mean", "throughput_uph_ci95", "theoretical_max_uph"]].copy()
    d["lo"] = d["throughput_uph_mean"] - d["throughput_uph_ci95"].fillna(0)
    d["hi"] = d["throughput_uph_mean"] + d["throughput_uph_ci95"].fillna(0)
    order = list(d.sort_values("throughput_uph_mean", ascending=False)["scenario"])
    base = alt.Chart(d).encode(y=alt.Y("scenario:N", sort=order, title=None))
    chart = (base.mark_bar(color="#4C78A8").encode(x=alt.X("throughput_uph_mean:Q", title="throughput (units/h)"),
                                                   tooltip=["scenario", alt.Tooltip("throughput_uph_mean", format=".1f"),
                                                            alt.Tooltip("throughput_uph_ci95", format=".1f")])
             + base.mark_rule(color="black").encode(x="lo:Q", x2="hi:Q")
             + base.mark_tick(color="#C62828", thickness=2, size=22).encode(x="theoretical_max_uph:Q"))
    st.altair_chart(chart.properties(height=40 * len(d) + 40), **W_CH)
    st.caption("Bars = mean throughput, black line = 95 % confidence interval, red tick = analytic upper bound.")
    st.download_button("⬇️ Per-replication results (CSV)", cmpd["per"].to_csv(index=False).encode(),
                       "comparison_replications.csv", "text/csv")


# =============================================================================
# Tab 7 - Logs & export
# =============================================================================


def tab_logs():
    res = ss.result
    if res is None:
        st.info("Run a simulation to see the event logs.")
        return
    sets = {"Events": res.events, "Jobs (per station)": res.jobs, "Units": res.units,
            "Picks - ground truth": res.picks, "Bin levels": res.bin_levels, "Camera events": res.camera,
            "Station state timeline": res.state_timeline}
    name = st.selectbox("Log", list(sets), key="log_sel")
    df = sets[name]
    st.caption(f"{len(df):,} rows")
    st.dataframe(df.head(5000), hide_index=True, height=420, **W_DF)
    c1, c2 = st.columns(2)
    c1.download_button(f"⬇️ {name} (CSV)", df.to_csv(index=False).encode(), f"{name.split()[0].lower()}.csv",
                       "text/csv", **W_DL)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("line_config.yaml", cm.dump_yaml(res.config))
        z.writestr("line_data.yaml", cm.dump_yaml(res.data))
        z.writestr("kpis.json", json.dumps({k: (None if isinstance(v, float) and math.isnan(v) else v)
                                            for k, v in res.kpis.items()}, indent=2, default=str))
        for fname, frame in [("stations.csv", res.stations), ("operators.csv", res.operators),
                             ("buffers.csv", res.buffers), ("bins.csv", res.bins), ("events.csv", res.events),
                             ("jobs.csv", res.jobs), ("units.csv", res.units), ("picks.csv", res.picks),
                             ("bin_levels.csv", res.bin_levels), ("camera.csv", res.camera)]:
            z.writestr(fname, frame.to_csv(index=False))
    c2.download_button("⬇️ Complete run (config + data + KPIs + logs, ZIP)", buf.getvalue(), "simulation_run.zip",
                       "application/zip", **W_DL)


# =============================================================================
# Page
# =============================================================================


def main():
    init_state()
    sidebar()
    st.title("🏭 Reconfigurable Assembly Line - Digital Twin & Simulator")
    st.caption("A common, machine-readable description of the line - its objects and relationships - reused by the "
               "sensor pipeline, the digital twin and the simulation. Change the description, not the software.")
    for m in ss.flash or []:
        st.toast(m)
    ss.flash = []
    architecture_strip()
    tabs = st.tabs(["🏭 Line structure", "🔧 Reconfigure", "📊 Behaviour data", "▶️ Simulation results",
                    "📡 Sensors & calibration", "⚖️ Compare configurations", "🧾 Logs & export"])
    with tabs[0]:
        tab_structure()
    with tabs[1]:
        tab_reconfigure()
    with tabs[2]:
        tab_data()
    with tabs[3]:
        tab_results()
    with tabs[4]:
        tab_sensors()
    with tabs[5]:
        tab_compare()
    with tabs[6]:
        tab_logs()
    st.divider()
    st.caption("VIPER / Innovation Factory assembly-line project · configuration file (structure) + data file "
               "(behaviour) → simulation, sensors, dashboard")


main()
