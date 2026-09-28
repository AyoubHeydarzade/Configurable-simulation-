"""
run_headless.py - use the library without Streamlit (e.g. from a notebook, a
scheduled job, or to feed results to FlexSim / a digital-twin dashboard).

    python examples/run_headless.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from line_sim import behavior as bh, config_model as cm, sensing as sn  # noqa: E402
from line_sim.simulator import LineSimulator, compare_scenarios  # noqa: E402

cfg = cm.load_yaml_file(ROOT / "configs" / "line_config.yaml")   # STRUCTURE
data = cm.load_yaml_file(ROOT / "configs" / "line_data.yaml")    # BEHAVIOUR

# 1) Validate the structure and check that the data covers it
print("structure:", cm.validate(cfg).summary(), "| data:", bh.check_data(data, cfg).summary())

# 2) Simulate
res = LineSimulator(cfg, data, horizon_s=3600, warmup_s=300, seed=42).run()
k = res.kpis
print(f"throughput {k['throughput_uph']:.1f} u/h | bottleneck {k['bottleneck']} | "
      f"lead time {k['lead_time_mean_s'] / 60:.2f} min | WIP {k['wip_avg']:.2f}")
print(res.stations[["station", "busy", "blocked", "starved", "wait_material", "down"]].round(2).to_string(index=False))

# 3) Reconfigure by editing relationships only, then compare
moved, msgs = cm.move_process(cfg, "P3", "S2")
print("\n".join(msgs))
print(cm.diff_relationships(cfg, moved).to_string(index=False))
_, summary, _ = compare_scenarios({"baseline": cfg, "P3 at S2": moved}, data, reps=5)
print(summary[["scenario", "throughput_uph_mean", "throughput_uph_ci95", "bottleneck"]].round(1).to_string(index=False))

# 4) Sensors: raw load-cell signals -> pick events -> process times -> calibrated data file
signals = sn.synthesize_load_cell_signals(res)                 # Phase 1 (replace with sn.read_signal_csv(...))
picks = sn.detect_all(signals, cfg, data)                      # Phase 2
scores, _ = sn.score_detection(picks, res.picks, cfg, data)
print(f"pick recall {scores.matched.sum() / scores.true_picks.sum():.1%}, "
      f"precision {scores.matched.sum() / scores.detected.sum():.1%}")
times, _ = sn.estimate_process_times(picks, res.camera, cfg, data, events=res.events)   # Phase 3
print(times[["station", "processes", "samples", "est_mean_s", "model_mean_s"]].round(2).to_string(index=False))
calibrated, notes = sn.calibrate_data(data, cfg, times)
(ROOT / "configs" / "line_data_calibrated.yaml").write_text(cm.dump_yaml(calibrated, header="Calibrated from sensors"))
print("wrote configs/line_data_calibrated.yaml")
