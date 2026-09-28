"""line_sim - configuration-driven digital twin / simulation of a reconfigurable manual assembly line.

Modules
  config_model  structure: objects + relationships, validation, reconfiguration, diagram
  behavior      behaviour: distributions, operators, failures, weights, sensor parameters
  simulator     discrete-event simulation built from config + data, KPIs, replications
  sensing       load-cell signals, pick detection, cycle-time estimation and calibration
"""
from . import behavior, config_model, sensing, simulator  # noqa: F401
from .config_model import ConfigError, LineStructure, dump_yaml, load_yaml_file, parse_yaml, validate  # noqa: F401
from .simulator import LineSimulator, SimResult, compare_scenarios, simulate  # noqa: F401

__version__ = "1.0.0"
