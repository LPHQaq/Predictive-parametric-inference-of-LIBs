"""Self-contained Slurm/server entrypoint for parallel PyBaMM data generation.

This file keeps all generation settings locally so it can run on a server with only
this script and the parameter JSON files present. It preserves the same output layout as
pybamm_test.ipynb and the local generator:
- one summary CSV that tracks completed simulations and their output files
- one cycle detail CSV per successful simulation for cycle-10 / cycle-50 Q-V data
- one life detail CSV per successful simulation for SOH vs cycle data
- one metadata JSON describing the run configuration

Failed simulations are logged but are not written to disk.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import logging
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import qmc

sys.path.insert(0, ".../PyBaMM")
import pybamm

SCRIPT_DIR = Path(__file__).resolve().parent
DQ_REFERENCE_CYCLE = 10
DQ_TARGET_CYCLE = 50
SOH_THRESHOLDS = np.arange(99.0, 79.0, -1.0)
VOLTAGE_WINDOW = (2.5, 4.2)
N_DQ_POINTS = 200
OUTPUT_ROOT_DIRNAME = "generated_pybamm_data"
# OUTPUT_ROOT_DIRNAME = "debug"
CYCLE_DETAIL_DIRNAME = "cycle_qv_csv"
LIFE_DETAIL_DIRNAME = "life_soh_csv"
SUMMARY_FILENAME = "scan_summary.csv"
METADATA_FILENAME = "scan_metadata.json"

PM_VARIATION_LIST = [
    "Initial SEI thickness [m]",
    "SEI solvent diffusivity [m2.s-1]",
    # "SEI resistivity [Ohm.m]",

    "Lithium plating kinetic rate constant [m.s-1]",
    "Dead lithium decay constant [s-1]",

    "Negative electrode critical stress [Pa]",
    # "Negative electrode LAM constant exponential term",
    "Negative electrode reaction-driven LAM factor [m3.mol-1]",
    "Negative electrode LAM constant proportional term [s-1]",
]

VARIATION_RANGES = {
    "Initial SEI thickness [m]": (0.25, 1.0),
    "SEI solvent diffusivity [m2.s-1]": (5, 30),
    # "SEI resistivity [Ohm.m]": (1.0, 5.0),

    "Lithium plating kinetic rate constant [m.s-1]": (5, 50),
    "Dead lithium decay constant [s-1]": (5, 100),

    "Negative electrode critical stress [Pa]": (0.8, 1.2),
    # "Negative electrode LAM constant exponential term": (0.9, 1.1),
    "Negative electrode reaction-driven LAM factor [m3.mol-1]": (0.5, 5),
    "Negative electrode LAM constant proportional term [s-1]": (1.0, 2.5),
}

def find_repo_root(start: Path) -> Path | None:
    for parent in [start, *start.parents]:
        if (parent / "pybamm_surrogate").exists() and (parent / "pybamm").exists():
            return parent
    return None


def configure_pybamm_import():
    repo_root = find_repo_root(SCRIPT_DIR)
    env_src = os.environ.get("PYBAMM_SRC")
    env_root = os.environ.get("PYBAMM_ROOT")

    candidates = []
    if env_src:
        candidates.append(Path(env_src).expanduser())
    if env_root:
        env_root_path = Path(env_root).expanduser()
        candidates.append(env_root_path / "src")
        candidates.append(env_root_path)
    if repo_root is not None:
        repo_pybamm_root = repo_root / "pybamm" / "PyBaMM"
        candidates.append(repo_pybamm_root / "src")
        candidates.append(repo_pybamm_root)

    server_default_root = Path(".../PyBaMM")
    candidates.append(server_default_root / "src")
    candidates.append(server_default_root)

    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

def setup_logging(log_level=logging.INFO, log_file=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def _parse_slurm_int(value: str | None) -> int | None:
    if not value:
        return None
    token = value.split("(")[0].split(",")[0].strip()
    try:
        parsed = int(token)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def slurm_default_workers() -> int:
    for env_name in ("SLURM_CPUS_PER_TASK", "SLURM_NTASKS", "SLURM_JOB_CPUS_PER_NODE"):
        parsed = _parse_slurm_int(os.environ.get(env_name))
        if parsed is not None:
            return parsed
    return max(1, os.cpu_count() or 1)


def set_model():
    return pybamm.lithium_ion.DFN(
        {
            "SEI": "solvent-diffusion limited",
            "SEI porosity change": "true",
            "lithium plating": "partially reversible",
            "lithium plating porosity change": "true",
            "particle mechanics": ("swelling and cracking", "swelling only"),
            "SEI on cracks": "true",
            "loss of active material": "stress-driven",
        }
    )


def load_parameter_from_file(parameter_file_path):
    if not os.path.isfile(parameter_file_path):
        raise FileNotFoundError(f"Parameter file not found: {parameter_file_path}")
    with open(parameter_file_path, "r", encoding="utf-8") as handle:
        param_json = json.load(handle)

    parameter = pybamm.ParameterValues("OKane2022")
    parameter.update(param_json)
    return parameter, param_json


def resolve_parameter_file(paramset_name=None, explicit_file=None):
    candidates = []
    if explicit_file:
        candidates.append(explicit_file)
    if paramset_name:
        filename = paramset_name if paramset_name.endswith(".json") else f"{paramset_name}.json"
        candidates.append(str(SCRIPT_DIR / "parameterset_config" / filename))
        candidates.append(str(SCRIPT_DIR / "parameterset" / filename))
        candidates.append(str(Path.cwd() / "parameterset_config" / filename))
        candidates.append(str(Path.cwd() / "parameterset" / filename))
        candidates.append(str(Path.cwd() / filename))

    for candidate in candidates:
        if not candidate:
            continue
        candidate_abs = os.path.abspath(candidate)
        if os.path.isfile(candidate_abs):
            return candidate_abs
    return None


def to_number(value):
    if callable(value):
        raise TypeError(f"value is callable (function); cannot convert to float: {value!r}")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped)
        except Exception as exc:
            raise TypeError(f"Cannot convert string parameter value to float: {value!r}") from exc
    raise TypeError(f"Unsupported parameter value type for numeric conversion: {type(value)} ({value!r})")


def make_lhs_sampler(dim: int, seed: int):
    return qmc.LatinHypercube(d=dim, seed=seed)


def scale_unit_to_bounds(unit_points: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    lows = np.array([item[0] for item in bounds], dtype=float)
    highs = np.array([item[1] for item in bounds], dtype=float)
    return lows + unit_points * (highs - lows)


def generate_multipliers_batch(pm_variation_list, variation_ranges, n: int, sampling: str, rng: np.random.Generator, lhs_sampler=None):
    bounds = [(variation_ranges[name][0], variation_ranges[name][1]) for name in pm_variation_list]
    dim = len(pm_variation_list)

    if sampling == "lhs":
        if lhs_sampler is None:
            raise ValueError("lhs_sampler is required when sampling='lhs'")
        unit_points = lhs_sampler.random(n)
        scaled = scale_unit_to_bounds(unit_points, bounds)
        return [list(map(float, scaled[idx, :])) for idx in range(n)]

    if sampling == "random":
        scaled = np.empty((n, dim), dtype=float)
        for dim_idx, (low, high) in enumerate(bounds):
            scaled[:, dim_idx] = rng.uniform(low, high, size=n)
        return [list(map(float, scaled[idx, :])) for idx in range(n)]

    raise ValueError("sampling must be 'lhs' or 'random'")


def generate_extreme_multipliers(pm_variation_list, variation_ranges):
    upper = [float(variation_ranges[name][1]) for name in pm_variation_list]
    lower = [float(variation_ranges[name][0]) for name in pm_variation_list]
    return [upper, lower]


def jitter_multipliers(pm_variation_list, variation_ranges, multipliers, retry_jitter: float, rng: np.random.Generator):
    jittered = []
    for name, multiplier in zip(pm_variation_list, multipliers):
        low, high = variation_ranges[name]
        new_multiplier = float(multiplier) * float(rng.normal(1.0, retry_jitter))
        jittered.append(float(np.clip(new_multiplier, low, high)))
    return jittered


def extract_cycle_frame(solution, cycle_number):
    if cycle_number >= len(solution.cycles):
        raise IndexError(f"Cycle {cycle_number} is unavailable. The solution has {len(solution.cycles) - 1} usable cycles.")
    cycle = solution.cycles[cycle_number]
    return pd.DataFrame(
        {
            "Voltage": np.asarray(cycle["Voltage [V]"].data, dtype=float),
            "DischargeCapacity": np.asarray(cycle["Discharge capacity [A.h]"].data, dtype=float),
            "Current": np.asarray(cycle["Current [A]"].data, dtype=float),
            "Time": np.asarray(cycle["Time [s]"].data, dtype=float),
        }
    )


def integrate_discharge_capacity(current, time):
    current = np.asarray(current, dtype=float)
    time = np.asarray(time, dtype=float)
    valid = np.isfinite(current) & np.isfinite(time) & (current < 0.0)
    if np.sum(valid) < 2:
        return float("nan")
    return float(np.trapezoid(np.abs(current[valid]), time[valid]) / 3600.0)


def build_soh_series_from_solution(solution):
    capacities = np.asarray(solution.summary_variables["Capacity [A.h]"], dtype=float)
    cycle_numbers = np.asarray(solution.summary_variables["Cycle number"], dtype=int)
    if capacities.size == 0 or capacities.size != cycle_numbers.size:
        raise RuntimeError("PyBaMM summary variables for cycle capacity are unavailable or inconsistent.")

    baseline_capacity = capacities[0]
    if not np.isfinite(baseline_capacity) or baseline_capacity <= 0.0:
        soh = np.full_like(capacities, np.nan, dtype=float)
        return capacities, soh, cycle_numbers

    soh = capacities / baseline_capacity * 100.0
    return capacities, soh, cycle_numbers


def extract_discharge_branch(cycle_df):
    voltage = np.asarray(cycle_df["Voltage"], dtype=float)
    capacity = np.asarray(cycle_df["DischargeCapacity"], dtype=float)
    current = np.asarray(cycle_df["Current"], dtype=float)

    valid = np.isfinite(voltage) & np.isfinite(capacity) & np.isfinite(current) & (current < 0.0)
    if np.sum(valid) < 3:
        return None, None

    voltage = voltage[valid]
    capacity = capacity[valid]
    order = np.argsort(voltage)
    voltage = voltage[order]
    capacity = capacity[order]
    capacity = capacity - float(np.nanmin(capacity))

    unique_voltage, unique_index = np.unique(voltage, return_index=True)
    capacity = capacity[unique_index]
    if unique_voltage.size < 3:
        return None, None

    return unique_voltage.astype(float), capacity.astype(float)


def build_cycle_payload(solution, reference_cycle, target_cycle, voltage_window, n_points):
    ref_df = extract_cycle_frame(solution, reference_cycle)
    tgt_df = extract_cycle_frame(solution, target_cycle)

    ref_voltage, ref_capacity = extract_discharge_branch(ref_df)
    tgt_voltage, tgt_capacity = extract_discharge_branch(tgt_df)
    if ref_voltage is None or tgt_voltage is None:
        return None

    v_lower = max(float(np.nanmin(ref_voltage)), float(np.nanmin(tgt_voltage)), float(voltage_window[0]))
    v_upper = min(float(np.nanmax(ref_voltage)), float(np.nanmax(tgt_voltage)), float(voltage_window[1]))
    if not np.isfinite(v_lower) or not np.isfinite(v_upper) or v_upper <= v_lower:
        return None

    common_voltage = np.linspace(v_lower, v_upper, n_points, dtype=float)
    ref_interp = interp1d(ref_voltage, ref_capacity, kind="linear", bounds_error=False, fill_value=np.nan)(common_voltage)
    tgt_interp = interp1d(tgt_voltage, tgt_capacity, kind="linear", bounds_error=False, fill_value=np.nan)(common_voltage)

    dq_curve = (tgt_interp - ref_interp).astype(float)
    negative_dq = np.where(np.isnan(dq_curve) | (dq_curve < 0.0), dq_curve, 0.0).astype(float)

    return {
        "common_voltage": common_voltage,
        "ref_capacity_interp": ref_interp,
        "tgt_capacity_interp": tgt_interp,
        "dq_curve": dq_curve,
        "negative_dq": negative_dq,
    }


def build_summary_row(simulation_index, result, cycle_detail_rel_path, life_detail_rel_path):
    row = {
        "simulation_index": int(simulation_index),
        "param_file": result["param_file"],
        "cycle_detail_csv": cycle_detail_rel_path,
        "life_detail_csv": life_detail_rel_path,
        "80soh_cycle": result.get("soh_threshold_cycles", {}).get(80),
    }

    for name, value in result["multipliers"].items():
        row[f"{name}__multiplier"] = float(value)
    for name, value in result["varied_params"].items():
        row[f"{name}__value"] = float(value)

    return row


def write_csv_atomic(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def save_cycle_detail_csv(detail_path, payload, reference_cycle, target_cycle):
    detail_df = pd.DataFrame(
        {
            "common_voltage": payload["common_voltage"],
            f"Q_cycle_{reference_cycle}": payload["ref_capacity_interp"],
            f"Q_cycle_{target_cycle}": payload["tgt_capacity_interp"],
            "dQ": payload["dq_curve"],
            "negative_dQ": payload["negative_dq"],
        }
    )
    write_csv_atomic(detail_df, detail_path)


def save_life_detail_csv(detail_path, payload):
    detail_df = pd.DataFrame(
        {
            "cycle_number": payload["cycle_numbers"],
            "capacity_ah": payload["capacities"],
            "soh": payload["soh"],
        }
    )
    write_csv_atomic(detail_df, detail_path)


def default_output_dir(param_file_path: str | Path) -> Path:
    param_stem = Path(param_file_path).stem
    return SCRIPT_DIR / OUTPUT_ROOT_DIRNAME / f"{param_stem}"


def resolve_server_output_dir(explicit_output_dir: str | None, param_file_path: str) -> Path:
    if explicit_output_dir:
        return Path(explicit_output_dir).expanduser().resolve()
    return default_output_dir(param_file_path)


def build_default_log_file(output_dir: Path, explicit_log_file: str | None) -> str:
    if explicit_log_file:
        return str(Path(explicit_log_file).expanduser().resolve())
    job_id = os.environ.get("SLURM_JOB_ID")
    suffix = f"_job{job_id}" if job_id else ""
    return str((output_dir / f"slurm_generation{suffix}.log").resolve())


def build_run_metadata(param_file_path, output_dir, target_successes, experiment_cycles, sampling, seed, workers, batch_size):
    return {
        "parameter_file": str(param_file_path),
        "pm_variation_list": list(PM_VARIATION_LIST),
        "variation_ranges": {name: [float(low), float(high)] for name, (low, high) in VARIATION_RANGES.items()},
        "target_successes": int(target_successes),
        "experiment_cycles": int(experiment_cycles),
        "dq_reference_cycle": int(DQ_REFERENCE_CYCLE),
        "dq_target_cycle": int(DQ_TARGET_CYCLE),
        "n_dq_points": int(N_DQ_POINTS),
        "voltage_window": [float(VOLTAGE_WINDOW[0]), float(VOLTAGE_WINDOW[1])],
        "soh_reference": "PyBaMM summary variable Capacity [A.h] normalized by first cycle",
        "soh_thresholds": [int(value) for value in SOH_THRESHOLDS],
        "sampling": sampling,
        "seed": int(seed),
        "workers": int(workers),
        "batch_size": int(batch_size),
        "lhs_base_points_consumed": 0,
        "summary_csv": str(output_dir / SUMMARY_FILENAME),
        "cycle_detail_dir": str(output_dir / CYCLE_DETAIL_DIRNAME),
        "life_detail_dir": str(output_dir / LIFE_DETAIL_DIRNAME),
    }


def validate_resume_metadata(metadata_path: Path, expected_metadata: dict):
    if not metadata_path.exists():
        return
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            existing_metadata = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Failed to read existing metadata file for resume: {metadata_path}") from exc

    keys_to_validate = [
        "parameter_file",
        "pm_variation_list",
        "variation_ranges",
        "experiment_cycles",
        "dq_reference_cycle",
        "dq_target_cycle",
        "n_dq_points",
        "voltage_window",
        "soh_reference",
        "soh_thresholds",
    ]
    mismatches = []
    for key in keys_to_validate:
        if existing_metadata.get(key) != expected_metadata.get(key):
            mismatches.append(key)
    if mismatches:
        mismatch_text = ", ".join(mismatches)
        raise RuntimeError(
            f"Resume metadata mismatch for {mismatch_text}. Start a fresh output directory or remove --resume."
        )


def persist_run_metadata(metadata_path: Path, metadata: dict):
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = metadata_path.with_name(f"{metadata_path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    os.replace(tmp_path, metadata_path)


def read_existing_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_existing_summary(summary_path: Path):
    if not summary_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(summary_path)
    except Exception:
        logging.warning("Failed to read existing summary CSV at %s; starting from an empty table", summary_path)
        return pd.DataFrame()


def is_completed_row(row, output_dir: Path):
    if pd.isna(row.get("simulation_index")):
        return False
    cycle_detail_rel_path = row.get("cycle_detail_csv") or row.get("detail_csv")
    life_detail_rel_path = row.get("life_detail_csv")
    if pd.isna(cycle_detail_rel_path) or pd.isna(life_detail_rel_path):
        return False
    cycle_detail_path = output_dir / str(cycle_detail_rel_path)
    life_detail_path = output_dir / str(life_detail_rel_path)
    return cycle_detail_path.exists() and life_detail_path.exists()


def flush_summary_rows(summary_rows, summary_path: Path):
    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        write_csv_atomic(summary_df, summary_path)
        return summary_df
    summary_df = summary_df.sort_values("simulation_index").reset_index(drop=True)
    write_csv_atomic(summary_df, summary_path)
    return summary_df


def persist_successful_result(output_dir: Path, summary_rows, summary_path: Path, result: dict):
    simulation_index = int(result["simulation_index"])
    cycle_detail_rel_path = f"{CYCLE_DETAIL_DIRNAME}/sim_{simulation_index:06d}.csv"
    life_detail_rel_path = f"{LIFE_DETAIL_DIRNAME}/sim_{simulation_index:06d}.csv"
    save_cycle_detail_csv(output_dir / cycle_detail_rel_path, result["cycle_detail_payload"], DQ_REFERENCE_CYCLE, DQ_TARGET_CYCLE)
    save_life_detail_csv(output_dir / life_detail_rel_path, result["life_detail_payload"])
    summary_rows.append(build_summary_row(simulation_index, result, cycle_detail_rel_path, life_detail_rel_path))
    flush_summary_rows(summary_rows, summary_path)


def clear_result_payloads(result: dict | None):
    if not isinstance(result, dict):
        return
    result["cycle_detail_payload"] = None
    result["life_detail_payload"] = None


def submit_single_simulation(
    executor,
    simulation_index,
    param_file_path,
    pm_variation_list,
    experiment_cycles,
    multipliers,
    soh_threshold,
    retry_attempt=0,
    original_multipliers=None,
):
    return executor.submit(
        run_single_simulation,
        simulation_index,
        param_file_path,
        pm_variation_list,
        experiment_cycles,
        multipliers,
        soh_threshold,
        retry_attempt,
        original_multipliers,
    )


def prevalidate_parameter_file(param_file_path: str):
    try:
        _, param_json = load_parameter_from_file(param_file_path)
    except Exception:
        logging.exception("Failed to load parameter JSON during pre-validation")
        sys.exit(1)

    for name in PM_VARIATION_LIST:
        if name not in param_json:
            logging.error("Parameter '%s' not present in parameter JSON (%s).", name, param_file_path)
            sys.exit(1)
        _ = to_number(param_json[name])


def extract_soh_threshold_cycles(soh_series, thresholds=SOH_THRESHOLDS):
    soh_arr = np.asarray(soh_series, dtype=float) if soh_series is not None else np.array([], dtype=float)
    threshold_cycles = {}
    for threshold in thresholds:
        idxs = np.where(soh_arr <= float(threshold))[0] if soh_arr.size else np.array([])
        threshold_cycles[int(threshold)] = int(idxs[0] + 1) if idxs.size else None
    return threshold_cycles


def run_single_simulation(
    simulation_index,
    param_file_path,
    pm_variation_list,
    experiment_cycles,
    multipliers,
    soh_threshold=80.0,
    retry_attempt=0,
    original_multipliers=None,
):
    parameter = None
    param_json = None
    model = None
    param_copy = None
    experiment = None
    solver = None
    sim = None
    solution = None
    capacities = None
    soh_series = None
    cycle_numbers = None
    cycle_detail_payload = None
    result = {
        "simulation_index": int(simulation_index),
        "success": False,
        "error": None,
        "param_file": param_file_path,
        "multipliers": {},
        "original_multipliers": {},
        "retry_attempt": int(retry_attempt),
        "varied_params": {},
        "cycle_detail_payload": None,
        "life_detail_payload": None,
        "soh_threshold_cycles": {},
    }

    try:
        if original_multipliers is None:
            original_multipliers = multipliers

        for name, multiplier in zip(pm_variation_list, original_multipliers):
            result["original_multipliers"][name] = float(multiplier)

        parameter, param_json = load_parameter_from_file(param_file_path)
        model = set_model()
        param_copy = parameter.copy()

        for name, multiplier in zip(pm_variation_list, multipliers):
            base_raw = param_json.get(name, None)
            if base_raw is None:
                try:
                    base_raw = parameter[name]
                except Exception:
                    base_raw = None
            if base_raw is None:
                raise KeyError(f"Base parameter '{name}' not found in parameter JSON or ParameterValues.")

            base_value = to_number(base_raw)
            varied_value = base_value * float(multiplier)
            param_copy.update({name: varied_value})
            result["multipliers"][name] = float(multiplier)
            result["varied_params"][name] = float(varied_value)

        experiment = pybamm.Experiment(
            [
                (
                    "Charge at 1C until 4.2V",
                    "Hold at 4.2 V until C/20",
                    "Discharge at 1C until 2.5V",
                )
            ]
            * experiment_cycles,
            termination="79% capacity",
            temperature="25 oC",
        )
        solver = pybamm.IDAKLUSolver()
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param_copy, solver=solver)
        solution = sim.solve(t_eval=None)

        capacities, soh_series, cycle_numbers = build_soh_series_from_solution(solution)
        cycle_detail_payload = build_cycle_payload(
            solution=solution,
            reference_cycle=DQ_REFERENCE_CYCLE,
            target_cycle=DQ_TARGET_CYCLE,
            voltage_window=VOLTAGE_WINDOW,
            n_points=N_DQ_POINTS,
        )
        if cycle_detail_payload is None:
            raise RuntimeError(f"Failed to build cycle-{DQ_REFERENCE_CYCLE} to cycle-{DQ_TARGET_CYCLE} payload for simulation {simulation_index}.")

        result["cycle_detail_payload"] = cycle_detail_payload
        result["life_detail_payload"] = {
            "cycle_numbers": cycle_numbers.astype(int),
            "capacities": capacities.astype(float),
            "soh": soh_series.astype(float),
        }
        threshold_cycles = extract_soh_threshold_cycles(soh_series)
        result["soh_threshold_cycles"] = threshold_cycles
        result["success"] = threshold_cycles.get(int(soh_threshold)) is not None
        return result

    except Exception as exc:
        tb = traceback.format_exc()
        logging.exception("Simulation failed in worker for simulation_index=%s", simulation_index)
        result["error"] = f"{str(exc)}\n{tb}"
        result["success"] = False
        return result
    finally:
        del cycle_detail_payload
        del cycle_numbers
        del soh_series
        del capacities
        del solution
        del sim
        del solver
        del experiment
        del param_copy
        del model
        del param_json
        del parameter
        gc.collect()


def run_until_target(
    param_file_path,
    pm_variation_list,
    variation_ranges,
    experiment_cycles,
    target_successes=200,
    workers=16,
    batch_size=16,
    soh_threshold=80.0,
    resume=False,
    sampling="lhs",
    seed=123,
    max_retries=0,
    retry_jitter=0.02,
    retry_seed_offset=100000,
    output_dir=None,
):
    if not os.path.isfile(param_file_path):
        logging.error("Parameter file disappeared before running: %s", param_file_path)
        raise FileNotFoundError(param_file_path)

    if batch_size <= 0:
        batch_size = workers

    output_dir = Path(output_dir) if output_dir else default_output_dir(param_file_path)
    cycle_detail_dir = output_dir / CYCLE_DETAIL_DIRNAME
    life_detail_dir = output_dir / LIFE_DETAIL_DIRNAME
    summary_path = output_dir / SUMMARY_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    output_dir.mkdir(parents=True, exist_ok=True)
    cycle_detail_dir.mkdir(parents=True, exist_ok=True)
    life_detail_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_run_metadata(
        param_file_path=param_file_path,
        output_dir=output_dir,
        target_successes=target_successes,
        experiment_cycles=experiment_cycles,
        sampling=sampling,
        seed=seed,
        workers=workers,
        batch_size=batch_size,
    )
    if resume:
        validate_resume_metadata(metadata_path, metadata)
    existing_metadata = read_existing_metadata(metadata_path) if resume else {}

    existing_df = load_existing_summary(summary_path) if resume else pd.DataFrame()
    summary_rows = []
    completed_indices = set()
    next_simulation_index = 0
    if not existing_df.empty:
        for _, existing_row in existing_df.iterrows():
            if is_completed_row(existing_row, output_dir):
                summary_rows.append(existing_row.to_dict())
                completed_indices.add(int(existing_row["simulation_index"]))
        if completed_indices:
            next_simulation_index = max(completed_indices) + 1

    success_count = len(completed_indices)
    total_tried = 0
    failed_count = 0
    lhs_base_points_consumed = int(existing_metadata.get("lhs_base_points_consumed", 0)) if sampling == "lhs" else 0
    metadata["lhs_base_points_consumed"] = lhs_base_points_consumed
    persist_run_metadata(metadata_path, metadata)

    rng = np.random.default_rng(seed=seed)
    retry_rng = np.random.default_rng(seed=seed + retry_seed_offset)
    lhs_sampler = make_lhs_sampler(dim=len(pm_variation_list), seed=seed) if sampling == "lhs" else None
    if lhs_sampler is not None and lhs_base_points_consumed > 0:
        logging.info("Advancing LHS sampler by %s previously assigned base points", lhs_base_points_consumed)
        lhs_sampler.random(lhs_base_points_consumed)

    logging.info(
        "Starting run: target_successes=%s, workers=%s, batch_size=%s, existing_successes=%s, sampling=%s, seed=%s, lhs_base_points_consumed=%s, output_dir=%s",
        target_successes,
        workers,
        batch_size,
        success_count,
        sampling,
        seed,
        lhs_base_points_consumed,
        output_dir,
    )

    def submit_batch(executor, multipliers_batch, start_index):
        future_map = {}
        for offset, multipliers in enumerate(multipliers_batch):
            simulation_index = start_index + offset
            future = submit_single_simulation(
                executor,
                simulation_index,
                param_file_path,
                pm_variation_list,
                experiment_cycles,
                multipliers,
                soh_threshold,
            )
            future_map[future] = {
                "simulation_index": simulation_index,
                "multipliers": list(multipliers),
            }
        return future_map

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        if success_count == 0:
            logging.info("Running deterministic extreme points first: all-max and all-min multipliers")
            future_map = submit_batch(
                executor,
                generate_extreme_multipliers(pm_variation_list, variation_ranges),
                next_simulation_index,
            )
            next_simulation_index += len(future_map)

            for future in concurrent.futures.as_completed(future_map):
                future_info = future_map.pop(future)
                total_tried += 1
                try:
                    result = future.result()
                except Exception as exc:
                    logging.exception("Worker future raised unhandled exception for extreme point")
                    result = {
                        "simulation_index": int(future_info["simulation_index"]),
                        "success": False,
                        "error": f"Unhandled worker exception: {exc}",
                        "multipliers": {
                            name: float(multiplier)
                            for name, multiplier in zip(pm_variation_list, future_info["multipliers"])
                        },
                        "original_multipliers": {
                            name: float(multiplier)
                            for name, multiplier in zip(pm_variation_list, future_info["multipliers"])
                        },
                    }

                original_multipliers = [
                    result.get("original_multipliers", {}).get(name, result.get("multipliers", {}).get(name))
                    for name in pm_variation_list
                ]
                current_multipliers = [
                    result.get("multipliers", {}).get(name)
                    for name in pm_variation_list
                ]
                if all(value is not None for value in original_multipliers) and all(value is not None for value in current_multipliers):
                    retry_attempt = 0
                    while (not result.get("success")) and retry_attempt < max_retries:
                        retry_attempt += 1
                        total_tried += 1
                        retry_multipliers = jitter_multipliers(
                            pm_variation_list=pm_variation_list,
                            variation_ranges=variation_ranges,
                            multipliers=current_multipliers,
                            retry_jitter=retry_jitter,
                            rng=retry_rng,
                        )
                        logging.info(
                            "Retrying failed extreme simulation_index=%s attempt=%s/%s",
                            result.get("simulation_index"),
                            retry_attempt,
                            max_retries,
                        )
                        retry_future = submit_single_simulation(
                            executor,
                            result["simulation_index"],
                            param_file_path,
                            pm_variation_list,
                            experiment_cycles,
                            retry_multipliers,
                            soh_threshold,
                            retry_attempt,
                            original_multipliers,
                        )
                        result = retry_future.result()
                        current_multipliers = retry_multipliers

                if result.get("success"):
                    persist_successful_result(output_dir, summary_rows, summary_path, result)
                    success_count += 1
                    logging.info("SUCCESS saved (extreme): simulation_index=%s total_success=%s", result["simulation_index"], success_count)
                else:
                    failed_count += 1
                    logging.warning("FAILED extreme simulation_index=%s error=%s", result.get("simulation_index"), result.get("error"))
                clear_result_payloads(result)
                gc.collect()

        while success_count < target_successes:
            sims_to_submit = min(batch_size, max(1, (target_successes - success_count) * 2))
            multipliers_batch = generate_multipliers_batch(
                pm_variation_list=pm_variation_list,
                variation_ranges=variation_ranges,
                n=sims_to_submit,
                sampling=sampling,
                rng=rng,
                lhs_sampler=lhs_sampler,
            )
            future_map = submit_batch(executor, multipliers_batch, next_simulation_index)
            next_simulation_index += len(future_map)
            if sampling == "lhs":
                lhs_base_points_consumed += len(future_map)
                metadata["lhs_base_points_consumed"] = lhs_base_points_consumed
                persist_run_metadata(metadata_path, metadata)

            for future in concurrent.futures.as_completed(future_map):
                future_info = future_map.pop(future)
                total_tried += 1
                try:
                    result = future.result()
                except Exception as exc:
                    logging.exception("Worker future raised unhandled exception")
                    result = {
                        "simulation_index": int(future_info["simulation_index"]),
                        "success": False,
                        "error": f"Unhandled worker exception: {exc}",
                        "multipliers": {
                            name: float(multiplier)
                            for name, multiplier in zip(pm_variation_list, future_info["multipliers"])
                        },
                        "original_multipliers": {
                            name: float(multiplier)
                            for name, multiplier in zip(pm_variation_list, future_info["multipliers"])
                        },
                    }

                original_multipliers = [
                    result.get("original_multipliers", {}).get(name, result.get("multipliers", {}).get(name))
                    for name in pm_variation_list
                ]
                current_multipliers = [
                    result.get("multipliers", {}).get(name)
                    for name in pm_variation_list
                ]
                if all(value is not None for value in original_multipliers) and all(value is not None for value in current_multipliers):
                    retry_attempt = 0
                    while (not result.get("success")) and retry_attempt < max_retries:
                        retry_attempt += 1
                        total_tried += 1
                        retry_multipliers = jitter_multipliers(
                            pm_variation_list=pm_variation_list,
                            variation_ranges=variation_ranges,
                            multipliers=current_multipliers,
                            retry_jitter=retry_jitter,
                            rng=retry_rng,
                        )
                        logging.info(
                            "Retrying failed simulation_index=%s attempt=%s/%s",
                            result.get("simulation_index"),
                            retry_attempt,
                            max_retries,
                        )
                        retry_future = submit_single_simulation(
                            executor,
                            result["simulation_index"],
                            param_file_path,
                            pm_variation_list,
                            experiment_cycles,
                            retry_multipliers,
                            soh_threshold,
                            retry_attempt,
                            original_multipliers,
                        )
                        result = retry_future.result()
                        current_multipliers = retry_multipliers

                if result.get("success"):
                    persist_successful_result(output_dir, summary_rows, summary_path, result)
                    success_count += 1
                    logging.info("SUCCESS saved: simulation_index=%s total_success=%s", result["simulation_index"], success_count)
                else:
                    failed_count += 1
                    logging.warning("FAILED simulation_index=%s error=%s", result.get("simulation_index"), result.get("error"))
                clear_result_payloads(result)
                gc.collect()

                if success_count >= target_successes:
                    break

    summary_df = flush_summary_rows(summary_rows, summary_path)
    logging.info("Finished run: total_tried=%s total_success=%s total_failed=%s", total_tried, success_count, failed_count)
    return summary_df


def main():

    parser = argparse.ArgumentParser(
        description="Run PyBaMM data generation on a Slurm server with all settings embedded in this single file."
    )
    parser.add_argument("--paramset", type=str, default="set2", help="Parameter set name (JSON file stem)")
    parser.add_argument("--param-file", type=str, default=None, help="Full path to parameter JSON file")
    parser.add_argument("--target-success", "--target_success", dest="target_success", type=int, default=100, help="Number of successful simulations to collect")
    parser.add_argument("--workers", type=int, default=16, help="Number of worker processes; defaults to Slurm CPU allocation")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=16, help="Number of simulations submitted per batch; defaults to workers")
    parser.add_argument("--soh-threshold", "--soh_threshold", dest="soh_threshold", type=float, default=0.80, help="SOH threshold (e.g. 0.8 means 80%%) required for a simulation to count as successful")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing summary/cycle-detail/life-detail output directory")
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file path")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--experiment-cycles", type=int, default=1500, help="Number of experiment cycles")
    parser.add_argument("--sampling", type=str, default="lhs", choices=["lhs", "random"], help="Sampling method for parameter multipliers")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for sampling")
    parser.add_argument("--max-retries", "--max_retries", dest="max_retries", type=int, default=5, help="Retry each failed parameter set up to this many times with local multiplier jitter")
    parser.add_argument("--retry-jitter", "--retry_jitter", dest="retry_jitter", type=float, default=0.02, help="Relative Gaussian jitter applied to failed multipliers during retries, e.g. 0.02 means about 2%%")
    parser.add_argument("--retry-seed-offset", "--retry_seed_offset", dest="retry_seed_offset", type=int, default=100000, help="Offset added to --seed for retry jitter RNG")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional explicit output directory")
    args = parser.parse_args()

    param_file_path = resolve_parameter_file(paramset_name=args.paramset, explicit_file=args.param_file)
    if not param_file_path:
        print(
            f"Cannot resolve parameter JSON file for paramset={args.paramset}. Use --param-file or provide a valid parameter set.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.path.isfile(param_file_path):
        print(f"Cannot find parameter JSON file: {param_file_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = resolve_server_output_dir(args.output_dir, param_file_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = build_default_log_file(output_dir, args.log_file)
    setup_logging(log_level=getattr(logging, args.log_level.upper(), logging.INFO), log_file=log_file)
    logging.info("PyBaMM version: %s", pybamm.__version__)
    logging.info("PyBaMM path %s:", os.path.dirname(pybamm.__file__))

    workers = args.workers if args.workers is not None else slurm_default_workers()
    batch_size = args.batch_size if args.batch_size is not None else workers

    logging.info("Using parameter file: %s", param_file_path)
    logging.info("Using output directory: %s", output_dir)
    logging.info(
        "Slurm context: job_id=%s cpus_per_task=%s ntasks=%s job_cpus_per_node=%s",
        os.environ.get("SLURM_JOB_ID"),
        os.environ.get("SLURM_CPUS_PER_TASK"),
        os.environ.get("SLURM_NTASKS"),
        os.environ.get("SLURM_JOB_CPUS_PER_NODE"),
    )
    logging.info("Parallel settings: workers=%s batch_size=%s", workers, batch_size)

    prevalidate_parameter_file(param_file_path)
    pybamm.set_logging_level("NOTICE")

    run_until_target(
        param_file_path=param_file_path,
        pm_variation_list=PM_VARIATION_LIST,
        variation_ranges=VARIATION_RANGES,
        experiment_cycles=args.experiment_cycles,
        target_successes=args.target_success,
        workers=workers,
        batch_size=batch_size,
        soh_threshold=args.soh_threshold * 100.0,
        resume=args.resume,
        sampling=args.sampling,
        seed=args.seed,
        max_retries=args.max_retries,
        retry_jitter=args.retry_jitter,
        retry_seed_offset=args.retry_seed_offset,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
