"""
io.py

Input/output utilities for the non-thermal nucleosynthesis network.

This module is responsible for reading simulation inputs from JSON files,
validating and sanitizing user-provided data, converting those inputs into
internal objects such as NetworkState, and writing simulation outputs back
to disk.

This module does not perform any physics calculations. It does
not evolve abundances, compute reaction rates, calculate stopping powers, or
choose timesteps. Its role is limited to file-based input/output and data
preparation for the solver.

Expected responsibilities:
    - Load a run configuration from JSON
    - Validate required fields and basic value ranges
    - Construct the initial NetworkState
    - Write final simulation outputs to JSON
    - Write optional abundance histories to CSV/JSON

Typical usage:
    config = load_run_config("inputs/run.json")
    state = build_initial_state(config)
    ...
    write_final_state("outputs/final_state.json", state, metadata)
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

# Adjust this import to match your actual project structure.
# Example alternatives:
# from state import NetworkState
# from .state import NetworkState
from state import NetworkState


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------

class InputValidationError(ValueError):
    """Raised when the user input file is missing required fields or contains invalid values."""


# ---------------------------------------------------------------------
# JSON loading / saving helpers
# ---------------------------------------------------------------------

def load_json(path: str | Path) -> Dict[str, Any]:
    """
    Load a JSON file and return its contents as a dictionary.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON contents.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    TypeError
        If the top-level JSON object is not a dictionary.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Top-level JSON object in '{path}' must be a dictionary.")

    return data


def write_json(path: str | Path, data: Mapping[str, Any], indent: int = 2) -> None:
    """
    Write a dictionary to a JSON file.

    Parameters
    ----------
    path : str or Path
        Output file path.
    data : mapping
        Data to serialize.
    indent : int, optional
        JSON indentation level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=indent, sort_keys=False)


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def _require_keys(mapping: Mapping[str, Any], required_keys: Iterable[str], context: str) -> None:
    """Raise InputValidationError if any required keys are missing."""
    missing = [key for key in required_keys if key not in mapping]
    if missing:
        raise InputValidationError(
            f"Missing required key(s) in {context}: {', '.join(missing)}"
        )


def _ensure_mapping(obj: Any, name: str) -> Mapping[str, Any]:
    """Ensure an object is a dictionary-like mapping."""
    if not isinstance(obj, Mapping):
        raise InputValidationError(f"'{name}' must be a JSON object/dictionary.")
    return obj


def _ensure_positive_number(value: Any, name: str) -> float:
    """Ensure a value is a positive int/float and return it as float."""
    if not isinstance(value, (int, float)):
        raise InputValidationError(f"'{name}' must be a number.")
    value = float(value)
    if value <= 0.0:
        raise InputValidationError(f"'{name}' must be > 0.")
    return value


def _ensure_nonnegative_number(value: Any, name: str) -> float:
    """Ensure a value is a nonnegative int/float and return it as float."""
    if not isinstance(value, (int, float)):
        raise InputValidationError(f"'{name}' must be a number.")
    value = float(value)
    if value < 0.0:
        raise InputValidationError(f"'{name}' must be >= 0.")
    return value


def _ensure_string(value: Any, name: str) -> str:
    """Ensure a value is a string."""
    if not isinstance(value, str):
        raise InputValidationError(f"'{name}' must be a string.")
    return value


def _validate_abundance_mapping(abundances: Mapping[str, Any], name: str = "abundances") -> Dict[str, float]:
    """
    Validate a species->abundance dictionary.

    Notes
    -----
    This function only checks basic structure and non-negativity.
    It does not enforce a specific normalization convention, because
    that may depend on whether you are using number fractions, mass
    fractions, or abundances per baryon.
    """
    abundances = _ensure_mapping(abundances, name)

    clean: Dict[str, float] = {}
    for species, value in abundances.items():
        if not isinstance(species, str):
            raise InputValidationError(f"All keys in '{name}' must be strings.")
        clean[species] = _ensure_nonnegative_number(value, f"{name}['{species}']")

    if len(clean) == 0:
        raise InputValidationError(f"'{name}' cannot be empty.")

    return clean


def _validate_species_list(species_list: Any, name: str) -> list[str]:
    """Validate a list of species names."""
    if not isinstance(species_list, list):
        raise InputValidationError(f"'{name}' must be a list of strings.")
    clean = []
    for i, item in enumerate(species_list):
        if not isinstance(item, str):
            raise InputValidationError(f"'{name}[{i}]' must be a string.")
        clean.append(item)
    return clean


def validate_run_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize the full run configuration.

    Expected top-level keys
    -----------------------
    physics :
        Cloud/environment and injection parameters.
    state :
        Initial abundances and optional initial time.
    numerics :
        Runtime controls such as end time and energy grid parameters.
    output :
        Optional output controls.

    Returns
    -------
    dict
        Sanitized configuration dictionary.

    Raises
    ------
    InputValidationError
        If the configuration is incomplete or invalid.
    """
    config = _ensure_mapping(config, "config")

    _require_keys(config, ["physics", "state", "numerics"], "top-level config")

    physics = _ensure_mapping(config["physics"], "physics")
    state = _ensure_mapping(config["state"], "state")
    numerics = _ensure_mapping(config["numerics"], "numerics")
    output = _ensure_mapping(config.get("output", {}), "output")

    # -------------------------
    # Validate state section
    # -------------------------
    _require_keys(state, ["abundances"], "state")
    clean_state: Dict[str, Any] = {
        "t0_s": _ensure_nonnegative_number(state.get("t0_s", 0.0), "state.t0_s"),
        "abundances": _validate_abundance_mapping(state["abundances"], "state.abundances"),
    }

    # Optional additional state fields
    if "temperature_K" in state:
        clean_state["temperature_K"] = _ensure_positive_number(state["temperature_K"], "state.temperature_K")

    # -------------------------
    # Validate numerics section
    # -------------------------
    _require_keys(numerics, ["t_end_s"], "numerics")
    clean_numerics: Dict[str, Any] = {
        "t_end_s": _ensure_positive_number(numerics["t_end_s"], "numerics.t_end_s"),
    }

    # Optional energy-grid info
    if "energy_grid" in numerics:
        energy_grid = _ensure_mapping(numerics["energy_grid"], "numerics.energy_grid")
        _require_keys(energy_grid, ["E_min_MeV", "E_max_MeV", "n_bins"], "numerics.energy_grid")

        E_min = _ensure_positive_number(energy_grid["E_min_MeV"], "numerics.energy_grid.E_min_MeV")
        E_max = _ensure_positive_number(energy_grid["E_max_MeV"], "numerics.energy_grid.E_max_MeV")

        n_bins = energy_grid["n_bins"]
        if not isinstance(n_bins, int) or n_bins < 2:
            raise InputValidationError("'numerics.energy_grid.n_bins' must be an integer >= 2.")

        if E_max <= E_min:
            raise InputValidationError("'numerics.energy_grid.E_max_MeV' must be greater than 'E_min_MeV'.")

        clean_numerics["energy_grid"] = {
            "E_min_MeV": E_min,
            "E_max_MeV": E_max,
            "n_bins": n_bins,
            "spacing": _ensure_string(energy_grid.get("spacing", "log"), "numerics.energy_grid.spacing"),
        }

    # Optional history controls
    clean_numerics["store_history"] = bool(numerics.get("store_history", True))

    # -------------------------
    # Validate physics section
    # -------------------------
    clean_physics: Dict[str, Any] = {}

    # Optional cloud composition
    if "cloud_abundances" in physics:
        clean_physics["cloud_abundances"] = _validate_abundance_mapping(
            physics["cloud_abundances"],
            "physics.cloud_abundances"
        )

    if "ionization_fraction" in physics:
        x_ion = _ensure_nonnegative_number(physics["ionization_fraction"], "physics.ionization_fraction")
        if x_ion > 1.0:
            raise InputValidationError("'physics.ionization_fraction' must be between 0 and 1.")
        clean_physics["ionization_fraction"] = x_ion

    if "electron_density_cm3" in physics:
        clean_physics["electron_density_cm3"] = _ensure_positive_number(
            physics["electron_density_cm3"],
            "physics.electron_density_cm3"
        )

    if "electron_temperature_K" in physics:
        clean_physics["electron_temperature_K"] = _ensure_positive_number(
            physics["electron_temperature_K"],
            "physics.electron_temperature_K"
        )

    # Optional injected / non-thermal particle settings
    if "injection" in physics:
        injection = _ensure_mapping(physics["injection"], "physics.injection")
        _require_keys(injection, ["species"], "physics.injection")

        clean_injection: Dict[str, Any] = {
            "species": _ensure_string(injection["species"], "physics.injection.species"),
        }

        if "abundance" in injection:
            clean_injection["abundance"] = _ensure_nonnegative_number(
                injection["abundance"],
                "physics.injection.abundance"
            )

        if "energy_grid_MeV" in injection:
            eg = injection["energy_grid_MeV"]
            if not isinstance(eg, list) or len(eg) == 0:
                raise InputValidationError("'physics.injection.energy_grid_MeV' must be a non-empty list.")
            clean_injection["energy_grid_MeV"] = [float(x) for x in eg]

        if "spectrum" in injection:
            spec = injection["spectrum"]
            if not isinstance(spec, list) or len(spec) == 0:
                raise InputValidationError("'physics.injection.spectrum' must be a non-empty list.")
            clean_injection["spectrum"] = [float(x) for x in spec]

            if "energy_grid_MeV" in clean_injection:
                if len(clean_injection["spectrum"]) != len(clean_injection["energy_grid_MeV"]):
                    raise InputValidationError(
                        "Lengths of 'physics.injection.spectrum' and "
                        "'physics.injection.energy_grid_MeV' must match."
                    )

        clean_physics["injection"] = clean_injection

    # -------------------------
    # Validate output section
    # -------------------------
    clean_output: Dict[str, Any] = {
        "output_dir": str(output.get("output_dir", "outputs")),
        "write_final_json": bool(output.get("write_final_json", True)),
        "write_history_csv": bool(output.get("write_history_csv", True)),
        "write_history_json": bool(output.get("write_history_json", False)),
    }

    return {
        "physics": clean_physics,
        "state": clean_state,
        "numerics": clean_numerics,
        "output": clean_output,
    }


# ---------------------------------------------------------------------
# Energy grid construction
# ---------------------------------------------------------------------

def build_energy_grid(energy_grid_config: Mapping[str, Any]) -> np.ndarray:
    """
    Construct an energy grid from the validated numerics.energy_grid config.

    Parameters
    ----------
    energy_grid_config : mapping
        Validated energy grid config with keys:
        E_min_MeV, E_max_MeV, n_bins, spacing

    Returns
    -------
    np.ndarray
        1D array of energy bin values in MeV.
    """
    E_min = float(energy_grid_config["E_min_MeV"])
    E_max = float(energy_grid_config["E_max_MeV"])
    n_bins = int(energy_grid_config["n_bins"])
    spacing = energy_grid_config.get("spacing", "log").lower()

    if spacing == "log":
        return np.logspace(np.log10(E_min), np.log10(E_max), n_bins)
    if spacing == "linear":
        return np.linspace(E_min, E_max, n_bins)

    raise InputValidationError(
        f"Unsupported energy-grid spacing '{spacing}'. Use 'log' or 'linear'."
    )


# ---------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------

def build_initial_state(config: Mapping[str, Any]) -> NetworkState:
    """
    Construct the initial NetworkState from a validated configuration dictionary.

    Parameters
    ----------
    config : mapping
        Validated run configuration.

    Returns
    -------
    NetworkState
        Initialized network state.

    Notes
    -----
    You may need to adjust this function depending on the exact fields expected
    by your NetworkState constructor.
    """
    # Build optional energy grid
    energy_grid = None
    if "energy_grid" in config["numerics"]:
        energy_grid = build_energy_grid(config["numerics"]["energy_grid"])

    state_kwargs = {
        "t_s": config["state"]["t0_s"],
        "abundances": dict(config["state"]["abundances"]),
        "t_end_s": config["numerics"]["t_end_s"],
        "energy_grid_MeV": energy_grid,
    }

    # Optional state/environment fields passed through if your NetworkState supports them
    if "temperature_K" in config["state"]:
        state_kwargs["temperature_K"] = config["state"]["temperature_K"]

    if "cloud_abundances" in config["physics"]:
        state_kwargs["cloud_abundances"] = dict(config["physics"]["cloud_abundances"])

    if "ionization_fraction" in config["physics"]:
        state_kwargs["ionization_fraction"] = config["physics"]["ionization_fraction"]

    if "electron_density_cm3" in config["physics"]:
        state_kwargs["electron_density_cm3"] = config["physics"]["electron_density_cm3"]

    if "electron_temperature_K" in config["physics"]:
        state_kwargs["electron_temperature_K"] = config["physics"]["electron_temperature_K"]

    if "injection" in config["physics"]:
        state_kwargs["injection"] = dict(config["physics"]["injection"])

    # IMPORTANT:
    # Adjust this constructor call to match your actual NetworkState signature.
    return NetworkState(**state_kwargs)


def load_run_config(path: str | Path) -> Dict[str, Any]:
    """
    Load and validate a run configuration JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the input JSON file.

    Returns
    -------
    dict
        Validated and sanitized configuration dictionary.
    """
    raw = load_json(path)
    return validate_run_config(raw)


def load_initial_state(path: str | Path) -> NetworkState:
    """
    Load a JSON run configuration and construct the initial NetworkState.

    Parameters
    ----------
    path : str or Path
        Path to the input JSON file.

    Returns
    -------
    NetworkState
        Initialized state object.
    """
    config = load_run_config(path)
    return build_initial_state(config)


# ---------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------

def state_to_dict(state: NetworkState) -> Dict[str, Any]:
    """
    Convert a NetworkState object into a plain dictionary for output.

    Parameters
    ----------
    state : NetworkState
        State object to serialize.

    Returns
    -------
    dict
        JSON-friendly representation of the state.
    """
    if is_dataclass(state):
        data = asdict(state)
    elif hasattr(state, "__dict__"):
        data = dict(state.__dict__)
    else:
        raise TypeError("Unable to serialize NetworkState; unsupported object type.")

    return _to_jsonable(data)


def _to_jsonable(obj: Any) -> Any:
    """
    Recursively convert objects into JSON-serializable forms.
    """
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# ---------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------

def write_final_state(
    path: str | Path,
    state: NetworkState,
    metadata: Optional[Mapping[str, Any]] = None
) -> None:
    """
    Write the final state and optional metadata to a JSON file.

    Parameters
    ----------
    path : str or Path
        Output JSON file path.
    state : NetworkState
        Final evolved state.
    metadata : mapping, optional
        Additional run metadata such as stopping reason, number of steps,
        min/max timestep, etc.
    """
    payload = {
        "final_state": state_to_dict(state),
        "metadata": _to_jsonable(dict(metadata)) if metadata is not None else {},
    }
    write_json(path, payload)


def write_history_json(path: str | Path, history: Mapping[str, Any]) -> None:
    """
    Write simulation history to a JSON file.

    Parameters
    ----------
    path : str or Path
        Output JSON file path.
    history : mapping
        History dictionary, e.g. containing time series of abundances and diagnostics.
    """
    write_json(path, history)


def write_history_csv(path: str | Path, history: Mapping[str, Any]) -> None:
    """
    Write abundance history to a CSV file.

    Expected history format
    -----------------------
    history = {
        "t_s": [t0, t1, t2, ...],
        "abundances": {
            "p":   [ ... ],
            "n":   [ ... ],
            "he4": [ ... ],
        }
    }

    Parameters
    ----------
    path : str or Path
        Output CSV file path.
    history : mapping
        History dictionary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if "t_s" not in history:
        raise InputValidationError("History dictionary must contain key 't_s'.")
    if "abundances" not in history:
        raise InputValidationError("History dictionary must contain key 'abundances'.")

    t_values = history["t_s"]
    abundances = history["abundances"]

    if not isinstance(t_values, list):
        raise InputValidationError("'history[\"t_s\"]' must be a list.")
    abundances = _ensure_mapping(abundances, 'history["abundances"]')

    species_names = sorted(abundances.keys())

    n_rows = len(t_values)
    for species in species_names:
        series = abundances[species]
        if not isinstance(series, list):
            raise InputValidationError(f"Abundance history for species '{species}' must be a list.")
        if len(series) != n_rows:
            raise InputValidationError(
                f"Abundance history length mismatch for species '{species}': "
                f"expected {n_rows}, got {len(series)}."
            )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", *species_names])

        for i in range(n_rows):
            row = [t_values[i]] + [abundances[species][i] for species in species_names]
            writer.writerow(row)


def write_outputs(
    output_dir: str | Path,
    state: NetworkState,
    metadata: Optional[Mapping[str, Any]] = None,
    history: Optional[Mapping[str, Any]] = None,
    write_final_json_flag: bool = True,
    write_history_csv_flag: bool = True,
    write_history_json_flag: bool = False,
) -> None:
    """
    Convenience function to write all requested outputs.

    Parameters
    ----------
    output_dir : str or Path
        Output directory.
    state : NetworkState
        Final evolved state.
    metadata : mapping, optional
        Run metadata.
    history : mapping, optional
        Time history data.
    write_final_json_flag : bool
        Whether to write final_state.json.
    write_history_csv_flag : bool
        Whether to write history.csv.
    write_history_json_flag : bool
        Whether to write history.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if write_final_json_flag:
        write_final_state(output_dir / "final_state.json", state, metadata=metadata)

    if history is not None:
        if write_history_csv_flag:
            write_history_csv(output_dir / "history.csv", history)
        if write_history_json_flag:
            write_history_json(output_dir / "history.json", history)