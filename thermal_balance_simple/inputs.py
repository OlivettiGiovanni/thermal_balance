"""Read one explicit configuration and validated absorption / optical tables."""

import csv
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 uses the small tomli backport.
    import tomli as tomllib


SCHEMA = {
    "absorption": {"file", "selection", "prominence_fraction"},
    "material": {"file"},
    "geometry": {"height_nm"},
    "gas": {"mass_amu", "gamma", "accommodation"},
    "radiation": {"integration_points"},
    "analysis": {"environment_temperature_k", "reference_pressure_mbar",
                 "decomposition_temperature_k", "temperature_max_k", "temperature_points"},
    "plot": {"dpi"},
}


def _number(value, name, minimum=0.0, inclusive=False, integer=False):
    """Require an ordinary finite number with its declared physical range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if value < minimum or (value == minimum and not inclusive):
        relation = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be {relation} {minimum}.")


def load_config(path):
    """Load only this TOML; relative input filenames belong to its directory."""
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        cfg = tomllib.load(stream)
    if set(cfg) != set(SCHEMA):
        raise ValueError(f"Configuration sections: missing {sorted(set(SCHEMA) - set(cfg))}; "
                         f"unknown {sorted(set(cfg) - set(SCHEMA))}.")
    for section, keys in SCHEMA.items():
        values = cfg[section]
        if not isinstance(values, dict) or set(values) != keys:
            raise ValueError(f"[{section}] requires exactly these keys: {', '.join(sorted(keys))}.")
        for key, value in values.items():
            if key not in {"file", "selection"}:
                minimum = {"gamma": 1, "temperature_points": 3, "integration_points": 100}.get(key, 0)
                inclusive = key in {"reference_pressure_mbar", "prominence_fraction", "temperature_points", "integration_points"}
                _number(value, f"{section}.{key}", minimum, inclusive,
                        integer=key in {"temperature_points", "integration_points", "dpi"})
    if cfg["absorption"]["selection"] not in {"first_peak", "max_peak"}:
        raise ValueError("absorption.selection must be 'first_peak' or 'max_peak'.")
    if cfg["absorption"]["prominence_fraction"] > 1 or cfg["gas"]["accommodation"] > 1:
        raise ValueError("prominence_fraction and accommodation must not exceed 1.")
    analysis = cfg["analysis"]
    if not (analysis["environment_temperature_k"] < analysis["decomposition_temperature_k"]
            <= analysis["temperature_max_k"]):
        raise ValueError("Require environment_temperature_k < decomposition_temperature_k <= temperature_max_k.")
    for section in ("absorption", "material"):
        filename = cfg[section]["file"]
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"{section}.file must be a nonempty filename.")
        cfg[section]["file"] = str((path.parent / Path(filename).expanduser()).resolve())
    return cfg


def _read_csv(path):
    """Read a CSV, tolerating UTF-8 BOMs, blank lines, and whole-line comments."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(line for line in stream if line.strip() and not line.lstrip().startswith("#"))
        names = [name.strip() for name in (reader.fieldnames or [])]
        if not names or len(names) != len(set(names)) or "" in names:
            raise ValueError(f"{path}: missing or duplicate CSV column names.")
        reader.fieldnames = names
        rows = list(reader)
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError(f"{path}: empty table or inconsistent CSV row lengths.")
    return set(names), rows


def _column(rows, name):
    """Convert one required CSV column to finite numeric values."""
    try:
        values = np.array([float(row[name]) for row in rows])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"CSV column '{name}' must contain numbers in every row.") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"CSV column '{name}' contains nonfinite values.")
    return values


def _converged(value):
    """Interpret only explicit convergence flags; malformed flags are errors."""
    value = value.strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError("converged must be True, False, 1, or 0.")
    return value in {"true", "1"}


def read_absorption(path, height_nm):
    """Read density or legacy power, using the configured cylinder height."""
    _number(height_nm, "height_nm")
    names, rows = _read_csv(path)
    if "radius_nm" not in names or not {"absorption_density_W_m3", "p_abs_W"} & names:
        raise ValueError("Absorption CSV requires radius_nm and absorption_density_W_m3 (or legacy p_abs_W).")
    retained = [row for row in rows if "converged" not in names or _converged(row["converged"])]
    omitted_rows = len(rows) - len(retained)
    if not retained:
        raise ValueError("No converged absorption samples remain.")
    radius = _column(retained, "radius_nm")
    if np.any(radius <= 0) or len(np.unique(radius)) != len(radius):
        raise ValueError("Absorption radii must be positive and unique.")
    if "height_nm" in names and not np.allclose(_column(retained, "height_nm"), height_nm, rtol=1e-8, atol=0):
        raise ValueError("Absorption CSV height_nm disagrees with geometry.height_nm.")
    volume = np.pi * (radius * 1e-9) ** 2 * height_nm * 1e-9
    density = (_column(retained, "absorption_density_W_m3") if "absorption_density_W_m3" in names
               else _column(retained, "p_abs_W") / volume)
    power = density * volume
    if np.any(density < 0) or not np.all(np.isfinite(power)):
        raise ValueError("Absorption density and power must be finite and nonnegative.")
    if "p_abs_W" in names and not np.allclose(power, _column(retained, "p_abs_W"), rtol=1e-6, atol=0):
        raise ValueError("Absorption density and p_abs_W disagree for the configured cylinder volume.")
    order = np.argsort(radius)
    return {"radius_nm": radius[order], "density_w_m3": density[order],
            "p_abs_w": power[order], "omitted_rows": omitted_rows}


def select_peak(radius_nm, density_w_m3, selection, prominence_fraction):
    """Select the global maximum or first sufficiently prominent interior peak."""
    radius, density = np.asarray(radius_nm), np.asarray(density_w_m3)
    _number(prominence_fraction, "prominence_fraction", inclusive=True)
    if prominence_fraction > 1:
        raise ValueError("prominence_fraction must not exceed 1.")
    if radius.ndim != 1 or radius.shape != density.shape or not radius.size:
        raise ValueError("Peak selection requires matching, nonempty radius and density arrays.")
    if not np.all(np.isfinite(density)) or np.any(density < 0) or not np.all(np.isfinite(radius)) or np.any(np.diff(radius) <= 0):
        raise ValueError("Peak selection requires increasing finite radii and finite nonnegative densities.")
    if selection == "max_peak":
        return int(np.argmax(density))  # Sorted radii make a tie choose the smaller radius.
    if selection != "first_peak":
        raise ValueError("selection must be 'first_peak' or 'max_peak'.")
    peaks, _ = find_peaks(density, prominence=prominence_fraction * density.max())
    if not peaks.size:
        raise ValueError("No qualifying interior density peak; change the input or explicitly use max_peak.")
    return int(peaks[0])


def read_material(path, integration_points):
    """Resample fixed or temperature-dependent n,k onto their common spectral band."""
    _number(integration_points, "integration_points", 100, inclusive=True, integer=True)
    names, rows = _read_csv(path)
    if not {"wavelength_um", "n", "k"} <= names:
        raise ValueError("Material CSV requires wavelength_um,n,k and optionally temperature_k.")
    wavelength, n, k = (_column(rows, key) for key in ("wavelength_um", "n", "k"))
    if np.any(wavelength <= 0) or np.any(n < 0) or np.any(k < 0):
        raise ValueError("Material wavelengths must be positive; n and k must be nonnegative.")
    temperature = _column(rows, "temperature_k") if "temperature_k" in names else np.zeros(len(rows))
    temperatures = np.unique(temperature)
    if "temperature_k" in names and (temperatures.size < 2 or np.any(temperatures <= 0)):
        raise ValueError("A temperature-dependent material table needs at least two positive temperatures.")
    slices = []
    for value in temperatures:
        indices = np.flatnonzero(temperature == value)
        indices = indices[np.argsort(wavelength[indices])]
        wave = wavelength[indices]
        if wave.size < 2 or np.any(np.diff(wave) <= 0):
            raise ValueError("Each material temperature needs at least two unique wavelengths.")
        slices.append((wave, n[indices], k[indices]))
    lower, upper = max(s[0][0] for s in slices), min(s[0][-1] for s in slices)
    if lower >= upper:
        raise ValueError("Material temperature slices have no shared wavelength interval.")
    original = wavelength[(wavelength >= lower) & (wavelength <= upper)]
    grid = np.unique(np.concatenate((np.geomspace(lower, upper, integration_points), original)))
    return {"wavelength_m": grid * 1e-6,
            "temperature_k": temperatures if "temperature_k" in names else None,
            "n": np.array([np.interp(grid, s[0], s[1]) for s in slices]),
            "k": np.array([np.interp(grid, s[0], s[2]) for s in slices])}
