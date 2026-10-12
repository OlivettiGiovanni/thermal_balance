"""Read one explicit TOML and small numeric CSVs; paths belong to the TOML."""
import csv
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


DEFAULTS = dict(intensity_w_um2=.1, reference_pressure_mbar=.1,
                temperature_max_k=1600., temperature_points=401,
                environment_temperature_k=300., accommodation=.8,
                selection="max_peak", height_nm=400.,
                decomposition_temperature_k=1950., nonlinear_file="")
FILES = ("absorption_file", "spectrum_file", "nonlinear_file")


def load_config(path):
    """Apply documented defaults to this file only; reject unknown keys."""
    path = Path(path).resolve()
    with path.open("rb") as stream:
        supplied = tomllib.load(stream)
    if set(supplied) - (set(DEFAULTS) | set(FILES) | {"material"}):
        raise ValueError("Unknown configuration keys: " + str(sorted(set(supplied)-set(DEFAULTS)-set(FILES)-{"material"})))
    material = supplied.get("material")
    if material not in ("Si", "SiN"):
        raise ValueError("material must be 'Si' or 'SiN'")
    irrelevant = ({"intensity_w_um2"} if material == "SiN" else
                  {"absorption_file", "spectrum_file", "nonlinear_file", "height_nm", "selection", "decomposition_temperature_k"})
    if set(supplied) & irrelevant:
        raise ValueError(f"Unused keys for {material}: {sorted(set(supplied) & irrelevant)}")
    config = {**DEFAULTS, "temperature_max_k": 1600. if material == "Si" else 2500., **supplied}
    for key in (set(DEFAULTS) - {"selection", "nonlinear_file"}):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite nonnegative number")
    if config["height_nm"] <= 0 or not 0 < config["accommodation"] <= 1:
        raise ValueError("height_nm > 0 and 0 < accommodation <= 1 are required")
    if not isinstance(config["temperature_points"], int) or config["temperature_points"] < 30:
        raise ValueError("temperature_points must be an integer >= 30")
    if not 0 < config["environment_temperature_k"] < config["temperature_max_k"]:
        raise ValueError("Require 0 < bath temperature < temperature_max_k")
    if config["selection"] not in ("first_peak", "max_peak"):
        raise ValueError("selection must be first_peak or max_peak")
    if material == "Si" and (config["environment_temperature_k"] != 300 or config["temperature_max_k"] > 1600):
        raise ValueError("The Si reference benchmark requires a 300 K bath and T_max <= 1600 K")
    if material == "SiN" and not config["environment_temperature_k"] < config["decomposition_temperature_k"] <= config["temperature_max_k"]:
        raise ValueError("Require bath < decomposition temperature <= temperature_max_k")
    for key in FILES:
        if key in config and not isinstance(config[key], str):
            raise ValueError(f"{key} must be a filename string")
        if config.get(key):
            config[key] = str((path.parent / config[key]).resolve())
    if material == "SiN" and not all(config.get(key) for key in FILES[:2]):
        raise ValueError("SiN requires absorption_file and spectrum_file")
    return config


def read_csv(path):
    """Read numeric columns, retaining NaN only where the input contains blanks."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(line for line in stream if line.strip() and not line.lstrip().startswith("#"))
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"{path}: missing or duplicate headers")
        rows = list(reader)
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError(f"{path}: empty table or malformed rows")
    if "converged" in reader.fieldnames:
        if any(row["converged"].strip().lower() not in {"true", "false", "0", "1"} for row in rows):
            raise ValueError("converged must be true/false or 1/0")
        rows = [row for row in rows if row["converged"].strip().lower() in {"true", "1"}]
    if not rows:
        raise ValueError("No converged absorption rows")
    # Legacy sweeps contain metadata columns; retain only columns that are numeric.
    result = {}
    for key in reader.fieldnames:
        try:
            result[key] = np.array([float(row[key]) if row[key].strip() else np.nan for row in rows])
        except ValueError:
            continue
    return result


def finite_columns(table, columns):
    """Require the named numeric columns to be present and finite."""
    for column in columns:
        if column not in table or np.any(~np.isfinite(table[column])):
            raise ValueError(f"Missing or nonfinite CSV column: {column}")


def select_absorption(path, height_nm, selection):
    """Select one maximum of absorption/V, never a maximum of total power."""
    table = read_csv(path)
    finite_columns(table, ["radius_nm"])
    order = np.argsort(table["radius_nm"])
    table = {key: values[order] for key, values in table.items()}
    radius = table["radius_nm"]
    if np.any(radius <= 0) or np.any(np.diff(radius) <= 0):
        raise ValueError("Radii must be positive and unique")
    if "height_nm" in table and not np.allclose(table["height_nm"], height_nm):
        raise ValueError("CSV and configuration heights disagree")
    volume = np.pi*(radius*1e-9)**2*height_nm*1e-9
    key = "absorption_density_W_m3" if "absorption_density_W_m3" in table else "p_abs_W"
    finite_columns(table, [key])
    density = table[key] if key == "absorption_density_W_m3" else table[key]/volume
    if np.any(density < 0):
        raise ValueError("Absorption must be nonnegative")
    power = density*volume
    if "p_abs_W" in table and not np.allclose(table["p_abs_W"], power, rtol=1e-6, atol=0):
        raise ValueError("CSV density and total power disagree")
    index = int(np.argmax(density))
    if selection == "first_peak":
        peaks, _ = find_peaks(density, prominence=.03*density.max())
        if not peaks.size:
            raise ValueError("No interior first peak with 3% prominence; use max_peak explicitly")
        index = int(peaks[0])
    selected = {key: float(values[index]) for key, values in table.items()}
    selected.update(radius_nm=float(radius[index]), p_abs_w=float(power[index]))
    return selected, dict(radius_nm=radius, density_w_m3=density, p_abs_w=power,
                         selected=(np.arange(len(radius)) == index).astype(int))


def read_spectrum(path, points=1600):
    """Resample n,k on a common wavelength band; preserve temperature slices."""
    table = read_csv(path)
    finite_columns(table, ["wavelength_um", "n", "k"])
    if np.any(table["wavelength_um"] <= 0) or np.any(table["n"] < 0) or np.any(table["k"] < 0):
        raise ValueError("Spectrum requires positive wavelengths and nonnegative n,k")
    temperatures = np.unique(table.get("temperature_k", [0.]))
    if "temperature_k" in table:
        finite_columns(table, ["temperature_k"])
        if len(temperatures) < 2 or np.any(temperatures <= 0):
            raise ValueError("Temperature-dependent spectra require at least two positive temperatures")
    slices = []
    for temperature in temperatures:
        mask = table["temperature_k"] == temperature if "temperature_k" in table else np.ones(len(table["n"]), bool)
        order = np.argsort(table["wavelength_um"][mask])
        wave, index, loss = [table[key][mask][order] for key in ("wavelength_um", "n", "k")]
        if len(wave) < 2 or np.any(np.diff(wave) <= 0):
            raise ValueError("Each spectrum needs at least two unique wavelengths")
        slices.append((wave, index, loss))
    lo, hi = max(row[0][0] for row in slices), min(row[0][-1] for row in slices)
    if lo >= hi:
        raise ValueError("Temperature spectra have no common wavelength band")
    grid = np.unique(np.r_[np.geomspace(lo, hi, points), table["wavelength_um"][(table["wavelength_um"] >= lo)&(table["wavelength_um"] <= hi)]])
    return dict(wavelength_m=grid*1e-6, temperature_k=temperatures if "temperature_k" in table else None,
                n=np.array([np.interp(grid, wave, index) for wave, index, loss in slices]),
                k=np.array([np.interp(grid, wave, loss) for wave, index, loss in slices]))


def nonlinear_coefficients(path, temperature):
    """Read material-specific beta and thermal FCA; an absent file means both zero."""
    if not path:
        return np.zeros_like(temperature), np.zeros_like(temperature)
    table = read_csv(path)
    finite_columns(table, ["temperature_k", "beta_m_w", "alpha_fca_m_inv"])
    order = np.argsort(table["temperature_k"])
    samples = table["temperature_k"][order]
    if len(samples) < 2 or np.any(np.diff(samples) <= 0) or temperature.min() < samples[0] or temperature.max() > samples[-1]:
        raise ValueError("Nonlinear table needs unique temperatures covering the full calculation range")
    if np.any(table["beta_m_w"] < 0) or np.any(table["alpha_fca_m_inv"] < 0):
        raise ValueError("Nonlinear absorption coefficients must be nonnegative")
    return tuple(np.interp(temperature, samples, table[key][order]) for key in ("beta_m_w", "alpha_fca_m_inv"))
