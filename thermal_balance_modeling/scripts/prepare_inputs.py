"""Prepare model inputs; unsupported temperature-dependent optics use reference values."""
from pathlib import Path
import csv
import json
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 uses the compatible lightweight backport.
    import tomli as tomllib
import numpy as np


_FIELDS = {
    "material": "name wavelength_nm n_0 reference_temperature_k",
    "data": "optical_file absorption_file",
    "geometry": "radius_nm height_nm",
    "linear": "simulation_alpha_m_inv simulation_intensity_w_um2",
    "tpa": "weights",
    "fca": "conductivity_masses_m0 density_prefactors activation_ev relaxation_times_s temperature_exponents",
    "bga": "sqrt_weights quadratic_weights",
    "experiment": "intensity_w_um2 bath_temperature_k reference_pressure_mbar temperature_max_k",
    "gas": "mass_amu gamma accommodation",
    "radiation": "wavelength_range_um",
}
_GAP = {"phonon_temperatures_k", "gap0_ev", "gap_slope_ev_k", "gap_temperature_k"}
_OPTIONAL = {"data": {"fallback_file"}, "tpa": _GAP,
             "bga": _GAP | {"quadratic_offset_ev"},
             "experiment": {"decomposition_temperature_k"}}


def _numbers(value, label, shape=None, minimum=None, positive=False):
    """Reject booleans, nonfinite values, and wrong coefficient dimensions."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "ifu" or any(isinstance(v, bool) for v in np.asarray(value, dtype=object).flat):
        raise ValueError(f"{label} must contain numbers")
    a = raw.astype(float)
    if not np.all(np.isfinite(a)) or (shape is not None and a.shape != shape):
        raise ValueError(f"{label} has nonfinite values or incorrect dimensions")
    if (positive and np.any(a <= 0)) or (minimum is not None and np.any(a < minimum)):
        raise ValueError(f"{label} is outside its allowed range")
    return a


def _configuration(path):
    with path.open("rb") as stream:
        c = tomllib.load(stream)
    if set(c) != set(_FIELDS):
        raise ValueError(f"Configuration sections must be exactly: {', '.join(_FIELDS)}")
    for section, fields in _FIELDS.items():
        required = set(fields.split())
        if not isinstance(c[section], dict):
            raise ValueError(f"{section} must be a TOML table")
        keys = set(c[section])
        missing, unknown = required - keys, keys - required - _OPTIONAL.get(section, set())
        if missing or unknown:
            raise ValueError(f"{section}: missing {sorted(missing)}, unknown {sorted(unknown)}")
    if not isinstance(c["material"]["name"], str) or not c["material"]["name"].strip():
        raise ValueError("material.name must be a nonempty name")
    for section, fields in {
        "material": "wavelength_nm n_0 reference_temperature_k",
        "geometry": "height_nm", "linear": _FIELDS["linear"],
        "experiment": "bath_temperature_k temperature_max_k",
        "gas": "mass_amu gamma accommodation",
    }.items():
        for field in fields.split():
            _numbers(c[section][field], f"{section}.{field}", (), positive=True)
    radius = c["geometry"]["radius_nm"]
    if isinstance(radius, str):
        if radius not in ("max_peak", "first_peak"):
            raise ValueError("geometry.radius_nm must be positive, max_peak, or first_peak")
    else:
        _numbers(radius, "geometry.radius_nm", (), positive=True)
    ex = c["experiment"]
    _numbers(ex["intensity_w_um2"], "experiment.intensity_w_um2", (), minimum=0)
    _numbers(ex["reference_pressure_mbar"], "experiment.reference_pressure_mbar", (), minimum=0)
    if ex["temperature_max_k"] <= ex["bath_temperature_k"]:
        raise ValueError("temperature_max_k must exceed bath_temperature_k")
    if "decomposition_temperature_k" in ex:
        _numbers(ex["decomposition_temperature_k"], "decomposition_temperature_k", (), positive=True)
        if ex["decomposition_temperature_k"] <= ex["bath_temperature_k"]:
            raise ValueError("decomposition_temperature_k must exceed the bath temperature")
        if ex["decomposition_temperature_k"] > ex["temperature_max_k"]:
            raise ValueError("decomposition_temperature_k must not exceed temperature_max_k")
    if c["gas"]["gamma"] <= 1 or c["gas"]["accommodation"] > 1:
        raise ValueError("gas.gamma must exceed 1 and accommodation must lie in (0, 1]")
    band = _numbers(c["radiation"]["wavelength_range_um"], "wavelength_range_um", (2,), positive=True)
    if band[0] >= band[1]:
        raise ValueError("wavelength_range_um must have increasing endpoints")
    for section, names in (("tpa", ["weights"]), ("bga", ["sqrt_weights", "quadratic_weights"])):
        arrays = [_numbers(c[section][name], f"{section}.{name}", minimum=0) for name in names]
        if any(a.ndim != 1 or not a.size or a.shape != arrays[0].shape for a in arrays):
            raise ValueError(f"{section} weights must be nonempty lists of equal length")
        enabled = any(np.any(a) for a in arrays)
        needed = _OPTIONAL[section] if enabled else set()
        actual = set(c[section]) - set(names)
        if actual != needed:
            raise ValueError(f"{section}: include gap/phonon settings only when weights are nonzero")
        if enabled:
            for field in needed:
                shape = arrays[0].shape if field == "phonon_temperatures_k" else ()
                _numbers(c[section][field], f"{section}.{field}", shape,
                         positive=field not in ("gap_slope_ev_k", "quadratic_offset_ev"), minimum=0)
    fca = c["fca"]
    for field in ("conductivity_masses_m0", "density_prefactors", "activation_ev"):
        _numbers(fca[field], f"fca.{field}", (2,), positive=field == "conductivity_masses_m0", minimum=0)
    tau = _numbers(fca["relaxation_times_s"], "fca.relaxation_times_s", positive=True)
    if tau.ndim != 2 or tau.shape[1] != 2 or tau.shape[0] == 0:
        raise ValueError("fca.relaxation_times_s must have one [electron, hole] row per term")
    _numbers(fca["temperature_exponents"], "fca.temperature_exponents", tau.shape)
    for field, value in c["data"].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"data.{field} must be a file path")
        resolved = (path.parent / Path(value).expanduser()).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"data.{field}: file not found: {resolved}")
        c["data"][field] = str(resolved)
    return c


def _read_table(path):
    """Blank optical cells are unknown; explicit NaN/inf and duplicate cells are errors."""
    columns = ["temperature_k", "wavelength_um", "n", "k"]
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != columns:
            raise ValueError(f"{path}: expected CSV columns {','.join(columns)}")
        values = []
        for line, row in enumerate(reader, 2):
            if None in row or any(row[key] is None for key in columns):
                raise ValueError(f"{path}:{line}: malformed CSV row")
            try:
                parsed = [float(row[key]) if row[key].strip() else np.nan for key in columns]
            except ValueError as exc:
                raise ValueError(f"{path}:{line}: invalid numeric cell") from exc
            for j, value in enumerate(parsed):
                if row[columns[j]].strip() and not np.isfinite(value):
                    raise ValueError(f"{path}:{line}: nonfinite numeric cell")
                if (j < 2 and (not np.isfinite(value) or value <= 0)) or (
                    j == 2 and np.isfinite(value) and value <= 0) or (j == 3 and value < 0):
                    raise ValueError(f"{path}:{line}: invalid temperature, wavelength, n, or k")
            values.append(parsed)
    a = np.asarray(values, dtype=float).reshape(-1, 4)
    if not len(a) or not np.any(np.isfinite(a[:, 2:])):
        raise ValueError(f"{path}: no usable optical constants; populate this material table first")
    if len(np.unique(a[:, :2], axis=0)) != len(a):
        raise ValueError(f"{path}: duplicate temperature/wavelength rows")
    return a[np.lexsort((a[:, 1], a[:, 0]))]


def _interp(x, y, target):
    """Linear interpolation between adjacent rows, retaining NaN gaps and no extrapolation."""
    if len(x) == 0:
        return np.full(np.asarray(target).shape, np.nan)
    return np.interp(target, x, y, left=np.nan, right=np.nan)


def _at_wavelength(table, wavelength, column):
    temperatures = np.unique(table[:, 0])
    values = []
    for temperature in temperatures:
        rows = table[table[:, 0] == temperature]
        values.append(_interp(rows[:, 1], rows[:, column], wavelength))
    return temperatures, np.asarray(values)


def _reference_slice(table, reference_temperature_k):
    """Use the nearest documented room-temperature slice, within 10 K."""
    if not len(table):
        return np.empty((0, 4)), None
    temperatures = np.unique(table[:, 0])
    chosen = temperatures[np.argmin(abs(temperatures - reference_temperature_k))]
    if abs(chosen - reference_temperature_k) > 10:
        return np.empty((0, 4)), None
    return table[table[:, 0] == chosen], float(chosen)


def _temperature_spectrum(table, temperature_k, wavelength_um, column):
    """Interpolate adjacent wavelength and temperature rows without filling NaN gaps."""
    temperatures, spectra = np.unique(table[:, 0]), []
    if not len(temperatures):
        return np.full(temperature_k.shape + wavelength_um.shape, np.nan)
    for temperature in temperatures:
        rows = table[table[:, 0] == temperature]
        values = rows[:, column].copy()
        valid = np.isfinite(values)
        adjacent = valid[:-1] & valid[1:]
        supported = np.r_[False, adjacent] | np.r_[adjacent, False]
        values[~supported] = np.nan  # A lone laser-wavelength point has no spectral bandwidth.
        spectra.append(_interp(rows[:, 1], values, wavelength_um))
    spectra = np.asarray(spectra)
    return np.stack([_interp(temperatures, spectrum, temperature_k)
                     for spectrum in spectra.T], axis=-1)


def _radiation_fallback_note(result, table, wavelength, reference_temperature):
    """Describe available broadband temperature coverage in one short plot note."""
    if not np.any(result["n_room_fallback"] | result["k_room_fallback"]):
        return ""
    temperatures = np.unique(table[:, 0])
    n = _temperature_spectrum(table, temperatures, wavelength, 2)
    k = _temperature_spectrum(table, temperatures, wavelength, 3)
    complete = np.all(np.isfinite(n) & np.isfinite(k), axis=-1)
    covered = temperatures[complete]
    reference = f"{reference_temperature:g} K"
    if covered.size > 1:
        return f"BB: {reference} n,k outside {covered[0]:g}–{covered[-1]:g} K or where data are missing."
    hot = ~np.isclose(temperatures, reference_temperature)
    if np.any(np.isfinite(n[hot]) | np.isfinite(k[hot])):
        return f"BB: {reference} values where spectral T data are missing."
    return f"BB: {reference} n,k (no broadband T data)."


def _coverage_breaks(temperature, fallback, reference_temperature):
    """Flag coverage changes, omitting the continuous room-temperature join."""
    breaks = np.zeros(temperature.shape, dtype=bool)
    if temperature.ndim == 1 and temperature.size > 1:
        changed = np.diff(fallback, axis=0)
        if changed.ndim > 1:
            changed = np.any(changed, axis=tuple(range(1, changed.ndim)))
        near_room = np.isclose(temperature[:-1], reference_temperature) | np.isclose(temperature[1:], reference_temperature)
        breaks[1:] = changed & ~near_room
    return breaks


def prepare_radiation_optics(table, temperature_k, wavelength_um,
                             reference_temperature_k=300.0, fallback_table=None):
    """Resolve n(lambda,T), k(lambda,T); unsupported cells use room-temperature data.

    Each property is interpolated independently, only between adjacent supplied
    rows. A single-wavelength temperature series therefore does not become a
    broadband correction. An explicitly declared other material supplies missing
    reference values only; its hot data are never transferred to this material.
    """
    temperature = np.asarray(temperature_k, dtype=float)
    wavelength = np.asarray(wavelength_um, dtype=float)
    reference_temperature_k = float(reference_temperature_k)
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0) or not np.isfinite(reference_temperature_k) or reference_temperature_k <= 0:
        raise ValueError("Optical temperatures must be finite and positive.")
    if wavelength.ndim != 1 or not wavelength.size or np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0) or np.any(np.diff(wavelength) <= 0):
        raise ValueError("Optical wavelengths must be a positive, increasing 1D grid.")
    table = np.asarray(table, dtype=float)
    fallback = np.empty((0, 4)) if fallback_table is None else np.asarray(fallback_table, dtype=float)
    for data in (table, fallback):
        if data.ndim != 2 or data.shape[1] != 4:
            raise ValueError("Optical tables need temperature_k,wavelength_um,n,k columns.")
        if np.any(~np.isfinite(data[:, :2])) or np.any(data[:, :2] <= 0) or np.any(np.isinf(data[:, 2:])) or np.any(data[:, 2] <= 0) or np.any(data[:, 3] < 0):
            raise ValueError("Invalid optical table coordinates or constants.")
        if len(np.unique(data[:, :2], axis=0)) != len(data):
            raise ValueError("Duplicate temperature/wavelength rows in optical table.")
    table = table[np.lexsort((table[:, 1], table[:, 0]))]
    fallback = fallback[np.lexsort((fallback[:, 1], fallback[:, 0]))]
    room, room_k = _reference_slice(table, reference_temperature_k)
    other_room, other_k = _reference_slice(fallback, reference_temperature_k)
    result = {"reference_temperature_k": room_k, "fallback_temperature_k": other_k}
    for name, column in (("n", 2), ("k", 3)):
        reference = _interp(room[:, 1], room[:, column], wavelength)
        other = _interp(other_room[:, 1], other_room[:, column], wavelength)
        borrowed = ~np.isfinite(reference) & np.isfinite(other)
        reference = np.where(borrowed, other, reference)
        if np.any(~np.isfinite(reference)):
            raise ValueError(f"Unresolved reference-spectrum {name} in requested radiation band; provide a documented fallback")
        hot = _temperature_spectrum(table, temperature, wavelength, column)
        missing = ~np.isfinite(hot)
        result[name + "_reference"] = reference
        result[name] = np.where(missing, reference, hot)
        result[name + "_room_fallback"] = missing & ~np.isclose(temperature[..., None], reference_temperature_k)
        result[name + "_borrowed_reference"] = borrowed
    result["footnote"] = _radiation_fallback_note(result, table, wavelength, reference_temperature_k)
    masks = np.stack((result["n_room_fallback"], result["k_room_fallback"]), axis=-1)
    result["break_before"] = _coverage_breaks(temperature, masks, reference_temperature_k)
    return result


def _metadata(path):
    sidecar = Path(path).with_suffix(".json")
    return json.loads(sidecar.read_text()) if sidecar.is_file() else {"csv_file": Path(path).name}


def prepare_inputs(config_path):
    """Return resolved settings and inspectable spectral/laser optical arrays."""
    c = _configuration(Path(config_path).expanduser().resolve())
    primary = _read_table(c["data"]["optical_file"])
    metadata = _metadata(c["data"]["optical_file"])
    fallback_path = c["data"].get("fallback_file")
    fallback = _read_table(fallback_path) if fallback_path else np.empty((0, 4))
    fallback_metadata = _metadata(fallback_path) if fallback_path else {}
    room, room_k = _reference_slice(primary, c["material"]["reference_temperature_k"])
    fallback_room, fallback_k = _reference_slice(fallback, c["material"]["reference_temperature_k"])
    lo, hi = c["radiation"]["wavelength_range_um"]
    wavelengths = np.unique(np.r_[lo, hi, primary[:, 1], fallback_room[:, 1]])
    wavelengths = wavelengths[(wavelengths >= lo) & (wavelengths <= hi)]
    # Resolve the Planck integral even for sparse tables, while probing every source
    # interval so disjoint tables cannot hide an uncovered spectral gap.
    wavelengths = np.unique(np.r_[wavelengths, (wavelengths[:-1] + wavelengths[1:]) / 2,
                                  np.geomspace(lo, hi, 900)])
    ex, material = c["experiment"], c["material"]
    temperature = np.linspace(ex["bath_temperature_k"], ex["temperature_max_k"], 601)
    limit = ex.get("decomposition_temperature_k")
    if limit is not None and temperature[0] < limit < temperature[-1]:
        temperature = np.unique(np.r_[temperature, limit])
    radiation = prepare_radiation_optics(primary, temperature, wavelengths,
                                         material["reference_temperature_k"], fallback)
    borrowed = bool(np.any(radiation["n_borrowed_reference"] | radiation["k_borrowed_reference"]))
    wavelength = material["wavelength_nm"] / 1000
    tn, nv = _at_wavelength(primary, wavelength, 2)
    n_reference = _interp(tn, nv, material["reference_temperature_k"])
    n_increment = _interp(tn, nv, temperature) - n_reference
    n_laser = material["n_0"] + np.where(np.isfinite(n_increment), n_increment, 0)
    n_laser_breaks = _coverage_breaks(temperature, ~np.isfinite(n_increment), material["reference_temperature_k"])
    tk, kv = _at_wavelength(primary, wavelength, 3)
    k_reference = _interp(room[:, 1], room[:, 3], wavelength)
    reference_borrowed = not np.isfinite(k_reference)
    if reference_borrowed:
        k_reference = _interp(fallback_room[:, 1], fallback_room[:, 3], wavelength)
    if not np.isfinite(k_reference):
        raise ValueError("No reference-spectrum k at the laser wavelength in the primary or declared fallback")
    hot_k = _interp(tk, kv, temperature)
    k_measured = np.isfinite(hot_k)  # Table coverage includes published fits, not only raw measurements.
    k_laser = np.where(k_measured, hot_k, k_reference)
    if np.any(n_laser <= 0):
        raise ValueError("The calibrated laser refractive index must remain positive")
    provenance = {"material": metadata, "fallback": fallback_metadata,
                  "reference_spectrum_temperature_k": room_k, "fallback_temperature_k": fallback_k,
                  "k_reference_borrowed": reference_borrowed,
                  "borrowed_spectrum": borrowed}
    return {"config": c, "temperature_k": temperature, "wavelength_m": wavelengths * 1e-6,
            "n_spectrum": radiation["n_reference"], "k_spectrum": radiation["k_reference"],
            "n_radiation": radiation["n"], "k_radiation": radiation["k"],
            "n_room_fallback": radiation["n_room_fallback"], "k_room_fallback": radiation["k_room_fallback"],
            "radiation_footnote": radiation["footnote"], "radiation_break_before": radiation["break_before"],
            "n_laser": n_laser, "n_laser_break_before": n_laser_breaks,
            "k_laser": k_laser, "k_measured": k_measured, "k_reference": float(k_reference),
            "borrowed_spectrum": borrowed, "provenance": provenance}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    inputs = prepare_inputs(parser.parse_args().config)
    print(f"Prepared {len(inputs['temperature_k'])} temperatures and {len(inputs['wavelength_m'])} radiation wavelengths.")
    print(f"Radiation uses a borrowed fallback: {inputs['borrowed_spectrum']}")
    if inputs["radiation_footnote"]:
        print(inputs["radiation_footnote"])
