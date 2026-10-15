"""Absorption powers from imported optical fields and explicit material parameters.

Local intensity is frozen at the simulated geometry and optical state. Changing
temperature or laser power does not re-solve the resonant electromagnetic field.
"""

from pathlib import Path
import csv
import json

import numpy as np
from scipy.constants import c, e, epsilon_0, h, k as K_B, m_e
from scipy.signal import find_peaks

K_B_EV = K_B / e


def _temperatures(T):
    """Return a finite, strictly positive temperature array in kelvin."""
    T = np.asarray(T, dtype=float)
    if np.any(~np.isfinite(T)) or np.any(T <= 0):
        raise ValueError("Temperatures must be finite and greater than zero kelvin.")
    return T


def _gap(T, params):
    """Varshni gap in eV; coefficients must be supplied by the material."""
    gap0, slope, scale = (float(params[key]) for key in
                         ("gap0_ev", "gap_slope_ev_k", "gap_temperature_k"))
    if not np.all(np.isfinite([gap0, slope, scale])) or min(gap0, slope, scale) < 0:
        raise ValueError("Varshni parameters must be finite and nonnegative.")
    gap = gap0 - slope * T**2 / (T + scale)
    if np.any(gap <= 0):
        raise ValueError("The supplied Varshni model gives a nonpositive band gap.")
    return gap


def _phonons(T, params, count):
    """Return phonon energies (eV) and Bose occupations for each mode."""
    theta = np.asarray(params["phonon_temperatures_k"], dtype=float)
    if theta.shape != (count,) or np.any(~np.isfinite(theta)) or np.any(theta <= 0):
        raise ValueError("One positive phonon temperature is required per weight.")
    x = theta / T[..., None]
    # Equivalent to 1/expm1(x), but does not overflow for a frozen-out mode.
    occupation = np.exp(-x) / -np.expm1(-x)
    return K_B_EV * theta, occupation


def _photon_energy(wavelength_m):
    """Photon energy in eV for one laser wavelength in metres."""
    wavelength_m = float(wavelength_m)
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("Laser wavelength must be finite and positive.")
    return h * c / (wavelength_m * e)


def tpa_beta(T, wavelength_m, params):
    """Indirect TPA coefficient in m/W, with weights in m/(W eV**3.5).

    Phonon emission has the higher threshold and factor N+1; absorption has
    the lower threshold and factor N. No material coefficients are implicit.
    """
    T = _temperatures(T)
    weights = np.asarray(params["weights"], dtype=float)
    if weights.ndim != 1 or not len(weights) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("TPA weights must be a nonempty, finite, nonnegative list.")
    if not np.any(weights):
        return np.zeros_like(T)  # Disabled channel needs no unused fit parameters.
    gap = _gap(T, params)
    phonon, occupation = _phonons(T, params, len(weights))
    excess = 2 * _photon_energy(wavelength_m) - gap[..., None]
    emission = np.maximum(excess - phonon, 0)**2
    absorption = np.maximum(excess + phonon, 0)**2
    return gap**1.5 * np.sum(weights * ((occupation + 1) * emission + occupation * absorption), axis=-1)


def bandgap_alpha(T, wavelength_m, params):
    """Parameterized indirect-edge absorption in inverse metres.

    sqrt_weights have units m^-1 eV^-1/2; quadratic_weights have units
    m^-1 eV^-2. These parameters describe the requested material only.
    """
    T = _temperatures(T)
    a = np.asarray(params["sqrt_weights"], dtype=float)
    b = np.asarray(params["quadratic_weights"], dtype=float)
    if a.ndim != 1 or not len(a) or b.shape != a.shape or np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(a < 0) or np.any(b < 0):
        raise ValueError("BGA weights must be equal-length, finite, nonnegative lists.")
    if not np.any(a) and not np.any(b):
        return np.zeros_like(T)
    gap = _gap(T, params)
    phonon, occupation = _phonons(T, params, len(a))
    offset = float(params["quadratic_offset_ev"])
    if not np.isfinite(offset) or offset < 0:
        raise ValueError("The quadratic band-edge offset must be finite and nonnegative.")
    excess = _photon_energy(wavelength_m) - gap[..., None]
    emission = a * np.sqrt(np.maximum(excess - phonon, 0)) + b * np.maximum(excess - phonon - offset, 0)**2
    absorption = a * np.sqrt(np.maximum(excess + phonon, 0)) + b * np.maximum(excess + phonon - offset, 0)**2
    return np.sum((occupation + 1) * emission + occupation * absorption, axis=-1)


def drude_alpha(T, wavelength_m, n, T0, params):
    """Drude absorption using explicit carrier populations and relaxation times.

    Density prefactors have units m^-3 K^-3/2. The final axis of density and
    damping contains electron and hole channels, respectively. This is not a
    dopant, charge-neutrality, or photocarrier solver.
    """
    T = _temperatures(T)
    prefactors = np.asarray(params["density_prefactors"], dtype=float)
    if prefactors.shape != (2,) or np.any(~np.isfinite(prefactors)) or np.any(prefactors < 0):
        raise ValueError("Two finite nonnegative carrier-density prefactors are required.")
    if not np.any(prefactors):
        zeros = np.zeros(T.shape + (2,))
        return {"alpha_m_inv": np.zeros_like(T), "density_m3": zeros, "damping_s_inv": zeros.copy()}
    masses = np.asarray(params["conductivity_masses_m0"], dtype=float)
    activation = np.asarray(params["activation_ev"], dtype=float)
    tau0 = np.asarray(params["relaxation_times_s"], dtype=float)
    powers = np.asarray(params["temperature_exponents"], dtype=float)
    if masses.shape != (2,) or activation.shape != (2,) or np.any(~np.isfinite(masses)) or np.any(masses <= 0) or np.any(~np.isfinite(activation)) or np.any(activation < 0):
        raise ValueError("Supply two positive conductivity masses and two nonnegative activation energies.")
    if tau0.ndim != 2 or tau0.shape[1] != 2 or not tau0.shape[0] or powers.shape != tau0.shape or np.any(~np.isfinite(tau0)) or np.any(tau0 < 0) or np.any(~np.isfinite(powers)):
        raise ValueError("Relaxation times and exponents must be matching [term][electron,hole] arrays.")
    n = np.asarray(n, dtype=float)
    T0 = float(T0)
    if np.any(~np.isfinite(n)) or np.any(n <= 0) or not np.isfinite(T0) or T0 <= 0:
        raise ValueError("Refractive index and reference temperature must be positive.")
    density = prefactors * T[..., None]**1.5 * np.exp(-activation / (K_B_EV * T[..., None]))
    tau = np.sum(tau0 * (T[..., None, None] / T0)**powers, axis=-2)
    if np.any(~np.isfinite(tau)) or np.any(tau <= 0):
        raise ValueError("Each carrier must have a finite positive total relaxation time.")
    damping = 1 / tau
    omega = _photon_energy(wavelength_m) * e / (h / (2 * np.pi))
    alpha = np.sum(density * e**2 * damping / (epsilon_0 * c * n[..., None] * masses * m_e * (omega**2 + damping**2)), axis=-1)
    return {"alpha_m_inv": alpha, "density_m3": density, "damping_s_inv": damping}


def _read_optical_table(path):
    """Read a named-column optical CSV without silently skipping malformed rows."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "radius_nm" not in reader.fieldnames:
            raise ValueError("Optical CSV requires a radius_nm column.")
        if any(not name.strip() for name in reader.fieldnames) or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Optical CSV headers must be nonempty and unique.")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Optical CSV row {reader.line_num} has a different number of fields than its header.")
            rows.append(row)
    if not rows:
        raise ValueError("Optical CSV contains no simulation rows.")
    return rows


def _number(row, key):
    """Read one finite numeric CSV value with its column in any error."""
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Optical CSV needs a numeric {key} value in every applicable row.") from exc
    if not np.isfinite(value):
        raise ValueError(f"Optical CSV {key} values must be finite.")
    return value


def _verify_metadata(path, rows, config):
    """Check every supplied metadata value; record absent fields as unverified."""
    expected = {
        "wavelength_nm": config["material"]["wavelength_nm"],
        "n_0": config["material"]["n_0"],
        "simulation_alpha_m_inv": config["linear"]["simulation_alpha_m_inv"],
        "simulation_intensity_w_um2": config["linear"]["simulation_intensity_w_um2"],
        "height_nm": config["geometry"]["height_nm"],
    }
    sources, sidecar_notes = [], []
    for candidate in dict.fromkeys([path.with_suffix(".json"), Path(str(path) + ".json")]):
        if candidate.is_file():
            with candidate.open(encoding="utf-8") as stream:
                metadata = json.load(stream)
            if not isinstance(metadata, dict):
                raise ValueError(f"Optical metadata must be a JSON object: {candidate}")
            source = metadata.get("metadata", metadata)
            sources.append(source)
            for record in ([metadata, source] if source is not metadata else [metadata]):
                if isinstance(record, dict):
                    for key in ("note", "status"):
                        if isinstance(record.get(key), (str, int, float, bool)):
                            entry = f"Simulation sidecar {key}: {record[key]}"
                            if entry not in sidecar_notes:
                                sidecar_notes.append(entry)
    sources.extend(rows)
    verified = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Optical metadata must contain a JSON object.")
        for key, target in expected.items():
            if key in source and source[key] not in (None, ""):
                value = _number(source, key)
                if not np.isclose(value, float(target), rtol=1e-8, atol=1e-12):
                    raise ValueError(f"Optical metadata {key}={value:g} disagrees with config value {float(target):g}.")
                verified.add(key)
    notes = ["Imported simulation field is frozen with temperature and incident intensity; no resonant re-solve or standing-wave multiplier is applied."]
    notes.extend(sidecar_notes)
    missing = sorted(set(expected) - verified)
    if missing:
        notes.append("Unverified simulation metadata (not supplied): " + ", ".join(missing) + ".")
    if verified:
        notes.append("Verified supplied simulation metadata: " + ", ".join(sorted(verified)) + ".")
    return notes


def _select_particle(rows, config):
    """Select the exact requested radius or a peak of absorbed-power density."""
    height = float(config["geometry"]["height_nm"])
    radius = np.array([_number(row, "radius_nm") for row in rows])
    if not np.isfinite(height) or height <= 0 or np.any(radius <= 0) or len(np.unique(radius)) != len(radius):
        raise ValueError("Simulation radii must be distinct and positive; height must be positive.")
    volumes = np.pi * (radius * 1e-9)**2 * height * 1e-9
    powers = []
    for row, volume in zip(rows, volumes):
        has_power = row.get("p_abs_W") not in (None, "")
        has_density = row.get("absorption_density_W_m3") not in (None, "")
        if not has_power and not has_density:
            raise ValueError("Optical CSV needs p_abs_W or absorption_density_W_m3.")
        power = _number(row, "p_abs_W") if has_power else _number(row, "absorption_density_W_m3") * volume
        if power < 0:
            raise ValueError("Simulated absorbed power cannot be negative.")
        if has_power and has_density and not np.isclose(power, _number(row, "absorption_density_W_m3") * volume, rtol=1e-6, atol=0):
            raise ValueError("p_abs_W and absorption_density_W_m3 disagree for the stated cylinder volume.")
        powers.append(power)
    powers = np.array(powers)
    choice = config["geometry"]["radius_nm"]
    order = np.argsort(radius)
    if choice == "max_peak":
        selected = order[np.argmax((powers / volumes)[order])]
    elif choice == "first_peak":
        density = (powers / volumes)[order]
        peaks, _ = find_peaks(density, prominence=0.03 * np.max(density))
        if not len(peaks):
            raise ValueError("No interior density peak has at least 3% of the maximum density in prominence.")
        selected = order[peaks[0]]
    else:
        try:
            matches = np.flatnonzero(radius == float(choice))
        except (TypeError, ValueError) as exc:
            raise ValueError("radius_nm must be a number, max_peak, or first_peak.") from exc
        if len(matches) != 1:
            raise ValueError(f"Requested radius_nm={choice} is absent from the optical CSV; no interpolation is performed.")
        selected = matches[0]
    row = rows[selected]
    if row.get("converged") not in (None, "") and str(row["converged"]).strip().lower() not in {"true", "1", "1.0", "yes"}:
        raise ValueError("The selected optical simulation is not marked converged.")
    return row, {"radius_nm": float(radius[selected]), "height_nm": height,
                 "volume_m3": float(volumes[selected]), "p_abs_W": float(powers[selected])}


def absorption_powers(prepared):
    """Build absorption channels from simulation power and validated field moments."""
    config = prepared["config"]
    T = _temperatures(prepared["temperature_k"])
    path = Path(config["data"]["absorption_file"])
    rows = _read_optical_table(path)
    notes = _verify_metadata(path, rows, config)
    row, particle = _select_particle(rows, config)
    alpha_sim = float(config["linear"]["simulation_alpha_m_inv"])
    intensity_sim = float(config["linear"]["simulation_intensity_w_um2"])
    intensity = float(config["experiment"]["intensity_w_um2"])
    if not np.all(np.isfinite([alpha_sim, intensity_sim, intensity])) or alpha_sim <= 0 or intensity_sim <= 0 or intensity < 0:
        raise ValueError("Simulation alpha and intensity must be positive; experiment intensity must be nonnegative.")
    moment1 = particle["p_abs_W"] / alpha_sim
    if row.get("intensity_integral_w_m") not in (None, "") and not np.isclose(_number(row, "intensity_integral_w_m"), moment1, rtol=1e-6, atol=0):
        raise ValueError("intensity_integral_w_m disagrees with imported P_abs / simulation alpha.")
    moment2 = None
    if row.get("intensity_squared_integral_w2_per_m") not in (None, ""):
        moment2 = _number(row, "intensity_squared_integral_w2_per_m")
        lower_bound = moment1**2 / particle["volume_m3"]
        if moment2 < 0 or moment2 < lower_bound * (1 - 1e-6):
            raise ValueError("Squared field moment violates integral(I^2 dV) >= integral(I dV)^2 / volume.")
    wavelength = float(config["material"]["wavelength_nm"]) * 1e-9
    beta = tpa_beta(T, wavelength, config["tpa"])
    if np.any(np.asarray(config["tpa"]["weights"]) != 0) and moment2 is None:
        raise ValueError("Enabled TPA requires intensity_squared_integral_w2_per_m from the optical simulation; it cannot be inferred from absorbed power.")
    fca = drude_alpha(T, wavelength, prepared["n_laser"], config["material"]["reference_temperature_k"], config["fca"])
    bga = bandgap_alpha(T, wavelength, config["bga"])
    scale = intensity / intensity_sim
    linear = np.full_like(T, particle["p_abs_W"] * scale)
    tpa = np.zeros_like(T) if moment2 is None else beta * moment2 * scale**2
    free_carrier = fca["alpha_m_inv"] * moment1 * scale
    bandgap = bga * moment1 * scale
    prepared.update(particle=particle, coefficients={"beta_m_w": beta,
                    "alpha_fca_m_inv": fca["alpha_m_inv"], "alpha_bga_m_inv": bga,
                    "density_m3": fca["density_m3"], "damping_s_inv": fca["damping_s_inv"]},
                    optical_notes=notes, field_integral_w_m=moment1,
                    field_squared_integral_w2_m=moment2)
    return {"temperature_k": T, "p_linear_w": linear, "p_tpa_w": tpa,
            "p_fca_w": free_carrier, "p_bga_w": bandgap,
            "p_abs_w": linear + tpa + free_carrier + bandgap}
