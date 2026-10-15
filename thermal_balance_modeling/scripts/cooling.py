"""Free-molecular gas cooling and spectral thermal radiation of a cylinder."""

import numpy as np
from scipy.constants import atomic_mass, c, h, k as K_B
from scipy.integrate import trapezoid


def cylinder_area(radius_nm, height_nm):
    """Cylinder sidewall plus both end caps, in square metres."""
    radius, height = float(radius_nm) * 1e-9, float(height_nm) * 1e-9
    if not np.all(np.isfinite([radius, height])) or min(radius, height) <= 0:
        raise ValueError("Cylinder radius and height must be finite and positive.")
    return 2 * np.pi * radius * (radius + height)


def gas_coefficient(area_m2, bath_k, gas):
    """Free-molecular conductance G in W/(mbar K), evaluated at the bath."""
    area, bath = float(area_m2), float(bath_k)
    mass = float(gas["mass_amu"]) * atomic_mass
    gamma = float(gas["gamma"])
    accommodation = float(gas["accommodation"])
    if not np.all(np.isfinite([area, bath, mass, gamma, accommodation])) or min(area, bath, mass) <= 0 or gamma <= 1 or not 0 <= accommodation <= 1:
        raise ValueError("Gas needs positive area, bath, mass, gamma > 1, and accommodation in [0,1].")
    mean_speed = np.sqrt(8 * K_B * bath / (np.pi * mass))
    return accommodation * area * 100 * mean_speed / (8 * bath) * (gamma + 1) / (gamma - 1)


def gas_power(T, p_mbar, bath, G):
    """Net power into a fixed-temperature gas bath; 1 mbar = 100 Pa."""
    T = np.asarray(T, dtype=float)
    p_mbar, bath, G = float(p_mbar), float(bath), float(G)
    if np.any(~np.isfinite(T)) or np.any(T <= 0) or not np.all(np.isfinite([p_mbar, bath, G])) or p_mbar < 0 or bath <= 0 or G < 0:
        raise ValueError("Gas temperatures must be positive; pressure and conductance must be nonnegative.")
    return G * p_mbar * (T - bath)


def _depolarization(aspect):
    """Spheroid axial and transverse factors for aspect = cylinder H/(2R)."""
    if abs(aspect - 1) < 1e-6:
        axial = 1 / 3
    elif aspect > 1:
        eccentricity = np.sqrt(1 - aspect**-2)
        axial = (1 - eccentricity**2) / (2 * eccentricity**3) * (np.log1p(eccentricity) - np.log1p(-eccentricity) - 2 * eccentricity)
    else:
        eccentricity = np.sqrt(aspect**-2 - 1)
        axial = (1 + eccentricity**2) / eccentricity**3 * (eccentricity - np.arctan(eccentricity))
    return axial, (1 - axial) / 2


def dipole_cross_section(wavelength_m, n, k, volume_m3, depolarization_factors):
    """Orientation-averaged electric-dipole absorption in square metres.

    Supply the particle's actual volume and three positive depolarization
    factors summing to one. Radiation reaction is included; scattering is
    excluded from absorbed power. A sphere uses [1/3, 1/3, 1/3].
    """
    wavelength, n, k = np.broadcast_arrays(np.asarray(wavelength_m, dtype=float),
                                          np.asarray(n, dtype=float), np.asarray(k, dtype=float))
    if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0) or np.any(~np.isfinite(n)) or np.any(n < 0) or np.any(~np.isfinite(k)) or np.any(k < 0):
        raise ValueError("Wavelengths must be positive and optical constants finite and nonnegative.")
    volume = float(volume_m3)
    factors = np.asarray(depolarization_factors, dtype=float)
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("Dipole volume must be finite and positive.")
    if factors.shape != (3,) or np.any(~np.isfinite(factors)) or np.any(factors <= 0) or not np.isclose(factors.sum(), 1., rtol=1e-10, atol=1e-12):
        raise ValueError("Three positive depolarization factors summing to one are required.")
    contrast = (n + 1j * k)**2 - 1
    wavenumber = 2 * np.pi / wavelength
    # Algebraically k Im(a)/|1-i k^3 a/(6pi)|^2, a=V contrast/(1+L contrast).
    # This form stays finite at an electrostatic pole and excludes scattering.
    reaction = 1j * wavenumber**3 * volume * contrast / (6 * np.pi)
    absorption = sum(wavenumber * volume * contrast.imag / abs(1 + factor * contrast - reaction)**2
                     for factor in factors)
    return np.maximum(absorption / 3, 0)


def cylinder_cross_section(wavelength_m, n, k, radius_nm, height_nm):
    """Dipole approximation using actual cylinder volume and spheroid shape factors.

    The spheroid factors use aspect H/(2R); this is not an exact cylinder
    Maxwell solution. Radiation reaction and scattering subtraction are shared
    with dipole_cross_section.
    """
    cylinder_area(radius_nm, height_nm)  # Share the geometry validation.
    radius, height = float(radius_nm) * 1e-9, float(height_nm) * 1e-9
    axial, transverse = _depolarization(height / (2 * radius))
    return dipole_cross_section(wavelength_m, n, k, np.pi * radius**2 * height,
                                [axial, transverse, transverse])


def _planck(wavelength_m, T):
    """Planck spectral radiance in W m^-3 sr^-1 without exponential overflow."""
    x = h * c / (wavelength_m * K_B * T[..., None])
    occupation = np.exp(-x) / -np.expm1(-x)
    return 2 * h * c**2 / wavelength_m**5 * occupation


def radiation_power(T, bath, wavelength_m, cross_section_m2):
    """Net radiation with C_abs(lambda) or C_abs(T,lambda).

    The particle's cross section at its current temperature multiplies both
    emission and absorbed bath radiation: 4 pi integral C_abs(T) [B(T)-B(bath)].
    """
    T = np.asarray(T, dtype=float)
    wavelength = np.asarray(wavelength_m, dtype=float)
    cross_section = np.asarray(cross_section_m2, dtype=float)
    bath = float(bath)
    if np.any(~np.isfinite(T)) or np.any(T <= 0) or not np.isfinite(bath) or bath <= 0:
        raise ValueError("Radiation temperatures must be finite and positive.")
    if wavelength.ndim != 1 or wavelength.size < 2 or np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0) or np.any(np.diff(wavelength) <= 0):
        raise ValueError("Radiation wavelengths must be a strictly increasing positive 1D grid.")
    if cross_section.shape not in (wavelength.shape, T.shape + wavelength.shape) or np.any(~np.isfinite(cross_section)) or np.any(cross_section < 0):
        raise ValueError("Cross sections must be finite, nonnegative, and have shape (wavelength,) or temperature.shape + (wavelength,).")
    radiance = _planck(wavelength, T) - _planck(wavelength, np.asarray(bath))
    return 4 * np.pi * trapezoid(cross_section * radiance, wavelength, axis=-1)


def _resample_spectrum(grid, wavelength, spectrum):
    """Interpolate only the last (wavelength) axis, preserving temperature axes."""
    spectrum = np.asarray(spectrum, dtype=float)
    rows = spectrum.reshape(-1, len(wavelength))
    return np.asarray([np.interp(grid, wavelength, row) for row in rows]).reshape(spectrum.shape[:-1] + grid.shape)


def cooling_powers(prepared):
    """Evaluate gas and spectral dipole cooling for the selected optical particle."""
    config, particle = prepared["config"], prepared["particle"]
    bath = float(config["experiment"]["bath_temperature_k"])
    pressure = float(config["experiment"]["reference_pressure_mbar"])
    area = cylinder_area(particle["radius_nm"], particle["height_nm"])
    G = gas_coefficient(area, bath, config["gas"])
    wavelength = np.asarray(prepared["wavelength_m"], dtype=float)
    temperature = np.asarray(prepared["temperature_k"], dtype=float)
    n = np.asarray(prepared.get("n_radiation", prepared["n_spectrum"]))
    k = np.asarray(prepared.get("k_radiation", prepared["k_spectrum"]))
    limits = np.asarray(config["radiation"]["wavelength_range_um"], dtype=float) * 1e-6
    allowed = (wavelength.shape, temperature.shape + wavelength.shape)
    if wavelength.ndim != 1 or len(wavelength) < 2 or np.any(~np.isfinite(wavelength)) or np.any(np.diff(wavelength) <= 0) or n.shape not in allowed or k.shape not in allowed:
        raise ValueError("Prepared optical spectra must share a strictly increasing wavelength grid.")
    if limits.shape != (2,) or np.any(~np.isfinite(limits)) or limits[0] <= 0 or limits[1] <= limits[0]:
        raise ValueError("Radiation wavelength_range_um must be [positive minimum, larger maximum].")
    if limits[0] < wavelength[0] * (1 - 1e-10) or limits[1] > wavelength[-1] * (1 + 1e-10):
        raise ValueError("Optical spectrum does not cover the full configured radiation wavelength range.")
    grid = np.r_[limits[0], wavelength[(wavelength > limits[0]) & (wavelength < limits[1])], limits[1]]
    cross_section = cylinder_cross_section(grid, _resample_spectrum(grid, wavelength, n),
                                          _resample_spectrum(grid, wavelength, k),
                                          particle["radius_nm"], particle["height_nm"])
    prepared["gas_coefficient_w_mbar_k"] = G
    prepared["radiation_cross_section_m2"] = cross_section
    prepared["radiation_wavelength_m"] = grid
    return {"p_gas_w": gas_power(temperature, pressure, bath, G),
            "p_bb_w": radiation_power(temperature, bath, grid, cross_section)}
