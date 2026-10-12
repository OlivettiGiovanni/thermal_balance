"""Laser heating, free-molecular gas cooling and dipole-spheroid radiation (SI)."""
import numpy as np
from scipy.constants import atomic_mass, c, h, k
from scipy.integrate import trapezoid


def cylinder_volume(radius_nm, height_nm):
    """Cylinder volume in m³."""
    return np.pi * (radius_nm * 1e-9)**2 * height_nm * 1e-9


def cylinder_area(radius_nm, height_nm):
    """Exposed cylinder area in m², including both ends."""
    return 2 * np.pi * radius_nm * (radius_nm + height_nm) * 1e-18


def absorption_terms(temperature_k, linear_power_w):
    """Named heating contributions in W; the imported linear term is fixed in T."""
    temperature = np.asarray(temperature_k, dtype=float)
    terms = {"linear": np.full_like(temperature, linear_power_w)}
    # Add future terms here, e.g. terms["free_carrier"] = free_carrier_power(temperature).
    return terms


def absorbed_power(temperature_k, linear_power_w):
    """Sum every implemented heating contribution."""
    return np.sum(list(absorption_terms(temperature_k, linear_power_w).values()), axis=0)


def gas_coefficient(radius_nm, height_nm, bath_k, mass_amu, gamma, accommodation):
    """Free-molecular gas coefficient in W/(mbar K); properties are fixed."""
    speed = np.sqrt(8 * k * bath_k / (np.pi * mass_amu * atomic_mass))
    area = cylinder_area(radius_nm, height_nm)
    return accommodation * area * 100 * speed / (8 * bath_k) * (gamma + 1) / (gamma - 1)


def gas_power(temperature_k, pressure_mbar, coefficient, bath_k):
    """Net gas cooling in W; 100 Pa/mbar is already included in coefficient."""
    return coefficient * pressure_mbar * (np.asarray(temperature_k) - bath_k)


def planck_lambda(wavelength_m, temperature_k):
    """Planck radiance in W m⁻³ sr⁻¹; wavelength occupies the final axis."""
    temperature = np.asarray(temperature_k, dtype=float)
    exponent = h * c / (wavelength_m * k * temperature[..., None])
    occupation = np.exp(-exponent) / -np.expm1(-exponent)  # Stable in the Wien tail.
    return 2 * h * c**2 / wavelength_m**5 * occupation


def depolarization_factors(aspect):
    """Parallel and perpendicular factors for spheroid c/a = aspect."""
    delta = aspect - 1
    if abs(delta) < 1e-4:
        parallel = 1 / 3 - 4 * delta / 15 + 6 * delta**2 / 35  # Smooth sphere limit.
    elif aspect > 1:
        eccentricity = np.sqrt(1 - aspect**-2)
        parallel = (1 - eccentricity**2) / (2 * eccentricity**3) * (
            np.log1p(eccentricity) - np.log1p(-eccentricity) - 2 * eccentricity)
    else:
        eccentricity = np.sqrt(aspect**-2 - 1)
        parallel = (1 + eccentricity**2) / eccentricity**3 * (
            eccentricity - np.arctan(eccentricity))
    return np.array([parallel, (1 - parallel) / 2])


def optical_indices(temperature_k, material):
    """Interpolate n,k at particle temperature; never extrapolate measured T data."""
    temperature = np.asarray(temperature_k, dtype=float)
    samples = material["temperature_k"]
    if samples is None:
        shape = temperature.shape + (len(material["wavelength_m"]),)
        return tuple(np.broadcast_to(material[name][0], shape) for name in ("n", "k"))
    if np.any(temperature < samples[0]) or np.any(temperature > samples[-1]):
        raise ValueError(f"Optical temperatures must lie within {samples[0]:g}–{samples[-1]:g} K")
    index = np.clip(np.searchsorted(samples, temperature) - 1, 0, len(samples) - 2)
    weight = ((temperature - samples[index]) / (samples[index + 1] - samples[index]))[..., None]
    return tuple((1 - weight) * material[name][index] + weight * material[name][index + 1]
                 for name in ("n", "k"))


def spheroid_cross_section(wavelength_m, n, extinction, radius_nm, height_nm):
    """Direction/polarization-averaged dipole absorption in m²; alpha uses m³."""
    epsilon = (n + 1j * extinction)**2
    factors = depolarization_factors(height_nm / (2 * radius_nm))
    wavevector = 2 * np.pi / wavelength_m
    alpha = cylinder_volume(radius_nm, height_nm) * (epsilon[..., None] - 1) / (
        1 + factors * (epsilon[..., None] - 1))
    denominator = np.abs(1 - 1j * wavevector[:, None]**3 * alpha / (6 * np.pi))**2
    absorption = wavevector[:, None] * alpha.imag / denominator  # Radiation reaction; lossless => zero.
    return (absorption[..., 0] + 2 * absorption[..., 1]) / 3


def radiation_power(temperature_k, material, radius_nm, height_nm, bath_k):
    """Net radiative cooling in W, integrated only over the supplied spectral band."""
    wavelength = material["wavelength_m"]
    index, extinction = optical_indices(temperature_k, material)
    cross_section = spheroid_cross_section(wavelength, index, extinction, radius_nm, height_nm)
    # Both emission and incoming-bath absorption use the particle's CURRENT optical properties.
    difference = planck_lambda(wavelength, temperature_k) - planck_lambda(wavelength, bath_k)
    return 4 * np.pi * trapezoid(cross_section * difference, wavelength, axis=-1)


def critical_pressure(heating_w, radiation_w, coefficient, bath_k, limit_k):
    """Signed balance pressure in mbar at T_limit; a negative value is unphysical."""
    if coefficient <= 0 or limit_k <= bath_k:
        raise ValueError("Critical pressure requires positive gas coupling and T_limit > T_bath")
    return float((heating_w - radiation_w) / (coefficient * (limit_k - bath_k)))


def emission_fraction_outside_dipole(temperature_k, material, radius_nm, height_nm, limit=0.3):
    """Emitted-power fraction failing a heuristic dipole-size test; not an error bound."""
    wavelength = material["wavelength_m"]
    index, extinction = optical_indices(temperature_k, material)
    cross_section = spheroid_cross_section(wavelength, index, extinction, radius_nm, height_nm)
    semiaxis = 1.5**(1 / 3) * max(radius_nm, height_nm / 2) * 1e-9  # Match V and aspect.
    size = 2 * np.pi * semiaxis / wavelength * np.maximum(1, np.abs(index + 1j * extinction))
    emission = cross_section * planck_lambda(wavelength, temperature_k)
    total = trapezoid(emission, wavelength, axis=-1)
    suspect = trapezoid(emission * (size > limit), wavelength, axis=-1)
    return np.divide(suspect, total, out=np.zeros_like(total), where=total > 0)
