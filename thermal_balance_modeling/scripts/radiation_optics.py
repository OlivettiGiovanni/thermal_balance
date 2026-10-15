"""Optional thermal-radiation optics with two fitted temperature parameters.

All carrier parameters are explicit material inputs. This module neither changes
laser heating nor supplies default semiconductor properties for other materials.
The positive-imaginary convention is epsilon = (n + i*k)**2.
"""

import numpy as np
from scipy.constants import c, e, epsilon_0, h, k as K_B, m_e


def _positive(value, name):
    """Validate a finite, positive scalar or array without modifying its input."""
    value = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(value)) or np.any(value <= 0):
        raise ValueError(f"{name} must be finite and positive.")
    return value


def temperature_scale(T, p, theta, T0=300.0, T1=500.0):
    """Anchored empirical growth: [f(T)-1]/[f(T1)-1], theta in kelvin.

    f(T)=(T/T0)**p * exp(theta*(1/T0-1/T)). Logarithmic evaluation avoids
    overflow in f itself; an unrepresentable final extrapolation is rejected.
    """
    T = _positive(T, "Temperature")
    T0, T1 = float(_positive(T0, "T0")), float(_positive(T1, "T1"))
    p, theta = float(p), float(theta)
    if not np.all(np.isfinite([p, theta])) or min(p, theta) < 0 or p + theta == 0 or T1 <= T0:
        raise ValueError("Require p, theta >= 0 (not both zero), and T1 > T0.")
    x = p * np.log(T / T0) + theta * (1 / T0 - 1 / T)
    x1 = p * np.log(T1 / T0) + theta * (1 / T0 - 1 / T1)
    if np.any(~np.isfinite(x)) or not np.isfinite(x1) or x1 <= 0:
        raise ValueError("Temperature scaling exponent is not representable.")
    # log|exp(x)-1|, evaluated separately on either side of zero.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        log_numerator = np.where(x > 0, x + np.log(-np.expm1(-np.maximum(x, 0))),
                                 np.log(-np.expm1(np.minimum(x, 0))))
        log_denominator = x1 + np.log(-np.expm1(-x1))
        growth = np.sign(x) * np.exp(log_numerator - log_denominator)
    if np.any(~np.isfinite(growth)):
        raise ValueError("Temperature extrapolation exceeds floating-point range.")
    return np.where(T == T0, 0., np.where(T == T1, 1., growth))


def carrier_densities(T, params):
    """Equilibrium [electrons, holes] in m^-3, with fully ionized net doping.

    DOS masses must include valley degeneracy consistently. The Varshni gap
    parameters are in eV, eV/K and K; net_doping_m3 is signed Nd-Na.
    This is a nondegenerate equilibrium model, not a photo-carrier solver.
    """
    T = _positive(T, "Temperature")
    masses = _positive(params["dos_masses_m0"], "DOS masses")
    if masses.shape != (2,):
        raise ValueError("Supply two DOS masses: electron and hole, in units of m0.")
    gap0, slope, scale, doping = (float(params[key]) for key in
        ("gap0_ev", "gap_slope_ev_k", "gap_temperature_k", "net_doping_m3"))
    if not np.all(np.isfinite([gap0, slope, scale, doping])) or min(gap0, slope, scale) < 0:
        raise ValueError("Gap parameters must be finite and nonnegative; net doping must be finite.")
    gap = gap0 - slope * T**2 / (T + scale)
    if np.any(gap <= 0) or np.any(~np.isfinite(gap)):
        raise ValueError("The supplied Varshni model gives a nonpositive or nonfinite band gap.")
    states = 2 * (2 * np.pi * masses * m_e * K_B * T[..., None] / h**2)**1.5
    intrinsic = np.sqrt(states[..., 0]) * np.sqrt(states[..., 1]) * np.exp(-gap * e / (2 * K_B * T))
    majority = np.hypot(abs(doping) / 2, intrinsic) + abs(doping) / 2
    minority = intrinsic * np.divide(intrinsic, majority, out=np.zeros_like(intrinsic), where=majority > 0)
    density = np.stack((majority, minority) if doping >= 0 else (minority, majority), axis=-1)
    if np.any(~np.isfinite(density)):
        raise ValueError("Carrier density exceeds the model's numerical range.")
    return density


def drude_permittivity(T, wavelength_m, params, T0=300.0):
    """Complex Drude susceptibility with shape T.shape + (wavelength_count,).

    Relaxation-time terms follow the existing absorption model:
    tau_s(T) = sum_j tau_js(T0)*(T/T0)**p_js, gamma_s=1/tau_s.
    Times and exponents are [term][electron,hole] arrays; masses are in m0.
    """
    T = _positive(T, "Temperature")
    T0 = float(_positive(T0, "T0"))
    wavelength = _positive(wavelength_m, "Wavelength")
    masses = _positive(params["conductivity_masses_m0"], "Conductivity masses")
    times = np.asarray(params["relaxation_times_s"], dtype=float)
    powers = np.asarray(params["temperature_exponents"], dtype=float)
    if wavelength.ndim != 1 or not len(wavelength) or masses.shape != (2,):
        raise ValueError("Supply a 1D wavelength array and two conductivity masses.")
    if times.ndim != 2 or times.shape[1] != 2 or not times.shape[0] or powers.shape != times.shape or np.any(~np.isfinite(times)) or np.any(times < 0) or np.any(~np.isfinite(powers)):
        raise ValueError("Times and exponents must be matching [term][electron,hole] arrays, with nonnegative times.")
    with np.errstate(over="ignore", invalid="ignore"):
        tau = np.sum(times * (T[..., None, None] / T0)**powers, axis=-2)
    tau = _positive(tau, "Total relaxation time")
    plasma_squared = carrier_densities(T, params) * e**2 / (epsilon_0 * masses * m_e)
    omega = 2 * np.pi * c / wavelength
    denominator = omega[:, None] * (omega[:, None] + 1j / tau[..., None, :])
    result = -np.sum(plasma_squared[..., None, :] / denominator, axis=-1)
    if np.any(~np.isfinite(result)):
        raise ValueError("Drude response exceeds the model's numerical range.")
    return result


def epsilon_to_nk(epsilon):
    """Passive principal square root, retaining extremely weak absorption.

    No absolute value or clipping repairs negative loss: an extrapolation with
    epsilon.imag < 0 is rejected, including small negative values.
    """
    epsilon = np.asarray(epsilon, dtype=complex)
    if np.any(~np.isfinite(epsilon)) or np.any(epsilon.imag < 0):
        raise ValueError("Permittivity must be finite and passive (imaginary part >= 0).")
    optical = np.sqrt(epsilon)
    return optical.real, optical.imag


def extrapolate_permittivity(T, epsilon0, epsilon1, drude_T, drude0, drude1,
                            p, theta, T0=300.0, T1=500.0):
    """Anchor both spectra and evolve their non-Drude difference with g(T).

    Baseline spectra have one wavelength axis. Drude arrays broadcast to that
    axis or to T.shape + (wavelength_count,). Use epsilon_to_nk to reject
    extrapolated gain before using the result for radiation.
    """
    T = _positive(T, "Temperature")
    epsilon0, epsilon1 = np.asarray(epsilon0, complex), np.asarray(epsilon1, complex)
    if epsilon0.ndim != 1 or not len(epsilon0) or epsilon1.shape != epsilon0.shape:
        raise ValueError("Both anchor spectra must share one nonempty wavelength axis.")
    shape = T.shape + epsilon0.shape
    d0, d1 = np.broadcast_to(np.asarray(drude0, complex), epsilon0.shape), np.broadcast_to(np.asarray(drude1, complex), epsilon0.shape)
    dT = np.broadcast_to(np.asarray(drude_T, complex), shape)
    if any(np.any(~np.isfinite(array)) for array in (epsilon0, epsilon1, d0, d1, dT)):
        raise ValueError("All anchor and Drude spectra must be finite.")
    remainder = epsilon1 - epsilon0 - (d1 - d0)
    epsilon = epsilon0 + temperature_scale(T, p, theta, T0, T1)[..., None] * remainder + dT - d0
    # Preserve the data exactly at either anchor despite cancellation roundoff.
    epsilon = np.where((T == T0)[..., None], epsilon0, epsilon)
    return np.where((T == T1)[..., None], epsilon1, epsilon)
