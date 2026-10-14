"""Loss of the cold stable thermal branch; a fold does not prove T -> infinity."""
import numpy as np
from scipy.interpolate import PchipInterpolator, PPoly
from scipy.optimize import brentq


def _roots(interpolant, lower, upper):
    """List distinct real roots inside the available temperature interval."""
    # Permit only roundoff beyond the endpoint, e.g. a boundary zero at T+1 ulp.
    roots = interpolant.roots(extrapolate=True)
    roots = np.sort(roots[np.isfinite(roots) & (roots > lower)
                         & (roots <= upper + 8*np.spacing(upper))])
    roots = np.minimum(roots, upper)
    return roots[np.r_[True, np.diff(roots) > 1e-8]] if len(roots) else roots


def _first_fold(stationarity, bath, upper):
    """Find the first negative-to-positive crossing of (T-Tbath) Q' - Q."""
    roots = _roots(stationarity, bath, upper)
    roots = roots[roots < upper]  # A domain endpoint cannot establish a two-sided fold.
    boundaries = np.r_[bath, roots, upper]
    for index, root in enumerate(roots):
        left = (boundaries[index] + root) / 2
        right = (root + boundaries[index + 2]) / 2
        if stationarity(left) < 0 < stationarity(right):
            return float(brentq(stationarity, left, right, xtol=1e-9))
    return None


def critical_pressure(temperature, absorbed, radiation, coefficient, bath, limit=None):
    """Follow equilibria from high pressure toward vacuum at fixed laser intensity.

    Q=P_abs-P_rad and p_eq=Q/[g(T-Tbath)]. The cold branch is stable while
    dp_eq/dT<0; its first positive minimum is the fold pressure. A later hot
    stable branch may exist, and behaviour above the supplied T range is unknown.
    An optional material temperature limit is reported separately from a fold.
    """
    temperature, absorbed, radiation = [np.asarray(value, dtype=float)
                                        for value in (temperature, absorbed, radiation)]
    if (temperature.ndim != 1 or len(temperature) < 3
            or absorbed.shape != temperature.shape or radiation.shape != temperature.shape
            or any(np.any(~np.isfinite(value)) for value in (temperature, absorbed, radiation))
            or np.any(np.diff(temperature) <= 0)):
        raise ValueError("Power curves need matching finite arrays and increasing temperatures")
    if (not np.isfinite(coefficient) or coefficient <= 0 or not np.isfinite(bath)
            or not temperature[0] <= bath < temperature[-1]):
        raise ValueError("Gas coefficient must be positive and bath must lie inside the T domain")
    upper = float(temperature[-1] if limit is None else limit)
    if not np.isfinite(upper) or not bath < upper <= temperature[-1]:
        raise ValueError("Temperature limit must be above the bath and inside the T domain")
    net = PchipInterpolator(temperature, absorbed - radiation, extrapolate=False)
    base = {"critical_pressure_mbar": None, "critical_temperature_k": None}
    if net(bath) <= 0:
        return {**base, "status": "no_heating", "critical_pressure_mbar": 0.0}

    # In each interval x=T-Tj: Q=a*x^3+b*x^2+c*x+d. H=(x+s)*Q'-Q.
    a, b, c, d = net.c
    offset = net.x[:-1] - bath
    coefficients = np.array([2*a, 3*a*offset+b, 2*b*offset, c*offset-d])
    stationarity = PPoly(coefficients, net.x, extrapolate=False)
    fold = _first_fold(stationarity, bath, upper)
    zeros = _roots(net, bath, upper)
    vacuum = float(zeros[0]) if len(zeros) else None
    if vacuum is not None and (fold is None or vacuum <= fold):
        return {**base, "status": "stable_to_vacuum", "critical_pressure_mbar": 0.0,
                "vacuum_temperature_k": vacuum}
    if fold is not None:
        pressure = float(net(fold) / (coefficient * (fold - bath)))
        return {"status": "cold_branch_fold", "critical_pressure_mbar": pressure,
                "critical_temperature_k": fold}
    if limit is not None:
        return {**base, "status": "temperature_limit_crossing",
                "threshold_pressure_mbar": float(net(upper) / (coefficient * (upper - bath))),
                "threshold_temperature_k": upper}
    return {**base, "status": "unresolved", "temperature_max_k": upper}
