"""Combine the separate powers and examine equilibria within the calculated range."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from absorption import absorption_powers
from cooling import cooling_powers


def find_equilibria(temperature, net, break_before=None):
    """An equilibrium is stable when a small temperature increase gives net cooling."""
    if break_before is not None and np.any(break_before):
        # A return to room optical data is a data boundary, not a physical zero.
        groups = np.split(np.arange(len(temperature)), np.flatnonzero(break_before))
        return [root for group in groups if len(group) > 1
                for root in find_equilibria(temperature[group], net[group])]
    curve = PchipInterpolator(temperature, net, extrapolate=False)
    roots = curve.roots(extrapolate=False)
    roots = np.unique(roots[np.isfinite(roots)])
    return [{"temperature_k": float(t), "stable": bool(curve.derivative()(t) < 0)}
            for t in roots if temperature[0] <= t <= temperature[-1]]


def pressure_report(temperature, absorbed, radiated, coefficient, bath, limit=None, break_before=None):
    """Distinguish balance at a chosen temperature from loss of the cold stable branch."""
    if break_before is not None and np.any(break_before):
        report = {"cold_branch_loss": None, "status": "unresolved_optical_fallback",
                  "search_range_k": [float(temperature[0]), float(temperature[-1])],
                  "optical_breaks_before_k": temperature[break_before].tolist(),
                  "note": "Optical data return to reference values at coverage boundaries; a continuous cold branch cannot be established across these jumps."}
        if limit is not None:
            groups = np.split(np.arange(len(temperature)), np.flatnonzero(break_before))
            group = next((g for g in groups if len(g) > 1 and temperature[g[0]] <= limit <= temperature[g[-1]]), None)
            value = None if group is None else float(PchipInterpolator(temperature[group], (absorbed-radiated)[group])(limit)/(coefficient*(limit-bath)))
            report["temperature_limit"] = {"temperature_k": float(limit),
                "balance_pressure_mbar": value, "on_cold_stable_branch": None}
        return report
    hot = temperature > bath
    t = temperature[hot]
    p = (absorbed[hot]-radiated[hot])/(coefficient*(t-bath))
    curve = PchipInterpolator(t, p, extrapolate=False)
    slope = curve.derivative()
    zeroes = curve.roots(extrapolate=False)
    zeroes = zeroes[np.isfinite(zeroes) & (zeroes >= t[0])]
    first_vacuum = min(zeroes, default=np.inf)
    turning_points = slope.roots(extrapolate=False)
    fold = None
    for root in sorted(turning_points[np.isfinite(turning_points)]):
        step = min(.01, (root-t[0])/2, (t[-1]-root)/2)
        if step > 0 and root < first_vacuum and curve(root) > 0:
            if slope(root-step) < 0 < slope(root+step):
                fold = {"temperature_k": float(root), "pressure_mbar": float(curve(root))}
                break  # First minimum reached when reducing pressure from the cold branch.
    report = {"cold_branch_loss": fold,
              "search_range_k": [float(temperature[0]), float(temperature[-1])],
              "note": "No extrapolation: null means no branch loss found in this interval."}
    if limit is not None:
        heating = float(PchipInterpolator(temperature, absorbed-radiated)(limit))
        pressure = heating/(coefficient*(limit-bath))
        report["temperature_limit"] = {
            "temperature_k": float(limit), "balance_pressure_mbar": pressure,
            "on_cold_stable_branch": bool(pressure >= 0 and limit <= first_vacuum and slope(limit) < 0
                and (fold is None or limit <= fold["temperature_k"]))}
        if pressure < 0:
            report["temperature_limit"]["note"] = "At this temperature radiation already exceeds absorption in vacuum."
    return report


def optical_temperature_estimate(prepared, powers):
    """A separate k(T) estimate, never added to the modeled heating contributions."""
    observed = prepared["k_measured"]
    hot = prepared["temperature_k"] > prepared["config"]["material"]["reference_temperature_k"]
    reference = prepared["k_reference"]
    if np.any(powers["p_tpa_w"] > 0) or np.count_nonzero(observed & hot) < 2:
        return {}  # Ordinary low-intensity optical tables cannot supply a TPA measurement.
    if not np.isfinite(reference) or reference <= 0:
        return {}
    # Fixed local intensity: alpha(T)/alpha(T0) = k(T)/k(T0) at the same wavelength.
    estimate = powers["p_linear_w"]*prepared["k_laser"]/reference
    return {"p_optical_estimate_w": np.where(observed, estimate, np.nan)}


def solve_balance(prepared):
    """Return final plotting arrays and numerical diagnostics; no files are written here."""
    powers = absorption_powers(prepared)
    powers.update(cooling_powers(prepared))
    powers.update(optical_temperature_estimate(prepared, powers))
    powers["p_net_w"] = powers["p_abs_w"]-powers["p_gas_w"]-powers["p_bb_w"]
    for key, values in powers.items():
        if key != "p_optical_estimate_w" and not np.all(np.isfinite(values)):
            raise ValueError(f"Nonfinite calculated values: {key}")
    experiment = prepared["config"]["experiment"]
    t = powers["temperature_k"]
    breaks = np.asarray(prepared.get("radiation_break_before", np.zeros(len(t), dtype=bool))) | np.asarray(prepared.get("n_laser_break_before", np.zeros(len(t), dtype=bool)))
    report = {"equilibria": find_equilibria(t, powers["p_net_w"], breaks),
              "pressure": pressure_report(t, powers["p_abs_w"], powers["p_bb_w"],
                  prepared["gas_coefficient_w_mbar_k"], experiment["bath_temperature_k"],
                  experiment.get("decomposition_temperature_k"), breaks)}
    if np.any(breaks):
        report["equilibria_note"] = "Searched each continuous optical-data interval separately; roots across fallback jumps are excluded."
    # Keep plot annotations in the saved results, after checking numerical powers.
    powers["radiation_footnote"] = np.array(prepared.get("radiation_footnote", ""))
    powers["radiation_break_before"] = np.asarray(prepared.get("radiation_break_before", np.zeros(len(t), dtype=bool)))
    return powers, report
