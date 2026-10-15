"""Fixed Junnemann silicon-sphere benchmark using the shared production models.

Run from any working directory:
    python compare_Junneman.py --output /path/to/results --dpi 200

No user-material configuration is read. Mie scattering is confined to this
sphere benchmark; it is not the production cylinder absorption workflow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import miepython
import numpy as np
from scipy.constants import c, e, h, k as K_B
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str((HERE.parent / "scripts").resolve()))
from absorption import bandgap_alpha, drude_alpha, tpa_beta
from cooling import dipole_cross_section, gas_coefficient, gas_power, radiation_power
from plotting import plot_results
from prepare_inputs import _read_table, prepare_radiation_optics
from radiation_optics import drude_permittivity, epsilon_to_nk, extrapolate_permittivity


# This self-contained source case does not supply defaults to user materials.
RADIUS_M = 130e-9
WAVELENGTH_M = 1550e-9
INDEX = 3.48
INTENSITY_W_M2 = 0.1e12
BATH_K = 300.0
OPTICAL_REFERENCE_K = 300.0  # Frozen-optics reference, independent of particle temperature.
RADIATION_BAND_UM = (0.4, 100.0)  # Passive extrapolation band; the UV is excluded explicitly.
PRESSURE_MBAR = 0.1
ALPHA0_M_INV = 4.3e-4
OPTICAL_FILE = "si_franta_300_500k.csv"
SOURCE_PHOTON_ENERGY_EV = 0.797
SOURCE_ENERGY_WAVELENGTH_M = h*c/(e*SOURCE_PHOTON_ENERGY_EV)

TPA = {"weights": [2.33e-12, 2.138e-11], "phonon_temperatures_k": [212, 670],
       "gap0_ev": 1.156, "gap_slope_ev_k": 7.021e-4, "gap_temperature_k": 1108}
FCA = {"conductivity_masses_m0": [0.3, 0.4],
       "density_prefactors": [3.87e22, 3.87e22], "activation_ev": [0.605, 0.605],
       "relaxation_times_s": [[1.5010063493845713e-14, 1.2349188601754881e-14],
                              [2.1352024221262788e-13, 9.25544073276513e-14]],
       "temperature_exponents": [[-0.57, -0.57], [-2.33, -2.23]]}
BGA = {"sqrt_weights": [50.4, 1808], "quadratic_weights": [39200, 576000],
       "quadratic_offset_ev": 0.0055, "phonon_temperatures_k": [212, 670],
       "gap0_ev": 1.155, "gap_slope_ev_k": 4.73e-4, "gap_temperature_k": 635}
GAS = {"mass_amu": 28.965, "gamma": 1.4, "accommodation": 0.8}

# BB-only equilibrium carrier model: all numbers are fixed, never fitted.
# DOS masses and Varshni law: LIGO T1500619 (Green 1990 background).
# Optical masses and relaxation times retain the benchmark's Arora-based inputs.
# Intrinsic material is an explicit approximation; no sample doping is inferred.
BB_CARRIERS = {"dos_masses_m0": [1.08, 0.81], "gap0_ev": 1.166,
               "gap_slope_ev_k": 4.73e-4, "gap_temperature_k": 636.,
               "net_doping_m3": 0.,
               **{key: FCA[key] for key in ("conductivity_masses_m0",
                                           "relaxation_times_s", "temperature_exponents")}}



def sinclair_as_written_beta(temperature, photon_energy_ev=SOURCE_PHOTON_ENERGY_EV):
    """Local reference only: Sinclair's written Bose-factor/energy pairing.

    Existing semi-empirical weights are retained without refitting:
    Eg**1.5 * sum(w * [N*(2E-Eg-Eph)_+**2 + (N+1)*(2E-Eg+Eph)_+**2]).
    The shared production function uses the physical emission/absorption
    pairing, (N+1) at the higher threshold and N at the lower threshold.
    """
    temperature = np.asarray(temperature, dtype=float)
    gap = TPA["gap0_ev"] - TPA["gap_slope_ev_k"]*temperature**2/(temperature+TPA["gap_temperature_k"])
    theta = np.asarray(TPA["phonon_temperatures_k"])
    phonon_ev = K_B/e*theta
    occupation = 1/np.expm1(theta/temperature[..., None])
    excess = 2*photon_energy_ev-gap[..., None]
    return gap**1.5*np.sum(np.asarray(TPA["weights"])*(
        occupation*np.maximum(excess-phonon_ev, 0)**2
        +(occupation+1)*np.maximum(excess+phonon_ev, 0)**2), axis=-1)


def sturm_empirical_alpha(temperature, density):
    """Source comparison in m^-1, using the shared electron/hole densities.

    sigma_e=1.01e-12*T*lambda_m**2, sigma_h=0.51e-12*T*lambda_m**2
    are cross sections in m^2. This empirical formula is not the Drude model.
    """
    cross_section = np.asarray(temperature)[..., None]*WAVELENGTH_M**2*np.array([1.01e-12, 0.51e-12])
    return np.sum(cross_section*density, axis=-1)


def sphere_absorption(wavelength_m, n, extinction):
    """Full Mie absorption cross section; scattering does not count as heat."""
    wavelength, n, extinction = np.broadcast_arrays(wavelength_m, n, extinction)
    if np.any(extinction < 0):
        raise ValueError("A passive sphere requires nonnegative extinction k")
    # miepython accepts a flat array; restore the caller's temperature/wavelength shape.
    qext, qsca, _, _ = miepython.efficiencies((n-1j*extinction).ravel(), 2*RADIUS_M, wavelength.ravel())
    qabs = np.asarray(qext)-np.asarray(qsca)
    if np.any(qabs < -1e-12*np.maximum(1, np.abs(qext))):
        raise ValueError("Mie calculation returned appreciably negative absorption")
    return (np.pi*RADIUS_M**2*np.maximum(qabs, 0)).reshape(wavelength.shape)


def uniform_loss_power(alpha):
    """Paper-style alpha0+beta*Iincident+FCA+BGA inserted as uniform bulk k.

    This method recomputes Mie scattering for the effective loss; it does not
    integrate beta*I_local**2 over the nonlinear internal field.
    """
    return INTENSITY_W_M2*sphere_absorption(WAVELENGTH_M, INDEX, alpha*WAVELENGTH_M/(4*np.pi))


def sphere_dipole(wavelength, n, k):
    """Use the shared dipole approximation with the benchmark sphere geometry."""
    return dipole_cross_section(wavelength, n, k, 4*np.pi*RADIUS_M**3/3, [1/3]*3)


def thermal_mie_cross_sections(wavelength, n, k):
    """Evaluate full Mie at every temperature; small batches report progress."""
    cross_sections = np.empty_like(n)
    for start in range(0, len(n), 100):
        stop = min(start+100, len(n))
        cross_sections[start:stop] = sphere_absorption(wavelength, n[start:stop], k[start:stop])
        print(f"BB_Mie(T): {stop}/{len(n)} temperatures", flush=True)
    return cross_sections


def fitted_epsilon(temperature, optics, drude, parameters):
    """Evaluate the same anchored optical model inside and above the table range."""
    return extrapolate_permittivity(temperature, optics["epsilon"][0], optics["epsilon"][-1],
        drude, optics["drude"][0], optics["drude"][-1], *parameters,
        T0=optics["temperature_k"][0], T1=optics["temperature_k"][-1])


def fit_radiation(wavelength, optics):
    """Fit only p and theta to dipole BB at native, interior table temperatures."""
    temperature = optics["temperature_k"]
    reference = radiation_power(temperature, BATH_K, wavelength,
        sphere_dipole(wavelength, optics["n"], optics["k"]))
    if len(temperature) < 4 or np.any(reference[1:] <= 0):
        raise ValueError("BB fitting needs two anchors and at least two positive interior BB powers.")
    fit_t, target = temperature[1:-1], reference[1:-1]

    def residual(parameters):
        # Optimize theta/1000 for comparable numerical scales; it remains ONE parameter.
        parameters = [parameters[0], 1000*parameters[1]]
        epsilon = fitted_epsilon(fit_t, optics, optics["drude"][1:-1], parameters)
        if np.any(epsilon.imag < 0):
            return np.full(target.shape, 1000.)  # Reject gain rather than repairing k.
        n, k = epsilon_to_nk(epsilon)
        predicted = radiation_power(fit_t, BATH_K, wavelength, sphere_dipole(wavelength, n, k))
        return np.log(predicted/target)  # Equal relative weight at each actual temperature.

    bounds = ([1e-6, 0.], [8., 6.])
    trials = [least_squares(residual, start, bounds=bounds, max_nfev=500,
              ftol=1e-11, xtol=1e-11, gtol=1e-11)
              for start in ([1., 0.], [3., 0.], [1., 1.])]
    best = min(trials, key=lambda trial: trial.cost if trial.success else np.inf)
    if not best.success or np.any(abs(best.fun) >= 100):
        raise ValueError("No passive, converged two-parameter BB fit was found.")
    p, theta = float(best.x[0]), float(1000*best.x[1])
    if theta < 1e-6:
        theta = 0.  # Report an active zero bound without meaningless numerical digits.
    fitted = reference.copy()
    fitted[1:-1] *= np.exp(residual([p, theta/1000]))
    relative = fitted[1:]/reference[1:]-1
    report = {"p": p, "theta_k": theta, "fitted_parameter_count": 2,
        "anchor_temperatures_k": [float(temperature[0]), float(temperature[-1])],
        "fit_temperatures_k": fit_t.tolist(), "objective": "sum(log(P_model/P_Franta)**2)",
        "bounds": {"p": [1e-6, 8.], "theta_k": [0., 6000.]},
        "native_temperatures_k": temperature.tolist(), "franta_power_w": reference.tolist(),
        "fitted_power_w": fitted.tolist(), "relative_errors_above_bath": relative.tolist(),
        "maximum_absolute_relative_error": float(np.max(abs(relative))),
        "parameters_at_bound": [name for name, active in zip(("p", "theta_k"), best.active_mask) if active],
        "jacobian_condition_number": float(np.linalg.cond(best.jac)),
        "note": "These are fitted dispersion-table powers, not independent BB measurements; high-temperature extrapolation is unvalidated."}
    return (p, theta), reference, report


def prepare_case():
    """Prepare fixed optics, native Franta points, and the two-parameter Si extrapolation."""
    spectrum = _read_table(HERE/OPTICAL_FILE)
    source_wavelength = np.unique(spectrum[:, 1])*1e-6
    low, high = np.asarray(RADIATION_BAND_UM)*1e-6
    wavelength = np.r_[low, source_wavelength[(source_wavelength > low)
                                             & (source_wavelength < high)], high]
    native_t = np.unique(spectrum[:, 0])
    if native_t[0] != OPTICAL_REFERENCE_K or native_t[-1] != 500.:
        raise ValueError("The Si benchmark requires the documented 300 and 500 K Franta anchors.")
    optics = prepare_radiation_optics(spectrum, native_t, wavelength*1e6, OPTICAL_REFERENCE_K)
    if np.any(optics["n_room_fallback"]) or np.any(optics["k_room_fallback"]):
        raise ValueError("Every benchmark fitting spectrum must cover the whole BB integration band.")
    optics.update(temperature_k=native_t, epsilon=(optics["n"]+1j*optics["k"])**2,
                  drude=drude_permittivity(native_t, wavelength, BB_CARRIERS))
    parameters, reference, fit = fit_radiation(wavelength, optics)
    print(f"BB fit: p={parameters[0]:.6g}, theta={parameters[1]:.6g} K; "
          f"maximum native-point error={100*fit['maximum_absolute_relative_error']:.3g}%", flush=True)
    temperature = np.arange(300.0, 1601.0)
    epsilon = fitted_epsilon(temperature, optics,
        drude_permittivity(temperature, wavelength, BB_CARRIERS), parameters)
    n, k = epsilon_to_nk(epsilon)  # Check the WHOLE modeled band/range, not just the dots.
    fit["minimum_epsilon_imag"] = float(np.min(epsilon.imag))
    fit["evaluated_temperature_range_k"] = [float(temperature[0]), float(temperature[-1])]
    print("Calculating BB_Mie(T) with the same fitted optics (this may take a few minutes)...", flush=True)
    mie_hot = thermal_mie_cross_sections(wavelength, n, k)
    print("Calculating fixed-optics Mie radiation and heating curves...", flush=True)
    return {"temperature_k": temperature,
            "coefficient_reference": np.genfromtxt(HERE/"junnemann_fig4.csv", delimiter=",", names=True),
            "power_reference": np.genfromtxt(HERE/"junnemann_fig6.csv", delimiter=",", names=True),
            "radiation_wavelength_m": wavelength, "radiation_n": n, "radiation_k": k,
            "radiation_optics": optics, "radiation_fit": fit,
            "franta_temperature_k": native_t, "p_bb_franta_w": reference,
            "radiation_mie_cross_section_m2": mie_hot,
            "radiation_mie_room_cross_section_m2": sphere_absorption(wavelength, optics["n"][0], optics["k"][0]),
            "radiation_dipole_room_cross_section_m2": sphere_dipole(wavelength, optics["n"][0], optics["k"][0]),
            "radiation_dipole_cross_section_m2": sphere_dipole(wavelength, n, k),
            "radiation_footnote": "Optics extrapolated above 500 K.",
            "radiation_break_before": np.zeros(temperature.shape, dtype=bool),
            "gas_coefficient_w_mbar_k": gas_coefficient(4*np.pi*RADIUS_M**2, BATH_K, GAS)}


def calculate(prepared):
    temperature = prepared["temperature_k"]
    drude = drude_alpha(temperature, WAVELENGTH_M, INDEX, BATH_K, FCA)
    # Both TPA conventions use the same source-fit photon energy, isolating
    # their Bose-factor pairing. Physical 1550 nm still sets Drude and Mie.
    tpa = tpa_beta(temperature, SOURCE_ENERGY_WAVELENGTH_M, TPA)*INTENSITY_W_M2
    tpa_reference = sinclair_as_written_beta(temperature)*INTENSITY_W_M2
    fca_reference = sturm_empirical_alpha(temperature, drude["density_m3"])
    bga = bandgap_alpha(temperature, SOURCE_ENERGY_WAVELENGTH_M, BGA)
    r4, r6 = prepared["coefficient_reference"], prepared["power_reference"]
    dipole_hot = radiation_power(temperature, BATH_K, prepared["radiation_wavelength_m"],
                                prepared["radiation_dipole_cross_section_m2"])
    return {"kind": np.array("junnemann"), "temperature_k": temperature,
            "radiation_footnote": np.array(prepared["radiation_footnote"]),
            "radiation_break_before": prepared["radiation_break_before"],
            "tpa_model_m_inv": tpa, "tpa_reference_model_m_inv": tpa_reference,
            "fca_model_m_inv": drude["alpha_m_inv"], "fca_reference_model_m_inv": fca_reference,
            "bga_model_m_inv": bga,
            "carrier_density_m3": drude["density_m3"], "carrier_damping_s_inv": drude["damping_s_inv"],
            "p_abs_w": uniform_loss_power(ALPHA0_M_INV+tpa+drude["alpha_m_inv"]+bga),
            "p_linear_w": np.full_like(temperature, uniform_loss_power(np.array(ALPHA0_M_INV))),
            "p_gas_w": gas_power(temperature, PRESSURE_MBAR, BATH_K, prepared["gas_coefficient_w_mbar_k"]),
            "p_bb_w": dipole_hot,  # Compatibility alias: the selected BB model is now hot dipole.
            "p_bb_dipole_w": dipole_hot,
            "p_bb_dipole_room_w": radiation_power(temperature, BATH_K, prepared["radiation_wavelength_m"],
                                                  prepared["radiation_dipole_room_cross_section_m2"]),
            "p_bb_mie_w": radiation_power(temperature, BATH_K, prepared["radiation_wavelength_m"],
                                          prepared["radiation_mie_cross_section_m2"]),
            "p_bb_mie_room_w": radiation_power(temperature, BATH_K, prepared["radiation_wavelength_m"],
                                               prepared["radiation_mie_room_cross_section_m2"]),
            "franta_temperature_k": prepared["franta_temperature_k"],
            "p_bb_franta_w": prepared["p_bb_franta_w"],
            "bb_fit_p": np.array(prepared["radiation_fit"]["p"]),
            "bb_fit_theta_k": np.array(prepared["radiation_fit"]["theta_k"]),
            "coefficient_reference_temperature_k": r4["temperature_k"],
            "tpa_digitized_m_inv": r4["tpa_low_cm_inv"]*100,
            "fca_digitized_m_inv": r4["fca_cm_inv"]*100,
            "bga_digitized_m_inv": r4["bga_cm_inv"]*100,
            "reference_temperature_k": r6["temperature_k"],
            "p_abs_reference_w": r6["p_abs_w"], "p_bb_reference_w": r6["p_bb_w"],
            "p_linear_reference_w": r6["p_linear_w"]}


def _comparison(temperature, model, reference_temperature, reference):
    values = np.interp(reference_temperature, temperature, model)
    mask = np.isfinite(reference) & (reference > 0)
    error = np.abs(values[mask]/reference[mask]-1)
    return {"finite_reference_count": int(mask.sum()),
            "mean_absolute_relative_error": float(np.mean(error)),
            "maximum_absolute_relative_error": float(np.max(error))}


def summarize(prepared, results):
    temperature = results["temperature_k"]
    comparisons = {}
    for mechanism in ["tpa", "fca", "bga"]:
        for model in (["model"] if mechanism == "bga" else ["model", "reference_model"]):
            comparisons[f"{mechanism}_{model}_vs_digitized"] = _comparison(temperature,
                results[f"{mechanism}_{model}_m_inv"], results["coefficient_reference_temperature_k"],
                results[f"{mechanism}_digitized_m_inv"])
    for model, reference in [("p_abs_w", "p_abs_reference_w"), ("p_bb_dipole_room_w", "p_bb_reference_w"),
                             ("p_bb_dipole_w", "p_bb_reference_w"), ("p_bb_mie_room_w", "p_bb_reference_w"),
                             ("p_bb_mie_w", "p_bb_reference_w"), ("p_linear_w", "p_linear_reference_w")]:
        comparisons[model+"_vs_digitized"] = _comparison(temperature, results[model],
            results["reference_temperature_k"], results[reference])
    notes = [
        "Figure 4 and Figure 6 dots are vector-curve digitizations, not author-supplied raw numerical measurements; NaN denotes unsupported or off-axis values.",
        "Production TPA uses physical phonon emission/absorption pairing with the existing Sinclair weights, without refitting. Local Sinclair as-written TPA is a distinct semi-empirical reference.",
        "Both benchmark TPA conventions and benchmark BGA use the source-fit photon energy 0.797 eV, supplied to the shared functions through its equivalent wavelength. This is not exact hc/1550 nm. Physical 1550 nm sets Drude, empirical FCA and Mie. Production user-material functions use their actual configured wavelength.",
        "Drude and local empirical Sturm FCA share the same explicit thermally generated electron/hole densities; they are different absorption laws.",
        "The scalar sphere model uses alpha0+beta*Iincident+FCA+BGA as uniform bulk loss in full Mie scattering. It is not an integration of beta times local intensity squared.",
        "All calculated BB powers use the same 0.4–100 um band, Si sphere radius and bath temperature. T0 means fixed 300 K optical constants, not a fixed particle temperature.",
        "BB_Franta dots are dipole powers computed at the native 300,350,400,450,500 K spectra, not measured powers. Only p and theta are fitted to their interior-temperature log power residuals. Both endpoint spectra are exact anchors.",
        "The anchored dielectric function subtracts the change in the fixed Drude model from the endpoint difference before adding its calculated temperature change. There is no second copy of baseline FCA and no change to laser heating.",
        "The BB carrier law uses intrinsic equilibrium populations, fixed DOS masses and a Varshni gap. The laser-heating carrier law is deliberately unchanged. Both relaxation-time laws are extrapolated above their literature ranges.",
        "The shared dipole includes radiation reaction and excludes elastic scattering. Both Mie curves include electric/magnetic responses and higher multipoles. BB_Mie(T) uses exactly the same fitted/extrapolated n and k as BB_dipole(T), without a separate fit. These BB curves are alternatives, not additive cooling terms.",
        "The paper states dipolar radiation; its digitized BB remains separate. No parameters are fitted to Junnemann BB. The two-parameter model is extrapolated above 500 K without resetting or freezing optics.",
        "The optical extrapolation is restricted to 0.4–100 um: extending it into the UV can produce nonphysical negative loss. Negative loss is rejected, never clipped. The discarded 0.3–0.4 um dipole power was below 0.01% at 1600 K with either endpoint spectrum; this is not validation of hot UV optics.",
        "The paper Fig. 6 gives no gas pressure or gas curve. The 0.1 mbar air curve is an explicitly added comparison using mass 28.965 u, gamma 1.4, accommodation 0.8, and sphere surface area.",
        "The source models are extrapolated across 300–1600 K for this benchmark; agreement with a plotted model does not establish experimental validity over that range."
    ]
    return {"benchmark": "Junnemann 2025 silicon sphere", "doi": "10.1103/PhysRevB.111.195304",
            "miepython_version": miepython.__version__,
            "case": {"radius_nm": 130, "wavelength_nm": 1550, "n": INDEX,
                     "incident_intensity_w_um2": 0.1, "bath_temperature_k": BATH_K,
                     "added_gas_pressure_mbar": PRESSURE_MBAR, "alpha0_m_inv": ALPHA0_M_INV,
                     "temperature_range_k": [300, 1600], "temperature_step_k": 1,
                     "exact_photon_energy_ev": h*c/(WAVELENGTH_M*e),
                     "source_fit_photon_energy_ev": SOURCE_PHOTON_ENERGY_EV},
            "explicit_parameters": {"tpa": TPA, "fca": FCA, "bga": BGA, "gas": GAS, "bb_carriers": BB_CARRIERS},
            "radiation": {"wavelength_range_um": list(RADIATION_BAND_UM),
                          "wavelength_samples": len(prepared["radiation_wavelength_m"]),
                          "optical_file": OPTICAL_FILE,
                          "models": {"p_bb_dipole_w": "Two-parameter extrapolated optics, dipole",
                                     "p_bb_dipole_room_w": "300 K optics, dipole",
                                     "p_bb_mie_w": "Same two-parameter extrapolated optics, full Mie",
                                     "p_bb_mie_room_w": "300 K optics, full Mie",
                                     "p_bb_franta_w": "Native Franta spectra, dipole"},
                          "tabulated_temperatures_k": prepared["franta_temperature_k"].tolist(),
                          "fit": prepared["radiation_fit"],
                          "footnote": prepared["radiation_footnote"],
                          "bb_carrier_sources": ["https://dcc.ligo.org/public/0123/T1500619/001/CarrierDensityIntrinsicSi.pdf",
                                                 "https://doi.org/10.1063/1.345414",
                                                 "Arora et al., IEEE TED 29, 292–295 (1982); existing benchmark relaxation times"]},
            "gas_coefficient_w_mbar_k": prepared["gas_coefficient_w_mbar_k"],
            "comparisons": comparisons, "notes": notes,
            "source_sha256": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                for name in ["Junnemann2025.pdf", "junnemann_fig4.csv", "junnemann_fig6.csv", OPTICAL_FILE]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Retain useful state for `python -i compare_Junneman.py --output ...`.
    global prepared, results
    print("Loading Franta spectra and fitting BB at the tabulated temperatures...", flush=True)
    prepared = prepare_case()
    results = calculate(prepared)
    np.savez_compressed(output/"results.npz", **results)
    summary = summarize(prepared, results)
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    plot_results(results, output, args.dpi)
    print(f"Saved results.npz, summary.json and two PNGs to {output}")


if __name__ == "__main__":
    main()
