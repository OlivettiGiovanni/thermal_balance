"""Small thermal-balance driver: explicit inputs, saved numbers, reproducible plots."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, minimize_scalar

from inputs import load_config, read_csv, read_spectrum, select_absorption, nonlinear_coefficients
from silicon import ALPHA_0, tpa_beta, fca_absorption, bandgap_absorption
from thermal import sphere_heating, gas_coefficient, gas_power, radiation_power, emission_fraction_outside_dipole
from plotting import plot_results

HERE = Path(__file__).resolve().parent
VARIANTS = [(tpa, fca) for tpa in ("printed", "physical") for fca in ("empirical", "drude")]


def write_csv(path, table):
    """Save named, equal-length numerical columns without hidden unit conversions."""
    with Path(path).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(table)
        writer.writerows(zip(*table.values()))


def reference_radiation(temperature):
    """Interpolate Fig. 6 radiation; the 300 K net-exchange boundary is imposed."""
    reference = read_csv(HERE / "data/junnemann_fig6.csv")
    valid = np.isfinite(reference["p_bb_w"]) & (reference["p_bb_w"] > 0)
    samples, power = reference["temperature_k"][valid], reference["p_bb_w"][valid]
    if np.any(temperature < 300) or np.any(temperature > samples[-1]):
        raise ValueError("Si radiation reference supports 300–1600 K only")
    result = np.exp(np.interp(temperature, samples, np.log(power)))
    # The published visible curve starts at 400 K; this interval is an explicit bridge.
    return np.where(temperature < samples[0], power[0]*(temperature-300)/(samples[0]-300), result)


def silicon_calculation(config, temperature):
    """Evaluate four effective-coefficient Mie variants at the benchmark geometry."""
    intensity = config["intensity_w_um2"]*1e12  # W/µm² -> W/m².
    coefficients = dict(temperature_k=temperature, alpha0_m_inv=np.full_like(temperature, ALPHA_0))
    for name in ("printed", "physical"):
        coefficients[f"tpa_{name}_m_inv"] = intensity*tpa_beta(temperature, name, photon_energy_ev=.797)
    for name in ("empirical", "drude"):
        coefficients[f"fca_{name}_m_inv"] = fca_absorption(temperature, name)
    coefficients["bga_m_inv"] = bandgap_absorption(temperature, photon_energy_ev=.797)
    coefficients["fca_sturm_corrected_m_inv"] = fca_absorption(temperature, density_model="corrected")
    powers = dict(temperature_k=temperature)
    for tpa, fca in VARIANTS:
        alpha = ALPHA_0 + coefficients[f"tpa_{tpa}_m_inv"] + coefficients[f"fca_{fca}_m_inv"] + coefficients["bga_m_inv"]
        powers[f"p_abs_{tpa}_{fca}_w"] = sphere_heating(alpha, intensity)
    powers["p_linear_w"] = np.full_like(temperature, sphere_heating(ALPHA_0, intensity))
    powers["p_bb_w"] = reference_radiation(temperature)
    summary = dict(profile="silicon", radius_nm=130., height_nm=None, wavelength_nm=1550.,
                   intensity_w_um2=config["intensity_w_um2"], photon_energy_fit_ev=.797,
                   reference_fig4_low_intensity_w_um2=.1, reference_fig4_high_intensity_w_um2=1.,
                   carrier_model="Sturm uncorrected thermal equilibrium; corrected curve diagnostic only",
                   radiation_model="digitized Fig. 6 INPUT; not independently reproduced",
                   geometry_model="homogeneous sphere; fixed real refractive index 3.48",
                   drude_mobility_validity_k=[250, 500], drude_extrapolated_above_k=500)
    return powers, {"coefficients.csv": coefficients}, summary, 4*np.pi*(130e-9)**2


def calibrated_heating(selected, temperature, nonlinear_file):
    """Anchor linear power; use field moments only when a nonlinear term needs them."""
    beta, fca = nonlinear_coefficients(nonlinear_file, temperature)
    moments = []
    for key, needed in [("intensity_integral_w_m", np.any(fca > 0)),
                        ("intensity_squared_integral_w2_per_m", np.any(beta > 0))]:
        value = selected.get(key)
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be finite and positive")
        if needed and value is None:
            raise ValueError(f"Nonzero nonlinear absorption requires CSV column {key}")
        moments.append(value)
    first, second = moments
    linear = np.full_like(temperature, selected["p_abs_w"])
    return dict(p_linear_w=linear, p_tpa_w=beta*(second or 0), p_fca_w=fca*(first or 0)), {
        "alpha0_m_inv": selected["p_abs_w"]/first if first else None,
        "linear_absorption_note": "alpha0=P_selected / integral(I_local dV); without that integral only power is identified",
        "intensity_integral_w_m": first, "intensity_squared_integral_w2_per_m": second}


def sin_calculation(config, temperature):
    """Use the selected cylinder's power and optional material-specific nonlinear data."""
    selected, selection = select_absorption(config["absorption_file"], config["height_nm"], config["selection"])
    radius, height = selected["radius_nm"], config["height_nm"]
    terms, calibration = calibrated_heating(selected, temperature, config["nonlinear_file"])
    material = read_spectrum(config["spectrum_file"])
    powers = dict(temperature_k=temperature, **terms)
    powers["p_abs_w"] = sum(terms.values())
    powers["p_bb_w"] = radiation_power(temperature, material, radius, height, config["environment_temperature_k"])
    fraction = emission_fraction_outside_dipole(temperature, material, radius, height)
    powers["emission_fraction_outside_dipole"] = fraction
    summary = dict(profile="sin", radius_nm=radius, height_nm=height,
                   decomposition_temperature_k=config["decomposition_temperature_k"],
                   selection=config["selection"], selected_linear_power_w=selected["p_abs_w"],
                   material_file=Path(config["spectrum_file"]).name,
                   radiation_model="provisional matched-volume spheroid dipole, supplied spectral band only",
                   radiation_wavelength_um=[float(material["wavelength_m"][0]*1e6), float(material["wavelength_m"][-1]*1e6)],
                   max_emission_fraction_outside_dipole=float(fraction.max()),
                   nonlinear_model="tabulated material-specific TPA/FCA" if config["nonlinear_file"] else "TPA=FCA=0 (coefficients not supplied)",
                   bga_model="zero by assumption for SiN", **calibration)
    return powers, {"selection.csv": selection}, summary, 2*np.pi*radius*(radius+height)*1e-18


def pressure_analysis(temperature, absorbed, radiation, coefficient, bath, limit=None):
    """Trace the cold stable branch; distinguish a fold from a decomposition crossing."""
    net = PchipInterpolator(temperature, absorbed-radiation)
    upper = limit if limit is not None else float(temperature[-1])
    samples = np.linspace(bath+min(1e-3, (upper-bath)/1000), upper, 6000)
    pressure = net(samples)/(coefficient*(samples-bath))
    if net(bath) <= 0:
        return {"status": "no_heating_at_bath", "critical_pressure_mbar": None}
    if net(samples[0]) <= 0:
        root = brentq(net, bath, samples[0])
        return {"status": "cold_branch_reaches_vacuum_equilibrium", "critical_pressure_mbar": None,
                "vacuum_temperature_k": float(root)}
    # With gas loss linear in p, dp_balance/dT < 0 is the stable portion of the branch.
    for index in range(1, len(samples)-1):
        if pressure[index] <= 0:
            root = brentq(net, samples[index-1], samples[index])
            return {"status": "cold_branch_reaches_vacuum_equilibrium", "critical_pressure_mbar": None,
                    "vacuum_temperature_k": float(root)}
        if pressure[index] < pressure[index-1] and pressure[index] < pressure[index+1]:
            result = minimize_scalar(lambda t: net(t)/(coefficient*(t-bath)),
                                     bounds=(samples[index-1], samples[index+1]), method="bounded")
            return {"status": "cold_branch_fold; high-temperature endpoint not established",
                    "critical_pressure_mbar": None, "fold_pressure_mbar": float(result.fun),
                    "fold_temperature_k": float(result.x)}
    if pressure[-1] <= 0:
        root = brentq(net, samples[-2], samples[-1])
        return {"status": "cold_branch_reaches_vacuum_equilibrium", "critical_pressure_mbar": None,
                "vacuum_temperature_k": float(root)}
    return {"status": "temperature_limit_crossing" if limit else "unresolved_above_temperature_range",
            "critical_pressure_mbar": float(pressure[-1]) if limit else None,
            "limit_temperature_k": float(upper), "balance_pressure_at_limit_mbar": float(pressure[-1])}


def comparison_metrics(coefficients, powers, intensity):
    """Report log-ratio residuals against independently digitized paper curves."""
    references = [("coefficients", read_csv(HERE/"data/junnemann_fig4.csv")),
                  ("powers", read_csv(HERE/"data/junnemann_fig6.csv"))]
    pairs = [("coefficients", "fca_empirical_m_inv", "fca_cm_inv", .01),
             ("coefficients", "bga_m_inv", "bga_cm_inv", .01)]
    if intensity in (.1, 1.):
        level = "low" if intensity == .1 else "high"
        pairs += [("coefficients", f"tpa_{name}_m_inv", f"tpa_{level}_cm_inv", .01) for name in ("printed", "physical")]
    if intensity == .1:
        pairs += [("powers", "p_linear_w", "p_linear_w", 1.)]
        pairs += [("powers", f"p_abs_{tpa}_{fca}_w", "p_abs_w", 1.) for tpa, fca in VARIANTS]
    result = {}
    for kind, model, target, factor in pairs:
        table = coefficients if kind == "coefficients" else powers
        reference = dict(references)[kind]
        valid = np.isfinite(reference[target]) & (reference[target] > 0)
        valid &= (reference["temperature_k"] >= table["temperature_k"][0]) & (reference["temperature_k"] <= table["temperature_k"][-1])
        if not np.any(valid):
            continue
        calculated = np.interp(reference["temperature_k"][valid], table["temperature_k"], table[model])*factor
        ratio = calculated/reference[target][valid]
        positive = ratio > 0
        result[model] = dict(median_model_to_reference=float(np.median(ratio)),
                            rms_log10_ratio=float(np.sqrt(np.mean(np.log10(ratio[positive])**2))),
                            reference_points=int(valid.sum()))
    return result


def calculate(config):
    """Assemble heating, cooling and branch diagnostics on one temperature grid."""
    temperature = np.linspace(config["environment_temperature_k"], config["temperature_max_k"], config["temperature_points"])
    method = silicon_calculation if config["material"] == "Si" else sin_calculation
    powers, tables, summary, area = method(config, temperature)
    coefficient = gas_coefficient(area, config["environment_temperature_k"], config["accommodation"])
    powers["p_gas_w"] = gas_power(temperature, config["reference_pressure_mbar"], coefficient, config["environment_temperature_k"])
    summary.update(reference_pressure_mbar=config["reference_pressure_mbar"],
                   environment_temperature_k=config["environment_temperature_k"],
                   gas_model="free-molecular air, mass=28.965 u, gamma=1.4",
                   accommodation=config["accommodation"], gas_coefficient_w_per_mbar_k=coefficient)
    summary["pressure_analysis"] = {name: pressure_analysis(temperature, power, powers["p_bb_w"], coefficient,
        config["environment_temperature_k"], config["decomposition_temperature_k"] if config["material"] == "SiN" else None)
        for name, power in powers.items() if name.startswith("p_abs")}
    if config["material"] == "Si":
        summary["paper_comparison"] = comparison_metrics(tables["coefficients.csv"], powers, config["intensity_w_um2"])
    return {"powers.csv": powers, **tables}, summary


def save_results(output, config_path, config, tables, summary):
    """Save the calculation and input copies so plotting needs no original paths."""
    output.mkdir(parents=True, exist_ok=True)
    inputs = output/"inputs"
    inputs.mkdir(exist_ok=True)
    shutil.copy2(config_path, inputs/"config.toml")
    for key in ("absorption_file", "spectrum_file", "nonlinear_file"):
        if config.get(key):
            shutil.copy2(config[key], inputs/(key+".csv"))
    for name, table in tables.items():
        write_csv(output/name, table)
    for name, value in [("summary.json", summary), ("resolved_config.json", config)]:
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    if config["material"] == "Si":
        for number in (4, 6):
            shutil.copy2(HERE/f"data/junnemann_fig{number}.csv", output/f"reference_fig{number}.csv")


def main():
    """Keep output location and PNG resolution in the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE/"config_si.toml")
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    try:
        if not 50 <= args.dpi <= 1200:
            raise ValueError("dpi must be between 50 and 1200")
        # New result folders avoid mixing stale files from a different material/run.
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Output folder must be empty or new; use plotting.py to replot saved data")
        config = load_config(args.config)
        tables, summary = calculate(config)
        save_results(args.output, args.config, config, tables, summary)
        plot_results(args.output, args.dpi)
        print(f"Saved {config['material']} results to {args.output.resolve()}")
        for variant, result in summary["pressure_analysis"].items():
            print(variant + ": " + json.dumps(result))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
