"""Calculate the decomposition-pressure threshold and three thermal power curves."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from inputs import load_config, read_absorption, read_material, select_peak
from physics import (absorbed_power, critical_pressure, emission_fraction_outside_dipole,
                     gas_coefficient, gas_power, radiation_power)
from plotting import plot_results


def temperature_grid(settings):
    """Include the exact limit and extra samples near the gas temperature."""
    bath = settings["environment_temperature_k"]
    span = settings["temperature_max_k"] - bath
    return np.unique(np.r_[np.linspace(bath, bath + span, settings["temperature_points"]),
                           bath + np.geomspace(min(1e-4, span / 1e4), min(10, span), 40),
                           settings["decomposition_temperature_k"]])


def threshold_result(raw_pressure, temperature_dependent, heating_w):
    """Separate a balance at the limit from a proven equilibrium crossing."""
    if temperature_dependent:
        return None, "threshold_balance_only", "Balance at T_decomp only; equilibrium branches are not determined."
    if heating_w == 0:
        return None, "no_heating", "No laser heating; no positive critical pressure."
    if raw_pressure < 0:
        return None, "no_positive_crossing", "Radiation alone prevents reaching T_decomp in this model."
    if raw_pressure == 0:
        return 0.0, "vacuum_at_limit", "The vacuum balance is exactly at T_decomp; positive pressure cools below it."
    return raw_pressure, "positive_crossing", "Below p_crit the equilibrium exceeds T_decomp, if a finite equilibrium exists."


def calculate(config):
    """Use the current inputs to calculate one selected particle."""
    analysis = config["analysis"]
    height = config["geometry"]["height_nm"]
    sweep = read_absorption(config["absorption"]["file"], height)
    chosen = select_peak(sweep["radius_nm"], sweep["density_w_m3"],
                         config["absorption"]["selection"], config["absorption"]["prominence_fraction"])
    radius, linear_power = float(sweep["radius_nm"][chosen]), float(sweep["p_abs_w"][chosen])
    material = read_material(config["material"]["file"], config["radiation"]["integration_points"])
    bath, limit = analysis["environment_temperature_k"], analysis["decomposition_temperature_k"]
    pressure = analysis["reference_pressure_mbar"]
    coefficient = gas_coefficient(radius, height, bath, **config["gas"])
    temperature = temperature_grid(analysis)
    heating = absorbed_power(temperature, linear_power)
    gas = gas_power(temperature, pressure, coefficient, bath)
    radiation = radiation_power(temperature, material, radius, height, bath)
    # Evaluate the threshold exactly at the requested T_decomp, not at a grid neighbour.
    heat_limit = float(absorbed_power(limit, linear_power))
    rad_limit = float(radiation_power(limit, material, radius, height, bath))
    raw_pressure = float(critical_pressure(heat_limit, rad_limit, coefficient, bath, limit))
    optical_t_dependence = material["temperature_k"] is not None
    outside_dipole = float(emission_fraction_outside_dipole(limit, material, radius, height))
    heating_t_dependence = bool(np.any(heating != heating[0]))
    crossing, status, interpretation = threshold_result(
        raw_pressure, optical_t_dependence or heating_t_dependence, heat_limit)
    notes = ["1950 K (or the configured value) is an assumed decomposition threshold, not a universal material constant.",
             "Gas cooling assumes the free-molecular regime and fixed gas properties.",
             "Radiation uses a volume/aspect-matched electric-dipole spheroid; finite-cylinder modes are omitted.",
             "Radiation is integrated only over the supplied spectral band; missing absorption is unknown."]
    if not optical_t_dependence:
        notes.append("The supplied optical constants are held fixed with temperature.")
    if outside_dipole > 0.1:
        notes.append(f"At T_decomp, {outside_dipole:.1%} of modeled emission fails the dipole-size criterion (0.3); full-wave validation is needed.")
    if chosen in (0, len(sweep["radius_nm"]) - 1):
        notes.append("The selected maximum is at a sweep boundary; an interior resonance is not established.")
    if sweep["omitted_rows"]:
        notes.append(f"Excluded {sweep['omitted_rows']} rows marked unconverged.")
    summary = {"radius_nm": radius, "height_nm": height, "p_abs_linear_w": linear_power,
               "reference_pressure_mbar": pressure, "environment_temperature_k": bath,
               "decomposition_temperature_k": limit, "threshold_pressure_mbar": raw_pressure,
               "critical_pressure_mbar": crossing, "critical_pressure_status": status,
               "threshold_interpretation": interpretation, "p_abs_at_limit_w": heat_limit,
               "p_bb_at_limit_w": rad_limit, "gas_coefficient_w_per_mbar_k": coefficient,
               "optical_properties_temperature_dependent": optical_t_dependence,
               "heating_temperature_dependent": heating_t_dependence,
               "emission_fraction_outside_dipole_at_limit": outside_dipole,
               "wavelength_band_um": (material["wavelength_m"][[0, -1]] * 1e6).tolist(),
               "selection": config["absorption"]["selection"], "dpi": config["plot"]["dpi"],
               "absorption_file": str(config["absorption"]["file"]),
               "material_file": str(config["material"]["file"]), "warnings": notes}
    powers = np.column_stack((temperature, heating, gas, radiation))
    selection = np.column_stack((sweep["radius_nm"], sweep["density_w_m3"], sweep["p_abs_w"],
                                 np.arange(len(sweep["radius_nm"])) == chosen))
    return powers, selection, summary


def save_results(folder, config_path, powers, selection, summary):
    """Write numerical results and exact input copies for this calculation."""
    folder = Path(folder).resolve()
    # Protect source inputs if a user chooses their input directory as the output.
    inputs = {Path(p).resolve() for p in (config_path, summary["absorption_file"], summary["material_file"])}
    names = ("power_balance.csv", "selected_absorption.csv", "summary.json", "config_used.toml",
             "material_input.csv", "absorption_input.csv", "power_balance.png", "absorption_selection.png")
    if any(folder / name in inputs for name in names):
        raise ValueError("Choose an output folder that does not overwrite an input file.")
    folder.mkdir(parents=True, exist_ok=True)
    np.savetxt(folder / "power_balance.csv", powers, delimiter=",", comments="", fmt="%.16g",
               header="temperature_k,p_abs_w,p_gas_w,p_bb_w")
    np.savetxt(folder / "selected_absorption.csv", selection, delimiter=",", comments="", fmt="%.16g",
               header="radius_nm,absorption_density_W_m3,p_abs_W,selected")
    summary["config_file"] = str(Path(config_path).resolve())
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    shutil.copyfile(config_path, folder / "config_used.toml")
    shutil.copyfile(summary["material_file"], folder / "material_input.csv")
    shutil.copyfile(summary["absorption_file"], folder / "absorption_input.csv")


def print_summary(summary, output):
    """Make the selected inputs and actual pressure visible in the terminal."""
    print(f"R = {summary['radius_nm']:g} nm; H = {summary['height_nm']:g} nm; "
          f"P_abs = {summary['p_abs_linear_w'] * 1e9:.6g} nW")
    status = summary["critical_pressure_status"]
    if status == "threshold_balance_only":
        print(f"Balance pressure at T_decomp = {summary['threshold_pressure_mbar']:.8g} mbar (signed; branch unverified)")
    elif summary["critical_pressure_mbar"] is None:
        print("No positive critical pressure.")
    else:
        print(f"Critical pressure = {summary['critical_pressure_mbar']:.8g} mbar")
    print(summary["threshold_interpretation"])
    print(f"At T_decomp: P_abs = {summary['p_abs_at_limit_w'] * 1e9:.6g} nW; "
          f"P_BB = {summary['p_bb_at_limit_w'] * 1e9:.6g} nW")
    print("Model: free-molecular gas; band-limited spheroid radiation. Assumptions are saved in summary.json.")
    if summary["emission_fraction_outside_dipole_at_limit"] > 0.1:
        print(f"Dipole-size diagnostic at T_decomp: {summary['emission_fraction_outside_dipole_at_limit']:.1%} "
              "of emission outside the small-size criterion; quantitative validation needed.")
    print(f"Saved: {Path(output).resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml", help="Only this TOML is loaded; paths are relative to it.")
    parser.add_argument("--output", "-o", required=True, help="Output folder, relative to the current shell directory.")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        print(f"Config: {Path(args.config).resolve()}")
        print(f"Absorption: {config['absorption']['file']}")
        print(f"Material: {config['material']['file']}")
        print(f"Reference pressure used: {config['analysis']['reference_pressure_mbar']:g} mbar")
        powers, selection, summary = calculate(config)
        save_results(args.output, args.config, powers, selection, summary)
        plot_results(args.output)
        print_summary(summary, args.output)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
