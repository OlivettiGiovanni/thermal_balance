"""Run the cylinder model. With python -i, prepared and results remain inspectable."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
from prepare_inputs import prepare_inputs
from balance import solve_balance
from plotting import plot_results


def save_results(prepared, results, report, output, config_path):
    """Save final plot arrays; do not export the prepared optical arrays."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = prepared["config"]
    results["kind"] = np.array("cylinder")
    source = prepared["provenance"]["material"]
    results["other_reference_used"] = np.array(prepared["borrowed_spectrum"] or
        source.get("room_spectrum_borrowed", False) or source.get("temperature_other_reference_used", False))
    np.savez_compressed(output/"results.npz", **results)
    report.update(material=config["material"]["name"], particle=prepared["particle"],
                  optical_notes=prepared["optical_notes"], provenance=prepared["provenance"],
                  radiation={"model": "Cylinder volume with spheroid dipole factors; spectral optical data at particle temperature when available",
                             "fallback": prepared.get("radiation_footnote", ""),
                             "reference_temperature_k": config["material"]["reference_temperature_k"]},
                  config=config)
    (output/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    snapshot = output/"config_used.toml"
    if Path(config_path).resolve() != snapshot.resolve():
        shutil.copyfile(config_path, snapshot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    try:
        if args.dpi <= 0:
            raise ValueError("dpi must be positive")
        prepared = prepare_inputs(args.config)
        results, report = solve_balance(prepared)
        save_results(prepared, results, report, args.output, args.config)
        plot_results(results, args.output, args.dpi)
        particle = prepared["particle"]
        print(f"{prepared['config']['material']['name']}: R={particle['radius_nm']:g} nm, H={particle['height_nm']:g} nm")
        print(f"Saved {args.output.resolve()}")
        for note in prepared["optical_notes"]:
            print(note)
        print("Equilibria in calculated interval:", report["equilibria"])
        if "temperature_limit" in report["pressure"]:
            limit = report["pressure"]["temperature_limit"]
            if limit["balance_pressure_mbar"] is not None:
                print(f"Balance pressure at {limit['temperature_k']:g} K: {limit['balance_pressure_mbar']:.6g} mbar")
        if report["pressure"].get("status") == "unresolved_optical_fallback":
            print(report["pressure"]["note"])
        return prepared, results
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    prepared, results = main()
