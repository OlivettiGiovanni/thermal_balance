"""Create PNG figures from saved results; changing a configuration requires a new run."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STYLE = {"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11,
         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
         "grid.alpha": .2, "figure.facecolor": "white", "axes.facecolor": "white",
         "legend.frameon": False, "legend.fontsize": 9, "savefig.facecolor": "white"}
COLORS = {"linear": "#885aa6", "tpa": "#be4b31", "fca": "#ce8c19",
          "bga": "#398957", "total": "#235d96", "bb": "#228665", "gas": "#777777"}
VARIANTS = [("printed", "empirical", "#b34935", "-"),
            ("physical", "empirical", "#225f9e", "-"),
            ("printed", "drude", "#b34935", "--"),
            ("physical", "drude", "#225f9e", "--")]


def read_table(path):
    """Read a numeric CSV; preserve a one-row table as an array."""
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8"))


def curve(ax, temperature, values, label, color, style="-", reference=False):
    """Plot positive finite values; physical zeros are never replaced by tiny numbers."""
    positive = np.isfinite(values) & (values > 0)
    if not np.any(positive):
        return False
    if reference:
        selected = np.flatnonzero(positive)[::max(1, positive.sum() // 24)]
        ax.plot(temperature[selected], values[selected], linestyle="none", marker="o",
                markersize=4, markerfacecolor="none", color=color, label=label, alpha=.75)
    else:
        ax.plot(temperature, np.where(positive, values, np.nan), color=color,
                linestyle=style, linewidth=2, label=label)
    return True


def logarithmic_axes(ax, temperature, ylabel):
    """Show low-temperature values while limiting numerical tails to twelve decades."""
    positive = [np.asarray(line.get_ydata())[np.isfinite(line.get_ydata()) & (line.get_ydata() > 0)]
                for line in ax.lines]
    positive = [values for values in positive if values.size]
    if positive:
        upper = max(values.max() for values in positive) * 2
        ax.set_ylim(max(min(values.min() for values in positive) / 2, upper / 1e12), upper)
        ax.legend(loc="best")
    else:
        ax.set_ylim(1e-3, 1)
        ax.text(.5, .5, "No positive values to display", ha="center", transform=ax.transAxes)
    ax.set(yscale="log", xlabel="Particle temperature (K)", ylabel=ylabel,
           xlim=(temperature.min(), temperature.max()))


def context(summary):
    """Short geometry and input-pressure label, using mbar throughout."""
    radius = summary.get("radius_nm")
    geometry = f"R = {radius:g} nm" if radius is not None else ""
    if summary.get("height_nm") is not None:
        geometry += f", H = {summary['height_nm']:g} nm"
    intensity = summary.get("intensity_w_um2")
    if intensity is not None:
        geometry += f" | I = {intensity:g} W/µm²"
    pressure = summary.get("reference_pressure_mbar")
    return geometry + (f" | p = {pressure:g} mbar" if pressure is not None else "")


def save(fig, output, name, dpi, footer=""):
    """Save one PNG and close it."""
    if footer:
        fig.text(.5, .025, footer, ha="center", va="bottom", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .10 if footer else 0, 1, .95))
    path = output / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def reference_curve(ax, reference, name, label, color):
    """Add an available digitized reference column, keeping its native units."""
    if reference is not None and name in reference.dtype.names:
        curve(ax, reference["temperature_k"], reference[name], label, color, reference=True)


def intensity_label(summary, level):
    """Identify the reference intensity instead of implying a like-for-like overlay."""
    value = summary.get(f"reference_fig4_{level}_intensity_w_um2")
    return f"Fig. 4: {value:g} W/µm²" if value is not None else f"Fig. 4: {level} intensity"


def coefficient_plot(data, reference, summary, output, dpi):
    """Separate component physics from the four choices of total absorption model."""
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 5.2))
    temperature = data["temperature_k"]
    for column, label, color in [("alpha0_m_inv", "Linear", COLORS["linear"]),
                                ("tpa_physical_m_inv", "TPA: physical", COLORS["tpa"]),
                                ("fca_empirical_m_inv", "FCA: empirical", COLORS["fca"]),
                                ("bga_m_inv", "Band-gap", COLORS["bga"])]:
        curve(axes[0], temperature, data[column] / 100, label, color)  # m⁻¹ -> cm⁻¹.
    for tpa, fca, color, style in VARIANTS:
        total = data["alpha0_m_inv"] + data[f"tpa_{tpa}_m_inv"] + data[f"fca_{fca}_m_inv"] + data["bga_m_inv"]
        curve(axes[1], temperature, total / 100, f"{tpa} TPA + {fca} FCA", color, style)
    reference_curve(axes[0], reference, "bga_cm_inv", "Fig. 4 band-gap (digitized)", COLORS["bga"])
    for level, color in [("low", "#444444"), ("high", "#999999")]:
        reference_curve(axes[1], reference, f"total_{level}_cm_inv", intensity_label(summary, level), color)
    for ax, title in zip(axes, ["Absorption contributions", "Total coefficient: four variants"]):
        logarithmic_axes(ax, temperature, r"Absorption coefficient (cm$^{-1}$)")
        ax.set_title(title)
    fig.suptitle("Silicon | " + context(summary))
    return save(fig, output, "coefficient_comparison.png", dpi,
                "Open circles: approximate paper references, at their stated low/high intensities.\n"
                "Model curves use the configured intensity. Drude mobility is extrapolated above 500 K.")


def tpa_plot(data, reference, summary, output, dpi):
    """Compare the two implemented two-photon-absorption expressions."""
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    temperature = data["temperature_k"]
    for model, color, style in [("printed", "#b34935", "--"), ("physical", "#225f9e", "-")]:
        curve(ax, temperature, data[f"tpa_{model}_m_inv"] / 100, f"TPA: {model} expression", color, style)
    for level, color in [("low", "#444444"), ("high", "#999999")]:
        reference_curve(ax, reference, f"tpa_{level}_cm_inv", intensity_label(summary, level), color)
    logarithmic_axes(ax, temperature, r"TPA absorption coefficient (cm$^{-1}$)")
    fig.suptitle("Silicon: two-photon absorption | " + context(summary))
    return save(fig, output, "tpa_comparison.png", dpi,
                "Printed and physical expressions are evaluated at the same configured intensity.\n"
                "Open circles: approximate paper curves at their original intensities.")


def fca_plot(data, reference, summary, output, dpi):
    """Compare empirical and Drude FCA, with the Sturm correction as a diagnostic."""
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    temperature = data["temperature_k"]
    for column, label, color, style in [("fca_empirical_m_inv", "Empirical FCA", "#b34935", "-"),
                                      ("fca_drude_m_inv", "Drude FCA", "#225f9e", "--"),
                                      ("fca_sturm_corrected_m_inv", "Sturm corrected (diagnostic)", "#7e6095", ":")]:
        if column in data.dtype.names:
            curve(ax, temperature, data[column] / 100, label, color, style)
    reference_curve(ax, reference, "fca_cm_inv", "Fig. 4 FCA (digitized)", "#444444")
    logarithmic_axes(ax, temperature, r"FCA absorption coefficient (cm$^{-1}$)")
    fig.suptitle("Silicon: free-carrier absorption | " + context(summary))
    return save(fig, output, "fca_comparison.png", dpi,
                "Drude mobility is extrapolated above 500 K; the empirical model uses its fitted coefficients.\n"
                "The corrected Sturm curve is a diagnostic; it is not a fifth heating combination.")


def silicon_power_plot(data, reference, summary, output, dpi):
    """Compare four heating predictions with cooling and the digitized Fig. 6."""
    fig, ax = plt.subplots(figsize=(12.3, 6.4))
    temperature = data["temperature_k"]
    for tpa, fca, color, style in VARIANTS:
        curve(ax, temperature, data[f"p_abs_{tpa}_{fca}_w"], f"{tpa} TPA + {fca} FCA", color, style)
    for column, label, color, style in [("p_linear_w", "Linear heating only", COLORS["linear"], ":"),
                                      ("p_gas_w", "Gas cooling", COLORS["gas"], "-."),
                                      ("p_bb_w", r"$P_{\rm BB}$: paper reference (input)", COLORS["bb"], "-")]:
        curve(ax, temperature, data[column], label, color, style)
    for column, label, color in [("p_abs_w", "Fig. 6 heating", "#ca8a16"),
                                 ("p_linear_w", "Fig. 6 linear", "#9851a0"),
                                 ("p_bb_w", "Fig. 6 radiation", "#222222")]:
        reference_curve(ax, reference, column, label + " (digitized)", color)
    logarithmic_axes(ax, temperature, "Power (W)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.suptitle("Silicon: absorbed and dissipated power\n" + context(summary))
    bb_note = "No radiation input available." if not np.any(np.isfinite(data["p_bb_w"])) else "Silicon radiation is interpolated from the paper; it is not independently validated."
    return save(fig, output, "power_comparison.png", dpi,
                bb_note + "\nDrude mobility is extrapolated above 500 K.\n"
                "Open circles: approximate Fig. 6 data; original diameter 260 nm, I = 0.1 W/µm².")


def sin_power_plot(data, summary, output, dpi):
    """Show exactly the three total powers, with optional heating terms kept in the CSV."""
    fig, ax = plt.subplots(figsize=(9, 5.8))
    temperature = data["temperature_k"]
    zeros = []
    for column, label, color, style in [("p_abs_w", r"$P_{\rm abs}$", "#bd4c28", "-"),
                                      ("p_gas_w", r"$P_{\rm gas}$", "#2776ad", "--"),
                                      ("p_bb_w", r"$P_{\rm BB}^{\rm net}$", "#398657", "-.")]:
        if not curve(ax, temperature, data[column] * 1e9, label, color, style):
            zeros.append(label + (" = 0" if np.all(data[column] == 0) else " unavailable"))
    logarithmic_axes(ax, temperature, "Power (nW)")
    threshold = summary.get("decomposition_temperature_k")
    if threshold is not None and temperature.min() <= threshold <= temperature.max():
        ax.axvline(threshold, color="#777777", linestyle=":", linewidth=1.2)
        ax.text(threshold, .98, r" $T_d$", transform=ax.get_xaxis_transform(), va="top")
    if zeros:
        ax.text(.02, .04, "; ".join(zeros), transform=ax.transAxes)
    material = Path(summary.get("material_file", "SiN")).stem
    fig.suptitle(material + " | " + context(summary))
    fraction = summary.get("max_emission_fraction_outside_dipole")
    diagnostic = f" Up to {100 * fraction:.2f}% of modeled emission fails the dipole-size criterion." if fraction is not None else ""
    return save(fig, output, "power_balance.png", dpi,
                "Provisional dipole radiation; full-wave validation is required." + diagnostic + "\n"
                "Gas cooling uses the saved reference pressure; powers and heating contributions are in powers.csv.")


def selection_plot(data, summary, output, dpi):
    """Make the selected absorption-density sample directly inspectable."""
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(data["radius_nm"], data["density_w_m3"], ".-", color="#2776ad", markersize=4)
    chosen = data["selected"] > 0
    ax.scatter(data["radius_nm"][chosen], data["density_w_m3"][chosen], s=55, color="#bd4c28",
               zorder=3, label="Selected sample")
    ax.set(xlabel="Cylinder radius (nm)", ylabel=r"Absorbed power / volume (W m$^{-3}$)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
    ax.legend()
    fig.suptitle("SiN: absorption selection")
    return save(fig, output, "absorption_selection.png", dpi)


def pressure_balance_plot(data, summary, output, dpi):
    """Plot equilibrium balance pressure from saved powers, without solving new roots."""
    silicon = summary["profile"] == "silicon"
    fig, ax = plt.subplots(figsize=(12.3 if silicon else 9.3, 5.8))
    temperature = data["temperature_k"]
    denominator = summary["gas_coefficient_w_per_mbar_k"] * (
        temperature - summary["environment_temperature_k"])
    models = [(f"p_abs_{tpa}_{fca}_w", f"{tpa} TPA + {fca} FCA", color, style)
              for tpa, fca, color, style in VARIANTS] if silicon else [
                  ("p_abs_w", "SiN power balance", "#2776ad", "-")]
    folds = []
    for column, label, color, style in models:
        pressure = np.full(temperature.shape, np.nan)
        np.divide(data[column] - data["p_bb_w"], denominator,
                  out=pressure, where=denominator > 0)
        curve(ax, temperature, pressure, label, color, style)
        analysis = summary.get("pressure_analysis", {}).get(column, {})
        if analysis.get("fold_pressure_mbar") is not None:
            folds.append((analysis["fold_temperature_k"], analysis["fold_pressure_mbar"], color))
        if analysis.get("vacuum_temperature_k") is not None:
            ax.text(.98, .95, f"Cold branch reaches vacuum equilibrium at "
                    f"{analysis['vacuum_temperature_k']:.1f} K", ha="right", transform=ax.transAxes)
    logarithmic_axes(ax, temperature, "Equilibrium balance pressure (mbar)")
    reference = summary["reference_pressure_mbar"]
    if reference > 0:
        ax.axhline(reference, color="#777777", linestyle=":", linewidth=1.5,
                   label=f"Reference pressure: {reference:g} mbar")
        lower, upper = ax.get_ylim()
        ax.set_ylim(min(lower, reference / 2), max(upper, reference * 2))
    for i, (fold_temperature, fold_pressure, color) in enumerate(folds):
        if fold_pressure > 0:
            ax.scatter(fold_temperature, fold_pressure, color=color, edgecolor="white", s=65,
                       zorder=4, label="Cold-branch folds" if i == 0 else None)
    threshold = summary.get("decomposition_temperature_k")
    if not silicon and threshold is not None and temperature.min() <= threshold <= temperature.max():
        ax.axvline(threshold, color="#777777", linestyle=":", linewidth=1.2)
        ax.text(threshold, .86, r" $T_d$", transform=ax.get_xaxis_transform(), va="top")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0) if silicon else ax.legend(loc="lower right")
    fig.suptitle(("Silicon" if silicon else "SiN") + ": pressure required for equilibrium\n" + context(summary))
    return save(fig, output, "pressure_balance.png", dpi,
                "Only positive balance pressures are shown. Turning points mark cold-branch instability;\n"
                "a fold alone does not establish decomposition. Free-molecular gas law: high-pressure extrapolation is unvalidated.\n"
                "Si radiation uses the paper reference; SiN radiation remains provisional.")


def plot_results(output, dpi=200):
    """Replot saved CSV/JSON data only; no configuration loading or recalculation."""
    output = Path(output)
    summary = json.loads((output / "summary.json").read_text())
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    powers = read_table(output / "powers.csv")
    with plt.rc_context(STYLE):
        if summary["profile"] == "sin":
            paths = [sin_power_plot(powers, summary, output, dpi),
                     pressure_balance_plot(powers, summary, output, dpi)]
            if (output / "selection.csv").exists():
                paths.append(selection_plot(read_table(output / "selection.csv"), summary, output, dpi))
            return paths
        if summary["profile"] != "silicon":
            raise ValueError("Unknown saved profile")
        coefficients = read_table(output / "coefficients.csv")
        references = [read_table(output / name) if (output / name).exists() else None
                      for name in ("reference_fig4.csv", "reference_fig6.csv")]
        return [function(coefficients, references[0], summary, output, dpi)
                for function in (coefficient_plot, tpa_plot, fca_plot)] + [
                    silicon_power_plot(powers, references[1], summary, output, dpi),
                    pressure_balance_plot(powers, summary, output, dpi)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", required=True, help="Existing result folder")
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution (default: 200)")
    args = parser.parse_args()
    for path in plot_results(args.output, args.dpi):
        print(path)


if __name__ == "__main__":
    main()
