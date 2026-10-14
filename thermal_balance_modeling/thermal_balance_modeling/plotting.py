"""Plot saved absorption mechanisms and thermal powers; no model recalculation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


STYLE = {"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
         "grid.alpha": .18, "figure.facecolor": "white", "axes.facecolor": "white",
         "legend.frameon": False, "legend.fontsize": 9, "savefig.facecolor": "white"}
COLORS = {"tpa": "#bc5733", "fca": "#246ca0", "bga": "#368058"}
OBSOLETE_PLOTS = ("coefficient_comparison.png", "tpa_comparison.png", "fca_comparison.png",
                  "power_comparison.png", "critical_pressure.png", "absorption_selection.png",
                  "pressure_balance.png")


def read_table(path):
    """Retain string status/model columns and the shape of one-row CSV tables."""
    with Path(path).open() as stream:
        columns = stream.readline().strip().split(",")
    # Force numerical empty columns to NaN; dtype inference otherwise calls them False.
    converters = {name: lambda value: float(value) if value.strip() else np.nan
                  for name in columns if name not in ("variant", "status")}
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=None,
                                     encoding="utf-8", converters=converters))


def save(fig, output, name, dpi):
    """Save one figure and close it, including the external legend and notes."""
    path = output / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def absorption_curve(ax, temperature, values, color, style="-", dots=False, label=None):
    """Preserve zero/missing-value gaps on logarithmic axes; thin reference dots."""
    positive = np.isfinite(values) & (values > 0)
    if dots:
        indices = np.flatnonzero(positive)
        indices = indices[::max(1, int(np.ceil(len(indices) / 32)))]
        ax.plot(temperature[indices], values[indices], linestyle="none", marker="o",
                markersize=4.2, markerfacecolor="white", markeredgewidth=1.1,
                color=color, zorder=4, label=label)
    else:
        ax.plot(temperature, np.where(positive, values, np.nan), color=color,
                linestyle=style, linewidth=2, label=label)


def comparison_legend(fig):
    """Lay out one row per mechanism: name, digitized points, first and second model."""
    rows = [("TPA:", "tpa", "Sinclair (semi-empirical)", "Physical pairing"),
            ("FCA:", "fca", "Sturm empirical", "Drude"),
            ("Band gap:", "bga", "Sturm empirical", "")]
    cells = []
    for title, mechanism, first, second in rows:
        color = COLORS[mechanism]
        empty = Line2D([], [], linestyle="none", color="none")
        cells.append([(empty, title),
                      (Line2D([], [], linestyle="none", marker="o", color=color,
                              markerfacecolor="white", markersize=4.5), "Digitized"),
                      (Line2D([], [], color=color, linewidth=2), first),
                      (Line2D([], [], color=color, linestyle="--", linewidth=2)
                       if second else empty, second)])
    # Matplotlib fills multi-column legends column-first, so transpose the rows.
    entries = [cells[row][column] for column in range(4) for row in range(3)]
    legend = fig.legend(*zip(*entries), loc="center", bbox_to_anchor=(.50, .17),
                        ncol=4, handlelength=2.7, columnspacing=1.7, labelspacing=.9)
    for text in legend.get_texts()[:3]:
        text.set_weight("bold")


def absorption_comparison(data, reference, summary, output, dpi):
    """Compare all three absorption mechanisms against the corresponding Fig. 4 data."""
    fig, ax = plt.subplots(figsize=(10.2, 7.3))
    fig.subplots_adjust(left=.11, right=.98, bottom=.31, top=.88)
    temperature = data["temperature_k"]
    for mechanism, models in [("tpa", ("printed", "physical")),
                              ("fca", ("empirical", "drude")), ("bga", (None,))]:
        for index, model in enumerate(models):
            column = f"{mechanism}_{model}_m_inv" if model else "bga_m_inv"
            absorption_curve(ax, temperature, data[column] / 100, COLORS[mechanism],
                             "--" if index else "-")  # m⁻¹ -> cm⁻¹.
    intensity = summary["intensity_w_um2"]
    low = summary.get("reference_fig4_low_intensity_w_um2", .1)
    high = summary.get("reference_fig4_high_intensity_w_um2", 1.)
    level = "high" if np.isclose(intensity, high, rtol=1e-10, atol=0) else "low"
    reference_intensity = high if level == "high" else low
    scaled = not np.isclose(intensity, reference_intensity, rtol=1e-10, atol=0)
    note = "Dots: digitized Jünnemann Fig. 4."
    if reference is not None:
        visible = ((reference["temperature_k"] >= temperature.min()) &
                   (reference["temperature_k"] <= temperature.max()))
        for mechanism, column, scale in [("tpa", f"tpa_{level}_cm_inv", intensity / reference_intensity),
                                         ("fca", "fca_cm_inv", 1.), ("bga", "bga_cm_inv", 1.)]:
            absorption_curve(ax, reference["temperature_k"][visible], reference[column][visible] * scale,
                             COLORS[mechanism], dots=True)
        if scaled:
            note += f" TPA dots rescaled from {reference_intensity:g} W/µm²."
    else:
        note = "Digitized reference file unavailable; only model lines are shown."
    positive = [line.get_ydata()[line.get_ydata() > 0] for line in ax.lines]
    positive = [values[np.isfinite(values)] for values in positive if len(values)]
    positive = [values for values in positive if len(values)]
    if positive:
        maximum = max(values.max() for values in positive) * 2
        minimum = max(min(values.min() for values in positive) / 2, maximum / 1e10)
        ax.set_ylim(minimum, maximum)
    ax.set(yscale="log", xlim=(temperature.min(), temperature.max()),
           xlabel="Particle temperature (K)", ylabel=r"Absorption coefficient (cm$^{-1}$)")
    fig.suptitle("Silicon absorption: model and paper", y=.97, fontsize=13)
    fig.text(.55, .925, f"I = {intensity:g} W/µm²  ·  λ = 1550 nm", ha="center", color="#555555")
    comparison_legend(fig)
    fig.text(.5, .055, note + "\nDrude mobility is extrapolated above 500 K.",
             ha="center", va="center", fontsize=8.5, color="#555555")
    return save(fig, output, "absorption_comparison.png", dpi)


def power_balance(data, reference, summary, output, dpi):
    """Compare one selected total absorption with cooling and the paper's total."""
    silicon = summary["profile"] == "silicon"
    temperature = data["temperature_k"]
    fig, ax = plt.subplots(figsize=(9.4, 7.0))
    fig.subplots_adjust(left=.11, right=.98, bottom=.31, top=.87)
    column = summary.get("selected_absorption_column", "p_abs_physical_drude_w" if silicon else "p_abs_w")
    absorption_curve(ax, temperature, data[column], "#246ca0", label="Our model: total absorption")
    pressure = summary["reference_pressure_mbar"]
    absorption_curve(ax, temperature, data["p_gas_w"], "#368058",
                     label=f"Gas cooling: p = {pressure:g} mbar")
    absorption_curve(ax, temperature, data["p_bb_w"], "#8d63b4", style="--",
                     label="BB radiation (reference input)" if silicon else "BB radiation (dipole model)")
    if silicon:
        paper_intensity = summary.get("reference_fig6_intensity_w_um2", .1)
        note = "Our model: linear + physical-pairing TPA + Drude FCA + Sturm BGA."
        if reference is not None:
            visible = ((reference["temperature_k"] >= temperature.min()) &
                       (reference["temperature_k"] <= temperature.max()))
            absorption_curve(ax, reference["temperature_k"][visible], reference["p_abs_w"][visible],
                             "#bc5733", dots=True,
                             label=f"Jünnemann model (digitized): I = {paper_intensity:g} W/µm²")
            if not np.isclose(summary["intensity_w_um2"], paper_intensity, rtol=1e-10, atol=0):
                note += f"\nPaper dots stay at {paper_intensity:g} W/µm²; they are not rescaled to the model intensity."
        else:
            note += "\nDigitized total-absorption reference unavailable."
        note += "\nBB: digitized Fig. 6 input, not an independent radiation calculation."
        subtitle = (f"R = {summary['radius_nm']:g} nm  ·  λ = 1550 nm  ·  "
                    f"Model I = {summary['intensity_w_um2']:g} W/µm²")
    else:
        note = "Absorption anchored to the selected optical CSV; nonlinear terms use supplied material data."
        note += "\nBB: provisional dipole approximation. No silicon reference curves are assigned to SiN."
        subtitle = f"R = {summary['radius_nm']:g} nm  ·  H = {summary['height_nm']:g} nm"
    subtitle += f"  ·  Gas bath = {summary['environment_temperature_k']:g} K"
    ax.set(yscale="log", xlim=(temperature.min(), temperature.max()),
           xlabel="Particle temperature (K)", ylabel="Power (W)")
    fig.suptitle(("Silicon" if silicon else "SiN") + ": absorption and cooling", y=.97, fontsize=13)
    fig.text(.55, .92, subtitle, ha="center", fontsize=9.5, color="#555555")
    handles, labels = ax.get_legend_handles_labels()
    if silicon and reference is not None:
        # Put modeled/reference absorption together, followed by the cooling terms.
        order = [3, 0, 1, 2]
        handles, labels = [handles[i] for i in order], [labels[i] for i in order]
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(.5, .18),
               ncol=2, handlelength=2.7, columnspacing=2., labelspacing=.9)
    fig.text(.5, .055, note, ha="center", va="center", fontsize=8.3, color="#555555")
    return save(fig, output, "power_balance.png", dpi)


def plot_results(output, dpi=200):
    """Produce the requested figures from saved CSV/JSON data only."""
    output = Path(output)
    summary = json.loads((output / "summary.json").read_text())
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if summary["profile"] not in ("silicon", "sin"):
        raise ValueError("Unknown saved profile")
    powers = read_table(output / "powers.csv")
    with plt.rc_context(STYLE):
        paths = []
        if summary["profile"] == "silicon":
            reference = output / "reference_fig4.csv"
            paths.append(absorption_comparison(read_table(output / "coefficients.csv"),
                         read_table(reference) if reference.exists() else None, summary, output, dpi))
        reference = output / "reference_fig6.csv"
        paths.append(power_balance(powers, read_table(reference) if reference.exists() else None,
                                   summary, output, dpi))
    for name in OBSOLETE_PLOTS:
        (output / name).unlink(missing_ok=True)  # Remove only superseded, known project plots.
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", required=True, help="Existing result folder")
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution (default: 200)")
    args = parser.parse_args()
    try:
        for path in plot_results(args.output, args.dpi):
            print(path)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
