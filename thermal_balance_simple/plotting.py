"""Plot saved CSV/JSON results; replotting never reloads or changes the TOML inputs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STYLE = {
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.7,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "savefig.facecolor": "white",
}


def read_table(path):
    """Read a named numeric CSV, including the one-row case."""
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8"))


def threshold_label(summary):
    """Describe the threshold without implying stability for future T-dependent heating."""
    branch_unverified = (summary.get("critical_pressure_status") == "threshold_balance_only"
                         or summary.get("optical_properties_temperature_dependent", False)
                         or summary.get("heating_temperature_dependent", False))
    if branch_unverified:
        threshold = summary.get("threshold_pressure_mbar")
        if threshold is not None and threshold > 0:
            return f"Threshold balance: p = {threshold:.4g} mbar (equilibrium branches unverified)"
        if threshold == 0:
            return "Threshold balance in vacuum (equilibrium branches unverified)"
        return "No nonnegative threshold-balance pressure; equilibrium branches unverified"
    pressure = summary.get("critical_pressure_mbar")
    if pressure is None or pressure <= 0:
        return "No positive critical pressure for the configured model"
    return f"Critical pressure: {pressure:.4g} mbar (below it, equilibrium exceeds the threshold)"


def save_figure(fig, path, dpi):
    """Write one PNG and release its memory."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_power_balance(table, summary, folder, dpi):
    """Show the three physical powers at the saved reference gas pressure."""
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    temperature = table["temperature_k"]
    curves = (
        ("p_abs_w", r"$P_{\rm abs}$", "#bd4c28", "-"),
        ("p_gas_w", r"$P_{\rm gas}$", "#2776ad", "--"),
        ("p_bb_w", r"$P_{\rm BB}^{\rm net}$", "#398657", "-."),
    )
    zero_curves, peaks, has_positive = [], [], False
    for column, label, color, style in curves:
        values = table[column] * 1e9  # W -> nW; physical zeros stay zero in the CSV.
        positive = np.isfinite(values) & (values > 0)
        if np.any(positive):
            has_positive = True
            peaks.append(values[positive].max())
            ax.plot(temperature, np.where(positive, values, np.nan),
                    color=color, linestyle=style, linewidth=2.2, label=label)
        elif np.all(values == 0):
            note = label + " = 0"
            if column == "p_gas_w" and summary["reference_pressure_mbar"] == 0:
                note += " (vacuum)"
            zero_curves.append((note, color))
    if has_positive:
        ax.set_ylim(min(peaks) / 100, max(peaks) * 2)  # Near-bath tails must not set the scale.
    else:
        ax.set_ylim(1e-3, 1)  # Defined display scale only; no artificial nonzero data.
        ax.text(0.5, 0.5, "All powers are zero.\nZero cannot appear on a logarithmic axis.",
                ha="center", transform=ax.transAxes, color="#555555")
    for i, (note, color) in enumerate(zero_curves):
        ax.text(0.02, 0.04 + 0.06 * i, note, transform=ax.transAxes, color=color)
    td = summary["decomposition_temperature_k"]
    if temperature.min() <= td <= temperature.max():
        ax.axvline(td, color="#7a7a7a", linestyle=":", linewidth=1.3)
        ax.text(td, 0.98, r" $T_d$", transform=ax.get_xaxis_transform(), va="top")
    ax.set(xlabel="Particle temperature (K)", ylabel="Power (nW)", yscale="log",
           xlim=(temperature.min(), temperature.max()))
    material = Path(summary["material_file"]).stem
    ax.set_title(f"{material} | R = {summary['radius_nm']:g} nm, H = {summary['height_nm']:g} nm\n"
                 f"Gas pressure = {summary['reference_pressure_mbar']:g} mbar; "
                 f"surroundings = {summary['environment_temperature_k']:g} K", pad=13)
    if has_positive:
        ax.legend(loc="best")
    fig.text(0.5, 0.045, threshold_label(summary), ha="center", fontsize=10)
    fig.text(0.5, 0.012, f"Assumed decomposition threshold: {td:g} K. "
             "Radiation includes exchange with the surroundings.", ha="center", fontsize=9,
             color="#555555")
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    return save_figure(fig, folder / "power_balance.png", dpi)


def plot_absorption_selection(table, summary, folder, dpi):
    """Verify which measured/simulated absorption-density sample was selected."""
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    radius, density = table["radius_nm"], table["absorption_density_W_m3"]
    ax.plot(radius, density, color="#2776ad", linewidth=1.8, marker=".", markersize=4)
    selected = table["selected"] > 0
    ax.scatter(radius[selected], density[selected], color="#bd4c28", s=58, zorder=3,
               label=f"Selected: R = {summary['radius_nm']:g} nm")
    ax.set(xlabel="Cylinder radius (nm)", ylabel=r"Absorbed power / volume (W m$^{-3}$)",
           title=f"Absorption selection | {Path(summary['absorption_file']).name}")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, folder / "absorption_selection.png", dpi)


def plot_results(folder, dpi=None):
    """Create PNGs from saved calculations; TOML changes require rerunning run.py."""
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text())
    dpi = int(summary["dpi"] if dpi is None else dpi)
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    with plt.rc_context(STYLE):
        paths = [plot_power_balance(read_table(folder / "power_balance.csv"), summary, folder, dpi)]
        if (folder / "selected_absorption.csv").is_file():
            paths.append(plot_absorption_selection(read_table(folder / "selected_absorption.csv"),
                                                   summary, folder, dpi))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Folder with previously calculated results")
    parser.add_argument("--dpi", type=int, help="PNG resolution; default: value saved by the run")
    args = parser.parse_args()
    for path in plot_results(args.results, args.dpi):
        print(path)


if __name__ == "__main__":
    main()
