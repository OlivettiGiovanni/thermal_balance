"""Plot an in-memory result or replot an output folder without rerunning the physics."""
from pathlib import Path
import argparse
from textwrap import fill
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = {"abs": "#153f69", "linear": "#777777", "tpa": "#d47e20", "fca": "#9463ad",
          "bga": "#38946c", "gas": "#c84b50", "bb": "#2493a5"}


def draw_positive(ax, x, y, break_before=None, **style):
    """Hide nonpositive values and separate changes in the optical-data regime."""
    y = np.asarray(y)
    y = np.where(np.isfinite(y) & (y > 0), y, np.nan)
    if break_before is not None:
        breaks = np.asarray(break_before, dtype=bool)
        if breaks.shape != y.shape:
            raise ValueError("Plot break mask must match the plotted values")
        indices = np.flatnonzero(breaks)
        # Insert separators, preserving both values on either side of a fallback.
        x = np.insert(np.asarray(x, dtype=float), indices, np.nan)
        y = np.insert(y, indices, np.nan)
    return ax.plot(x, y, **style)


def format_axes(ax, temperature, ylabel):
    """Keep labels short and the same style on both figures."""
    ax.set(xlabel="Particle temperature (K)", ylabel=ylabel, yscale="log",
           xlim=(temperature[0], temperature[-1]))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(which="major", color="#dbe0e6", linewidth=.6)


def radiation_footnote(fig, results):
    """Reserve space below the axes for optical-data limitations, when present."""
    notes = [str(results.get("radiation_footnote", "")).strip()]
    if bool(results.get("other_reference_used", False)):
        notes.append("* other reference used")
    note = " · ".join(text for text in notes if text)
    if note:
        fig.supxlabel(fill(note, width=110), x=.08, ha="left", fontsize=8, color="#555555")



def benchmark_power_lines(ax, results):
    """Plot the benchmark; all legend names and colors are listed below."""
    t = results["temperature_k"]
    t_junn = results["reference_temperature_k"]
    t_franta = results["franta_temperature_k"]

    # Each row: data key, LEGEND NAME, COLOR, temperatures, line style, marker.
    curves = [
        ("p_abs_reference_w", "$P_{\mathrm{abs}}^{\mathrm{tot}}(T)$ Junn.", "#f5bc00", t_junn, "--", None),
        ("p_bb_reference_w", "$P_{\mathrm{BB}}(T)$ Junn.", "#8a0ebb", t_junn, "--", None),
        ("p_abs_w", "$P_{\mathrm{abs}}^{\mathrm{tot}}(T)$", "#ec6404", t, "-", None),
        ("p_gas_w", "$P_{\mathrm{gas}}$", "#3de30fbc", t, "-", None),
        ("p_bb_franta_w", "$P_{\mathrm{BB}}(T)$ Franta", "#202020", t_franta, "none", "o"),
        ("p_bb_dipole_room_w", "$P_{\mathrm{BB}}(T_0)$ dipole", "#0EB7F5", t, "-", None),
        ("p_bb_dipole_w", "$P_{\mathrm{BB}}(T)$ dipole", "#050bc1", t, "-", None),
        ("p_bb_mie_room_w", "$P_{\mathrm{BB}}(T_0)$ Mie", "#ff8f8f", t, "-", None),
        ("p_bb_mie_w", "$P_{\mathrm{BB}}(T)$ Mie", "#fe0505ff", t, "-", None),
        ]

    for key, label, color, temperature, style, marker in curves:
        if key not in results:  # Older saved results may lack hot Mie.
            continue

        draw_positive(
            ax, temperature, results[key],
            label=label,
            color=color,
            linestyle=style,
            marker=marker,
            linewidth=1.8,
            markersize=4.5,
            zorder=5 if marker else 2,
        )


def power_plot(results, destination, dpi):
    """One power plot, with individual heating contributions for the cylinder calculation."""
    fig, ax = plt.subplots(figsize=(8, 5.3), layout="constrained")
    temperature = results["temperature_k"]
    benchmark = str(results.get("kind", "cylinder")) == "junnemann"
    compare_bb = benchmark and "p_bb_dipole_w" in results
    fitted_bb = benchmark and "p_bb_dipole_room_w" in results
    if fitted_bb:
        benchmark_power_lines(ax, results)
    else:  # Older saved results and cylinder runs keep their original available curves.
        for name, label in [("abs", "abs"), ("gas", "gas"), ("bb", "BB Mie" if benchmark else "BB")]:
            if name == "bb" and compare_bb:
                draw_positive(ax, temperature, results["p_bb_dipole_w"], color=COLORS["bb"],
                              linewidth=1.8, linestyle="-", label="BB dipole",
                              break_before=results.get("radiation_break_before"))
            draw_positive(ax, temperature, results[f"p_{name}_w"], color=COLORS[name],
                          linewidth=2.2 if name == "abs" else 1.8, label=label,
                          linestyle="--" if name == "bb" and compare_bb else "-",
                          break_before=results.get("radiation_break_before") if name == "bb" else None)
    if benchmark and not fitted_bb:
        paper_t = results["reference_temperature_k"][::3]
        for key, label, color in [("abs", "abs Junn.", "#d98627"), ("bb", "BB Junn.", COLORS["bb"])]:
            draw_positive(ax, paper_t, results[f"p_{key}_reference_w"][::3],
                          color=color, marker="o", markersize=3, linestyle="none", label=label)
    elif not benchmark:
        for name, label in [("linear", "Linear"), ("tpa", "TPA"), ("fca", "FCA"), ("bga", "BGA")]:
            if np.any(results[f"p_{name}_w"] > 0):
                draw_positive(ax, temperature, results[f"p_{name}_w"], color=COLORS[name],
                              linewidth=1.4, linestyle="--", label=label)
        if "p_optical_estimate_w" in results:
            draw_positive(ax, temperature, results["p_optical_estimate_w"], color="black",
                          linewidth=1.4, linestyle=":", label="k(T) estimate")
    format_axes(ax, temperature, "Power (W)")
    if not benchmark:
        largest = max((float(np.nanmax(value)) for key, value in results.items()
                       if key.startswith("p_") and key.endswith("_w") and np.any(np.isfinite(value))), default=0.)
        heating = float(np.max(results["p_abs_w"]))
        if largest > 0:
            ax.set_ylim((heating if heating > 0 else largest)*1e-7, largest*2)
    ax.legend(loc="lower right", ncol=3, fontsize=9, framealpha=.93)
    radiation_footnote(fig, results)
    fig.savefig(destination, dpi=dpi)
    plt.close(fig)


def coefficient_plot(results, destination, dpi):
    """Legend rows are: mechanism dots, first model solid, second model dashed."""
    fig, ax = plt.subplots(figsize=(8.4, 5.5), layout="constrained")
    temperature = results["temperature_k"]
    rows = []
    models = {
        "tpa": [("tpa_reference_model_m_inv", "Sinclair (semi-empirical)", "-"),
                ("tpa_model_m_inv", "Physical pairing", "--")],
        "fca": [("fca_reference_model_m_inv", "Sturm (empirical)", "-"),
                ("fca_model_m_inv", "Drude", "--")],
        "bga": [("bga_model_m_inv", "Sturm (semi-empirical)", "-")],
    }
    for mechanism, alternatives in models.items():
        color = COLORS[mechanism]
        draw_positive(ax, results["coefficient_reference_temperature_k"][::3],
                      results[f"{mechanism}_digitized_m_inv"][::3], color=color,
                      marker="o", markersize=3, linestyle="none")
        handles = [Line2D([], [], color=color, marker="o", linestyle="none", markersize=4,
                          label=mechanism.upper()+" Junn.")]
        for column, label, style in alternatives:
            draw_positive(ax, temperature, results[column], color=color, linewidth=1.8, linestyle=style)
            handles.append(Line2D([], [], color=color, linestyle=style, label=label))
        rows.append(handles+[Line2D([], [], color="none", label=" ")]*(3-len(handles)))
    format_axes(ax, temperature, "Absorption coefficient (m⁻¹)")
    ax.set_ylim(1e-3, 1e7)
    ax.legend(handles=[rows[r][c] for c in range(3) for r in range(3)],
              ncol=3, loc="upper left", fontsize=8, columnspacing=1, framealpha=.94)
    fig.savefig(destination, dpi=dpi)
    plt.close(fig)


def plot_results(results, output, dpi=200):
    """Output directory and PNG resolution are chosen by the caller."""
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 11, "axes.labelsize": 12, "savefig.facecolor": "white"}):
        power_plot(results, output/"power_balance.png", dpi)
        if str(results.get("kind", "cylinder")) == "junnemann":
            coefficient_plot(results, output/"absorption_comparison.png", dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Folder containing results.npz")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    try:
        with np.load(args.output/"results.npz", allow_pickle=False) as saved:
            plot_results(dict(saved), args.output, args.dpi)
        print(f"Replotted {args.output.resolve()}")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
